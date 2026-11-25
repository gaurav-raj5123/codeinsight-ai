"""Code-Aware Okapi BM25 Lexical Retriever (Day 4).

Key design guarantees:
1. V1 Header Label Hygiene: Strips only static V1 contextual field labels
   ('File:', 'Language:', 'Type:', 'Class:', 'Parent:', 'Symbol:', 'Part:',
   'Signature:') on V1 header lines while keeping all field values untouched
   so generic query words ('file', 'type', 'class') never match every chunk.
2. Code-Aware Tokenization: Preserves exact dotted/scoped/path/camelCase/
   snake_case code identifiers alongside their split components, retaining
   1-2 character code symbols ('g', 'ip', 'os', 'fg', 'bg'), and applying the
   exact same tokenizer to corpus documents and queries.
3. Zero-Overlap Eligibility Filter: Eligibility is determined strictly by
   actual query-token lexical overlap (`overlap_count >= 1`), never by `score > 0`,
   guaranteeing zero-overlap documents never enter candidates or pad Top-K.
4. Direct ChromaDB Hydration: Supports rebuilding synchronized BM25 indices
   directly from persisted ChromaDB collections without pickle files.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Sequence

V1_HEADER_LABELS = (
    "File:",
    "Language:",
    "Type:",
    "Class:",
    "Parent:",
    "Symbol:",
    "Part:",
    "Signature:",
)

_COMPOUND_TOKEN_RE = re.compile(
    r"[A-Za-z0-9_]+(?:(?:::|\.|/|\\|-)[A-Za-z0-9_]+)*"
)
_CAMEL_SPLIT_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)


def strip_v1_header_labels(text: str) -> str:
    """Remove static V1 contextual header labels on leading header lines only.

    Matches only the structured V1 header preamble:
      Line 0: File: <path> | Language: <lang> | Type: <chunk_type>
      Line 1 (optional): Class/Parent/Symbol/Part fields separated by ' | '
      Line 2 (optional): Signature: <sig>
    All field values and source body lines remain intact.
    """
    if not text:
        return text

    lines = text.split("\n")
    first = lines[0]
    if not (
        first.startswith("File: ")
        and " | Language: " in first
        and " | Type: " in first
    ):
        return text

    cleaned_header: list[str] = []
    # Strip Line 0 labels cleanly
    line0_parts = [re.sub(r"^(?:File|Language|Type):\s*", "", seg.strip()) for seg in first.split(" | ")]
    cleaned_header.append(" ".join(p for p in line0_parts if p))
    idx = 1

    # Strip optional Line 1 metadata details
    if idx < len(lines) and any(
        lines[idx].startswith(lbl) or f" | {lbl}" in lines[idx]
        for lbl in ("Class: ", "Parent: ", "Symbol: ", "Part: ")
    ):
        line1_parts = [
            re.sub(r"^(?:Class|Parent|Symbol|Part):\s*", "", seg.strip())
            for seg in lines[idx].split(" | ")
        ]
        cleaned_header.append(" ".join(p for p in line1_parts if p))
        idx += 1

    # Strip optional Line 2 signature prefix
    if idx < len(lines) and lines[idx].startswith("Signature: "):
        cleaned_header.append(lines[idx][len("Signature: ") :])
        idx += 1

    return "\n".join(cleaned_header + lines[idx:])


def _split_identifier_pieces(identifier: str) -> list[str]:
    """Split a single identifier by underscores and camelCase/acronym boundaries."""
    subparts: list[str] = []
    for piece in identifier.split("_"):
        if not piece:
            continue
        segments = _CAMEL_SPLIT_RE.findall(piece)
        if segments:
            for seg in segments:
                subparts.append(seg.lower())
        else:
            subparts.append(piece.lower())
    return subparts


def tokenize_code(text: str, strip_v1_labels: bool = True) -> list[str]:
    """Lightweight, deterministic code-aware tokenizer shared by indexing and search.

    Rules documented in DAY4_HYBRID_REPORT.md:
      - Strips static V1 header labels ('File:', 'Language:', 'Type:', etc.) if present.
      - Normalizes Windows backslashes in path-like tokens to forward slashes.
      - Preserves compound tokens lowercased ('res.cookie', 'fmt::format',
        'include/fmt/os.h', 'createapplication', 'handle_request', 'fmt_string').
      - Retains slash path components ('os.h') AND dotted/scoped/hyphenated
        sub-segments plus camelCase/snake_case sub-words without discarding
        1-2 character code identifiers ('g', 'ip', 'os', 'fg', 'bg').
    """
    if not text:
        return []

    processed = strip_v1_header_labels(text) if strip_v1_labels else text
    tokens: list[str] = []

    for raw_compound in _COMPOUND_TOKEN_RE.findall(processed):
        norm_compound = raw_compound.replace("\\", "/").lower()
        emitted_local: set[str] = set()

        def _emit(tok: str) -> None:
            if tok and tok not in emitted_local:
                emitted_local.add(tok)
                tokens.append(tok)

        _emit(norm_compound)

        # Preserve path segments (e.g., 'os.h' in 'include/fmt/os.h') before dot/colon splits
        slash_parts = raw_compound.replace("\\", "/").split("/")
        for s_part in slash_parts:
            if not s_part:
                continue
            _emit(s_part.lower())
            # Split scoped/dotted/hyphenated identifiers (e.g., Config.from_object, fmt::format)
            for leaf in re.split(r"::|\.|-" , s_part):
                if not leaf:
                    continue
                _emit(leaf.lower())
                for sub in _split_identifier_pieces(leaf):
                    _emit(sub)

    return tokens


def compute_corpus_fingerprint(
    chunk_ids: Sequence[str],
    documents: Sequence[str],
) -> str:
    """Compute a deterministic SHA-256 fingerprint over sorted (chunk_id, doc_sha256) pairs."""
    pairs = sorted(
        (str(cid), hashlib.sha256(doc.encode("utf-8")).hexdigest()[:16])
        for cid, doc in zip(chunk_ids, documents)
    )
    payload = "|".join(f"{cid}:{h}" for cid, h in pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class BM25Retriever:
    """Inverted-index Okapi BM25 retriever with zero-overlap exclusion guarantees."""

    def __init__(
        self,
        chunk_ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        idf_mode: str = "lucene",
        strip_v1_labels: bool = True,
    ) -> None:
        if not (len(chunk_ids) == len(documents) == len(metadatas)):
            raise ValueError(
                f"Mismatched lengths: ids={len(chunk_ids)}, docs={len(documents)}, metas={len(metadatas)}"
            )
        if idf_mode not in ("lucene", "raw_rsj"):
            raise ValueError(f"Unsupported idf_mode {idf_mode!r}; expected 'lucene' or 'raw_rsj'")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Duplicate chunk_ids provided to BM25Retriever")

        self.chunk_ids = [str(cid) for cid in chunk_ids]
        self.documents = list(documents)
        self.metadatas = [dict(m) for m in metadatas]
        self.k1 = float(k1)
        self.b = float(b)
        self.idf_mode = idf_mode
        self.strip_v1_labels = bool(strip_v1_labels)

        self.num_docs = len(self.documents)
        self.corpus_fingerprint = compute_corpus_fingerprint(self.chunk_ids, self.documents)

        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_freq: dict[str, int] = {}

        for doc_idx, doc_text in enumerate(self.documents):
            toks = tokenize_code(doc_text, strip_v1_labels=self.strip_v1_labels)
            self.doc_lengths.append(len(toks))
            counts = Counter(toks)
            for term, tf in counts.items():
                self.postings.setdefault(term, []).append((doc_idx, tf))
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self.avgdl = (sum(self.doc_lengths) / self.num_docs) if self.num_docs > 0 else 0.0
        self.idf: dict[str, float] = {}
        for term, df in self.doc_freq.items():
            if self.idf_mode == "lucene":
                # Strictly positive BM25 IDF (prevents negative weights when df > N/2)
                self.idf[term] = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))
            else:
                # Raw Robertson-Sparck Jones IDF (can produce zero/negative values on high df)
                self.idf[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 20,
        return_metadata: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        """Retrieve up to top_k lexically overlapping documents ranked by BM25 score.

        Documents with zero query-token overlap are strictly excluded. Never pads
        results with zero-overlap chunks; returns [] if no lexical matches exist.
        """
        query_tokens = tokenize_code(query, strip_v1_labels=self.strip_v1_labels)
        unique_q_counts = Counter(query_tokens)

        if self.num_docs == 0 or top_k <= 0 or not unique_q_counts:
            meta = {
                "query_tokens": query_tokens,
                "eligible_overlap_docs": 0,
                "returned_count": 0,
                "idf_mode": self.idf_mode,
            }
            return ([], meta) if return_metadata else []

        scores: dict[int, float] = {}
        matched_terms: dict[int, set[str]] = {}
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        for q_term, q_tf in unique_q_counts.items():
            term_postings = self.postings.get(q_term)
            if not term_postings:
                continue
            idf_val = self.idf[q_term]
            for doc_idx, tf in term_postings:
                dl = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                contrib = idf_val * ((tf * (self.k1 + 1.0)) / denom) * q_tf
                scores[doc_idx] = scores.get(doc_idx, 0.0) + contrib
                matched_terms.setdefault(doc_idx, set()).add(q_term)

        # Eligibility is determined strictly by actual lexical overlap (`matched_terms`), NOT `score > 0`
        eligible_indices = list(matched_terms.keys())
        eligible_indices.sort(
            key=lambda idx: (
                -scores[idx],
                -len(matched_terms[idx]),
                self.chunk_ids[idx],
            )
        )

        selected = eligible_indices[:top_k]
        results: list[dict[str, Any]] = []
        for rank_i, doc_idx in enumerate(selected, start=1):
            results.append(
                {
                    "id": self.chunk_ids[doc_idx],
                    "content": self.documents[doc_idx],
                    "metadata": self.metadatas[doc_idx],
                    "bm25_score": round(float(scores[doc_idx]), 6),
                    "bm25_rank": rank_i,
                    "overlap_count": len(matched_terms[doc_idx]),
                    "matched_terms": sorted(matched_terms[doc_idx]),
                }
            )

        meta = {
            "query_tokens": query_tokens,
            "eligible_overlap_docs": len(eligible_indices),
            "returned_count": len(results),
            "idf_mode": self.idf_mode,
        }
        return (results, meta) if return_metadata else results


def build_bm25_from_collection(
    collection: Any,
    k1: float = 1.5,
    b: float = 0.75,
    idf_mode: str = "lucene",
) -> tuple[BM25Retriever, str]:
    """Hydrate a synchronized BM25Retriever directly from a ChromaDB collection."""
    raw = collection.get(include=["documents", "metadatas"])
    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    # Guard against accidental duplicate IDs or malformed rows in persisted stores
    seen: set[str] = set()
    clean_ids: list[str] = []
    clean_docs: list[str] = []
    clean_metas: list[dict[str, Any]] = []
    for cid, doc, meta in zip(ids, docs, metas):
        s_cid = str(cid)
        if s_cid in seen:
            continue
        seen.add(s_cid)
        clean_ids.append(s_cid)
        clean_docs.append(doc if isinstance(doc, str) else "")
        clean_metas.append(meta if isinstance(meta, dict) else {})

    retriever = BM25Retriever(
        chunk_ids=clean_ids,
        documents=clean_docs,
        metadatas=clean_metas,
        k1=k1,
        b=b,
        idf_mode=idf_mode,
        strip_v1_labels=True,
    )
    return retriever, retriever.corpus_fingerprint
