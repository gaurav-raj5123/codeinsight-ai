"""Deterministic Citation Resolution, Validation & Fabrication Checks (Sections 13, 14, 16).

Strictly guarantees:
  - Evidence IDs (`[EVIDENCE_1]`, etc.) map deterministically from retrieved chunk metadata.
  - Invalid IDs, unretrieved IDs, duplicate IDs, and missing/invalid metadata are detected
    and flagged cleanly without ever turning an invalid citation into a valid-looking link.
  - Original raw text / parsed JSON is never overwritten by G4 post-processing.
"""

from __future__ import annotations

import os
import re
from typing import Any, Sequence

_BRACKET_GROUP_RE = re.compile(r"\[([^\[\]]*EVIDENCE_[^\[\]]+)\]", re.IGNORECASE)
_EVIDENCE_TOKEN_RE = re.compile(r"\b(EVIDENCE_(\d+))\b", re.IGNORECASE)
_VALID_ID_EXACT_RE = re.compile(r"^EVIDENCE_([1-9]\d*)$")

# Heuristic detector for prose mentioning code file paths / filenames with supported code extensions
_CODE_FILE_MENTION_RE = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.(?:py|js|ts|java|cpp|cc|c|h|hpp)|"
    r"`([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.(?:py|js|ts|java|cpp|cc|c|h|hpp))`)"
)
_LINE_MENTION_RE = re.compile(r"\b[Ll]ines?\s+(\d{1,5})(?:\s*[-–—to]+\s*(\d{1,5}))?\b")


def _is_valid_int_line(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and val >= 1


def build_evidence_map(retrieved_chunks: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build deterministic mapping `{ 'EVIDENCE_1': {...metadata...} }` from retrieved chunks."""
    evidence_map: dict[str, dict[str, Any]] = {}
    for idx, ch in enumerate(retrieved_chunks, start=1):
        eid = f"EVIDENCE_{idx}"
        meta = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
        file_path = str(meta.get("file_path") or ch.get("file_path") or "").strip().replace("\\", "/")
        symbol_name = str(meta.get("symbol_name") or ch.get("symbol_name") or "").strip()
        class_name = str(meta.get("class_name") or "").strip()
        raw_start = meta.get("start_line", ch.get("start_line"))
        raw_end = meta.get("end_line", ch.get("end_line"))
        chunk_id = str(ch.get("chunk_id") or ch.get("id") or meta.get("chunk_id") or "").strip()

        warnings: list[str] = []
        has_valid_file = bool(file_path) and file_path.lower() not in ("unknown", "none", "null")
        if not has_valid_file:
            warnings.append("missing_file_path")

        has_valid_lines = (
            _is_valid_int_line(raw_start)
            and _is_valid_int_line(raw_end)
            and raw_end >= raw_start
        )
        if not has_valid_lines:
            warnings.append("invalid_or_missing_line_range")

        start_line = int(raw_start) if has_valid_lines else None
        end_line = int(raw_end) if has_valid_lines else None

        if has_valid_file and has_valid_lines:
            rendered_ref = f"`{file_path}:{start_line}–{end_line}`"
            display_label = f"{file_path}:{start_line}–{end_line}"
        elif has_valid_file:
            rendered_ref = f"`{file_path}`"
            display_label = file_path
        else:
            rendered_ref = None
            display_label = None

        evidence_map[eid] = {
            "evidence_id": eid,
            "rank": idx,
            "chunk_id": chunk_id,
            "file_path": file_path if has_valid_file else "",
            "symbol_name": symbol_name,
            "class_name": class_name,
            "start_line": start_line,
            "end_line": end_line,
            "has_valid_file": has_valid_file,
            "has_valid_lines": has_valid_lines,
            "metadata_warnings": warnings,
            "rendered_ref": rendered_ref,
            "display_label": display_label,
        }
    return evidence_map


def extract_inline_evidence_ids(text: str) -> list[str]:
    """Extract evidence ID references (`EVIDENCE_1`, `[EVIDENCE_0]`, etc.) in appearance order."""
    if not isinstance(text, str) or not text:
        return []
    found: list[str] = []
    # Match both bracketed `[EVIDENCE_1, EVIDENCE_2]` and unbracketed `EVIDENCE_1`
    for m in _EVIDENCE_TOKEN_RE.finditer(text):
        found.append(m.group(1).upper())
    return found


def detect_unretrieved_file_mentions(
    answer_text: str,
    retrieved_chunks: Sequence[dict[str, Any]],
) -> list[str]:
    """Detect source file paths mentioned in `answer_text` that do not appear in retrieved metadata.

    Conservative heuristic for Layer 1 audit telemetry (does not assert hallucination alone,
    especially if the answer is explaining what is missing during abstention).
    """
    if not isinstance(answer_text, str) or not answer_text:
        return []

    retrieved_paths: set[str] = set()
    retrieved_basenames: set[str] = set()
    for ch in retrieved_chunks:
        meta = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
        fp = str(meta.get("file_path") or ch.get("file_path") or "").strip().replace("\\", "/").lower()
        if fp and fp not in ("unknown", "none"):
            retrieved_paths.add(fp)
            retrieved_basenames.add(os.path.basename(fp))

    unretrieved: list[str] = []
    seen: set[str] = set()
    for m in _CODE_FILE_MENTION_RE.finditer(answer_text):
        raw_path = (m.group(2) or m.group(1) or "").strip().replace("\\", "/")
        norm = raw_path.lstrip("./").lower()
        if not norm or norm in seen:
            continue
        matched = (
            norm in retrieved_paths
            or any(rp.endswith("/" + norm) or norm.endswith("/" + rp) for rp in retrieved_paths)
            or ("/" not in norm and norm in retrieved_basenames)
        )
        if not matched:
            seen.add(norm)
            unretrieved.append(raw_path)
    return unretrieved


def detect_unretrieved_line_mentions(
    answer_text: str,
    retrieved_chunks: Sequence[dict[str, Any]],
) -> list[str]:
    """Heuristic check for line numbers (`Lines A-B` / `line X`) absent from retrieved chunk line ranges."""
    if not isinstance(answer_text, str) or not answer_text:
        return []
    ranges: list[tuple[int, int]] = []
    for ch in retrieved_chunks:
        meta = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
        s = meta.get("start_line", ch.get("start_line"))
        e = meta.get("end_line", ch.get("end_line"))
        if _is_valid_int_line(s) and _is_valid_int_line(e) and e >= s:
            ranges.append((s, e))

    if not ranges:
        return []

    suspicious: list[str] = []
    for m in _LINE_MENTION_RE.finditer(answer_text):
        s_val = int(m.group(1))
        e_val = int(m.group(2)) if m.group(2) else s_val
        lo, hi = min(s_val, e_val), max(s_val, e_val)
        covered = any(r_s <= lo and hi <= r_e for r_s, r_e in ranges)
        if not covered:
            suspicious.append(m.group(0))
    return suspicious


def _render_single_eid(eid_norm: str, evidence_map: dict[str, dict[str, Any]]) -> str:
    entry = evidence_map.get(eid_norm)
    if entry is None:
        return f"[UNVERIFIED_CITATION: {eid_norm}]"
    if not entry["has_valid_file"]:
        return f"[{eid_norm} (metadata unavailable)]"
    return f"[{eid_norm} ({entry['display_label']})]"


def validate_and_render_citations(
    raw_answer: str,
    declared_citations: Sequence[Any] | None,
    retrieved_chunks: Sequence[dict[str, Any]],
    abstained: bool = False,
) -> dict[str, Any]:
    """Deterministically validate all cited IDs against `retrieved_chunks` and render readable citations.

    Never promotes invalid evidence IDs or incomplete metadata into valid-looking links.
    """
    evidence_map = build_evidence_map(retrieved_chunks)
    inline_tokens = extract_inline_evidence_ids(raw_answer)

    declared_raw_list = list(declared_citations) if isinstance(declared_citations, (list, tuple)) else []
    normalized_raw_citations: list[str] = []
    for item in declared_raw_list:
        s = str(item).strip().strip("[]").strip()
        if not s:
            continue
        # Handle comma-separated entries inside declared list defensively
        if "," in s:
            for sub in s.split(","):
                sub_clean = sub.strip().strip("[]").strip()
                if sub_clean:
                    normalized_raw_citations.append(sub_clean)
        else:
            normalized_raw_citations.append(s)

    combined_sequence = normalized_raw_citations + inline_tokens

    valid_citation_ids: list[str] = []
    seen_valid: set[str] = set()
    invalid_citation_ids: list[str] = []
    duplicate_citation_ids: list[str] = []
    metadata_warnings: list[str] = []

    # Track duplicates within declared_citations and inline tokens separately for accuracy
    def _classify_stream(tokens: list[str], track_dups: bool) -> None:
        seen_stream: set[str] = set()
        for tok in tokens:
            up = tok.upper()
            if track_dups and up in seen_stream:
                duplicate_citation_ids.append(tok)
            seen_stream.add(up)

            if _VALID_ID_EXACT_RE.match(up) and up in evidence_map:
                entry = evidence_map[up]
                if not entry["has_valid_file"]:
                    if up not in invalid_citation_ids:
                        invalid_citation_ids.append(tok)
                    metadata_warnings.extend(f"{up}:{w}" for w in entry["metadata_warnings"])
                else:
                    if entry["metadata_warnings"]:
                        metadata_warnings.extend(f"{up}:{w}" for w in entry["metadata_warnings"])
                    if up not in seen_valid:
                        seen_valid.add(up)
                        valid_citation_ids.append(up)
            else:
                if tok not in invalid_citation_ids:
                    invalid_citation_ids.append(tok)

    _classify_stream(normalized_raw_citations, track_dups=True)
    _classify_stream(inline_tokens, track_dups=False)

    # Render inline `[EVIDENCE_1]` / `[EVIDENCE_1, EVIDENCE_2]` / bare `EVIDENCE_1` safely
    answer_text = raw_answer if isinstance(raw_answer, str) else str(raw_answer or "")

    def _replace_bracket_group(match: re.Match[str]) -> str:
        inner = match.group(1)
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        # If every comma-segment is an EVIDENCE_ token, replace cleanly
        if parts and all(_EVIDENCE_TOKEN_RE.fullmatch(p) for p in parts):
            rendered_parts = []
            seen_local: set[str] = set()
            for p in parts:
                up = p.upper()
                if up in seen_local:
                    continue
                seen_local.add(up)
                rendered_parts.append(_render_single_eid(up, evidence_map))
            return " ".join(rendered_parts)
        return _EVIDENCE_TOKEN_RE.sub(
            lambda em: _render_single_eid(em.group(1).upper(), evidence_map),
            match.group(0),
        )

    rendered_body = _BRACKET_GROUP_RE.sub(_replace_bracket_group, answer_text)
    # Catch any remaining standalone `EVIDENCE_k` outside brackets (not already rewritten inside ` (`)
    # Only replace tokens not preceded by `[`
    rendered_body = re.sub(
        r"(?<!\[)\b(EVIDENCE_\d+)\b(?!\s*\()",
        lambda em: _render_single_eid(em.group(1).upper(), evidence_map),
        rendered_body,
        flags=re.IGNORECASE,
    )

    resolved_sources: list[dict[str, Any]] = []
    for eid in valid_citation_ids:
        entry = evidence_map[eid]
        resolved_sources.append(
            {
                "evidence_id": eid,
                "rank": entry["rank"],
                "chunk_id": entry["chunk_id"],
                "file_path": entry["file_path"],
                "symbol_name": entry["symbol_name"],
                "start_line": entry["start_line"],
                "end_line": entry["end_line"],
                "rendered_ref": entry["rendered_ref"],
                "display_label": entry["display_label"],
                "metadata_warnings": entry["metadata_warnings"],
            }
        )

    if resolved_sources:
        source_lines = []
        for src in resolved_sources:
            sym_note = f" — `{src['symbol_name']}`" if src["symbol_name"] else ""
            source_lines.append(f"- **[{src['evidence_id']}]** {src['rendered_ref']}{sym_note}")
        rendered_full = f"{rendered_body.rstrip()}\n\n**Sources:**\n" + "\n".join(source_lines)
    else:
        rendered_full = rendered_body.rstrip()

    total_unique_refs = len(valid_citation_ids) + len(invalid_citation_ids)
    citation_validity_rate = (
        round(len(valid_citation_ids) / total_unique_refs, 4) if total_unique_refs > 0 else None
    )

    if invalid_citation_ids:
        validation_status = "invalid_citations_detected"
    elif metadata_warnings:
        validation_status = "valid_with_metadata_warnings"
    elif valid_citation_ids:
        validation_status = "valid"
    elif abstained:
        validation_status = "valid_abstention_no_citations"
    else:
        validation_status = "valid_no_citations"

    unretrieved_files = detect_unretrieved_file_mentions(raw_answer, retrieved_chunks)

    return {
        "evidence_map": evidence_map,
        "all_cited_ids_raw": combined_sequence,
        "declared_citation_ids": normalized_raw_citations,
        "inline_citation_ids": inline_tokens,
        "valid_citation_ids": valid_citation_ids,
        "invalid_citation_ids": invalid_citation_ids,
        "duplicate_citation_ids": list(dict.fromkeys(duplicate_citation_ids)),
        "metadata_warnings": list(dict.fromkeys(metadata_warnings)),
        "resolved_sources": resolved_sources,
        "has_valid_citations": len(valid_citation_ids) > 0,
        "has_invalid_citations": len(invalid_citation_ids) > 0,
        "citation_validity_rate": citation_validity_rate,
        "unretrieved_file_mentions": unretrieved_files,
        "validation_status": validation_status,
        "rendered_body": rendered_body,
        "rendered_answer": rendered_full,
    }
