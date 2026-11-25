#!/usr/bin/env python3
"""
Day 3: Controlled Chunking Optimization Benchmark Runner.

Evaluates three chunking strategies under strict experimental control:
  C0: fixed_window_v0     (50 lines, 10-line overlap sliding window)
  C1: existing_chunker_v0 (unmodified V0 hybrid AST/regex/sliding-window chunker)
  C2: code_aware_v1       (token-budgeted AST + lexical brace-scanner chunker)

Fixed experimental constants (identical to Day 1/Day 2 V0):
  - Repositories: pallets/flask, expressjs/express, fmtlib/fmt (pinned commit SHAs)
  - Corpus: identical 253 filtered source files across all strategies
  - Embedding model: ONNX all-MiniLM-L6-v2 (mean pooling, 384d, L2-normalized, max_seq_length=256)
  - Vector DB: ChromaDB PersistentClient, hnsw:space="l2" (squared L2 distance), Top-K=5
  - Queries: 60 evaluation queries (54 answerable + 6 unanswerable), raw "question" strings submitted
  - Metrics: evaluation/retrieval_metrics.py + full 24 chunk quality metrics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import chromadb
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chunker import SUPPORTED_CHUNKING_STRATEGIES, chunk_code
from chunker_v1 import count_tokens
from evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    build_chunks_by_file_index,
    calc_distance_distribution,
    check_overlap,
    compute_latency_stats,
    compute_query_metrics,
    count_relevant_chunks_in_repo,
)
from experiments.embedding_experiments.run_embedding_comparison import (
    CONFIG_FILE as EMB_CONFIG_FILE,
    DirectONNXEmbedder,
    compute_breakdowns,
    get_file_sha256,
    get_rss_mb,
    run_preflight_checks,
)
from repo_loader import get_source_files

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EXP_DIR, "results")
PER_STRATEGY_DIR = os.path.join(OUT_DIR, "per_strategy")
CHROMA_DB_DIR = os.path.join(EXP_DIR, "chroma_day3_db")

DATASETS_DIR = os.path.join(PROJECT_ROOT, "experiments", "datasets")
CACHE_DIR = os.path.join(DATASETS_DIR, "cache")
REPOS_FILE = os.path.join(DATASETS_DIR, "repositories.json")
QUESTIONS_FILE = os.path.join(DATASETS_DIR, "evaluation_questions.json")
GROUND_TRUTH_FILE = os.path.join(DATASETS_DIR, "ground_truth.json")
DAY1_SUMMARY_FILE = os.path.join(PROJECT_ROOT, "experiments", "results", "baseline_summary.json")

TOP_K = 5
BATCH_SIZE = 32
CHROMA_INSERT_BATCH = 100

STRATEGIES = [
    {
        "strategy_id": "fixed_window_v0",
        "label": "C0: Fixed-Window Baseline (50L / 10L overlap)",
        "role": "Control C0",
    },
    {
        "strategy_id": "existing_chunker_v0",
        "label": "C1: Existing V0 Chunker",
        "role": "Control C1 (Production V0 Baseline)",
    },
    {
        "strategy_id": "code_aware_v1",
        "label": "C2: Code-Aware V1 Chunker (<=256 Tokens)",
        "role": "Experimental V1",
    },
]


def make_day3_collection_name(strategy_id: str, repo_alias: str, run_short: str) -> str:
    """Create a fresh <=63-char ChromaDB collection name isolated by strategy, repo, and run."""
    raw = f"d3_{strategy_id}_{repo_alias}_{run_short}"
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in raw)[:63]
    if len(safe) < 3:
        safe = safe.ljust(3, "_")
    return safe


def validate_chroma_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Verify every chunk metadata value is a primitive type accepted by ChromaDB."""
    clean: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            raise TypeError(
                f"Invalid Chroma metadata type for key {k!r}: {type(v).__name__} ({v!r})"
            )
    return clean


def load_source_corpus(repos_data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Load identical deterministic filtered source files for every benchmark repository."""
    files_by_repo: dict[str, list[dict[str, Any]]] = {}
    for r_info in repos_data:
        r_name = r_info["name"]
        alias = r_name.split("/")[-1]
        repo_dir = os.path.join(CACHE_DIR, alias)
        sfs = sorted(get_source_files(repo_dir), key=lambda f: f["file_path"])
        files_by_repo[r_name] = sfs
    return files_by_repo


def build_corpus_manifest(files_by_repo: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Build cryptographic file manifest to prove identical input files across all strategies."""
    manifest: dict[str, Any] = {"total_files": 0, "repositories": {}}
    for r_name, sfs in files_by_repo.items():
        items = [
            {
                "file_path": sf["file_path"],
                "extension": sf["extension"],
                "lines": len(sf["content"].splitlines()),
                "sha256": hashlib.sha256(sf["content"].encode("utf-8")).hexdigest()[:16],
            }
            for sf in sfs
        ]
        manifest["total_files"] += len(items)
        manifest["repositories"][r_name] = {
            "file_count": len(items),
            "manifest_sha256": hashlib.sha256(
                json.dumps(items, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
    return manifest


def _summary_stats(values: list[int | float], include_p95: bool = False) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    s_vals = sorted(values)
    out: dict[str, Any] = {
        "count": len(values),
        "mean": round(float(statistics.mean(values)), 2),
        "median": round(float(statistics.median(values)), 2),
        "min": s_vals[0],
        "max": s_vals[-1],
    }
    if include_p95:
        idx = max(0, min(int(0.95 * len(s_vals)), len(s_vals) - 1))
        out["p95"] = s_vals[idx]
    return out


def compute_corpus_chunk_metrics(
    strategy_id: str,
    chunks_by_repo: dict[str, list[dict[str, Any]]],
    files_by_repo: dict[str, list[dict[str, Any]]],
    ground_truth: list[dict[str, Any]],
    raw_tokenizer: Any,
    chunking_times_by_repo: dict[str, float],
    doc_embed_times_by_repo: dict[str, float],
    insert_times_by_repo: dict[str, float],
    runtime_truncation_counts_by_repo: dict[str, int],
    peak_rss_mb: float,
) -> dict[str, Any]:
    """Compute all 24 required chunk quality and indexing performance metrics."""
    all_chunks: list[dict[str, Any]] = []
    chunks_per_repo: dict[str, int] = {}
    chunks_per_lang: Counter[str] = Counter()
    chunk_types: Counter[str] = Counter()
    ext_by_repo_file: dict[tuple[str, str], str] = {}
    for r_name, sfs in files_by_repo.items():
        for sf in sfs:
            ext_by_repo_file[(r_name, sf["file_path"])] = sf.get("extension", "")

    total_files = 0
    files_with_chunks = 0
    total_raw_lines = 0
    covered_raw_lines = 0
    total_non_empty_lines = 0
    covered_non_empty_lines = 0

    for r_name, sfs in files_by_repo.items():
        repo_chunks = chunks_by_repo.get(r_name, [])
        chunks_per_repo[r_name] = len(repo_chunks)
        all_chunks.extend([(r_name, c) for c in repo_chunks])

        by_file: dict[str, list[dict[str, Any]]] = {}
        for ch in repo_chunks:
            fp = ch["metadata"]["file_path"]
            by_file.setdefault(fp, []).append(ch)
            lang = ch["metadata"].get("language") or ext_by_repo_file.get((r_name, fp), "unknown")
            chunks_per_lang[str(lang)] += 1
            chunk_types[str(ch["metadata"].get("chunk_type", "unknown"))] += 1

        for sf in sfs:
            total_files += 1
            fp = sf["file_path"]
            f_chunks = by_file.get(fp, [])
            if any(c["content"].strip() for c in f_chunks):
                files_with_chunks += 1
            f_lines = sf["content"].splitlines()
            n_lines = len(f_lines)
            total_raw_lines += n_lines
            non_empty_set = {idx + 1 for idx, ln_txt in enumerate(f_lines) if ln_txt.strip()}
            total_non_empty_lines += len(non_empty_set)

            cov_set: set[int] = set()
            for c in f_chunks:
                s_l = int(c["metadata"].get("start_line", 1))
                e_l = int(c["metadata"].get("end_line", s_l))
                for ln in range(max(1, s_l), min(n_lines, e_l) + 1):
                    cov_set.add(ln)
            covered_raw_lines += len(cov_set)
            covered_non_empty_lines += len(cov_set & non_empty_set)

    n_chunks = len(all_chunks)
    line_lengths: list[int] = []
    char_lengths: list[int] = []
    token_lengths: list[int] = []
    empty_chunks = 0
    single_line_chunks = 0
    lt5_line_chunks = 0
    lt15_tok_chunks = 0
    gt256_tok_chunks = 0
    gt512_tok_chunks = 0
    content_hashes: Counter[str] = Counter()
    syntax_parsed_count = 0
    fallback_count = 0

    for _r_name, c in all_chunks:
        text = c["content"]
        meta = c["metadata"]
        s_l = int(meta.get("start_line", 1))
        e_l = int(meta.get("end_line", s_l))
        l_cnt = max(0, e_l - s_l + 1) if text.strip() or (s_l <= e_l) else 0
        tok_cnt = count_tokens(text, raw_tokenizer)

        line_lengths.append(l_cnt)
        char_lengths.append(len(text))
        token_lengths.append(tok_cnt)

        if not text.strip():
            empty_chunks += 1
        if l_cnt <= 1:
            single_line_chunks += 1
        if l_cnt < 5:
            lt5_line_chunks += 1
        if tok_cnt < 15:
            lt15_tok_chunks += 1
        if tok_cnt > 256:
            gt256_tok_chunks += 1
        if tok_cnt > 512:
            gt512_tok_chunks += 1

        content_hashes[hashlib.sha256(text.encode("utf-8")).hexdigest()] += 1

        if strategy_id == "code_aware_v1":
            if meta.get("fallback_used", False):
                fallback_count += 1
            else:
                syntax_parsed_count += 1
        elif strategy_id == "existing_chunker_v0":
            ct = str(meta.get("chunk_type", ""))
            if ct == "sliding_window":
                fallback_count += 1
            elif ct.startswith(("FunctionDef", "AsyncFunctionDef", "ClassDef")):
                syntax_parsed_count += 1
        else:
            fallback_count += 1

    exact_dup_excess = sum(cnt - 1 for cnt in content_hashes.values() if cnt > 1)
    exact_dup_groups = sum(1 for cnt in content_hashes.values() if cnt > 1)

    # Answerable ground-truth span overlap audit
    chunks_by_repo_file_idx = {
        r_name: build_chunks_by_file_index(cs) for r_name, cs in chunks_by_repo.items()
    }
    raw_chunks_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r_name, cs in chunks_by_repo.items():
        for c in cs:
            raw_chunks_lookup.setdefault((r_name, c["metadata"]["file_path"]), []).append(c)

    relevant_counts_per_q: list[int] = []
    total_gt_spans = 0
    covered_gt_spans = 0
    untruncated_gt_spans = 0

    for gt in ground_truth:
        if not gt.get("answerable"):
            continue
        r_name = gt["repository"]
        spans = gt["evidence"]
        rel_cnt = count_relevant_chunks_in_repo(
            evidence_spans=spans,
            chunks_by_file=chunks_by_repo_file_idx.get(r_name, {}),
        )
        relevant_counts_per_q.append(rel_cnt)
        for sp in spans:
            total_gt_spans += 1
            cands = raw_chunks_lookup.get((r_name, sp["file_path"]), [])
            overlaps = [
                c
                for c in cands
                if check_overlap(
                    int(c["metadata"]["start_line"]),
                    int(c["metadata"]["end_line"]),
                    sp["start_line"],
                    sp["end_line"],
                )
            ]
            if overlaps:
                covered_gt_spans += 1
                if any(count_tokens(c["content"], raw_tokenizer) <= 256 for c in overlaps):
                    untruncated_gt_spans += 1

    total_chunk_s = sum(chunking_times_by_repo.values())
    total_embed_s = sum(doc_embed_times_by_repo.values())
    total_insert_s = sum(insert_times_by_repo.values())
    total_e2e_s = total_chunk_s + total_embed_s + total_insert_s
    runtime_trunc_total = sum(runtime_truncation_counts_by_repo.values())

    return {
        "strategy_id": strategy_id,
        "total_chunks": n_chunks,
        "chunks_per_repository": chunks_per_repo,
        "chunks_per_language": dict(sorted(chunks_per_lang.items())),
        "source_files_total": total_files,
        "source_files_with_non_empty_chunks": files_with_chunks,
        "source_file_coverage_pct": round(100.0 * files_with_chunks / max(1, total_files), 2),
        "line_coverage_pct": round(100.0 * covered_raw_lines / max(1, total_raw_lines), 2),
        "non_empty_line_coverage_pct": round(
            100.0 * covered_non_empty_lines / max(1, total_non_empty_lines), 2
        ),
        "chunk_line_length_stats": _summary_stats(line_lengths),
        "chunk_char_length_stats": _summary_stats(char_lengths),
        "chunk_token_length_stats": _summary_stats(token_lengths, include_p95=True),
        "empty_chunk_count": empty_chunks,
        "single_line_chunk_count": single_line_chunks,
        "single_line_chunk_pct": round(100.0 * single_line_chunks / max(1, n_chunks), 2),
        "short_lt_5_lines_count": lt5_line_chunks,
        "short_lt_5_lines_pct": round(100.0 * lt5_line_chunks / max(1, n_chunks), 2),
        "tiny_lt_15_tokens_count": lt15_tok_chunks,
        "tiny_lt_15_tokens_pct": round(100.0 * lt15_tok_chunks / max(1, n_chunks), 2),
        "over_256_tokens_count": gt256_tok_chunks,
        "over_256_tokens_pct": round(100.0 * gt256_tok_chunks / max(1, n_chunks), 2),
        "over_512_tokens_count": gt512_tok_chunks,
        "over_512_tokens_pct": round(100.0 * gt512_tok_chunks / max(1, n_chunks), 2),
        "runtime_truncation_count": runtime_trunc_total,
        "runtime_truncation_rate_pct": round(100.0 * runtime_trunc_total / max(1, n_chunks), 2),
        "exact_duplicate_chunks_excess": exact_dup_excess,
        "exact_duplicate_groups": exact_dup_groups,
        "exact_duplicate_rate_pct": round(100.0 * exact_dup_excess / max(1, n_chunks), 2),
        "syntax_parsed_count": syntax_parsed_count,
        "syntax_parsed_pct": round(100.0 * syntax_parsed_count / max(1, n_chunks), 2),
        "fallback_chunk_count": fallback_count,
        "fallback_chunk_pct": round(100.0 * fallback_count / max(1, n_chunks), 2),
        "chunk_type_distribution": dict(chunk_types.most_common()),
        "avg_relevant_chunks_per_answerable_query": round(
            sum(relevant_counts_per_q) / max(1, len(relevant_counts_per_q)), 2
        ),
        "ground_truth_span_coverage": {
            "total_spans": total_gt_spans,
            "covered_spans": covered_gt_spans,
            "covered_pct": round(100.0 * covered_gt_spans / max(1, total_gt_spans), 2),
            "untruncated_spans_le_256_tokens": untruncated_gt_spans,
            "untruncated_spans_pct": round(100.0 * untruncated_gt_spans / max(1, total_gt_spans), 2),
        },
        "peak_rss_mb": round(peak_rss_mb, 2),
        "timings_seconds": {
            "total_chunking_seconds": round(total_chunk_s, 4),
            "mean_chunking_per_repo_seconds": round(
                total_chunk_s / max(1, len(chunking_times_by_repo)), 4
            ),
            "total_doc_embed_seconds": round(total_embed_s, 4),
            "embedding_throughput_chunks_per_sec": round(n_chunks / max(1e-6, total_embed_s), 2),
            "total_chroma_insert_seconds": round(total_insert_s, 4),
            "total_indexing_end_to_end_seconds": round(total_e2e_s, 4),
            "by_repository": {
                r: {
                    "chunking_seconds": round(chunking_times_by_repo[r], 4),
                    "doc_embed_seconds": round(doc_embed_times_by_repo[r], 4),
                    "chroma_insert_seconds": round(insert_times_by_repo[r], 4),
                }
                for r in files_by_repo
            },
        },
    }


def build_chunking_error_analysis(
    all_strategy_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compare solved/regressed answerable queries pairwise across all chunking strategies."""
    pairs = [
        ("code_aware_v1", "existing_chunker_v0"),
        ("fixed_window_v0", "existing_chunker_v0"),
        ("code_aware_v1", "fixed_window_v0"),
    ]
    pairwise: dict[str, Any] = {}
    for target_id, base_id in pairs:
        if target_id not in all_strategy_records or base_id not in all_strategy_records:
            continue
        t_map = {r["question_id"]: r for r in all_strategy_records[target_id] if r["answerable"]}
        b_map = {r["question_id"]: r for r in all_strategy_records[base_id] if r["answerable"]}
        gained, regressed, both_hit, both_miss = [], [], [], []
        for qid, b_r in b_map.items():
            t_r = t_map[qid]
            b_hit = b_r["metrics"]["hit_rate_at_5"] == 1.0
            t_hit = t_r["metrics"]["hit_rate_at_5"] == 1.0
            entry = {
                "question_id": qid,
                "repository": t_r["repository"],
                "category": t_r["category"],
                "query_style": t_r["query_style"],
                "question": t_r["raw_question"],
                f"{base_id}_rank": b_r["first_relevant_rank"],
                f"{target_id}_rank": t_r["first_relevant_rank"],
                f"{base_id}_span_recall": b_r["metrics"]["span_recall_at_5"],
                f"{target_id}_span_recall": t_r["metrics"]["span_recall_at_5"],
                f"{base_id}_mrr": b_r["metrics"]["mrr_at_5"],
                f"{target_id}_mrr": t_r["metrics"]["mrr_at_5"],
            }
            if not b_hit and t_hit:
                gained.append(entry)
            elif b_hit and not t_hit:
                regressed.append(entry)
            elif b_hit and t_hit:
                both_hit.append(entry)
            else:
                both_miss.append(entry)

        pairwise[f"{target_id}_vs_{base_id}"] = {
            "gained_count": len(gained),
            "gained": gained,
            "regressed_count": len(regressed),
            "regressed": regressed,
            "both_hit_count": len(both_hit),
            "both_miss_count": len(both_miss),
        }

    # Universal failures across all evaluated strategies
    if not all_strategy_records:
        return {"pairwise_transitions": pairwise}

    first_recs = next(iter(all_strategy_records.values()))
    ans_qids = [r["question_id"] for r in first_recs if r["answerable"]]
    rec_maps = {
        sid: {r["question_id"]: r for r in recs if r["answerable"]}
        for sid, recs in all_strategy_records.items()
    }
    missed_by_all = []
    unique_wins: dict[str, list[dict[str, Any]]] = {sid: [] for sid in all_strategy_records}

    for qid in ans_qids:
        hits_by_s = {
            sid: rec_maps[sid][qid]["metrics"]["hit_rate_at_5"] == 1.0
            for sid in all_strategy_records
        }
        ref = rec_maps[next(iter(all_strategy_records))][qid]
        item = {
            "question_id": qid,
            "repository": ref["repository"],
            "category": ref["category"],
            "query_style": ref["query_style"],
            "question": ref["raw_question"],
        }
        if not any(hits_by_s.values()):
            missed_by_all.append(item)
        else:
            winners = [sid for sid, ok in hits_by_s.items() if ok]
            if len(winners) == 1:
                w = winners[0]
                item["rank"] = rec_maps[w][qid]["first_relevant_rank"]
                unique_wins[w].append(item)

    return {
        "pairwise_transitions": pairwise,
        "missed_by_all_strategies_count": len(missed_by_all),
        "missed_by_all_strategies": missed_by_all,
        "unique_wins_by_strategy": {
            sid: {"count": len(lst), "queries": lst} for sid, lst in unique_wins.items()
        },
    }


def evaluate_single_strategy(
    strategy_id: str,
    strategy_label: str,
    embedder: DirectONNXEmbedder,
    client: chromadb.PersistentClient,
    repos_data: list[dict[str, Any]],
    files_by_repo: dict[str, list[dict[str, Any]]],
    questions_data: list[dict[str, Any]],
    gt_map: dict[str, dict[str, Any]],
    ground_truth_list: list[dict[str, Any]],
    run_id: str,
    corpus_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Run indexing + retrieval evaluation for one chunking strategy in isolated ChromaDB collections."""
    run_short = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:6]
    rss_samples: list[float] = [get_rss_mb()]

    chunks_by_repo: dict[str, list[dict[str, Any]]] = {}
    chunks_by_repo_file_idx: dict[str, dict[str, list[tuple[int, int]]]] = {}
    collections: dict[str, Any] = {}
    collection_names: dict[str, str] = {}

    chunking_times: dict[str, float] = {}
    embed_times: dict[str, float] = {}
    insert_times: dict[str, float] = {}
    trunc_counts: dict[str, int] = {}

    for r_info in repos_data:
        r_name = r_info["name"]
        alias = r_name.split("/")[-1]
        sfs = files_by_repo[r_name]

        t_ch0 = time.perf_counter()
        repo_chunks: list[dict[str, Any]] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            for sf in sfs:
                repo_chunks.extend(
                    chunk_code(
                        sf,
                        strategy=strategy_id,
                        repository=r_name,
                        tokenizer=embedder.raw_len_tokenizer if strategy_id == "code_aware_v1" else None,
                    )
                )
        chunking_times[r_name] = time.perf_counter() - t_ch0
        rss_samples.append(get_rss_mb())

        docs = [c["content"] for c in repo_chunks]
        metas = [validate_chroma_metadata(c["metadata"]) for c in repo_chunks]
        if strategy_id == "code_aware_v1":
            ids = [str(m["chunk_id"]) for m in metas]
            assert len(ids) == len(set(ids)), f"Duplicate chunk_id encountered in {r_name} ({strategy_id})"
        else:
            ids = [f"chunk_{i}" for i in range(len(repo_chunks))]

        vecs, emb_stats = embedder.encode_texts(
            docs,
            prefix=embedder.document_prefix,
            batch_size=BATCH_SIZE,
            track_truncation=True,
            rss_tracker=rss_samples,
        )
        embed_times[r_name] = float(emb_stats["duration_seconds"])
        trunc_counts[r_name] = int(emb_stats["truncated_count"] or 0)

        col_name = make_day3_collection_name(strategy_id, alias, run_short)
        collection_names[r_name] = col_name
        try:
            client.delete_collection(name=col_name)
        except Exception:
            pass
        col = client.create_collection(name=col_name, metadata={"hnsw:space": "l2"})

        t_ins0 = time.perf_counter()
        for start in range(0, len(repo_chunks), CHROMA_INSERT_BATCH):
            end = start + CHROMA_INSERT_BATCH
            col.add(
                embeddings=vecs[start:end].tolist(),
                documents=docs[start:end],
                metadatas=metas[start:end],
                ids=ids[start:end],
            )
        insert_times[r_name] = time.perf_counter() - t_ins0
        rss_samples.append(get_rss_mb())

        collections[r_name] = col
        chunks_by_repo[r_name] = repo_chunks
        chunks_by_repo_file_idx[r_name] = build_chunks_by_file_index(repo_chunks)

        print(
            f"[{strategy_id}][{r_name}] files={len(sfs)} chunks={len(repo_chunks)} "
            f"chunk={chunking_times[r_name]:.2f}s embed={embed_times[r_name]:.2f}s "
            f"insert={insert_times[r_name]:.2f}s trunc={trunc_counts[r_name]}",
            flush=True,
        )

    # Evaluate all 60 queries
    per_question_records: list[dict[str, Any]] = []
    answerable_records: list[dict[str, Any]] = []
    unanswerable_top1_l2: list[float] = []
    answerable_top1_l2: list[float] = []
    all_q_embed_ms: list[float] = []
    all_vec_search_ms: list[float] = []
    all_total_ms: list[float] = []
    ans_total_ms: list[float] = []
    unans_total_ms: list[float] = []
    failed_queries = 0

    for q in questions_data:
        qid = q["question_id"]
        r_name = q["repository"]
        is_ans = q["category"] != "unanswerable"
        col = collections[r_name]

        err_msg: str | None = None
        failed = False
        q_emb_ms = 0.0
        vec_ms = 0.0
        ret_chunks: list[dict[str, Any]] = []
        top_results_meta: list[dict[str, Any]] = []

        try:
            q_vec, q_emb_s, _ = embedder.encode_single_query(q["question"])
            q_emb_ms = q_emb_s * 1000.0
            t_s0 = time.perf_counter()
            res = col.query(query_embeddings=[q_vec.tolist()], n_results=TOP_K)
            vec_ms = (time.perf_counter() - t_s0) * 1000.0

            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            ret_ids = res.get("ids", [[]])[0]
            if not metas:
                failed = True
                err_msg = "Empty Chroma query metadata response"
            else:
                for rank_i, (cid, meta, dist) in enumerate(zip(ret_ids, metas, dists), start=1):
                    ret_chunks.append({"id": cid, "metadata": meta, "distance": float(dist)})
                    top_results_meta.append(
                        {
                            "rank": rank_i,
                            "chunk_id": cid,
                            "file_path": meta.get("file_path"),
                            "start_line": meta.get("start_line"),
                            "end_line": meta.get("end_line"),
                            "chunk_type": meta.get("chunk_type"),
                            "symbol_name": meta.get("symbol_name", ""),
                            "distance": round(float(dist), 6),
                        }
                    )
        except Exception as exc:
            failed = True
            err_msg = str(exc)

        if failed:
            failed_queries += 1

        total_ms = q_emb_ms + vec_ms
        all_q_embed_ms.append(q_emb_ms)
        all_vec_search_ms.append(vec_ms)
        all_total_ms.append(total_ms)

        top1_dist = float(ret_chunks[0]["distance"]) if (not failed and ret_chunks) else None
        if top1_dist is not None:
            if is_ans:
                answerable_top1_l2.append(top1_dist)
            else:
                unanswerable_top1_l2.append(top1_dist)

        if is_ans:
            ans_total_ms.append(total_ms)
            spans = gt_map[qid]["evidence"]
            rel_in_repo = count_relevant_chunks_in_repo(
                evidence_spans=spans, chunks_by_file=chunks_by_repo_file_idx[r_name]
            )
            qm = compute_query_metrics(
                ret_chunks, spans, rel_in_repo, top_k=TOP_K, failed=failed
            )
            rec = {
                "question_id": qid,
                "repository": r_name,
                "category": q["category"],
                "query_style": q.get("query_style", "identifier"),
                "answerable": True,
                "raw_question": q["question"],
                "query_embed_ms": round(q_emb_ms, 2),
                "vector_search_ms": round(vec_ms, 2),
                "retrieval_latency_ms": round(total_ms, 2),
                "failed": failed,
                "error": err_msg,
                "first_relevant_rank": qm["first_relevant_rank"],
                "matched_evidence_spans_count": qm["matched_evidence_spans_count"],
                "total_evidence_spans_count": qm["total_evidence_spans_count"],
                "total_relevant_chunks_in_repo": qm["total_relevant_chunks_in_repo"],
                "per_chunk_relevance": qm["per_chunk_relevance"],
                "metrics": qm["metrics"],
                "top_k_results": top_results_meta,
            }
            answerable_records.append(rec)
            per_question_records.append(rec)
        else:
            unans_total_ms.append(total_ms)
            per_question_records.append(
                {
                    "question_id": qid,
                    "repository": r_name,
                    "category": q["category"],
                    "query_style": q.get("query_style", "unanswerable"),
                    "answerable": False,
                    "raw_question": q["question"],
                    "query_embed_ms": round(q_emb_ms, 2),
                    "vector_search_ms": round(vec_ms, 2),
                    "retrieval_latency_ms": round(total_ms, 2),
                    "failed": failed,
                    "error": err_msg,
                    "top1_l2_distance": top1_dist,
                    "top_k_results": top_results_meta,
                }
            )

    rss_samples.append(get_rss_mb())
    peak_rss_mb = max(rss_samples)

    chunk_metrics = compute_corpus_chunk_metrics(
        strategy_id=strategy_id,
        chunks_by_repo=chunks_by_repo,
        files_by_repo=files_by_repo,
        ground_truth=ground_truth_list,
        raw_tokenizer=embedder.raw_len_tokenizer,
        chunking_times_by_repo=chunking_times,
        doc_embed_times_by_repo=embed_times,
        insert_times_by_repo=insert_times,
        runtime_truncation_counts_by_repo=trunc_counts,
        peak_rss_mb=peak_rss_mb,
    )

    overall_metrics = aggregate_retrieval_metrics(answerable_records)
    breakdowns = compute_breakdowns(answerable_records, repos_data)
    validity_status = "VALID" if failed_queries == 0 else "INVALID_OR_INCOMPLETE"

    summary = {
        "strategy_id": strategy_id,
        "label": strategy_label,
        "validity_status": validity_status,
        "failed_queries": failed_queries,
        "total_questions": len(questions_data),
        "answerable_questions": len(answerable_records),
        "unanswerable_questions": len(questions_data) - len(answerable_records),
        "collections": collection_names,
        "corpus_manifest_digest": hashlib.sha256(
            json.dumps(corpus_manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "overall_retrieval_metrics": overall_metrics,
        "breakdowns": breakdowns,
        "latency_breakdown_ms": {
            "all_60_queries_total": compute_latency_stats(all_total_ms),
            "all_60_query_embed": compute_latency_stats(all_q_embed_ms),
            "all_60_vector_search": compute_latency_stats(all_vec_search_ms),
            "answerable_54_total": compute_latency_stats(ans_total_ms),
            "unanswerable_6_total": compute_latency_stats(unans_total_ms),
        },
        "distance_distributions": {
            "answerable_top1_l2": calc_distance_distribution(answerable_top1_l2),
            "unanswerable_top1_l2": calc_distance_distribution(unanswerable_top1_l2),
        },
        "chunk_quality_highlights": {
            "total_chunks": chunk_metrics["total_chunks"],
            "over_256_tokens_count": chunk_metrics["over_256_tokens_count"],
            "over_256_tokens_pct": chunk_metrics["over_256_tokens_pct"],
            "empty_chunk_count": chunk_metrics["empty_chunk_count"],
            "single_line_chunk_count": chunk_metrics["single_line_chunk_count"],
            "tiny_lt_15_tokens_count": chunk_metrics["tiny_lt_15_tokens_count"],
            "line_coverage_pct": chunk_metrics["line_coverage_pct"],
            "non_empty_line_coverage_pct": chunk_metrics["non_empty_line_coverage_pct"],
            "exact_duplicate_rate_pct": chunk_metrics["exact_duplicate_rate_pct"],
            "ground_truth_span_coverage": chunk_metrics["ground_truth_span_coverage"],
            "timings_seconds": chunk_metrics["timings_seconds"],
        },
    }

    return {
        "strategy_id": strategy_id,
        "summary": summary,
        "chunk_metrics": chunk_metrics,
        "per_question": per_question_records,
    }


def main(selected_strategies: list[str] | None = None) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PER_STRATEGY_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    repos = json.load(open(REPOS_FILE, encoding="utf-8"))
    questions = json.load(open(QUESTIONS_FILE, encoding="utf-8"))
    ground_truth = json.load(open(GROUND_TRUTH_FILE, encoding="utf-8"))
    day1_summary = json.load(open(DAY1_SUMMARY_FILE, encoding="utf-8"))
    emb_suite = json.load(open(EMB_CONFIG_FILE, encoding="utf-8"))
    v0_cfg = next(m for m in emb_suite["models"] if m["experiment_id"] == "E0_v0_minilm_l6_v2")

    preflight_stats, commit_verifs = run_preflight_checks(repos, questions, ground_truth)
    gt_map = {g["question_id"]: g for g in ground_truth}

    files_by_repo = load_source_corpus(repos)
    corpus_manifest = build_corpus_manifest(files_by_repo)
    assert corpus_manifest["total_files"] == 253, f"Expected 253 source files, got {corpus_manifest['total_files']}"

    embedder = DirectONNXEmbedder(v0_cfg)
    warmup_s = embedder.warmup()
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    strategy_defs = [
        s for s in STRATEGIES if not selected_strategies or s["strategy_id"] in selected_strategies
    ]

    all_summaries: dict[str, Any] = {}
    all_chunk_metrics: dict[str, Any] = {}
    all_per_question: dict[str, list[dict[str, Any]]] = {}

    for s_def in strategy_defs:
        sid = s_def["strategy_id"]
        res = evaluate_single_strategy(
            strategy_id=sid,
            strategy_label=s_def["label"],
            embedder=embedder,
            client=client,
            repos_data=repos,
            files_by_repo=files_by_repo,
            questions_data=questions,
            gt_map=gt_map,
            ground_truth_list=ground_truth,
            run_id=run_id,
            corpus_manifest=corpus_manifest,
        )
        all_summaries[sid] = res["summary"]
        all_chunk_metrics[sid] = res["chunk_metrics"]
        all_per_question[sid] = res["per_question"]

        with open(os.path.join(PER_STRATEGY_DIR, f"{sid}_evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(res["per_question"], f, indent=2)
        with open(os.path.join(PER_STRATEGY_DIR, f"{sid}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(res["summary"], f, indent=2)
        with open(os.path.join(PER_STRATEGY_DIR, f"{sid}_chunk_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(res["chunk_metrics"], f, indent=2)

    error_analysis = build_chunking_error_analysis(all_per_question)

    # Verify V0 reproduction against Day 1 historical metrics if existing_chunker_v0 ran
    v0_deltas_vs_day1: dict[str, float] | None = None
    if "existing_chunker_v0" in all_summaries:
        d1_m = day1_summary["overall_retrieval_metrics"]
        v0_m = all_summaries["existing_chunker_v0"]["overall_retrieval_metrics"]
        v0_deltas_vs_day1 = {
            k: round(v0_m[k] - d1_m[k], 6)
            for k in [
                "hit_rate_at_1",
                "hit_rate_at_3",
                "hit_rate_at_5",
                "precision_at_5",
                "span_recall_at_5",
                "mrr_at_5",
                "ndcg_at_5",
            ]
        }

    # Build pairwise metric deltas vs existing_chunker_v0 and fixed_window_v0
    deltas_vs_v0: dict[str, Any] = {}
    if "existing_chunker_v0" in all_summaries:
        base = all_summaries["existing_chunker_v0"]["overall_retrieval_metrics"]
        for sid, summ in all_summaries.items():
            if sid == "existing_chunker_v0":
                continue
            cur = summ["overall_retrieval_metrics"]
            deltas_vs_v0[sid] = {
                k: round(cur[k] - base[k], 4)
                for k in [
                    "hit_rate_at_1",
                    "hit_rate_at_3",
                    "hit_rate_at_5",
                    "precision_at_5",
                    "span_recall_at_5",
                    "mrr_at_5",
                    "ndcg_at_5",
                ]
            }

    provenance = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
            "chromadb_version": chromadb.__version__,
            "numpy_version": np.__version__,
        },
        "experimental_controls": {
            "embedding_experiment_id": embedder.experiment_id,
            "embedding_model_version": embedder.model_version,
            "embedding_dimension": embedder.expected_dimension,
            "max_seq_length": embedder.max_seq_length,
            "distance_metric": "l2",
            "top_k": TOP_K,
            "model_load_duration_seconds": embedder.model_load_duration_seconds,
            "warmup_duration_seconds": warmup_s,
            "preflight_stats": preflight_stats,
            "commit_verifications": commit_verifs,
        },
        "script_sha256": get_file_sha256(os.path.abspath(__file__)),
        "chunker_v0_sha256": get_file_sha256(os.path.join(PROJECT_ROOT, "chunker.py")),
        "chunker_v1_sha256": get_file_sha256(os.path.join(PROJECT_ROOT, "chunker_v1.py")),
    }

    master_summary = {
        "provenance": provenance,
        "corpus_fairness_manifest": corpus_manifest,
        "day1_v0_parity_deltas": v0_deltas_vs_day1,
        "retrieval_deltas_vs_existing_chunker_v0": deltas_vs_v0,
        "strategies_summary": all_summaries,
        "chunk_quality_metrics": all_chunk_metrics,
        "error_analysis": error_analysis,
    }

    with open(os.path.join(OUT_DIR, "chunking_summary.json"), "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    full_payload = {
        **master_summary,
        "per_question_by_strategy": all_per_question,
    }
    with open(os.path.join(OUT_DIR, "chunking_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("\n=== DAY 3 CHUNKING COMPARISON COMPLETE ===", flush=True)
    for sid, summ in all_summaries.items():
        m = summ["overall_retrieval_metrics"]
        print(
            f"  {sid:20s} | H@1={m['hit_rate_at_1']:.4f} H@3={m['hit_rate_at_3']:.4f} "
            f"H@5={m['hit_rate_at_5']:.4f} P@5={m['precision_at_5']:.4f} "
            f"SR@5={m['span_recall_at_5']:.4f} MRR={m['mrr_at_5']:.4f} "
            f"NDCG={m['ndcg_at_5']:.4f} | valid={summ['validity_status']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 3 Controlled Chunking Benchmark")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(SUPPORTED_CHUNKING_STRATEGIES),
        default=None,
        help="Optional subset of strategy IDs to evaluate.",
    )
    args = parser.parse_args()
    sys.exit(main(args.strategies))
