#!/usr/bin/env python3
"""
Day 4: Controlled Hybrid Retrieval (BM25 + Dense Vector + RRF) Benchmark Runner.

Strict experimental controls:
  - Fixed chunking: Day 3 code_aware_v1
  - Fixed embeddings: ONNX E0_v0_minilm_l6_v2 (all-MiniLM-L6-v2, 384d, fixed_256, L2 norm)
  - Vector DB: ChromaDB PersistentClient in experiments/day4_hybrid/chroma_day4_db/, squared L2
  - Single-pass indexing per repo (day4_v1_<repo>), reused identically for Dense Top-5,
    Dense Top-20, BM25 hydration verification, and Hybrid RRF (k=60).
  - BM25 indexes the exact same V1 chunk IDs and V1 content (with static V1 header labels
    stripped so generic words like 'file', 'type', 'class' do not pollute lexical overlap).
  - Zero-overlap BM25 filtering + non-padded Top-20 candidates + empty-BM25 graceful fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import sys
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Sequence

import chromadb
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chunker import chunk_code
from evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    build_chunks_by_file_index,
    calc_distance_distribution,
    compute_latency_stats,
    compute_query_metrics,
    count_relevant_chunks_in_repo,
    evaluate_retrieved_chunks,
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
from retrieval.bm25_retriever import BM25Retriever, build_bm25_from_collection
from retrieval.hybrid_retriever import HybridRetriever

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
PER_METHOD_DIR = os.path.join(RESULTS_DIR, "per_method")
CHROMA_DAY4_DIR = os.path.join(EXP_DIR, "chroma_day4_db")

DATASETS_DIR = os.path.join(PROJECT_ROOT, "experiments", "datasets")
CACHE_DIR = os.path.join(DATASETS_DIR, "cache")
REPOS_FILE = os.path.join(DATASETS_DIR, "repositories.json")
QUESTIONS_FILE = os.path.join(DATASETS_DIR, "evaluation_questions.json")
GROUND_TRUTH_FILE = os.path.join(DATASETS_DIR, "ground_truth.json")
DAY3_V1_SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "chunking_experiments",
    "results",
    "per_strategy",
    "code_aware_v1_summary.json",
)

CANDIDATE_K = 20
FINAL_K = 5
RRF_K = 60
BATCH_SIZE = 32
CHROMA_INSERT_BATCH = 100
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10000


def calc_extended_distribution(values: Sequence[float]) -> dict[str, Any]:
    """Compute min, q25, median, mean, q75, IQR, max for score distributions."""
    if not values:
        return {"count": 0}
    sorted_v = sorted(float(v) for v in values)
    n = len(sorted_v)
    if n >= 2:
        qs = statistics.quantiles(sorted_v, n=4, method="inclusive")
        q25, q50, q75 = qs[0], qs[1], qs[2]
    else:
        q25 = q50 = q75 = sorted_v[0]
    iqr = q75 - q25
    base = calc_distance_distribution(sorted_v)
    return {
        **base,
        "q25": round(float(q25), 6),
        "q75": round(float(q75), 6),
        "iqr": round(float(iqr), 6),
    }


def paired_bootstrap_ci(
    diffs: Sequence[float],
    num_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute reproducible percentile bootstrap confidence interval for mean paired delta."""
    n = len(diffs)
    if n == 0:
        return {"count": 0}
    obs_mean = sum(diffs) / n
    rng = random.Random(seed)
    boot_means = []
    for _ in range(num_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo_idx = max(0, int((alpha / 2.0) * num_resamples))
    hi_idx = min(num_resamples - 1, int((1.0 - alpha / 2.0) * num_resamples))
    return {
        "count": n,
        "observed_mean_delta": round(float(obs_mean), 4),
        "ci_95_lower": round(float(boot_means[lo_idx]), 4),
        "ci_95_upper": round(float(boot_means[hi_idx]), 4),
        "num_resamples": num_resamples,
        "seed": seed,
        "method": "percentile paired bootstrap (exploratory descriptive interval)",
    }


def compute_paired_comparison(
    dense_records: Sequence[dict[str, Any]],
    hybrid_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build exact paired Hit@5, MRR@5, and query-level transition analysis (n=54)."""
    d_map = {r["question_id"]: r for r in dense_records if r["answerable"]}
    h_map = {r["question_id"]: r for r in hybrid_records if r["answerable"]}

    both_hit = []
    dense_hit_hybrid_miss = []
    dense_miss_hybrid_hit = []
    both_miss = []

    improved_mrr = []
    regressed_mrr = []
    unchanged_mrr = []

    hit5_diffs: list[float] = []
    mrr5_diffs: list[float] = []

    for qid, d_r in d_map.items():
        h_r = h_map[qid]
        d_hit = d_r["metrics"]["hit_rate_at_5"] == 1.0
        h_hit = h_r["metrics"]["hit_rate_at_5"] == 1.0
        d_mrr = float(d_r["metrics"]["mrr_at_5"])
        h_mrr = float(h_r["metrics"]["mrr_at_5"])
        d_sr = float(d_r["metrics"]["span_recall_at_5"])
        h_sr = float(h_r["metrics"]["span_recall_at_5"])

        hit5_diffs.append((1.0 if h_hit else 0.0) - (1.0 if d_hit else 0.0))
        mrr_delta = h_mrr - d_mrr
        mrr5_diffs.append(mrr_delta)

        entry = {
            "question_id": qid,
            "repository": h_r["repository"],
            "category": h_r["category"],
            "query_style": h_r["query_style"],
            "question": h_r["raw_question"],
            "dense_rank": d_r["first_relevant_rank"],
            "hybrid_rank": h_r["first_relevant_rank"],
            "dense_mrr": d_mrr,
            "hybrid_mrr": h_mrr,
            "mrr_delta": round(mrr_delta, 4),
            "dense_span_recall": d_sr,
            "hybrid_span_recall": h_sr,
        }

        if d_hit and h_hit:
            both_hit.append(entry)
        elif d_hit and not h_hit:
            dense_hit_hybrid_miss.append(entry)
        elif not d_hit and h_hit:
            dense_miss_hybrid_hit.append(entry)
        else:
            both_miss.append(entry)

        # Compare composite retrieval progress (rank/MRR or span recall improvement)
        if mrr_delta > 1e-9 or (abs(mrr_delta) <= 1e-9 and h_sr > d_sr + 1e-9):
            improved_mrr.append(entry)
        elif mrr_delta < -1e-9 or (abs(mrr_delta) <= 1e-9 and h_sr < d_sr - 1e-9):
            regressed_mrr.append(entry)
        else:
            unchanged_mrr.append(entry)

    improved_mrr.sort(key=lambda e: (-e["mrr_delta"], e["question_id"]))
    regressed_mrr.sort(key=lambda e: (e["mrr_delta"], e["question_id"]))

    return {
        "answerable_count": len(d_map),
        "hit_at_5_quadrants": {
            "dense_hit_hybrid_hit": len(both_hit),
            "dense_hit_hybrid_miss": len(dense_hit_hybrid_miss),
            "dense_miss_hybrid_hit": len(dense_miss_hybrid_hit),
            "dense_miss_hybrid_miss": len(both_miss),
        },
        "mrr_and_span_transitions": {
            "hybrid_improved_count": len(improved_mrr),
            "hybrid_regressed_count": len(regressed_mrr),
            "unchanged_count": len(unchanged_mrr),
            "hybrid_gained_hit5": dense_miss_hybrid_hit,
            "hybrid_lost_hit5": dense_hit_hybrid_miss,
            "top_improved_queries": improved_mrr[:15],
            "top_regressed_queries": regressed_mrr[:15],
        },
        "paired_bootstrap_95pct_ci": {
            "delta_hit_rate_at_5": paired_bootstrap_ci(hit5_diffs),
            "delta_mrr_at_5": paired_bootstrap_ci(mrr5_diffs),
        },
    }


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PER_METHOD_DIR, exist_ok=True)
    os.makedirs(CHROMA_DAY4_DIR, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    repos = json.load(open(REPOS_FILE, encoding="utf-8"))
    questions = json.load(open(QUESTIONS_FILE, encoding="utf-8"))
    ground_truth = json.load(open(GROUND_TRUTH_FILE, encoding="utf-8"))
    emb_suite = json.load(open(EMB_CONFIG_FILE, encoding="utf-8"))
    v0_cfg = next(m for m in emb_suite["models"] if m["experiment_id"] == "E0_v0_minilm_l6_v2")
    day3_v1_summary = json.load(open(DAY3_V1_SUMMARY_FILE, encoding="utf-8"))

    preflight_stats, commit_verifs = run_preflight_checks(repos, questions, ground_truth)
    gt_map = {g["question_id"]: g for g in ground_truth}

    embedder = DirectONNXEmbedder(v0_cfg)
    warmup_s = embedder.warmup()
    client = chromadb.PersistentClient(path=CHROMA_DAY4_DIR)

    rss_samples: list[float] = [get_rss_mb()]
    collections: dict[str, Any] = {}
    bm25_indices: dict[str, BM25Retriever] = {}
    chunks_by_repo_file_idx: dict[str, dict[str, list[tuple[int, int]]]] = {}
    indexing_stats_by_repo: dict[str, Any] = {}

    # 1. Single-pass Dense + BM25 V1 indexing across all 3 pinned repos
    for r_info in repos:
        r_name = r_info["name"]
        alias = r_name.split("/")[-1]
        repo_dir = os.path.join(CACHE_DIR, alias)
        sfs = sorted(get_source_files(repo_dir), key=lambda f: f["file_path"])

        t_ch0 = time.perf_counter()
        repo_chunks: list[dict[str, Any]] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            for sf in sfs:
                repo_chunks.extend(
                    chunk_code(
                        sf,
                        strategy="code_aware_v1",
                        repository=r_name,
                        tokenizer=embedder.raw_len_tokenizer,
                    )
                )
        chunk_s = time.perf_counter() - t_ch0

        docs = [c["content"] for c in repo_chunks]
        metas = [dict(c["metadata"]) for c in repo_chunks]
        ids = [str(m["chunk_id"]) for m in metas]
        assert len(ids) == len(set(ids)), f"Duplicate chunk_id in {r_name}"

        vecs, emb_stats = embedder.encode_texts(
            docs,
            prefix=embedder.document_prefix,
            batch_size=BATCH_SIZE,
            track_truncation=True,
            rss_tracker=rss_samples,
        )

        col_name = f"day4_v1_{alias}"
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
        ins_s = time.perf_counter() - t_ins0

        # Build BM25 AND verify exact ChromaDB hydration identity
        t_bm25_0 = time.perf_counter()
        hydrated_bm25, hydrated_fp = build_bm25_from_collection(col, k1=1.5, b=0.75, idf_mode="lucene")
        bm25_build_s = time.perf_counter() - t_bm25_0

        direct_bm25 = BM25Retriever(ids, docs, metas, k1=1.5, b=0.75, idf_mode="lucene")
        assert hydrated_fp == direct_bm25.corpus_fingerprint, (
            f"Corpus fingerprint mismatch between V1 chunks and Chroma hydration in {r_name}"
        )
        assert set(hydrated_bm25.chunk_ids) == set(ids)
        assert hydrated_bm25.num_docs == len(repo_chunks)

        collections[r_name] = col
        bm25_indices[r_name] = hydrated_bm25
        chunks_by_repo_file_idx[r_name] = build_chunks_by_file_index(repo_chunks)

        indexing_stats_by_repo[r_name] = {
            "collection_name": col_name,
            "files": len(sfs),
            "chunks": len(repo_chunks),
            "truncated_chunks": int(emb_stats["truncated_count"] or 0),
            "corpus_fingerprint_sha256": hydrated_fp,
            "bm25_vocab_size": len(hydrated_bm25.doc_freq),
            "bm25_avgdl_tokens": round(hydrated_bm25.avgdl, 2),
            "chunking_seconds": round(chunk_s, 4),
            "doc_embed_seconds": round(float(emb_stats["duration_seconds"]), 4),
            "chroma_insert_seconds": round(ins_s, 4),
            "bm25_hydration_and_build_seconds": round(bm25_build_s, 4),
        }
        print(
            f"[DAY4-INDEX][{r_name}] files={len(sfs)} chunks={len(repo_chunks)} "
            f"fp={hydrated_fp[:12]} vocab={len(hydrated_bm25.doc_freq)} "
            f"embed={emb_stats['duration_seconds']:.2f}s bm25_build={bm25_build_s:.3f}s",
            flush=True,
        )

    # 2. Evaluate Dense (D4-E0), BM25 (D4-E1), Hybrid RRF (D4-E2), and Stage-1 Pools
    dense_records: list[dict[str, Any]] = []
    bm25_records: list[dict[str, Any]] = []
    hybrid_records: list[dict[str, Any]] = []
    candidate_pool_records: list[dict[str, Any]] = []

    unans_dense_l2: list[float] = []
    ans_dense_l2: list[float] = []
    unans_bm25_top1: list[float] = []
    ans_bm25_top1: list[float] = []
    unans_rrf_top1: list[float] = []
    ans_rrf_top1: list[float] = []

    latencies: dict[str, list[float]] = {
        "dense_query_embed_ms": [],
        "dense_vector_search_20_ms": [],
        "dense_total_ms": [],
        "bm25_search_20_ms": [],
        "rrf_fusion_ms": [],
        "hybrid_total_ms": [],
    }

    failed_queries = {"dense": 0, "bm25": 0, "hybrid": 0}

    for q in questions:
        qid = q["question_id"]
        r_name = q["repository"]
        is_ans = q["category"] != "unanswerable"
        col = collections[r_name]
        bm25 = bm25_indices[r_name]

        dense_timing: dict[str, float] = {"q_embed_ms": 0.0, "vec_ms": 0.0}

        def _dense_search_fn(query_txt: str, n_res: int) -> list[dict[str, Any]]:
            q_vec, q_emb_s, _ = embedder.encode_single_query(query_txt)
            dense_timing["q_embed_ms"] = q_emb_s * 1000.0
            t_v0 = time.perf_counter()
            res = col.query(query_embeddings=[q_vec.tolist()], n_results=n_res)
            dense_timing["vec_ms"] = (time.perf_counter() - t_v0) * 1000.0
            res_ids = res.get("ids", [[]])[0]
            res_docs = res.get("documents", [[]])[0]
            res_metas = res.get("metadatas", [[]])[0]
            res_dists = res.get("distances", [[]])[0]
            if not res_metas:
                raise RuntimeError("Empty Chroma query metadata response")
            out = []
            for cid, doc, meta, dist in zip(res_ids, res_docs, res_metas, res_dists):
                out.append(
                    {
                        "id": str(cid),
                        "content": doc,
                        "metadata": meta,
                        "distance": float(dist),
                    }
                )
            return out

        hybrid = HybridRetriever(
            dense_search_fn=_dense_search_fn,
            bm25_retriever=bm25,
            candidate_k=CANDIDATE_K,
            rrf_k=RRF_K,
        )

        err_msg: str | None = None
        failed = False
        try:
            hybrid_top5, diag = hybrid.retrieve(
                q["question"], top_k=FINAL_K, candidate_k=CANDIDATE_K, return_diagnostics=True
            )
            dense_20 = diag["dense_candidates"]
            bm25_20 = diag["bm25_candidates"]
            union_cands = diag["candidate_union"]
            rrf_20 = diag["rrf_top_20"]
            rrf_all = diag["rrf_all_ranked"]
            bm25_meta = diag["bm25_meta"]
            rrf_stats = diag["rrf_stats"]
            t_info = diag["timings_ms"]
        except Exception as exc:
            failed = True
            err_msg = str(exc)
            for key in failed_queries:
                failed_queries[key] += 1
            dense_20, bm25_20, union_cands, rrf_20, rrf_all, hybrid_top5 = [], [], [], [], [], []
            bm25_meta = {"query_tokens": [], "eligible_overlap_docs": 0, "returned_count": 0}
            rrf_stats = {
                "dense_candidates_count": 0,
                "bm25_candidates_count": 0,
                "union_candidates_count": 0,
                "overlap_candidates_count": 0,
                "bm25_fallback_to_dense_only": True,
            }
            t_info = {"bm25_ms": 0.0, "rrf_ms": 0.0, "total_hybrid_ms": 0.0}

        d_total_ms = dense_timing["q_embed_ms"] + dense_timing["vec_ms"]
        latencies["dense_query_embed_ms"].append(dense_timing["q_embed_ms"])
        latencies["dense_vector_search_20_ms"].append(dense_timing["vec_ms"])
        latencies["dense_total_ms"].append(d_total_ms)
        latencies["bm25_search_20_ms"].append(float(t_info["bm25_ms"]))
        latencies["rrf_fusion_ms"].append(float(t_info["rrf_ms"]))
        latencies["hybrid_total_ms"].append(d_total_ms + float(t_info["bm25_ms"]) + float(t_info["rrf_ms"]))

        d_top1 = float(dense_20[0]["distance"]) if dense_20 else None
        b_top1 = float(bm25_20[0]["bm25_score"]) if bm25_20 else 0.0
        h_top1 = float(hybrid_top5[0]["rrf_score"]) if hybrid_top5 else None

        if is_ans:
            if d_top1 is not None:
                ans_dense_l2.append(d_top1)
            ans_bm25_top1.append(b_top1)
            if h_top1 is not None:
                ans_rrf_top1.append(h_top1)
        else:
            if d_top1 is not None:
                unans_dense_l2.append(d_top1)
            unans_bm25_top1.append(b_top1)
            if h_top1 is not None:
                unans_rrf_top1.append(h_top1)

        def _brief_top(items: Sequence[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
            return [
                {
                    "rank": idx,
                    "chunk_id": it.get("id"),
                    "file_path": it.get("metadata", {}).get("file_path"),
                    "start_line": it.get("metadata", {}).get("start_line"),
                    "end_line": it.get("metadata", {}).get("end_line"),
                    "symbol_name": it.get("metadata", {}).get("symbol_name", ""),
                    "chunk_type": it.get("metadata", {}).get("chunk_type"),
                    score_key: it.get(score_key),
                    "retrieval_sources": it.get("retrieval_sources"),
                    "dense_rank": it.get("dense_rank"),
                    "bm25_rank": it.get("bm25_rank"),
                }
                for idx, it in enumerate(items[:FINAL_K], start=1)
            ]

        if is_ans:
            spans = gt_map[qid]["evidence"]
            rel_repo = count_relevant_chunks_in_repo(
                evidence_spans=spans, chunks_by_file=chunks_by_repo_file_idx[r_name]
            )
            qm_dense = compute_query_metrics(dense_20[:FINAL_K], spans, rel_repo, top_k=FINAL_K, failed=failed)
            qm_bm25 = compute_query_metrics(bm25_20[:FINAL_K], spans, rel_repo, top_k=FINAL_K, failed=failed)
            qm_hybrid = compute_query_metrics(hybrid_top5, spans, rel_repo, top_k=FINAL_K, failed=failed)

            # Candidate pool metrics (explicitly avoid truncating union <=40 at 20!)
            dense20_eval = evaluate_retrieved_chunks(dense_20, spans, top_k=CANDIDATE_K)
            bm25_20_eval = evaluate_retrieved_chunks(bm25_20, spans, top_k=CANDIDATE_K)
            union_eval = evaluate_retrieved_chunks(
                union_cands, spans, top_k=max(1, len(union_cands))
            )
            rrf20_eval = evaluate_retrieved_chunks(rrf_all, spans, top_k=CANDIDATE_K)

            total_sp = max(1, len(spans))
            d20_hit = 1.0 if dense20_eval["first_relevant_rank"] is not None else 0.0
            d20_sr = len(dense20_eval["matched_span_indices"]) / total_sp
            b20_hit = 1.0 if bm25_20_eval["first_relevant_rank"] is not None else 0.0
            b20_sr = len(bm25_20_eval["matched_span_indices"]) / total_sp
            u_hit = 1.0 if union_eval["first_relevant_rank"] is not None else 0.0
            u_sr = len(union_eval["matched_span_indices"]) / total_sp
            rrf20_hit = 1.0 if rrf20_eval["first_relevant_rank"] is not None else 0.0
            rrf20_sr = len(rrf20_eval["matched_span_indices"]) / total_sp

            if qm_hybrid["metrics"]["hit_rate_at_5"] == 1.0:
                handoff_status = "solved_in_rrf_top5"
            elif rrf20_hit == 1.0:
                handoff_status = "in_rrf_top20_missed_top5"
            elif u_hit == 1.0:
                handoff_status = "in_union_missed_rrf_top20"
            else:
                handoff_status = "candidate_generation_failure"

            candidate_pool_records.append(
                {
                    "question_id": qid,
                    "repository": r_name,
                    "category": q["category"],
                    "query_style": q.get("query_style", "identifier"),
                    "question": q["question"],
                    "dense_candidates_count": len(dense_20),
                    "bm25_candidates_count": len(bm25_20),
                    "bm25_eligible_overlap_docs": bm25_meta["eligible_overlap_docs"],
                    "union_candidates_count": len(union_cands),
                    "overlap_candidates_count": rrf_stats["overlap_candidates_count"],
                    "dense_hit_at_20": d20_hit,
                    "dense_span_recall_at_20": round(d20_sr, 4),
                    "bm25_hit_at_20": b20_hit,
                    "bm25_span_recall_at_20": round(b20_sr, 4),
                    "candidate_union_hit_rate": u_hit,
                    "candidate_union_span_recall": round(u_sr, 4),
                    "rrf_hit_at_20": rrf20_hit,
                    "rrf_span_recall_at_20": round(rrf20_sr, 4),
                    "rrf_hit_at_5": qm_hybrid["metrics"]["hit_rate_at_5"],
                    "rrf_span_recall_at_5": qm_hybrid["metrics"]["span_recall_at_5"],
                    "best_dense_20_rank": dense20_eval["first_relevant_rank"],
                    "best_bm25_20_rank": bm25_20_eval["first_relevant_rank"],
                    "best_rrf_all_rank": evaluate_retrieved_chunks(
                        rrf_all, spans, top_k=max(1, len(rrf_all))
                    )["first_relevant_rank"],
                    "handoff_status": handoff_status,
                }
            )

            common_meta = {
                "question_id": qid,
                "repository": r_name,
                "category": q["category"],
                "query_style": q.get("query_style", "identifier"),
                "answerable": True,
                "raw_question": q["question"],
                "failed": failed,
                "error": err_msg,
            }
            dense_records.append(
                {
                    **common_meta,
                    "retrieval_latency_ms": round(d_total_ms, 2),
                    **qm_dense,
                    "top_k_results": _brief_top(dense_20, "distance"),
                }
            )
            bm25_records.append(
                {
                    **common_meta,
                    "retrieval_latency_ms": round(float(t_info["bm25_ms"]), 2),
                    "bm25_eligible_overlap_docs": bm25_meta["eligible_overlap_docs"],
                    "query_tokens": bm25_meta["query_tokens"],
                    **qm_bm25,
                    "top_k_results": _brief_top(bm25_20, "bm25_score"),
                }
            )
            hybrid_records.append(
                {
                    **common_meta,
                    "retrieval_latency_ms": round(
                        d_total_ms + float(t_info["bm25_ms"]) + float(t_info["rrf_ms"]), 2
                    ),
                    "bm25_eligible_overlap_docs": bm25_meta["eligible_overlap_docs"],
                    "union_candidates_count": len(union_cands),
                    **qm_hybrid,
                    "top_k_results": _brief_top(hybrid_top5, "rrf_score"),
                }
            )
        else:
            unans_base = {
                "question_id": qid,
                "repository": r_name,
                "category": q["category"],
                "query_style": q.get("query_style", "unanswerable"),
                "answerable": False,
                "raw_question": q["question"],
                "failed": failed,
                "error": err_msg,
            }
            dense_records.append(
                {
                    **unans_base,
                    "retrieval_latency_ms": round(d_total_ms, 2),
                    "top1_l2_distance": d_top1,
                    "top_k_results": _brief_top(dense_20, "distance"),
                }
            )
            bm25_records.append(
                {
                    **unans_base,
                    "retrieval_latency_ms": round(float(t_info["bm25_ms"]), 2),
                    "bm25_eligible_overlap_docs": bm25_meta["eligible_overlap_docs"],
                    "top1_bm25_score": b_top1,
                    "top_k_results": _brief_top(bm25_20, "bm25_score"),
                }
            )
            hybrid_records.append(
                {
                    **unans_base,
                    "retrieval_latency_ms": round(
                        d_total_ms + float(t_info["bm25_ms"]) + float(t_info["rrf_ms"]), 2
                    ),
                    "top1_rrf_score": h_top1,
                    "union_candidates_count": len(union_cands),
                    "top_k_results": _brief_top(hybrid_top5, "rrf_score"),
                }
            )

    rss_samples.append(get_rss_mb())
    peak_rss_mb = max(rss_samples)

    # 3. Aggregate metrics & check Day 3 Dense V1 baseline exact parity
    ans_dense_recs = [r for r in dense_records if r["answerable"]]
    ans_bm25_recs = [r for r in bm25_records if r["answerable"]]
    ans_hybrid_recs = [r for r in hybrid_records if r["answerable"]]

    dense_overall = aggregate_retrieval_metrics(ans_dense_recs)
    bm25_overall = aggregate_retrieval_metrics(ans_bm25_recs)
    hybrid_overall = aggregate_retrieval_metrics(ans_hybrid_recs)

    day3_ref = day3_v1_summary["overall_retrieval_metrics"]
    day3_parity_deltas = {
        k: round(dense_overall[k] - day3_ref[k], 6)
        for k in (
            "hit_rate_at_1",
            "hit_rate_at_3",
            "hit_rate_at_5",
            "precision_at_5",
            "span_recall_at_5",
            "mrr_at_5",
            "ndcg_at_5",
        )
    }

    union_sizes = [r["union_candidates_count"] for r in candidate_pool_records]
    overlap_sizes = [r["overlap_candidates_count"] for r in candidate_pool_records]
    n_ans = max(1, len(candidate_pool_records))
    candidate_pool_summary = {
        "answerable_query_count": len(candidate_pool_records),
        "candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "dense_hit_at_20": round(sum(r["dense_hit_at_20"] for r in candidate_pool_records) / n_ans, 4),
        "dense_span_recall_at_20": round(
            sum(r["dense_span_recall_at_20"] for r in candidate_pool_records) / n_ans, 4
        ),
        "bm25_hit_at_20": round(sum(r["bm25_hit_at_20"] for r in candidate_pool_records) / n_ans, 4),
        "bm25_span_recall_at_20": round(
            sum(r["bm25_span_recall_at_20"] for r in candidate_pool_records) / n_ans, 4
        ),
        "candidate_union_hit_rate": round(
            sum(r["candidate_union_hit_rate"] for r in candidate_pool_records) / n_ans, 4
        ),
        "candidate_union_span_recall": round(
            sum(r["candidate_union_span_recall"] for r in candidate_pool_records) / n_ans, 4
        ),
        "rrf_hit_at_20": round(sum(r["rrf_hit_at_20"] for r in candidate_pool_records) / n_ans, 4),
        "rrf_span_recall_at_20": round(
            sum(r["rrf_span_recall_at_20"] for r in candidate_pool_records) / n_ans, 4
        ),
        "rrf_hit_at_5": hybrid_overall["hit_rate_at_5"],
        "rrf_span_recall_at_5": hybrid_overall["span_recall_at_5"],
        "candidate_union_size_stats": calc_extended_distribution(union_sizes),
        "candidate_overlap_size_stats": calc_extended_distribution(overlap_sizes),
        "handoff_breakdown": {
            "solved_in_rrf_top5": [
                r["question_id"] for r in candidate_pool_records if r["handoff_status"] == "solved_in_rrf_top5"
            ],
            "in_rrf_top20_missed_top5": [
                r for r in candidate_pool_records if r["handoff_status"] == "in_rrf_top20_missed_top5"
            ],
            "in_union_missed_rrf_top20": [
                r for r in candidate_pool_records if r["handoff_status"] == "in_union_missed_rrf_top20"
            ],
            "candidate_generation_failure": [
                r for r in candidate_pool_records if r["handoff_status"] == "candidate_generation_failure"
            ],
        },
        "bm25_zero_match_queries": [
            r["question_id"] for r in bm25_records if r.get("bm25_eligible_overlap_docs", 0) == 0
        ],
    }

    paired_comparison = compute_paired_comparison(ans_dense_recs, ans_hybrid_recs)

    methods_payload = {
        "D4-E0_dense_v1": {
            "method_id": "D4-E0_dense_v1",
            "label": "D4-E0: Dense Vector Top-5 (all-MiniLM-L6-v2 + code_aware_v1)",
            "failed_queries": failed_queries["dense"],
            "overall_retrieval_metrics": dense_overall,
            "breakdowns": compute_breakdowns(ans_dense_recs, repos),
            "per_question": dense_records,
        },
        "D4-E1_bm25_v1": {
            "method_id": "D4-E1_bm25_v1",
            "label": "D4-E1: Code-Aware BM25 Top-5 (V1 header label stripped)",
            "failed_queries": failed_queries["bm25"],
            "overall_retrieval_metrics": bm25_overall,
            "breakdowns": compute_breakdowns(ans_bm25_recs, repos),
            "per_question": bm25_records,
        },
        "D4-E2_hybrid_rrf": {
            "method_id": "D4-E2_hybrid_rrf",
            "label": "D4-E2: Hybrid Dense@20 + BM25@20 -> RRF(k=60) Top-5",
            "failed_queries": failed_queries["hybrid"],
            "overall_retrieval_metrics": hybrid_overall,
            "breakdowns": compute_breakdowns(ans_hybrid_recs, repos),
            "per_question": hybrid_records,
        },
    }

    for mid, mdata in methods_payload.items():
        with open(os.path.join(PER_METHOD_DIR, f"{mid}_evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(mdata["per_question"], f, indent=2)
        with open(os.path.join(PER_METHOD_DIR, f"{mid}_summary.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in mdata.items() if k != "per_question"}, f, indent=2)

    summary = {
        "provenance": {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "chromadb_version": chromadb.__version__,
                "numpy_version": np.__version__,
            },
            "controls": {
                "chunking_strategy": "code_aware_v1",
                "embedding_experiment_id": embedder.experiment_id,
                "embedding_model_version": embedder.model_version,
                "distance_metric": "l2",
                "candidate_k": CANDIDATE_K,
                "final_k": FINAL_K,
                "rrf_k": RRF_K,
                "bm25_k1": 1.5,
                "bm25_b": 0.75,
                "bm25_idf_mode": "lucene",
                "bm25_document_representation": "V1 content directly with static header field labels stripped; identical tokenizer for docs and queries; zero-overlap filtering",
                "model_load_duration_seconds": embedder.model_load_duration_seconds,
                "warmup_duration_seconds": warmup_s,
                "preflight_stats": preflight_stats,
                "commit_verifications": commit_verifs,
            },
            "script_sha256": get_file_sha256(os.path.abspath(__file__)),
            "bm25_module_sha256": get_file_sha256(
                os.path.join(PROJECT_ROOT, "retrieval", "bm25_retriever.py")
            ),
            "hybrid_module_sha256": get_file_sha256(
                os.path.join(PROJECT_ROOT, "retrieval", "hybrid_retriever.py")
            ),
        },
        "validity_status": "VALID" if sum(failed_queries.values()) == 0 else "INVALID_OR_INCOMPLETE",
        "day3_v1_parity_deltas": day3_parity_deltas,
        "indexing_stats_by_repo": indexing_stats_by_repo,
        "peak_rss_mb": round(peak_rss_mb, 2),
        "methods_summary": {
            mid: {k: v for k, v in mdata.items() if k != "per_question"}
            for mid, mdata in methods_payload.items()
        },
        "deltas_hybrid_vs_dense": {
            k: round(hybrid_overall[k] - dense_overall[k], 4)
            for k in (
                "hit_rate_at_1",
                "hit_rate_at_3",
                "hit_rate_at_5",
                "precision_at_5",
                "span_recall_at_5",
                "mrr_at_5",
                "ndcg_at_5",
            )
        },
        "candidate_pool_and_handoff": candidate_pool_summary,
        "candidate_pool_per_query": candidate_pool_records,
        "paired_comparison_dense_vs_hybrid": paired_comparison,
        "score_distributions_answerable_vs_unanswerable": {
            "dense_top1_l2_distance": {
                "answerable_54": calc_extended_distribution(ans_dense_l2),
                "unanswerable_6": calc_extended_distribution(unans_dense_l2),
            },
            "bm25_top1_score": {
                "answerable_54": calc_extended_distribution(ans_bm25_top1),
                "unanswerable_6": calc_extended_distribution(unans_bm25_top1),
                "answerable_zero_match_count": sum(1 for x in ans_bm25_top1 if x == 0.0),
                "unanswerable_zero_match_count": sum(1 for x in unans_bm25_top1 if x == 0.0),
            },
            "rrf_top1_score": {
                "answerable_54": calc_extended_distribution(ans_rrf_top1),
                "unanswerable_6": calc_extended_distribution(unans_rrf_top1),
            },
        },
        "query_latency_ms": {
            name: compute_latency_stats(vals) for name, vals in latencies.items()
        },
    }

    with open(os.path.join(RESULTS_DIR, "hybrid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "hybrid_comparison.json"), "w", encoding="utf-8") as f:
        json.dump({**summary, "per_method_records": methods_payload}, f, indent=2)

    print("\n=== DAY 4 HYBRID BENCHMARK COMPLETE ===", flush=True)
    for mid, ms in summary["methods_summary"].items():
        m = ms["overall_retrieval_metrics"]
        print(
            f"  {mid:18s} | H@1={m['hit_rate_at_1']:.4f} H@3={m['hit_rate_at_3']:.4f} "
            f"H@5={m['hit_rate_at_5']:.4f} P@5={m['precision_at_5']:.4f} "
            f"SR@5={m['span_recall_at_5']:.4f} MRR={m['mrr_at_5']:.4f} NDCG={m['ndcg_at_5']:.4f}",
            flush=True,
        )
    print(
        f"  Candidate Pool@20  | Dense H@20={candidate_pool_summary['dense_hit_at_20']:.4f} "
        f"BM25 H@20={candidate_pool_summary['bm25_hit_at_20']:.4f} "
        f"Union H={candidate_pool_summary['candidate_union_hit_rate']:.4f} "
        f"Union SR={candidate_pool_summary['candidate_union_span_recall']:.4f} "
        f"RRF@20 H={candidate_pool_summary['rrf_hit_at_20']:.4f}",
        flush=True,
    )
    print(f"  Day 3 V1 Parity Deltas: {day3_parity_deltas}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
