#!/usr/bin/env python3
"""
Day 5: Controlled Cross-Encoder Reranking & Conversational Query Rewriting Benchmark.

Experiments executed under strict Day 4 corpus & control invariance:
  1. D5-Base (Hybrid RRF Baseline Parity): Reproduces Day 4 Hybrid RRF k=60 metrics.
  2. D5-E3a (Rerank RRF Top-20): Dense@20 + BM25@20 -> RRF -> RRF Top-20 -> ONNX Cross-Encoder -> Top-5.
  3. D5-E3b (Rerank Full Candidate Union): Dense@20 + BM25@20 -> Union (<=40) -> ONNX Cross-Encoder -> Top-5.
  4. D5-Q0 (Original Query Baseline on Best Fixed Day 5 Pipeline).
  5. D5-Q1 (Leakage-Free Gemini Query Rewriting via Deterministic Gate -> Best Fixed Day 5 Pipeline).
  6. D5-Q2-Oracle (Offline Oracle Reference ONLY — Not a Deployable Configuration).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Sequence

import chromadb
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    build_chunks_by_file_index,
    compute_latency_stats,
    compute_query_metrics,
    count_relevant_chunks_in_repo,
)
from experiments.day4_hybrid.run_hybrid_comparison import (
    calc_extended_distribution,
    compute_paired_comparison,
)
from experiments.embedding_experiments.run_embedding_comparison import (
    CONFIG_FILE as EMB_CONFIG_FILE,
    DirectONNXEmbedder,
    compute_breakdowns,
    get_file_sha256,
    get_rss_mb,
    run_preflight_checks,
)
from retrieval.bm25_retriever import BM25Retriever, build_bm25_from_collection
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_rewriter import rewrite_conversational_query
from retrieval.reranker import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_REVISION,
    ONNXCrossEncoderReranker,
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
PER_EXP_DIR = os.path.join(RESULTS_DIR, "per_experiment")

DAY4_CHROMA_DIR = os.path.join(PROJECT_ROOT, "experiments", "day4_hybrid", "chroma_day4_db")
DAY4_SUMMARY_FILE = os.path.join(
    PROJECT_ROOT, "experiments", "day4_hybrid", "results", "hybrid_summary.json"
)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "experiments", "datasets")
REPOS_FILE = os.path.join(DATASETS_DIR, "repositories.json")
QUESTIONS_FILE = os.path.join(DATASETS_DIR, "evaluation_questions.json")
GROUND_TRUTH_FILE = os.path.join(DATASETS_DIR, "ground_truth.json")

CANDIDATE_K = 20
FINAL_K = 5
RRF_K = 60


def _brief_reranked(items: Sequence[dict[str, Any]], score_field: str = "reranker_raw_logit") -> list[dict[str, Any]]:
    return [
        {
            "rank": idx,
            "chunk_id": it.get("id"),
            "file_path": it.get("metadata", {}).get("file_path"),
            "start_line": it.get("metadata", {}).get("start_line"),
            "end_line": it.get("metadata", {}).get("end_line"),
            "symbol_name": it.get("metadata", {}).get("symbol_name", ""),
            "chunk_type": it.get("metadata", {}).get("chunk_type"),
            "score": it.get(score_field),
            "reranker_raw_logit": it.get("reranker_raw_logit"),
            "reranker_sigmoid_score": it.get("reranker_sigmoid_score"),
            "original_retrieval_rank": it.get("original_retrieval_rank"),
            "retrieval_sources": it.get("retrieval_sources"),
            "dense_rank": it.get("dense_rank"),
            "bm25_rank": it.get("bm25_rank"),
        }
        for idx, it in enumerate(items[:FINAL_K], start=1)
    ]


def _score_distribution_handoff(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in records if r["answerable"] and r.get("top1_reranker_raw_logit") is not None]
    unans = [r for r in records if not r["answerable"] and r.get("top1_reranker_raw_logit") is not None]

    ans_logits = [float(r["top1_reranker_raw_logit"]) for r in ans]
    unans_logits = [float(r["top1_reranker_raw_logit"]) for r in unans]
    ans_sigs = [float(r["top1_reranker_sigmoid_score"]) for r in ans]
    unans_sigs = [float(r["top1_reranker_sigmoid_score"]) for r in unans]

    ans_min_l = min(ans_logits) if ans_logits else 0.0
    ans_max_l = max(ans_logits) if ans_logits else 0.0
    unans_min_l = min(unans_logits) if unans_logits else 0.0
    unans_max_l = max(unans_logits) if unans_logits else 0.0

    lo_ov = max(ans_min_l, unans_min_l)
    hi_ov = min(ans_max_l, unans_max_l)
    has_overlap = lo_ov <= hi_ov and bool(ans_logits) and bool(unans_logits)

    false_high_unans = sorted(
        [
            {
                "question_id": r["question_id"],
                "repository": r["repository"],
                "question": r["raw_question"],
                "top1_logit": r["top1_reranker_raw_logit"],
                "top1_sigmoid": r["top1_reranker_sigmoid_score"],
                "top1_file": r["top_k_results"][0]["file_path"] if r.get("top_k_results") else None,
            }
            for r in unans
        ],
        key=lambda x: -x["top1_logit"],
    )
    low_ans = sorted(
        [
            {
                "question_id": r["question_id"],
                "repository": r["repository"],
                "category": r["category"],
                "query_style": r["query_style"],
                "hit_at_5": r["metrics"]["hit_rate_at_5"],
                "question": r["raw_question"],
                "top1_logit": r["top1_reranker_raw_logit"],
                "top1_sigmoid": r["top1_reranker_sigmoid_score"],
            }
            for r in ans
        ],
        key=lambda x: x["top1_logit"],
    )[:10]

    return {
        "note": "Exploratory Day 6 handoff distributions only; sigmoid scores are NOT calibrated relevance probabilities.",
        "raw_logit_distribution": {
            "answerable": calc_extended_distribution(ans_logits),
            "unanswerable": calc_extended_distribution(unans_logits),
        },
        "sigmoid_score_distribution": {
            "answerable": calc_extended_distribution(ans_sigs),
            "unanswerable": calc_extended_distribution(unans_sigs),
        },
        "distribution_overlap": {
            "has_overlap": has_overlap,
            "overlap_logit_min": round(lo_ov, 6) if has_overlap else None,
            "overlap_logit_max": round(hi_ov, 6) if has_overlap else None,
            "answerable_below_unans_max_count": sum(1 for v in ans_logits if v <= unans_max_l) if unans_logits else 0,
            "unanswerable_above_ans_min_count": sum(1 for v in unans_logits if v >= ans_min_l) if ans_logits else 0,
        },
        "per_query_top1_scores": [
            {
                "question_id": r["question_id"],
                "answerable": r["answerable"],
                "hit_at_5": r["metrics"]["hit_rate_at_5"] if r["answerable"] else None,
                "top1_raw_logit": r["top1_reranker_raw_logit"],
                "top1_sigmoid": r["top1_reranker_sigmoid_score"],
            }
            for r in records
        ],
        "false_high_score_unanswerable_examples": false_high_unans,
        "lowest_score_answerable_examples": low_ans,
    }


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PER_EXP_DIR, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    repos = json.load(open(REPOS_FILE, encoding="utf-8"))
    questions = json.load(open(QUESTIONS_FILE, encoding="utf-8"))
    ground_truth = json.load(open(GROUND_TRUTH_FILE, encoding="utf-8"))
    emb_suite = json.load(open(EMB_CONFIG_FILE, encoding="utf-8"))
    v0_cfg = next(m for m in emb_suite["models"] if m["experiment_id"] == "E0_v0_minilm_l6_v2")
    day4_summary = json.load(open(DAY4_SUMMARY_FILE, encoding="utf-8"))

    preflight_stats, commit_verifs = run_preflight_checks(repos, questions, ground_truth)
    conv_turn2_count = sum(1 for q in questions if q.get("turn_number") == 2)
    assert conv_turn2_count == 12, f"Expected 12 Turn-2 conversational queries, found {conv_turn2_count}"
    gt_map = {g["question_id"]: g for g in ground_truth}

    # Verify Day 4 ChromaDB collections exist cleanly & match corpus fingerprints
    if not os.path.exists(DAY4_CHROMA_DIR):
        raise RuntimeError(f"Day 4 ChromaDB not found at {DAY4_CHROMA_DIR}")
    client = chromadb.PersistentClient(path=DAY4_CHROMA_DIR)

    embedder = DirectONNXEmbedder(v0_cfg)
    emb_warmup_s = embedder.warmup()

    reranker = ONNXCrossEncoderReranker(
        model_name=DEFAULT_CROSS_ENCODER_MODEL,
        revision=DEFAULT_CROSS_ENCODER_REVISION,
        batch_size=16,
    )
    reranker_warmup_s = reranker.warmup()

    collections: dict[str, Any] = {}
    bm25_indices: dict[str, BM25Retriever] = {}
    chunks_by_repo_file_idx: dict[str, dict[str, list[tuple[int, int]]]] = {}
    corpus_verifications: dict[str, Any] = {}

    for r_info in repos:
        r_name = r_info["name"]
        alias = r_name.split("/")[-1]
        col_name = f"day4_v1_{alias}"
        col = client.get_collection(name=col_name)
        raw = col.get(include=["documents", "metadatas"])
        bm25, fp = build_bm25_from_collection(col, k1=1.5, b=0.75, idf_mode="lucene")
        expected_fp = day4_summary["indexing_stats_by_repo"][r_name]["corpus_fingerprint_sha256"]
        assert fp == expected_fp, f"Day 4 corpus fingerprint mismatch for {r_name}: {fp} != {expected_fp}"
        assert col.count() == day4_summary["indexing_stats_by_repo"][r_name]["chunks"]

        repo_chunks = [
            {"id": cid, "content": doc, "metadata": meta}
            for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]
        collections[r_name] = col
        bm25_indices[r_name] = bm25
        chunks_by_repo_file_idx[r_name] = build_chunks_by_file_index(repo_chunks)
        corpus_verifications[r_name] = {
            "collection_name": col_name,
            "chunks_count": col.count(),
            "corpus_fingerprint_sha256": fp,
            "matches_day4_fingerprint": True,
        }
        print(f"[DAY5-CORPUS][{r_name}] verified {col.count()} chunks | fp={fp[:12]}", flush=True)

    rss_samples: list[float] = [get_rss_mb()]

    def _run_fixed_pipeline(
        query_text: str,
        r_name: str,
        pool_mode: str = "both",
    ) -> dict[str, Any]:
        """Run Stage-1 Hybrid (Dense@20 + BM25@20 + RRF) and requested Stage-2 rerank pool(s)."""
        col = collections[r_name]
        bm25 = bm25_indices[r_name]
        timing_d = {"q_embed_ms": 0.0, "vec_ms": 0.0}

        def _dense_fn(q_str: str, n_req: int) -> list[dict[str, Any]]:
            q_vec, q_emb_s, _ = embedder.encode_single_query(q_str)
            timing_d["q_embed_ms"] = q_emb_s * 1000.0
            t_v0 = time.perf_counter()
            res = col.query(query_embeddings=[q_vec.tolist()], n_results=n_req)
            timing_d["vec_ms"] = (time.perf_counter() - t_v0) * 1000.0
            r_ids = res.get("ids", [[]])[0]
            r_docs = res.get("documents", [[]])[0]
            r_metas = res.get("metadatas", [[]])[0]
            r_dists = res.get("distances", [[]])[0]
            if not r_metas:
                raise RuntimeError("Empty Chroma query metadata response")
            return [
                {"id": str(cid), "content": doc, "metadata": meta, "distance": float(dist)}
                for cid, doc, meta, dist in zip(r_ids, r_docs, r_metas, r_dists)
            ]

        hybrid = HybridRetriever(
            dense_search_fn=_dense_fn,
            bm25_retriever=bm25,
            candidate_k=CANDIDATE_K,
            rrf_k=RRF_K,
        )
        rrf_top5, h_diag = hybrid.retrieve(
            query_text, top_k=FINAL_K, candidate_k=CANDIDATE_K, return_diagnostics=True
        )
        out: dict[str, Any] = {
            "rrf_top5": rrf_top5,
            "h_diag": h_diag,
            "stage1_timings_ms": {
                "dense_embed_ms": round(timing_d["q_embed_ms"], 2),
                "dense_vector_search_ms": round(timing_d["vec_ms"], 2),
                "dense_total_ms": round(timing_d["q_embed_ms"] + timing_d["vec_ms"], 2),
                "bm25_ms": h_diag["timings_ms"]["bm25_ms"],
                "rrf_ms": h_diag["timings_ms"]["rrf_ms"],
                "stage1_hybrid_ms": h_diag["timings_ms"]["total_hybrid_ms"],
            },
        }

        if pool_mode in ("rrf_top_20", "both"):
            t_p0 = time.perf_counter()
            pool_20 = list(h_diag["rrf_top_20"])
            prep_20_ms = (time.perf_counter() - t_p0) * 1000.0
            r20_top5, r20_meta = reranker.rerank(query_text, pool_20, top_k=FINAL_K)
            out["rerank_rrf20"] = {
                "top5": r20_top5,
                "meta": {k: v for k, v in r20_meta.items() if k != "all_reranked"},
                "all_reranked": r20_meta["all_reranked"],
                "candidate_count": len(pool_20),
                "pool_prep_ms": round(prep_20_ms, 3),
                "total_pipeline_ms": round(
                    h_diag["timings_ms"]["total_hybrid_ms"] + prep_20_ms + r20_meta["total_rerank_ms"],
                    2,
                ),
            }

        if pool_mode in ("candidate_union", "both"):
            t_pu0 = time.perf_counter()
            pool_u = list(h_diag["candidate_union"])
            prep_u_ms = (time.perf_counter() - t_pu0) * 1000.0
            ru_top5, ru_meta = reranker.rerank(query_text, pool_u, top_k=FINAL_K)
            out["rerank_union"] = {
                "top5": ru_top5,
                "meta": {k: v for k, v in ru_meta.items() if k != "all_reranked"},
                "all_reranked": ru_meta["all_reranked"],
                "candidate_count": len(pool_u),
                "pool_prep_ms": round(prep_u_ms, 3),
                "total_pipeline_ms": round(
                    h_diag["timings_ms"]["total_hybrid_ms"] + prep_u_ms + ru_meta["total_rerank_ms"],
                    2,
                ),
            }
        return out

    # ==========================================================================
    # PART A: Reranking Candidate Pools (D5-Base, D5-E3a, D5-E3b) on Original Queries
    # ==========================================================================
    base_records: list[dict[str, Any]] = []
    e3a_records: list[dict[str, Any]] = []
    e3b_records: list[dict[str, Any]] = []

    raw_pipeline_cache: dict[str, dict[str, Any]] = {}
    failed_counts = {"D5-Base_hybrid_rrf": 0, "D5-E3a_rerank_rrf20": 0, "D5-E3b_rerank_union": 0}

    for idx_q_a, q in enumerate(questions, start=1):
        qid = q["question_id"]
        if idx_q_a % 15 == 0:
            print(f"[DAY5-PART-A] Processed {idx_q_a}/{len(questions)} queries...", flush=True)
        r_name = q["repository"]
        is_ans = q["category"] != "unanswerable"
        spans = gt_map[qid]["evidence"] if is_ans else []
        rel_repo = (
            count_relevant_chunks_in_repo(evidence_spans=spans, chunks_by_file=chunks_by_repo_file_idx[r_name])
            if is_ans
            else 0
        )

        failed = False
        err_msg: str | None = None
        try:
            pipe_res = _run_fixed_pipeline(q["question"], r_name, pool_mode="both")
            raw_pipeline_cache[qid] = pipe_res
        except Exception as exc:
            failed = True
            err_msg = str(exc)
            for k in failed_counts:
                failed_counts[k] += 1
            pipe_res = None

        for exp_id, target_list, mode_key in (
            ("D5-Base_hybrid_rrf", base_records, "rrf_top5"),
            ("D5-E3a_rerank_rrf20", e3a_records, "rerank_rrf20"),
            ("D5-E3b_rerank_union", e3b_records, "rerank_union"),
        ):
            if failed or pipe_res is None:
                rec = {
                    "question_id": qid,
                    "repository": r_name,
                    "category": q["category"],
                    "query_style": q.get("query_style", "unanswerable" if not is_ans else "identifier"),
                    "answerable": is_ans,
                    "turn_number": q.get("turn_number", 1),
                    "raw_question": q["question"],
                    "failed": True,
                    "error": err_msg,
                    "retrieval_latency_ms": 0.0,
                    "top_k_results": [],
                    "top1_reranker_raw_logit": None,
                    "top1_reranker_sigmoid_score": None,
                }
                if is_ans:
                    qm = compute_query_metrics([], spans, rel_repo, top_k=FINAL_K, failed=True)
                    rec.update(qm)
                target_list.append(rec)
                continue

            st1 = pipe_res["stage1_timings_ms"]
            if mode_key == "rrf_top5":
                top5 = pipe_res["rrf_top5"]
                lat_ms = st1["stage1_hybrid_ms"]
                extra = {
                    "stage1_timings_ms": st1,
                    "top_k_results": _brief_reranked(top5, score_field="rrf_score"),
                }
            else:
                sub = pipe_res[mode_key]
                top5 = sub["top5"]
                lat_ms = sub["total_pipeline_ms"]
                extra = {
                    "candidate_count": sub["candidate_count"],
                    "reranker_meta": sub["meta"],
                    "stage1_timings_ms": st1,
                    "pool_prep_ms": sub["pool_prep_ms"],
                    "top1_reranker_raw_logit": top5[0]["reranker_raw_logit"] if top5 else None,
                    "top1_reranker_sigmoid_score": top5[0]["reranker_sigmoid_score"] if top5 else None,
                    "top_k_results": _brief_reranked(top5, score_field="reranker_raw_logit"),
                }

            rec = {
                "question_id": qid,
                "repository": r_name,
                "category": q["category"],
                "query_style": q.get("query_style", "unanswerable" if not is_ans else "identifier"),
                "answerable": is_ans,
                "turn_number": q.get("turn_number", 1),
                "raw_question": q["question"],
                "failed": False,
                "error": None,
                "retrieval_latency_ms": round(float(lat_ms), 2),
                **extra,
            }
            if is_ans:
                qm = compute_query_metrics(top5, spans, rel_repo, top_k=FINAL_K, failed=False)
                rec.update(qm)
            target_list.append(rec)

        rss_samples.append(get_rss_mb())

    base_overall = aggregate_retrieval_metrics([r for r in base_records if r["answerable"]])
    e3a_overall = aggregate_retrieval_metrics([r for r in e3a_records if r["answerable"]])
    e3b_overall = aggregate_retrieval_metrics([r for r in e3b_records if r["answerable"]])

    day4_ref = day4_summary["methods_summary"]["D4-E2_hybrid_rrf"]["overall_retrieval_metrics"]
    day4_parity_deltas = {
        k: round(base_overall[k] - day4_ref[k], 6)
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

    # Select Best Fixed Day 5 Retrieval Pipeline deterministically from D5-E3a vs D5-E3b
    def _rank_key(m: dict[str, Any]) -> tuple[float, float, float, float]:
        return (m["hit_rate_at_5"], m["mrr_at_5"], m["ndcg_at_5"], m["span_recall_at_5"])

    if _rank_key(e3b_overall) >= _rank_key(e3a_overall):
        best_pool_mode = "candidate_union"
        best_pool_exp_id = "D5-E3b_rerank_union"
        q0_records = e3b_records
        q0_overall = e3b_overall
    else:
        best_pool_mode = "rrf_top_20"
        best_pool_exp_id = "D5-E3a_rerank_rrf20"
        q0_records = e3a_records
        q0_overall = e3a_overall

    print(
        f"[DAY5-RERANK] D5-Base H@5={base_overall['hit_rate_at_5']:.4f} MRR={base_overall['mrr_at_5']:.4f} "
        f"NDCG={base_overall['ndcg_at_5']:.4f} (parity={day4_parity_deltas})",
        flush=True,
    )
    print(
        f"[DAY5-RERANK] D5-E3a(RRF20) H@5={e3a_overall['hit_rate_at_5']:.4f} MRR={e3a_overall['mrr_at_5']:.4f} "
        f"NDCG={e3a_overall['ndcg_at_5']:.4f}",
        flush=True,
    )
    print(
        f"[DAY5-RERANK] D5-E3b(Union) H@5={e3b_overall['hit_rate_at_5']:.4f} MRR={e3b_overall['mrr_at_5']:.4f} "
        f"NDCG={e3b_overall['ndcg_at_5']:.4f} -> Selected {best_pool_exp_id}",
        flush=True,
    )

    # ==========================================================================
    # PART B: Leakage-Free Conversational Query Rewriting (D5-Q1) + Oracle Reference (D5-Q2-Oracle)
    # ==========================================================================
    def _sanitize_rewrite_input(question_obj: dict[str, Any]) -> dict[str, Any]:
        """Firewall: extract ONLY query, previous_context, repository (NEVER oracle or GT!)."""
        return {
            "question_id": question_obj["question_id"],
            "query": question_obj["question"],
            "previous_context": question_obj.get("previous_context"),
            "repository_name": question_obj["repository"],
        }

    q1_records: list[dict[str, Any]] = []
    q1_map_by_id = {r["question_id"]: r for r in q0_records}
    failed_counts["D5-Q1_rewrite_best"] = 0
    conversational_transitions: list[dict[str, Any]] = []

    for idx_q, q in enumerate(questions):
        qid = q["question_id"]
        is_ans = q["category"] != "unanswerable"
        spans = gt_map[qid]["evidence"] if is_ans else []
        rel_repo = (
            count_relevant_chunks_in_repo(evidence_spans=spans, chunks_by_file=chunks_by_repo_file_idx[q["repository"]])
            if is_ans
            else 0
        )

        safe_in = _sanitize_rewrite_input(q)
        # Rewriter receives ONLY query, previous_context, repository_name (zero oracle/GT access)
        rw = rewrite_conversational_query(
            query=safe_in["query"],
            previous_context=safe_in["previous_context"],
            repository_name=safe_in["repository_name"],
            max_retries=2,
            retry_delay_seconds=2.5,
            timeout_seconds=30.0,
        )
        if rw["gemini_calls"] > 0:
            print(
                f"[DAY5-REWRITE][{qid}] rewritten={rw['rewritten']} failed={rw['rewrite_failed']} "
                f"lat={rw['rewrite_latency_ms']}ms query={rw['retrieval_query'][:80]}",
                flush=True,
            )

        # For gated standalone questions, retrieval_query == original query -> reuse deterministic Q0 output identically
        if not rw["rewritten"] and qid in q1_map_by_id and not q1_map_by_id[qid]["failed"]:
            q0_copy = dict(q1_map_by_id[qid])
            q0_copy["rewriter"] = rw
            q1_records.append(q0_copy)
            if q.get("turn_number") == 2:
                conversational_transitions.append(
                    {
                        "question_id": qid,
                        "repository": q["repository"],
                        "original_query": q["question"],
                        "rewritten_query": rw["retrieval_query"],
                        "rewritten": rw["rewritten"],
                        "rewrite_failed": rw["rewrite_failed"],
                        "rewrite_error": rw["error"],
                        "rewrite_latency_ms": rw["rewrite_latency_ms"],
                        "q0_rank": q0_copy["first_relevant_rank"],
                        "q1_rank": q0_copy["first_relevant_rank"],
                        "q0_hit5": q0_copy["metrics"]["hit_rate_at_5"],
                        "q1_hit5": q0_copy["metrics"]["hit_rate_at_5"],
                        "q0_mrr": q0_copy["metrics"]["mrr_at_5"],
                        "q1_mrr": q0_copy["metrics"]["mrr_at_5"],
                    }
                )
            continue

        # Run identical fixed pipeline on rewritten query
        failed = False
        err_msg = None
        try:
            pipe_rw = _run_fixed_pipeline(rw["retrieval_query"], q["repository"], pool_mode=best_pool_mode)
            sub_key = "rerank_union" if best_pool_mode == "candidate_union" else "rerank_rrf20"
            sub = pipe_rw[sub_key]
            top5 = sub["top5"]
            lat_ms = sub["total_pipeline_ms"]
            st1 = pipe_rw["stage1_timings_ms"]
        except Exception as exc:
            failed = True
            err_msg = str(exc)
            failed_counts["D5-Q1_rewrite_best"] += 1
            sub = None
            top5 = []
            lat_ms = 0.0
            st1 = {}

        rec = {
            "question_id": qid,
            "repository": q["repository"],
            "category": q["category"],
            "query_style": q.get("query_style", "identifier"),
            "answerable": is_ans,
            "turn_number": q.get("turn_number", 1),
            "raw_question": q["question"],
            "rewriter": rw,
            "failed": failed,
            "error": err_msg,
            "retrieval_latency_ms": round(float(lat_ms), 2),
            "candidate_count": sub["candidate_count"] if sub else 0,
            "reranker_meta": sub["meta"] if sub else {},
            "stage1_timings_ms": st1,
            "pool_prep_ms": sub["pool_prep_ms"] if sub else 0.0,
            "top1_reranker_raw_logit": top5[0]["reranker_raw_logit"] if top5 else None,
            "top1_reranker_sigmoid_score": top5[0]["reranker_sigmoid_score"] if top5 else None,
            "top_k_results": _brief_reranked(top5, score_field="reranker_raw_logit"),
        }
        if is_ans:
            rec.update(compute_query_metrics(top5, spans, rel_repo, top_k=FINAL_K, failed=failed))
        q1_records.append(rec)

        if q.get("turn_number") == 2 and is_ans:
            q0_ref = q1_map_by_id[qid]
            conversational_transitions.append(
                {
                    "question_id": qid,
                    "repository": q["repository"],
                    "original_query": q["question"],
                    "rewritten_query": rw["retrieval_query"],
                    "rewritten": rw["rewritten"],
                    "rewrite_failed": rw["rewrite_failed"],
                    "rewrite_error": rw["error"],
                    "rewrite_latency_ms": rw["rewrite_latency_ms"],
                    "q0_rank": q0_ref["first_relevant_rank"],
                    "q1_rank": rec["first_relevant_rank"],
                    "q0_hit5": q0_ref["metrics"]["hit_rate_at_5"],
                    "q1_hit5": rec["metrics"]["hit_rate_at_5"],
                    "q0_mrr": q0_ref["metrics"]["mrr_at_5"],
                    "q1_mrr": rec["metrics"]["mrr_at_5"],
                }
            )
        # Polite small spacing between Gemini calls for free-tier stability
        if rw["gemini_calls"] > 0:
            time.sleep(0.5)

    q1_overall = aggregate_retrieval_metrics([r for r in q1_records if r["answerable"]])
    q0_conv_records = [r for r in q0_records if r.get("turn_number") == 2 and r["answerable"]]
    q1_conv_records = [r for r in q1_records if r.get("turn_number") == 2 and r["answerable"]]
    q0_conv_metrics = aggregate_retrieval_metrics(q0_conv_records)
    q1_conv_metrics = aggregate_retrieval_metrics(q1_conv_records)

    # ==========================================================================
    # PART C: Optional Offline Oracle Reference (D5-Q2-Oracle) — STRICTLY SEPARATED
    # Executed AFTER D5-Q1 completed; NEVER influences production rewriter or D5-Q1.
    # ==========================================================================
    q2_oracle_records: list[dict[str, Any]] = []
    failed_counts["D5-Q2_oracle_ref"] = 0
    for idx_q_a, q in enumerate(questions, start=1):
        qid = q["question_id"]
        if idx_q_a % 15 == 0:
            print(f"[DAY5-PART-A] Processed {idx_q_a}/{len(questions)} queries...", flush=True)
        is_ans = q["category"] != "unanswerable"
        spans = gt_map[qid]["evidence"] if is_ans else []
        rel_repo = (
            count_relevant_chunks_in_repo(evidence_spans=spans, chunks_by_file=chunks_by_repo_file_idx[q["repository"]])
            if is_ans
            else 0
        )
        oracle_q = q.get("expected_standalone_query")
        if not oracle_q:
            q0_copy = dict(q1_map_by_id[qid])
            q0_copy["oracle_query_used"] = q["question"]
            q2_oracle_records.append(q0_copy)
            continue

        try:
            pipe_or = _run_fixed_pipeline(oracle_q, q["repository"], pool_mode=best_pool_mode)
            sub_key = "rerank_union" if best_pool_mode == "candidate_union" else "rerank_rrf20"
            sub = pipe_or[sub_key]
            top5 = sub["top5"]
            rec = {
                "question_id": qid,
                "repository": q["repository"],
                "category": q["category"],
                "query_style": q.get("query_style", "conversational"),
                "answerable": is_ans,
                "turn_number": q.get("turn_number", 2),
                "raw_question": q["question"],
                "oracle_query_used": oracle_q,
                "failed": False,
                "error": None,
                "retrieval_latency_ms": round(float(sub["total_pipeline_ms"]), 2),
                "top1_reranker_raw_logit": top5[0]["reranker_raw_logit"] if top5 else None,
                "top1_reranker_sigmoid_score": top5[0]["reranker_sigmoid_score"] if top5 else None,
                "top_k_results": _brief_reranked(top5, score_field="reranker_raw_logit"),
            }
            if is_ans:
                rec.update(compute_query_metrics(top5, spans, rel_repo, top_k=FINAL_K, failed=False))
            q2_oracle_records.append(rec)
        except Exception as exc:
            failed_counts["D5-Q2_oracle_ref"] += 1
            rec = {
                "question_id": qid,
                "repository": q["repository"],
                "category": q["category"],
                "query_style": q.get("query_style", "conversational"),
                "answerable": is_ans,
                "turn_number": q.get("turn_number", 2),
                "raw_question": q["question"],
                "oracle_query_used": oracle_q,
                "failed": True,
                "error": str(exc),
                "retrieval_latency_ms": 0.0,
                "top_k_results": [],
            }
            if is_ans:
                rec.update(compute_query_metrics([], spans, rel_repo, top_k=FINAL_K, failed=True))
            q2_oracle_records.append(rec)

    q2_oracle_overall = aggregate_retrieval_metrics([r for r in q2_oracle_records if r["answerable"]])
    q2_oracle_conv = aggregate_retrieval_metrics(
        [r for r in q2_oracle_records if r.get("turn_number") == 2 and r["answerable"]]
    )

    rss_samples.append(get_rss_mb())
    peak_rss_mb = max(rss_samples)

    # Aggregate candidate count, truncation, and latency telemetry cleanly
    def _latency_and_trunc_summary(recs: Sequence[dict[str, Any]], include_rerank: bool) -> dict[str, Any]:
        all_tot = [r["retrieval_latency_ms"] for r in recs]
        dense_emb = [r.get("stage1_timings_ms", {}).get("dense_embed_ms", 0.0) for r in recs]
        dense_vec = [r.get("stage1_timings_ms", {}).get("dense_vector_search_ms", 0.0) for r in recs]
        bm25_l = [r.get("stage1_timings_ms", {}).get("bm25_ms", 0.0) for r in recs]
        rrf_l = [r.get("stage1_timings_ms", {}).get("rrf_ms", 0.0) for r in recs]
        out: dict[str, Any] = {
            "total_retrieval_latency_ms": compute_latency_stats(all_tot),
            "dense_embed_ms": compute_latency_stats(dense_emb),
            "dense_vector_search_ms": compute_latency_stats(dense_vec),
            "bm25_search_ms": compute_latency_stats(bm25_l),
            "rrf_fusion_ms": compute_latency_stats(rrf_l),
        }
        if include_rerank:
            cands = [r.get("candidate_count", 0) for r in recs]
            prep_l = [r.get("pool_prep_ms", 0.0) for r in recs]
            tok_l = [r.get("reranker_meta", {}).get("tokenization_ms", 0.0) for r in recs]
            inf_l = [r.get("reranker_meta", {}).get("inference_ms", 0.0) for r in recs]
            rer_l = [r.get("reranker_meta", {}).get("total_rerank_ms", 0.0) for r in recs]
            pairs_trunc = sum(r.get("reranker_meta", {}).get("pairs_truncated_count", 0) for r in recs)
            q_trunc = sum(1 for r in recs if r.get("reranker_meta", {}).get("query_truncated", False))
            c_trunc = sum(r.get("reranker_meta", {}).get("candidate_inputs_truncated_count", 0) for r in recs)
            total_pairs = sum(r.get("reranker_meta", {}).get("pairs_count", 0) for r in recs)
            out.update(
                {
                    "candidate_pool_size_stats": calc_extended_distribution(cands),
                    "total_cross_encoder_pairs_scored": total_pairs,
                    "candidate_pool_prep_ms": compute_latency_stats(prep_l),
                    "reranker_tokenization_ms": compute_latency_stats(tok_l),
                    "reranker_inference_ms": compute_latency_stats(inf_l),
                    "reranker_total_ms": compute_latency_stats(rer_l),
                    "truncation_telemetry": {
                        "total_pairs_scored": total_pairs,
                        "pairs_truncated_count": pairs_trunc,
                        "pairs_truncated_pct": round(100.0 * pairs_trunc / max(1, total_pairs), 2),
                        "query_inputs_truncated_count": q_trunc,
                        "candidate_inputs_truncated_count": c_trunc,
                    },
                }
            )
        return out

    rewrite_latencies = [
        r["rewriter"]["rewrite_latency_ms"] for r in q1_records if r.get("rewriter", {}).get("gate_triggered")
    ]
    rewritten_count = sum(1 for r in q1_records if r.get("rewriter", {}).get("rewritten"))
    rewrite_failures = sum(1 for r in q1_records if r.get("rewriter", {}).get("rewrite_failed"))
    standalone_unchanged = sum(
        1 for r in q1_records if not r.get("rewriter", {}).get("gate_triggered")
    )

    exp_payloads = {
        "D5-Base_hybrid_rrf": {
            "experiment_id": "D5-Base_hybrid_rrf",
            "label": "D5-Base: Day 4 Hybrid RRF Parity Baseline (k=60)",
            "failed_queries": failed_counts["D5-Base_hybrid_rrf"],
            "overall_retrieval_metrics": base_overall,
            "breakdowns": compute_breakdowns([r for r in base_records if r["answerable"]], repos),
            "latency_and_cost": _latency_and_trunc_summary(base_records, include_rerank=False),
            "records": base_records,
        },
        "D5-E3a_rerank_rrf20": {
            "experiment_id": "D5-E3a_rerank_rrf20",
            "label": "D5-E3a: Hybrid RRF Top-20 -> ONNX Cross-Encoder -> Top-5",
            "failed_queries": failed_counts["D5-E3a_rerank_rrf20"],
            "overall_retrieval_metrics": e3a_overall,
            "breakdowns": compute_breakdowns([r for r in e3a_records if r["answerable"]], repos),
            "latency_and_cost": _latency_and_trunc_summary(e3a_records, include_rerank=True),
            "paired_vs_hybrid_rrf": compute_paired_comparison(base_records, e3a_records),
            "day6_score_handoff": _score_distribution_handoff(e3a_records),
            "records": e3a_records,
        },
        "D5-E3b_rerank_union": {
            "experiment_id": "D5-E3b_rerank_union",
            "label": "D5-E3b: Full Deduplicated Candidate Union -> ONNX Cross-Encoder -> Top-5",
            "failed_queries": failed_counts["D5-E3b_rerank_union"],
            "overall_retrieval_metrics": e3b_overall,
            "breakdowns": compute_breakdowns([r for r in e3b_records if r["answerable"]], repos),
            "latency_and_cost": _latency_and_trunc_summary(e3b_records, include_rerank=True),
            "paired_vs_hybrid_rrf": compute_paired_comparison(base_records, e3b_records),
            "paired_vs_e3a": compute_paired_comparison(e3a_records, e3b_records),
            "day6_score_handoff": _score_distribution_handoff(e3b_records),
            "records": e3b_records,
        },
        "D5-Q1_rewrite_best": {
            "experiment_id": "D5-Q1_rewrite_best",
            "label": f"D5-Q1: Leakage-Free Gemini Rewrite Gate -> {best_pool_exp_id}",
            "fixed_pipeline_used": best_pool_exp_id,
            "failed_queries": failed_counts["D5-Q1_rewrite_best"],
            "overall_retrieval_metrics": q1_overall,
            "conversational_subset_metrics": {"Q0_original": q0_conv_metrics, "Q1_rewritten": q1_conv_metrics},
            "breakdowns": compute_breakdowns([r for r in q1_records if r["answerable"]], repos),
            "latency_and_cost": {
                **_latency_and_trunc_summary(q1_records, include_rerank=True),
                "query_rewriting_gate_stats": {
                    "standalone_unchanged_count": standalone_unchanged,
                    "conversational_gated_count": len(rewrite_latencies),
                    "successfully_rewritten_count": rewritten_count,
                    "rewrite_failures_count": rewrite_failures,
                    "rewrite_latency_ms": compute_latency_stats(rewrite_latencies),
                },
            },
            "paired_vs_q0_overall": compute_paired_comparison(q0_records, q1_records),
            "day6_score_handoff": _score_distribution_handoff(q1_records),
            "records": q1_records,
        },
        "D5-Q2_oracle_ref": {
            "experiment_id": "D5-Q2_oracle_ref",
            "label": "D5-Q2-Oracle: Offline Oracle Reference — Not a Deployable Configuration",
            "warning": "Offline Oracle Reference — Not a Deployable Configuration; strictly separate from production rewriter.",
            "fixed_pipeline_used": best_pool_exp_id,
            "failed_queries": failed_counts["D5-Q2_oracle_ref"],
            "overall_retrieval_metrics": q2_oracle_overall,
            "conversational_subset_metrics": q2_oracle_conv,
            "records": q2_oracle_records,
        },
    }

    for eid, edata in exp_payloads.items():
        with open(os.path.join(PER_EXP_DIR, f"{eid}_evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(edata["records"], f, indent=2)
        with open(os.path.join(PER_EXP_DIR, f"{eid}_summary.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in edata.items() if k != "records"}, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "conversational_subset_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "fixed_pipeline_used": best_pool_exp_id,
                "D5-Q0_conversational_metrics": q0_conv_metrics,
                "D5-Q1_conversational_metrics": q1_conv_metrics,
                "D5-Q2_oracle_conversational_metrics_NOT_DEPLOYABLE": q2_oracle_conv,
                "per_query_transitions": conversational_transitions,
            },
            f,
            indent=2,
        )

    day5_summary = {
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
                "embedder_load_seconds": embedder.model_load_duration_seconds,
                "embedder_warmup_seconds": emb_warmup_s,
                "cross_encoder_info": reranker.get_model_info(),
                "cross_encoder_warmup_seconds": reranker_warmup_s,
                "candidate_k": CANDIDATE_K,
                "final_k": FINAL_K,
                "rrf_k": RRF_K,
                "preflight_stats": preflight_stats,
                "conversational_turn2_count": conv_turn2_count,
                "commit_verifications": commit_verifs,
                "day4_corpus_verifications": corpus_verifications,
            },
            "script_sha256": get_file_sha256(os.path.abspath(__file__)),
            "reranker_module_sha256": get_file_sha256(
                os.path.join(PROJECT_ROOT, "retrieval", "reranker.py")
            ),
            "query_rewriter_module_sha256": get_file_sha256(
                os.path.join(PROJECT_ROOT, "retrieval", "query_rewriter.py")
            ),
        },
        "validity_status": "VALID" if sum(failed_counts.values()) == 0 else "INVALID_OR_INCOMPLETE",
        "day4_hybrid_parity_deltas": day4_parity_deltas,
        "selected_best_fixed_pipeline": best_pool_exp_id,
        "peak_rss_mb": round(peak_rss_mb, 2),
        "experiments_summary": {
            eid: {k: v for k, v in edata.items() if k != "records"}
            for eid, edata in exp_payloads.items()
        },
        "conversational_subset_summary": {
            "count": len(q0_conv_records),
            "Q0_original": q0_conv_metrics,
            "Q1_leakage_free_rewrite": q1_conv_metrics,
            "Q2_oracle_ref_NOT_DEPLOYABLE": q2_oracle_conv,
            "transitions": conversational_transitions,
        },
    }

    with open(os.path.join(RESULTS_DIR, "day5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(day5_summary, f, indent=2)

    print("\n=== DAY 5 RERANKING & QUERY REWRITING BENCHMARK COMPLETE ===", flush=True)
    for eid, es in day5_summary["experiments_summary"].items():
        m = es["overall_retrieval_metrics"]
        print(
            f"  {eid:22s} | H@1={m['hit_rate_at_1']:.4f} H@3={m['hit_rate_at_3']:.4f} "
            f"H@5={m['hit_rate_at_5']:.4f} P@5={m['precision_at_5']:.4f} "
            f"SR@5={m['span_recall_at_5']:.4f} MRR={m['mrr_at_5']:.4f} NDCG={m['ndcg_at_5']:.4f}",
            flush=True,
        )
    print(
        f"  Conv Subset (N={len(q0_conv_records)}) | Q0 H@5={q0_conv_metrics['hit_rate_at_5']:.4f} MRR={q0_conv_metrics['mrr_at_5']:.4f} "
        f"| Q1 H@5={q1_conv_metrics['hit_rate_at_5']:.4f} MRR={q1_conv_metrics['mrr_at_5']:.4f} "
        f"| Oracle(Ref) H@5={q2_oracle_conv['hit_rate_at_5']:.4f} MRR={q2_oracle_conv['mrr_at_5']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
