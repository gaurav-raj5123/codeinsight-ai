#!/usr/bin/env python3
"""
Task 2: Pre-Modification Day 3 Baseline Validation.
Runs the unmodified V0 chunker (`chunker.chunk_code`) with the V0 ONNX embedding
pipeline (`all-MiniLM-L6-v2`) into isolated validation collections and verifies
exact reproduction of historical Day 1 metrics before any V1 implementation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import chromadb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chunker import chunk_code
from evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    build_chunks_by_file_index,
    compute_latency_stats,
    compute_query_metrics,
    count_relevant_chunks_in_repo,
)
from experiments.embedding_experiments.run_embedding_comparison import (
    CONFIG_FILE as EMB_CONFIG_FILE,
    DirectONNXEmbedder,
    run_preflight_checks,
)
from repo_loader import get_source_files

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CHROMA_VAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_day3_db")
OUT_FILE = os.path.join(OUT_DIR, "baseline_validation.json")

DATASETS_DIR = os.path.join(PROJECT_ROOT, "experiments", "datasets")
CACHE_DIR = os.path.join(DATASETS_DIR, "cache")
REPOS_FILE = os.path.join(DATASETS_DIR, "repositories.json")
QUESTIONS_FILE = os.path.join(DATASETS_DIR, "evaluation_questions.json")
GROUND_TRUTH_FILE = os.path.join(DATASETS_DIR, "ground_truth.json")
DAY1_SUMMARY_FILE = os.path.join(PROJECT_ROOT, "experiments", "results", "baseline_summary.json")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CHROMA_VAL_DIR, exist_ok=True)

    repos = json.load(open(REPOS_FILE, encoding="utf-8"))
    questions = json.load(open(QUESTIONS_FILE, encoding="utf-8"))
    ground_truth = json.load(open(GROUND_TRUTH_FILE, encoding="utf-8"))
    day1_summary = json.load(open(DAY1_SUMMARY_FILE, encoding="utf-8"))
    emb_suite = json.load(open(EMB_CONFIG_FILE, encoding="utf-8"))
    v0_cfg = next(m for m in emb_suite["models"] if m["experiment_id"] == "E0_v0_minilm_l6_v2")

    preflight_stats, commit_verifs = run_preflight_checks(repos, questions, ground_truth)
    gt_map = {g["question_id"]: g for g in ground_truth}

    embedder = DirectONNXEmbedder(v0_cfg)
    warmup_s = embedder.warmup()

    client = chromadb.PersistentClient(path=CHROMA_VAL_DIR)
    collections = {}
    chunks_by_repo_file = {}
    repo_indexing_stats = {}

    for r in repos:
        r_name = r["name"]
        alias = r_name.split("/")[-1]
        repo_dir = os.path.join(CACHE_DIR, alias)
        t0 = time.perf_counter()
        sfs = get_source_files(repo_dir)
        extract_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        chunks = []
        for sf in sfs:
            chunks.extend(chunk_code(sf))
        chunk_s = time.perf_counter() - t1

        chunks_by_repo_file[r_name] = build_chunks_by_file_index(chunks)
        docs = [c["content"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        vecs, emb_stats = embedder.encode_texts(docs, prefix="", batch_size=32, track_truncation=True)

        col_name = f"day3_val_v0_{alias}"
        try:
            client.delete_collection(name=col_name)
        except Exception:
            pass
        col = client.create_collection(name=col_name, metadata={"hnsw:space": "l2"})

        t_ins = time.perf_counter()
        for start in range(0, len(chunks), 100):
            end = start + 100
            col.add(embeddings=vecs[start:end].tolist(), documents=docs[start:end], metadatas=metas[start:end], ids=ids[start:end])
        ins_s = time.perf_counter() - t_ins

        collections[r_name] = col
        repo_indexing_stats[r_name] = {
            "files": len(sfs),
            "chunks": len(chunks),
            "extract_s": round(extract_s, 4),
            "chunk_s": round(chunk_s, 4),
            "embed_s": emb_stats["duration_seconds"],
            "insert_s": round(ins_s, 4),
        }
        print(f"[VAL-V0][{r_name}] Indexed {len(chunks)} chunks | Embed: {emb_stats['duration_seconds']:.2f}s | Insert: {ins_s:.2f}s", flush=True)

    answerable_records = []
    all_records = []
    failed_queries = 0

    for q in questions:
        qid = q["question_id"]
        r_name = q["repository"]
        is_ans = q["category"] != "unanswerable"
        col = collections[r_name]
        try:
            q_vec, q_emb_s, _ = embedder.encode_single_query(q["question"])
            t_s = time.perf_counter()
            res = col.query(query_embeddings=[q_vec.tolist()], n_results=5)
            vec_s = time.perf_counter() - t_s
            total_ms = round((q_emb_s + vec_s) * 1000.0, 2)
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            ret_chunks = [{"metadata": m} for m in metas]
            err = None
            failed = False
        except Exception as exc:
            failed = True
            failed_queries += 1
            err = str(exc)
            total_ms = 0.0
            ret_chunks = []
            dists = []

        if is_ans:
            spans = gt_map[qid]["evidence"]
            rel_repo = count_relevant_chunks_in_repo(evidence_spans=spans, chunks_by_file=chunks_by_repo_file[r_name])
            qm = compute_query_metrics(ret_chunks, spans, rel_repo, top_k=5, failed=failed)
            rec = {
                "question_id": qid,
                "repository": r_name,
                "category": q["category"],
                "query_style": q.get("query_style", "identifier"),
                "retrieval_latency_ms": total_ms,
                "metrics": qm["metrics"],
                "failed": failed,
                "error": err,
            }
            answerable_records.append(rec)
        all_records.append({
            "question_id": qid,
            "answerable": is_ans,
            "top1_l2": dists[0] if dists else None,
            "retrieval_latency_ms": total_ms,
            "failed": failed,
        })

    reproduced = aggregate_retrieval_metrics(answerable_records)
    historical = day1_summary["overall_retrieval_metrics"]
    deltas = {
        k: round(reproduced[k] - historical[k], 6)
        for k in ["hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5", "precision_at_5", "span_recall_at_5", "mrr_at_5", "ndcg_at_5"]
    }
    exact_match = all(abs(v) == 0.0 for v in deltas.values()) and failed_queries == 0

    out = {
        "validation_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "VALID_EXACT_MATCH" if exact_match else "DISCREPANCY_DETECTED",
            "exact_metric_reproduction": exact_match,
            "failed_queries": failed_queries,
            "warmup_duration_seconds": warmup_s,
            "preflight_stats": preflight_stats,
            "commit_verifications": commit_verifs,
        },
        "indexing_stats_by_repo": repo_indexing_stats,
        "reproduced_metrics": reproduced,
        "historical_day1_metrics": historical,
        "deltas_vs_historical": deltas,
        "all_queries_latency_ms": compute_latency_stats([r["retrieval_latency_ms"] for r in all_records]),
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== TASK 2 BASELINE VALIDATION COMPLETE ===", flush=True)
    print("Reproduced:", {k: reproduced[k] for k in deltas}, flush=True)
    print("Historical:", {k: historical[k] for k in deltas}, flush=True)
    print("Deltas    :", deltas, flush=True)
    print("Exact match:", exact_match, flush=True)
    return 0 if exact_match else 1


if __name__ == "__main__":
    sys.exit(main())
