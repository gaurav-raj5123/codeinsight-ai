#!/usr/bin/env python3
"""
======================================================================
REPRODUCIBLE V0 BASELINE EVALUATION HARNESS (run_baseline_evaluation.py)
======================================================================
Runs the original (V0) dense retrieval system against the 60-question
evaluation benchmark dataset and produces reproducible baseline results.

Controls & Constraints:
- Pinned commit snapshots verified via .commit_sha against repositories.json
- Raw 'question' text evaluated as-is (no query rewriting for Turn-2)
- Zero Gemini generation calls (retrieval-only evaluation)
- Unmodified V0 pipeline: repo_loader -> chunker -> embedder (ChromaDB L2)
- Preflight integrity check of expected counts and commit SHAs before indexing
- Unique evidence span matching & formal interval-overlap proxy rule
- Corrected NDCG@5: IDCG computed using total relevant chunks in indexed repo (capped at 5)
- Strict failure accounting: failed_queries increments on exceptions and error metadata
- Dual latency reporting: separate stats for answerable (N=54) and all queries (N=60)
- Standard statistics.median for P50 and nearest rank index floor(0.95 * N) for P95
- Aggregated metrics with breakdowns by repo, category, query style, and span count
- Saves detailed per-question records and aggregated summary JSON
"""

import os
import sys
import json
import time
import math
import hashlib
import platform
import statistics
import argparse
from datetime import datetime, timezone

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from repo_loader import get_source_files
from chunker import chunk_code
import embedder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
CACHE_DIR = os.path.join(DATASETS_DIR, "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REPOS_FILE = os.path.join(DATASETS_DIR, "repositories.json")
QUESTIONS_FILE = os.path.join(DATASETS_DIR, "evaluation_questions.json")
GROUND_TRUTH_FILE = os.path.join(DATASETS_DIR, "ground_truth.json")

EVAL_OUTPUT_FILE = os.path.join(RESULTS_DIR, "baseline_evaluation.json")
SUMMARY_OUTPUT_FILE = os.path.join(RESULTS_DIR, "baseline_summary.json")

# Dedicated collection name mapping to ensure total isolation and stale-data prevention
COLLECTION_PREFIX = "v0_eval_"

def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_preflight_checks(repos: list[dict], questions: list[dict], ground_truth: list[dict]) -> tuple[dict, dict]:
    """Verifies expected dataset properties and pinned commit SHAs before execution."""
    print("Running Preflight Dataset & Commit SHA Verification...")
    total_q = len(questions)
    answerable_q = sum(1 for g in ground_truth if g.get("answerable"))
    unanswerable_q = sum(1 for g in ground_truth if not g.get("answerable"))
    total_spans = sum(len(g.get("evidence", [])) for g in ground_truth)
    multi_ev = sum(1 for g in ground_truth if len(g.get("evidence", [])) >= 2)

    repo_counts = {}
    for q in questions:
        r = q["repository"]
        repo_counts[r] = repo_counts.get(r, 0) + 1

    expected = {
        "total_questions": 60,
        "answerable_questions": 54,
        "unanswerable_questions": 6,
        "total_evidence_spans": 71,
        "multi_evidence_questions": 17,
        "questions_per_repo": 20
    }

    errors = []
    if total_q != expected["total_questions"]:
        errors.append(f"Total questions mismatch: found {total_q}, expected {expected['total_questions']}")
    if answerable_q != expected["answerable_questions"]:
        errors.append(f"Answerable questions mismatch: found {answerable_q}, expected {expected['answerable_questions']}")
    if unanswerable_q != expected["unanswerable_questions"]:
        errors.append(f"Unanswerable questions mismatch: found {unanswerable_q}, expected {expected['unanswerable_questions']}")
    if total_spans != expected["total_evidence_spans"]:
        errors.append(f"Total evidence spans mismatch: found {total_spans}, expected {expected['total_evidence_spans']}")
    if multi_ev != expected["multi_evidence_questions"]:
        errors.append(f"Multi-evidence questions mismatch: found {multi_ev}, expected {expected['multi_evidence_questions']}")
    for r, count in repo_counts.items():
        if count != expected["questions_per_repo"]:
            errors.append(f"Questions for repo '{r}' mismatch: found {count}, expected {expected['questions_per_repo']}")

    # Independent Pinned Commit SHA Verification
    commit_verifications = {}
    for r in repos:
        repo_name = r["name"]
        alias = repo_name.split("/")[-1]
        expected_sha = r["commit_sha"].strip().lower()
        sha_file = os.path.join(CACHE_DIR, alias, ".commit_sha")
        if not os.path.exists(sha_file):
            errors.append(f"Missing .commit_sha marker file for {repo_name} at {sha_file}")
            continue
        with open(sha_file, "r", encoding="utf-8") as f_sha:
            actual_sha = f_sha.read().strip().lower()
        if actual_sha != expected_sha:
            errors.append(f"Commit SHA mismatch for {repo_name}: expected {expected_sha}, found {actual_sha}")
        else:
            commit_verifications[repo_name] = {
                "verified": True,
                "commit_sha": actual_sha,
                "marker_path": sha_file
            }

    if errors:
        print("PREFLIGHT VERIFICATION FAILED:")
        for err in errors:
            print("  -", err)
        raise ValueError("Preflight dataset integrity check failed! Stopping benchmark.")

    print(f"Preflight Verified: {total_q} questions across {len(repo_counts)} repos.")
    print("Pinned commit SHAs verified against .commit_sha marker files for all repositories.")
    return {
        "total_questions": total_q,
        "answerable_questions": answerable_q,
        "unanswerable_questions": unanswerable_q,
        "total_evidence_spans": total_spans,
        "multi_evidence_questions": multi_ev
    }, commit_verifications

def check_overlap(c_start: int, c_end: int, e_start: int, e_end: int) -> bool:
    """Formal interval overlap condition: max(s1, s2) <= min(e1, e2)"""
    return max(c_start, e_start) <= min(c_end, e_end)

def calculate_dcg(relevance_vector: list[int]) -> float:
    """DCG@K = sum(rel_i / log2(i + 1)) for i in 1..K (rank 1 uses log2(2) = 1.0)"""
    dcg = 0.0
    for idx, rel in enumerate(relevance_vector, start=1):
        if rel > 0:
            dcg += rel / math.log2(idx + 1)
    return dcg

def calculate_ndcg(relevance_vector: list[int], total_relevant_chunks_in_repo: int) -> float:
    """Calculates NDCG@K for binary relevance.
    IDCG is computed using the total number of relevant chunks in the indexed repository,
    capped at K (here K = len(relevance_vector) = 5).
    NDCG = DCG / IDCG in [0.0, 1.0].
    """
    dcg = calculate_dcg(relevance_vector)
    if dcg == 0.0 or total_relevant_chunks_in_repo <= 0:
        return 0.0

    k = len(relevance_vector)
    ideal_rel_count = min(total_relevant_chunks_in_repo, k)
    ideal_vector = [1] * ideal_rel_count + [0] * (k - ideal_rel_count)
    idcg = calculate_dcg(ideal_vector)
    if idcg == 0.0:
        return 0.0
    return min(1.0, dcg / idcg)

def count_relevant_chunks_in_repo(chunks: list[dict], evidence_spans: list[dict]) -> int:
    """Counts how many chunks in the indexed repository overlap any evidence span for this query."""
    count = 0
    for ch in chunks:
        ch_meta = ch.get("metadata", {})
        c_file = ch_meta.get("file_path")
        c_start = ch_meta.get("start_line")
        c_end = ch_meta.get("end_line")
        if c_file is None or c_start is None or c_end is None:
            continue
        for span in evidence_spans:
            if c_file == span["file_path"] and check_overlap(c_start, c_end, span["start_line"], span["end_line"]):
                count += 1
                break
    return count

def compute_latency_stats(latencies: list[float]) -> dict:
    """Calculates latency statistics using standard statistics.median and floor(0.95 * N) for P95."""
    if not latencies:
        return {}
    n = len(latencies)
    sorted_lat = sorted(latencies)
    p50 = statistics.median(latencies)
    p95_idx = min(int(0.95 * n), n - 1)
    p95 = sorted_lat[p95_idx]
    return {
        "count": n,
        "mean": round(sum(latencies) / n, 2),
        "median_p50": round(p50, 2),
        "p95": round(p95, 2),
        "max": round(max(latencies), 2),
        "p95_rank_index": p95_idx,
        "p95_convention": "nearest rank index floor(0.95 * N)"
    }

def run_benchmark(force_reindex: bool = False, recompute_only: bool = False):
    run_start_time = time.time()
    utc_iso_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = f"v0_baseline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    print("=" * 75)
    print("   CODEBASE AI ASSISTANT — V0 BASELINE BENCHMARK EVALUATION")
    print("=" * 75)
    print(f"Run ID    : {run_id}")
    print(f"Timestamp : {utc_iso_timestamp}")

    # Load datasets
    with open(REPOS_FILE, "r", encoding="utf-8") as f:
        repos_data = json.load(f)
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    # 1. Preflight Verification (including commit SHAs)
    preflight_stats, commit_verifications = run_preflight_checks(repos_data, questions_data, ground_truth_data)
    gt_map = {item["question_id"]: item for item in ground_truth_data}

    # Record dataset fingerprints
    dataset_fingerprints = {
        "repositories.json_sha256": get_file_sha256(REPOS_FILE),
        "evaluation_questions.json_sha256": get_file_sha256(QUESTIONS_FILE),
        "ground_truth.json_sha256": get_file_sha256(GROUND_TRUTH_FILE)
    }

    # 2. Extract and Chunk Source Files (for repository chunks & indexing)
    print("\n" + "-" * 75)
    print("STAGE 1: PREPARING REPOSITORIES & V0 CHUNKING")
    print("-" * 75)

    repo_chunks = {}
    indexing_records = {}
    repo_collections = {}

    # Check for existing summary to preserve timing metadata when not force reindexing
    prev_indexing_summary = {}
    if os.path.exists(SUMMARY_OUTPUT_FILE):
        try:
            with open(SUMMARY_OUTPUT_FILE, "r", encoding="utf-8") as f_prev:
                prev_indexing_summary = json.load(f_prev).get("indexing_summary", {})
        except Exception:
            pass

    client = embedder.get_chroma_client()

    for repo_info in repos_data:
        repo_name = repo_info["name"]
        alias = repo_name.split("/")[-1]
        commit_sha = repo_info["commit_sha"]
        repo_path = os.path.join(CACHE_DIR, alias)

        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Pinned repository snapshot not found at {repo_path}. Run validate_dataset.py first.")

        dedicated_collection_name = f"{COLLECTION_PREFIX}{repo_name.replace('/', '_')}"
        repo_collections[repo_name] = dedicated_collection_name

        t_load_start = time.perf_counter()
        source_files = get_source_files(repo_path)
        load_duration = time.perf_counter() - t_load_start

        t_chunk_start = time.perf_counter()
        all_chunks = []
        for sf in source_files:
            chunks = chunk_code(sf)
            all_chunks.extend(chunks)
        chunk_duration = time.perf_counter() - t_chunk_start

        repo_chunks[repo_name] = all_chunks
        print(f"[{repo_name}] Extracted {len(source_files)} files -> Generated {len(all_chunks)} chunks in {chunk_duration:.4f}s")

        # Check if collection already exists in ChromaDB
        existing_col = None
        if not force_reindex:
            try:
                existing_col = client.get_collection(dedicated_collection_name)
            except Exception:
                pass

        if existing_col is not None and existing_col.count() == len(all_chunks) and not force_reindex:
            print(f"[{repo_name}] Found existing collection '{dedicated_collection_name}' with {existing_col.count()} chunks. Preserving index.")
            prev_record = prev_indexing_summary.get(repo_name, {})
            indexing_records[repo_name] = {
                "repository": repo_name,
                "commit_sha": commit_sha,
                "collection_name": dedicated_collection_name,
                "source_files_count": len(source_files),
                "chunks_count": len(all_chunks),
                "load_files_duration_seconds": prev_record.get("load_files_duration_seconds", round(load_duration, 4)),
                "chunking_duration_seconds": prev_record.get("chunking_duration_seconds", round(chunk_duration, 4)),
                "collection_add_duration_seconds": prev_record.get("collection_add_duration_seconds", 0.0),
                "total_index_function_duration_seconds": prev_record.get("total_index_function_duration_seconds", 0.0),
                "embedding_model": "DefaultEmbeddingFunction (all-MiniLM-L6-v2)",
                "embedding_dimension": 384,
                "distance_metric": "l2"
            }
        else:
            print(f"[{repo_name}] Indexing {len(all_chunks)} chunks into ChromaDB '{dedicated_collection_name}'...")
            total_indexed, index_meta = embedder.index_chunks(
                chunks=all_chunks,
                repo_name=dedicated_collection_name,
                return_metadata=True
            )
            indexing_records[repo_name] = {
                "repository": repo_name,
                "commit_sha": commit_sha,
                "collection_name": dedicated_collection_name,
                "source_files_count": len(source_files),
                "chunks_count": total_indexed,
                "load_files_duration_seconds": round(load_duration, 4),
                "chunking_duration_seconds": round(chunk_duration, 4),
                "collection_add_duration_seconds": index_meta["collection_add_duration_seconds"],
                "total_index_function_duration_seconds": index_meta["total_index_function_duration_seconds"],
                "embedding_model": index_meta["embedding_model"],
                "embedding_dimension": index_meta["embedding_dimension"],
                "distance_metric": index_meta["distance_metric"]
            }

    # 3. Query Execution & Relevance Evaluation
    print("\n" + "-" * 75)
    print("STAGE 2: EVALUATING QUERIES & COMPUTING RETRIEVAL METRICS")
    print("-" * 75)

    # If recompute_only and existing evaluation records exist, use them while updating NDCG & failure accounting
    existing_records_map = {}
    if recompute_only and os.path.exists(EVAL_OUTPUT_FILE):
        try:
            with open(EVAL_OUTPUT_FILE, "r", encoding="utf-8") as f_ev:
                for r in json.load(f_ev):
                    existing_records_map[r["question_id"]] = r
            print(f"Recompute Mode: Loaded {len(existing_records_map)} existing query records for recalculation.")
        except Exception as e:
            print(f"Warning: Could not load existing evaluation records ({e}). Running fresh queries.")

    evaluation_records = []
    top_k = 5
    failed_queries = 0

    for idx, q_item in enumerate(questions_data, start=1):
        qid = q_item["question_id"]
        repo = q_item["repository"]
        category = q_item["category"]
        query_style = q_item.get("query_style", "identifier")
        raw_question = q_item["question"]
        is_answerable = q_item["category"] != "unanswerable"
        gt_entry = gt_map[qid]
        col_name = repo_collections[repo]

        query_failed = False
        error_msg = None

        if qid in existing_records_map:
            prev_r = existing_records_map[qid]
            query_latency_ms = prev_r.get("retrieval_latency_ms", 0.0)
            query_latency_sec = prev_r.get("retrieval_latency_seconds", round(query_latency_ms / 1000.0, 4))
            formatted_retrieved = prev_r.get("retrieved_chunks", [])
            error_msg = prev_r.get("error")
            if error_msg:
                query_failed = True
                failed_queries += 1
        else:
            t_query_start = time.perf_counter()
            try:
                retrieved_chunks, ret_meta = embedder.search_similar(
                    query=raw_question,
                    repo_name=col_name,
                    n_results=top_k,
                    return_metadata=True
                )
                query_latency_sec = ret_meta.get("retrieval_latency_seconds", 0.0)
                query_latency_ms = round(query_latency_sec * 1000, 2)
                if ret_meta.get("error"):
                    error_msg = str(ret_meta["error"])
                    query_failed = True
                    failed_queries += 1
            except Exception as e:
                query_failed = True
                failed_queries += 1
                query_latency_sec = round(time.perf_counter() - t_query_start, 4)
                query_latency_ms = round(query_latency_sec * 1000, 2)
                retrieved_chunks = []
                error_msg = str(e)

            formatted_retrieved = []
            for r_idx, ch in enumerate(retrieved_chunks, start=1):
                ch_meta = ch.get("metadata", {})
                dist = ch.get("distance")
                formatted_retrieved.append({
                    "rank": r_idx,
                    "chunk_id": ch.get("id"),
                    "l2_distance": round(dist, 6) if dist is not None else None,
                    "file_path": ch_meta.get("file_path"),
                    "start_line": ch_meta.get("start_line"),
                    "end_line": ch_meta.get("end_line"),
                    "chunk_type": ch_meta.get("chunk_type"),
                    "content": ch.get("content")
                })

        l2_distances = [ch["l2_distance"] for ch in formatted_retrieved if ch.get("l2_distance") is not None]
        top1_l2_dist = l2_distances[0] if l2_distances else None
        mean_l2_dist = round(sum(l2_distances) / len(l2_distances), 6) if l2_distances else None

        if is_answerable:
            gt_spans = gt_entry.get("evidence", [])
            total_spans = len(gt_spans)
            matched_span_indices = set()
            per_chunk_relevance = []

            for ch in formatted_retrieved:
                c_file = ch["file_path"]
                c_start = ch["start_line"]
                c_end = ch["end_line"]
                chunk_is_relevant = False

                if c_file is not None and c_start is not None and c_end is not None:
                    for s_idx, span in enumerate(gt_spans):
                        if c_file == span["file_path"] and check_overlap(c_start, c_end, span["start_line"], span["end_line"]):
                            chunk_is_relevant = True
                            matched_span_indices.add(s_idx)

                per_chunk_relevance.append(chunk_is_relevant)

            while len(per_chunk_relevance) < top_k:
                per_chunk_relevance.append(False)

            binary_rel_vector = [1 if r else 0 for r in per_chunk_relevance]

            first_rel_rank = None
            for r_idx, is_rel in enumerate(per_chunk_relevance, start=1):
                if is_rel:
                    first_rel_rank = r_idx
                    break

            # Count total relevant chunks in repo for corrected NDCG@5
            total_rel_in_repo = count_relevant_chunks_in_repo(repo_chunks[repo], gt_spans)
            dcg_val = calculate_dcg(binary_rel_vector)
            ndcg_val = calculate_ndcg(binary_rel_vector, total_rel_in_repo)

            if query_failed:
                # Errored query must never count as successful
                hit_at_1 = 0.0
                hit_at_3 = 0.0
                hit_at_5 = 0.0
                precision_at_5 = 0.0
                mrr_at_5 = 0.0
                span_recall_at_5 = 0.0
                ndcg_at_5 = 0.0
            else:
                hit_at_1 = 1.0 if first_rel_rank == 1 else 0.0
                hit_at_3 = 1.0 if (first_rel_rank is not None and first_rel_rank <= 3) else 0.0
                hit_at_5 = 1.0 if (first_rel_rank is not None and first_rel_rank <= 5) else 0.0
                precision_at_5 = round(sum(binary_rel_vector) / 5.0, 4)
                mrr_at_5 = round(1.0 / first_rel_rank, 4) if first_rel_rank is not None else 0.0
                span_recall_at_5 = round(len(matched_span_indices) / total_spans, 4) if total_spans > 0 else 0.0
                ndcg_at_5 = round(ndcg_val, 4)

            eval_record = {
                "question_id": qid,
                "repository": repo,
                "category": category,
                "query_style": query_style,
                "answerable": True,
                "raw_question": raw_question,
                "conversation_id": q_item.get("conversation_id"),
                "turn_number": q_item.get("turn_number"),
                "retrieval_latency_ms": query_latency_ms,
                "retrieval_latency_seconds": query_latency_sec,
                "n_results_returned": len(formatted_retrieved),
                "top1_l2_distance": top1_l2_dist,
                "mean_top5_l2_distance": mean_l2_dist,
                "per_chunk_relevance": per_chunk_relevance,
                "first_relevant_rank": first_rel_rank,
                "matched_evidence_spans_count": len(matched_span_indices),
                "total_evidence_spans_count": total_spans,
                "total_relevant_chunks_in_repo": total_rel_in_repo,
                "dcg_at_5": round(dcg_val, 4),
                "failed": query_failed,
                "metrics": {
                    "hit_rate_at_1": hit_at_1,
                    "hit_rate_at_3": hit_at_3,
                    "hit_rate_at_5": hit_at_5,
                    "precision_at_5": precision_at_5,
                    "mrr_at_5": mrr_at_5,
                    "span_recall_at_5": span_recall_at_5,
                    "ndcg_at_5": ndcg_at_5
                },
                "retrieved_chunks": formatted_retrieved,
                "error": error_msg
            }
        else:
            eval_record = {
                "question_id": qid,
                "repository": repo,
                "category": category,
                "query_style": query_style,
                "answerable": False,
                "raw_question": raw_question,
                "conversation_id": q_item.get("conversation_id"),
                "turn_number": q_item.get("turn_number"),
                "retrieval_latency_ms": query_latency_ms,
                "retrieval_latency_seconds": query_latency_sec,
                "n_results_returned": len(formatted_retrieved),
                "top1_l2_distance": top1_l2_dist,
                "mean_top5_l2_distance": mean_l2_dist,
                "per_chunk_relevance": [False] * len(formatted_retrieved),
                "first_relevant_rank": None,
                "justification": gt_entry.get("justification"),
                "failed": query_failed,
                "retrieved_chunks": formatted_retrieved,
                "error": error_msg
            }

        evaluation_records.append(eval_record)
        status_symbol = "HIT" if is_answerable and eval_record.get("metrics", {}).get("hit_rate_at_5") == 1.0 else ("MISS" if is_answerable else "UNANS")
        if query_failed:
            status_symbol = "FAIL"
        print(f"[{idx:02d}/60] {qid:<12} | {category:<28} | {query_style:<14} | {status_symbol} | Latency: {query_latency_ms}ms")

    # 4. Aggregations & Breakdowns
    print("\n" + "-" * 75)
    print("STAGE 3: COMPUTING AGGREGATED METRICS & BREAKDOWNS")
    print("-" * 75)

    answerable_records = [r for r in evaluation_records if r["answerable"]]
    unanswerable_records = [r for r in evaluation_records if not r["answerable"]]

    def aggregate_metrics(records: list[dict]) -> dict:
        if not records:
            return {}
        n = len(records)
        m_list = [r["metrics"] for r in records]
        lat_list = [r["retrieval_latency_ms"] for r in records]
        lat_stats = compute_latency_stats(lat_list)

        return {
            "query_count": n,
            "hit_rate_at_1": round(sum(m["hit_rate_at_1"] for m in m_list) / n, 4),
            "hit_rate_at_3": round(sum(m["hit_rate_at_3"] for m in m_list) / n, 4),
            "hit_rate_at_5": round(sum(m["hit_rate_at_5"] for m in m_list) / n, 4),
            "precision_at_5": round(sum(m["precision_at_5"] for m in m_list) / n, 4),
            "mrr_at_5": round(sum(m["mrr_at_5"] for m in m_list) / n, 4),
            "span_recall_at_5": round(sum(m["span_recall_at_5"] for m in m_list) / n, 4),
            "ndcg_at_5": round(sum(m["ndcg_at_5"] for m in m_list) / n, 4),
            "latency_ms": {
                "mean": lat_stats["mean"],
                "median_p50": lat_stats["median_p50"],
                "p95": lat_stats["p95"],
                "max": lat_stats["max"]
            }
        }

    overall_metrics = aggregate_metrics(answerable_records)

    # Repository breakdown
    repo_breakdown = {}
    for repo_info in repos_data:
        r_name = repo_info["name"]
        subset = [r for r in answerable_records if r["repository"] == r_name]
        repo_breakdown[r_name] = aggregate_metrics(subset)

    # Category breakdown
    cat_breakdown = {}
    for cat in sorted(list({r["category"] for r in answerable_records})):
        subset = [r for r in answerable_records if r["category"] == cat]
        cat_breakdown[cat] = aggregate_metrics(subset)

    # Query Style breakdown
    style_breakdown = {}
    for style in sorted(list({r["query_style"] for r in answerable_records})):
        subset = [r for r in answerable_records if r["query_style"] == style]
        style_breakdown[style] = aggregate_metrics(subset)

    # Single vs Multi-evidence breakdown
    single_ev_records = [r for r in answerable_records if r["total_evidence_spans_count"] == 1]
    multi_ev_records = [r for r in answerable_records if r["total_evidence_spans_count"] >= 2]
    evidence_span_breakdown = {
        "single_evidence_queries": aggregate_metrics(single_ev_records),
        "multi_evidence_queries": aggregate_metrics(multi_ev_records)
    }

    # Top-1 L2 Distance Distribution: Answerable vs Unanswerable
    def calc_distance_distribution(dist_list: list[float]) -> dict:
        if not dist_list:
            return {}
        s = sorted(dist_list)
        return {
            "count": len(dist_list),
            "min": round(min(s), 6),
            "median": round(statistics.median(dist_list), 6),
            "mean": round(sum(s) / len(s), 6),
            "max": round(max(s), 6)
        }

    ans_top1_dists = [r["top1_l2_distance"] for r in answerable_records if r["top1_l2_distance"] is not None]
    unans_top1_dists = [r["top1_l2_distance"] for r in unanswerable_records if r["top1_l2_distance"] is not None]

    l2_distance_analysis = {
        "distance_metric": "l2 (squared Euclidean distance)",
        "interpretation": "Lower distance denotes closer embedding vectors under DefaultEmbeddingFunction (all-MiniLM-L6-v2)",
        "answerable_questions_top1_l2": calc_distance_distribution(ans_top1_dists),
        "unanswerable_questions_top1_l2": calc_distance_distribution(unans_top1_dists)
    }

    # Latency Summary distinguishing answerable queries (N=54) from all queries (N=60)
    all_lat_list = [r["retrieval_latency_ms"] for r in evaluation_records]
    ans_lat_list = [r["retrieval_latency_ms"] for r in answerable_records]
    latency_summary = {
        "p95_convention": "nearest rank index floor(0.95 * N)",
        "answerable_queries_latency_ms": compute_latency_stats(ans_lat_list),
        "all_queries_latency_ms": compute_latency_stats(all_lat_list)
    }

    # Package Versions
    import chromadb
    package_versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "chromadb": chromadb.__version__
    }
    try:
        import onnxruntime
        package_versions["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    try:
        import streamlit
        package_versions["streamlit"] = streamlit.__version__
    except ImportError:
        pass

    # Compile Summary Data
    summary_data = {
        "experiment_metadata": {
            "version": "V0",
            "run_id": run_id,
            "timestamp_utc": utc_iso_timestamp,
            "system": "Original Codebase AI Assistant (Dense Retrieval Baseline)",
            "status": "VALID_COMPLETED" if failed_queries == 0 else "INVALID_FAILED",
            "failed_queries_count": failed_queries,
            "total_queries_evaluated": len(evaluation_records),
            "answerable_queries_count": len(answerable_records),
            "unanswerable_queries_count": len(unanswerable_records),
            "top_k": top_k,
            "embedding_model": "DefaultEmbeddingFunction (all-MiniLM-L6-v2)",
            "embedding_dimension": 384,
            "distance_metric": "l2 (squared Euclidean distance)",
            "chunker_configuration": {
                "max_lines": 100,
                "sliding_window_size": 50,
                "sliding_window_overlap": 10,
                "parsers": "Python AST with regex and sliding-window fallback"
            },
            "package_versions": package_versions,
            "dataset_fingerprints": dataset_fingerprints,
            "dedicated_collections": repo_collections,
            "pinned_commit_verifications": commit_verifications
        },
        "indexing_summary": indexing_records,
        "overall_retrieval_metrics": overall_metrics,
        "latency_summary": latency_summary,
        "breakdowns": {
            "by_repository": repo_breakdown,
            "by_category": cat_breakdown,
            "by_query_style": style_breakdown,
            "by_evidence_spans": evidence_span_breakdown
        },
        "top1_l2_distance_analysis": l2_distance_analysis
    }

    # Write Output JSONs
    with open(EVAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(evaluation_records, f, indent=2)
    print(f"\nSaved detailed evaluation records (60 queries) to: {EVAL_OUTPUT_FILE}")

    with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved aggregated baseline summary to: {SUMMARY_OUTPUT_FILE}")

    # Print Summary Tables
    print("\n" + "=" * 75)
    print("                     V0 BASELINE EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Overall Metrics (N = {overall_metrics['query_count']} Answerable Questions):")
    print(f"  - Hit Rate@1     : {overall_metrics['hit_rate_at_1']:.4f} ({overall_metrics['hit_rate_at_1']*100:.1f}%)")
    print(f"  - Hit Rate@3     : {overall_metrics['hit_rate_at_3']:.4f} ({overall_metrics['hit_rate_at_3']*100:.1f}%)")
    print(f"  - Hit Rate@5     : {overall_metrics['hit_rate_at_5']:.4f} ({overall_metrics['hit_rate_at_5']*100:.1f}%)")
    print(f"  - Precision@5    : {overall_metrics['precision_at_5']:.4f}")
    print(f"  - MRR@5          : {overall_metrics['mrr_at_5']:.4f}")
    print(f"  - Span Recall@5  : {overall_metrics['span_recall_at_5']:.4f} ({overall_metrics['span_recall_at_5']*100:.1f}%)")
    print(f"  - NDCG@5         : {overall_metrics['ndcg_at_5']:.4f} (Corrected with total repo relevant chunks)")
    print(f"  - Answerable Latency : Mean={latency_summary['answerable_queries_latency_ms']['mean']}ms, Median={latency_summary['answerable_queries_latency_ms']['median_p50']}ms, P95={latency_summary['answerable_queries_latency_ms']['p95']}ms")
    print(f"  - All Queries Latency: Mean={latency_summary['all_queries_latency_ms']['mean']}ms, Median={latency_summary['all_queries_latency_ms']['median_p50']}ms, P95={latency_summary['all_queries_latency_ms']['p95']}ms")
    print("-" * 75)

    print("Breakdown by Repository:")
    print(f"  {'Repository':<22} | {'Hit@1':<7} | {'Hit@3':<7} | {'Hit@5':<7} | {'MRR@5':<7} | {'SpanRec@5':<9} | {'NDCG@5':<7}")
    for r_name, r_m in repo_breakdown.items():
        print(f"  {r_name:<22} | {r_m['hit_rate_at_1']:<7.4f} | {r_m['hit_rate_at_3']:<7.4f} | {r_m['hit_rate_at_5']:<7.4f} | {r_m['mrr_at_5']:<7.4f} | {r_m['span_recall_at_5']:<9.4f} | {r_m['ndcg_at_5']:<7.4f}")
    print("-" * 75)

    print("Breakdown by Query Style:")
    print(f"  {'Query Style':<22} | {'Count':<5} | {'Hit@1':<7} | {'Hit@5':<7} | {'MRR@5':<7} | {'SpanRec@5':<9} | {'NDCG@5':<7}")
    for s_name, s_m in style_breakdown.items():
        print(f"  {s_name:<22} | {s_m['query_count']:<5} | {s_m['hit_rate_at_1']:<7.4f} | {s_m['hit_rate_at_5']:<7.4f} | {s_m['mrr_at_5']:<7.4f} | {s_m['span_recall_at_5']:<9.4f} | {s_m['ndcg_at_5']:<7.4f}")
    print("-" * 75)

    print("Breakdown by Evidence Complexity:")
    for ev_k, ev_m in evidence_span_breakdown.items():
        print(f"  - {ev_k:<24} (N={ev_m['query_count']}): Hit@5={ev_m['hit_rate_at_5']:.4f}, SpanRec@5={ev_m['span_recall_at_5']:.4f}, NDCG@5={ev_m['ndcg_at_5']:.4f}")
    print("-" * 75)

    print("Top-1 L2 Distance Distribution (Exploratory Analysis):")
    a_d = l2_distance_analysis["answerable_questions_top1_l2"]
    u_d = l2_distance_analysis["unanswerable_questions_top1_l2"]
    print(f"  - Answerable   (N={a_d['count']:02d}) : Min={a_d['min']:.4f}, Median={a_d['median']:.4f}, Mean={a_d['mean']:.4f}, Max={a_d['max']:.4f}")
    print(f"  - Unanswerable (N={u_d['count']:02d}) : Min={u_d['min']:.4f}, Median={u_d['median']:.4f}, Mean={u_d['mean']:.4f}, Max={u_d['max']:.4f}")
    print("=" * 75)

    total_time = time.time() - run_start_time
    print(f"\nExecution finished in {total_time:.2f} seconds.")

    if failed_queries > 0:
        print(f"ERROR: {failed_queries} queries failed during execution. Benchmark is INCOMPLETE/INVALID.")
        sys.exit(1)
    else:
        print("SUCCESS: 100% of 60 evaluation queries completed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V0 Baseline Benchmark Runner")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindexing all repositories into ChromaDB")
    parser.add_argument("--recompute-only", action="store_true", default=True, help="Recompute metrics and summaries from verified records")
    args = parser.parse_args()

    run_benchmark(force_reindex=args.force_reindex, recompute_only=args.recompute_only)
