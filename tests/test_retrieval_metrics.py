"""Unit tests and Day 1 V0 sanity verification for evaluation/retrieval_metrics.py."""

import json
import math
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chunker import chunk_code
from evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    build_chunks_by_file_index,
    calculate_ndcg,
    check_overlap,
    compute_latency_stats,
    compute_query_metrics,
    count_relevant_chunks_in_repo,
    hit_rate_at_k,
    nearest_rank_p95,
    precision_at_k,
    reciprocal_rank_at_k,
    span_recall,
)
from repo_loader import get_source_files


def test_precision_fixed_denominator():
    # Fewer than 5 chunks returned still uses denominator K=5
    assert precision_at_k([True, False], k=5) == 0.2
    assert precision_at_k([True, True, True], k=5) == 0.6
    assert precision_at_k([], k=5) == 0.0


def test_duplicate_chunks_do_not_inflate_span_recall():
    evidence = [
        {"file_path": "a.py", "start_line": 10, "end_line": 30},
        {"file_path": "b.py", "start_line": 1, "end_line": 20},
    ]
    retrieved = [
        {"file_path": "a.py", "start_line": 10, "end_line": 20},
        {"file_path": "a.py", "start_line": 15, "end_line": 25},
        {"file_path": "a.py", "start_line": 20, "end_line": 30},
    ]
    res = compute_query_metrics(retrieved, evidence, total_relevant_chunks_in_repo=3, top_k=5)
    assert res["matched_evidence_spans_count"] == 1
    assert res["metrics"]["span_recall_at_5"] == 0.5
    assert res["metrics"]["precision_at_5"] == 0.6
    assert res["metrics"]["hit_rate_at_1"] == 1.0


def test_ndcg_uses_corpus_relevant_chunks_not_span_count():
    # One span may overlap 5 corpus chunks; ideal DCG@5 must use min(5, 5)
    ndcg = calculate_ndcg([1, 0, 0, 0, 0], total_relevant_chunks_in_repo=5, k=5)
    ideal_dcg_5 = sum(1.0 / math.log2(i + 1) for i in range(1, 6))
    assert abs(ndcg - (1.0 / ideal_dcg_5)) < 1e-9
    # When only 1 chunk exists in repo and it is rank 2:
    ndcg_1 = calculate_ndcg([0, 1, 0, 0, 0], total_relevant_chunks_in_repo=1, k=5)
    assert abs(ndcg_1 - (1.0 / math.log2(3))) < 1e-9


def test_latency_percentiles_and_empty_samples():
    assert nearest_rank_p95([]) is None
    assert compute_latency_stats([]) == {}

    # N=60 -> ceil(0.95 * 60) - 1 = 56 (57th sorted item, 0-based index 56)
    samples_60 = [float(i) for i in range(1, 61)]
    stats_60 = compute_latency_stats(samples_60)
    assert stats_60["median_p50"] == 30.5
    assert stats_60["p95"] == 57.0
    assert stats_60["p95_rank_index"] == 56

    # N=54 -> ceil(0.95 * 54) - 1 = 51 (52nd sorted item)
    samples_54 = [float(i) for i in range(1, 55)]
    stats_54 = compute_latency_stats(samples_54)
    assert stats_54["median_p50"] == 27.5
    assert stats_54["p95"] == 52.0
    assert stats_54["p95_rank_index"] == 51


def test_failed_query_zeroes_metrics():
    evidence = [{"file_path": "a.py", "start_line": 1, "end_line": 10}]
    retrieved = [{"file_path": "a.py", "start_line": 1, "end_line": 10}]
    res = compute_query_metrics(retrieved, evidence, total_relevant_chunks_in_repo=1, failed=True)
    assert all(v == 0.0 for v in res["metrics"].values())


def test_reproduces_validated_day1_v0_metrics():
    """Sanity check: recomputing from stored baseline_evaluation.json & ground_truth reproduces MRR@5=0.4534, NDCG@5=0.3339."""
    eval_path = os.path.join(PROJECT_ROOT, "experiments", "results", "baseline_evaluation.json")
    summary_path = os.path.join(PROJECT_ROOT, "experiments", "results", "baseline_summary.json")
    gt_path = os.path.join(PROJECT_ROOT, "experiments", "datasets", "ground_truth.json")
    repos_path = os.path.join(PROJECT_ROOT, "experiments", "datasets", "repositories.json")

    with open(eval_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_map = {item["question_id"]: item for item in json.load(f)}
    with open(repos_path, "r", encoding="utf-8") as f:
        repos_data = json.load(f)

    # Rebuild V0 chunks from pinned snapshots to independently recompute repo relevant chunk counts
    chunks_by_repo_file = {}
    for r in repos_data:
        alias = r["name"].split("/")[-1]
        repo_dir = os.path.join(PROJECT_ROOT, "experiments", "datasets", "cache", alias)
        chunks = []
        for sf in get_source_files(repo_dir):
            chunks.extend(chunk_code(sf))
        chunks_by_repo_file[r["name"]] = build_chunks_by_file_index(chunks)

    recomputed_answerable = []
    for rec in records:
        if not rec["answerable"]:
            continue
        qid = rec["question_id"]
        spans = gt_map[qid]["evidence"]
        rel_in_repo = count_relevant_chunks_in_repo(
            evidence_spans=spans,
            chunks_by_file=chunks_by_repo_file[rec["repository"]],
        )
        qm = compute_query_metrics(
            retrieved_chunks=rec["retrieved_chunks"],
            evidence_spans=spans,
            total_relevant_chunks_in_repo=rel_in_repo,
            top_k=5,
            failed=bool(rec.get("failed", False)),
        )
        recomputed_answerable.append(
            {
                "repository": rec["repository"],
                "category": rec["category"],
                "query_style": rec["query_style"],
                "retrieval_latency_ms": rec["retrieval_latency_ms"],
                "metrics": qm["metrics"],
            }
        )

    agg = aggregate_retrieval_metrics(recomputed_answerable)
    expected = summary["overall_retrieval_metrics"]

    assert agg["query_count"] == 54
    assert agg["hit_rate_at_1"] == expected["hit_rate_at_1"]
    assert agg["hit_rate_at_3"] == expected["hit_rate_at_3"]
    assert agg["hit_rate_at_5"] == expected["hit_rate_at_5"]
    assert agg["precision_at_5"] == expected["precision_at_5"]
    assert agg["span_recall_at_5"] == expected["span_recall_at_5"]
    assert agg["mrr_at_5"] == 0.4534 == expected["mrr_at_5"]
    assert agg["ndcg_at_5"] == 0.3339 == expected["ndcg_at_5"]
    assert agg["latency_ms"]["median_p50"] == expected["latency_ms"]["median_p50"]
    assert agg["latency_ms"]["p95"] == expected["latency_ms"]["p95"]


if __name__ == "__main__":
    test_precision_fixed_denominator()
    test_duplicate_chunks_do_not_inflate_span_recall()
    test_ndcg_uses_corpus_relevant_chunks_not_span_count()
    test_latency_percentiles_and_empty_samples()
    test_failed_query_zeroes_metrics()
    test_reproduces_validated_day1_v0_metrics()
    print("ALL RETRIEVAL METRICS UNIT & DAY 1 PARITY TESTS PASSED!")
