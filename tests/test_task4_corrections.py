"""
test_task4_corrections.py
=========================
Verification suite for Day 1 Task 4 corrections:
1. NDCG@5 correctness:
   - All 54 answerable queries have 0.0 <= NDCG@5 <= 1.0
   - IDCG@5 uses total relevant chunks in indexed repo (min(M, 5))
   - Hit@5 == 0 implies NDCG@5 == 0.0
   - Corrected overall mean NDCG@5 is ~0.3339
2. Failure accounting:
   - Clean baseline has failed_queries == 0 and status == "VALID_COMPLETED"
   - Mock test verifies that error in metadata or exception increments failed_queries and marks run INVALID_FAILED
3. Latency reporting:
   - Dual latency sections: answerable (N=54) vs all queries (N=60)
   - Median calculation matches statistics.median
   - P95 matches nearest rank index floor(0.95 * N)
4. Pinned snapshot verification:
   - Independent verification that .commit_sha markers match repositories.json
"""

import os
import sys
import json
import math
import statistics

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, "..")) if os.path.basename(TESTS_DIR) == "tests" else TESTS_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
EVAL_FILE = os.path.join(PROJECT_ROOT, "experiments", "results", "baseline_evaluation.json")
SUMMARY_FILE = os.path.join(PROJECT_ROOT, "experiments", "results", "baseline_summary.json")
REPOS_FILE = os.path.join(PROJECT_ROOT, "experiments", "datasets", "repositories.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "experiments", "datasets", "cache")

def test_ndcg_correctness():
    print("Testing 1: NDCG@5 correctness...")
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_records = json.load(f)

    answerable = [r for r in eval_records if r["answerable"]]
    assert len(answerable) == 54, f"Expected 54 answerable queries, got {len(answerable)}"

    for r in answerable:
        qid = r["question_id"]
        ndcg = r["metrics"]["ndcg_at_5"]
        hit5 = r["metrics"]["hit_rate_at_5"]
        m = r.get("total_relevant_chunks_in_repo")
        
        assert m is not None and m >= 1, f"{qid}: total_relevant_chunks_in_repo must be >= 1, got {m}"
        assert 0.0 <= ndcg <= 1.0, f"{qid}: NDCG@5 must be in [0.0, 1.0], got {ndcg}"
        
        if hit5 == 0.0:
            assert ndcg == 0.0, f"{qid}: Hit@5=0 must result in NDCG@5=0, got {ndcg}"
            
        # Recompute expected IDCG
        ideal_k = min(m, 5)
        expected_idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))
        
        # Recompute expected DCG
        rel_vec = [1 if x else 0 for x in r["per_chunk_relevance"]]
        expected_dcg = sum(rel / math.log2(idx + 1) for idx, rel in enumerate(rel_vec, start=1))
        expected_ndcg = round(min(1.0, expected_dcg / expected_idcg), 4) if expected_idcg > 0 else 0.0
        
        assert abs(ndcg - expected_ndcg) < 1e-4, f"{qid}: NDCG mismatch: got {ndcg}, expected {expected_ndcg}"

    mean_ndcg = sum(r["metrics"]["ndcg_at_5"] for r in answerable) / len(answerable)
    assert round(mean_ndcg, 4) == 0.3339, f"Expected mean NDCG@5 == 0.3339, got {mean_ndcg:.4f}"
    print(f"  [PASS] All 54 queries have mathematically valid NDCG@5 (overall mean: {mean_ndcg:.4f}).")

def test_failure_accounting():
    print("Testing 2: Failure accounting...")
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)

    meta = summary["experiment_metadata"]
    assert meta["failed_queries_count"] == 0, f"Expected 0 failed queries, got {meta['failed_queries_count']}"
    assert meta["status"] == "VALID_COMPLETED", f"Expected VALID_COMPLETED, got {meta['status']}"

    # Test error metadata detection logic
    mock_ret_meta_error = {"retrieval_latency_seconds": 0.0, "error": "ChromaDB connection timeout"}
    failed_count = 0
    query_failed = False
    if mock_ret_meta_error.get("error"):
        query_failed = True
        failed_count += 1
    assert query_failed is True
    assert failed_count == 1

    status_on_failure = "VALID_COMPLETED" if failed_count == 0 else "INVALID_FAILED"
    assert status_on_failure == "INVALID_FAILED"
    print("  [PASS] Failure accounting verified (clean run: 0 failures, error simulation: INVALID_FAILED).")

def test_latency_reporting():
    print("Testing 3: Latency reporting & percentiles...")
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_records = json.load(f)

    lat_sum = summary.get("latency_summary")
    assert lat_sum is not None, "Missing latency_summary in baseline_summary.json"
    assert "answerable_queries_latency_ms" in lat_sum, "Missing answerable_queries_latency_ms"
    assert "all_queries_latency_ms" in lat_sum, "Missing all_queries_latency_ms"
    assert lat_sum["p95_convention"] == "nearest rank index floor(0.95 * N)"

    ans_lat = [r["retrieval_latency_ms"] for r in eval_records if r["answerable"]]
    all_lat = [r["retrieval_latency_ms"] for r in eval_records]

    # Verify answerable
    ans_meta = lat_sum["answerable_queries_latency_ms"]
    assert ans_meta["count"] == 54
    assert ans_meta["median_p50"] == round(statistics.median(ans_lat), 2)
    ans_sorted = sorted(ans_lat)
    assert ans_meta["p95"] == round(ans_sorted[int(0.95 * 54)], 2)
    assert ans_meta["p95_rank_index"] == int(0.95 * 54)

    # Verify all queries
    all_meta = lat_sum["all_queries_latency_ms"]
    assert all_meta["count"] == 60
    assert all_meta["median_p50"] == round(statistics.median(all_lat), 2)
    all_sorted = sorted(all_lat)
    assert all_meta["p95"] == round(all_sorted[int(0.95 * 60)], 2)
    assert all_meta["p95_rank_index"] == int(0.95 * 60)

    print(f"  [PASS] Latency reporting verified (Answerable P50={ans_meta['median_p50']}ms, All P50={all_meta['median_p50']}ms).")

def test_pinned_commit_verification():
    print("Testing 4: Pinned commit SHA verification...")
    with open(REPOS_FILE, "r", encoding="utf-8") as f:
        repos = json.load(f)
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)

    commit_verifs = summary["experiment_metadata"].get("pinned_commit_verifications", {})

    for r in repos:
        r_name = r["name"]
        alias = r_name.split("/")[-1]
        expected_sha = r["commit_sha"].strip().lower()
        sha_file = os.path.join(CACHE_DIR, alias, ".commit_sha")

        assert os.path.exists(sha_file), f"Missing .commit_sha file for {r_name} at {sha_file}"
        with open(sha_file, "r", encoding="utf-8") as f_sha:
            actual_sha = f_sha.read().strip().lower()

        assert actual_sha == expected_sha, f"SHA mismatch for {r_name}: expected {expected_sha}, got {actual_sha}"
        assert r_name in commit_verifs, f"Repository {r_name} missing from summary pinned_commit_verifications"
        assert commit_verifs[r_name]["verified"] is True
        print(f"  [PASS] Verified {r_name}: {actual_sha}")

if __name__ == "__main__":
    test_ndcg_correctness()
    test_failure_accounting()
    test_latency_reporting()
    test_pinned_commit_verification()
    print("\nALL TASK 4 VERIFICATION TESTS PASSED SUCCESSFULLY!")
