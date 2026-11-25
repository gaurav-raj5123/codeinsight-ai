"""Verification tests for Day 3 chunking comparison benchmark harness and Day 1/2 preservation."""

import hashlib
import json
import os
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.chunking_experiments.run_chunking_comparison import (
    STRATEGIES,
    build_chunking_error_analysis,
    build_corpus_manifest,
    compute_corpus_chunk_metrics,
    load_source_corpus,
    make_day3_collection_name,
    validate_chroma_metadata,
)
from chunker import chunk_code
from chunker_v1 import get_embedding_tokenizer
from evaluation.retrieval_metrics import aggregate_retrieval_metrics, compute_query_metrics


def test_collection_naming_isolation_and_metadata_validation():
    names = {
        make_day3_collection_name(s["strategy_id"], repo, "run123")
        for s in STRATEGIES
        for repo in ("flask", "express", "fmt")
    }
    assert len(names) == 9
    for n in names:
        assert 3 <= len(n) <= 63

    clean = validate_chroma_metadata({"file_path": "a.py", "start_line": 1, "is_split": False, "score": 0.5})
    assert clean["start_line"] == 1
    with pytest.raises(TypeError):
        validate_chroma_metadata({"bad_none": None})


def test_corpus_fairness_manifest_253_files():
    repos = json.load(open(os.path.join(PROJECT_ROOT, "experiments/datasets/repositories.json"), encoding="utf-8"))
    files_by_repo = load_source_corpus(repos)
    manifest = build_corpus_manifest(files_by_repo)
    assert manifest["total_files"] == 253
    assert manifest["repositories"]["pallets/flask"]["file_count"] == 83
    assert manifest["repositories"]["expressjs/express"]["file_count"] == 142
    assert manifest["repositories"]["fmtlib/fmt"]["file_count"] == 28


def test_chunk_quality_metrics_calculation_all_fields():
    tok, _ = get_embedding_tokenizer()
    sample_file = {
        "file_path": "src/mod.py",
        "extension": ".py",
        "content": "CONST = 1\n\ndef f():\n    return CONST\n",
    }
    files_by_repo = {"org/repo": [sample_file]}
    chunks = chunk_code(sample_file, strategy="code_aware_v1", repository="org/repo", tokenizer=tok)
    gt = [
        {
            "question_id": "q1",
            "repository": "org/repo",
            "answerable": True,
            "evidence": [{"file_path": "src/mod.py", "start_line": 3, "end_line": 4}],
        }
    ]
    qm = compute_corpus_chunk_metrics(
        strategy_id="code_aware_v1",
        chunks_by_repo={"org/repo": chunks},
        files_by_repo=files_by_repo,
        ground_truth=gt,
        raw_tokenizer=tok,
        chunking_times_by_repo={"org/repo": 0.01},
        doc_embed_times_by_repo={"org/repo": 0.1},
        insert_times_by_repo={"org/repo": 0.02},
        runtime_truncation_counts_by_repo={"org/repo": 0},
        peak_rss_mb=120.5,
    )
    required_keys = {
        "total_chunks",
        "chunks_per_repository",
        "chunks_per_language",
        "source_file_coverage_pct",
        "line_coverage_pct",
        "non_empty_line_coverage_pct",
        "chunk_line_length_stats",
        "chunk_char_length_stats",
        "chunk_token_length_stats",
        "empty_chunk_count",
        "single_line_chunk_count",
        "short_lt_5_lines_count",
        "tiny_lt_15_tokens_count",
        "over_256_tokens_count",
        "over_512_tokens_count",
        "runtime_truncation_count",
        "runtime_truncation_rate_pct",
        "exact_duplicate_chunks_excess",
        "exact_duplicate_rate_pct",
        "syntax_parsed_count",
        "fallback_chunk_count",
        "chunk_type_distribution",
        "avg_relevant_chunks_per_answerable_query",
        "ground_truth_span_coverage",
        "peak_rss_mb",
        "timings_seconds",
    }
    assert required_keys.issubset(set(qm.keys()))
    assert qm["empty_chunk_count"] == 0
    assert qm["over_256_tokens_count"] == 0
    assert qm["non_empty_line_coverage_pct"] == 100.0
    assert qm["ground_truth_span_coverage"]["covered_spans"] == 1
    assert qm["ground_truth_span_coverage"]["untruncated_spans_le_256_tokens"] == 1


def test_failed_query_zero_metrics_and_error_analysis_transitions():
    failed_qm = compute_query_metrics(
        retrieved_chunks=[{"metadata": {"file_path": "a.py", "start_line": 1, "end_line": 5}}],
        evidence_spans=[{"file_path": "a.py", "start_line": 1, "end_line": 5}],
        total_relevant_chunks_in_repo=1,
        top_k=5,
        failed=True,
    )
    for k, v in failed_qm["metrics"].items():
        assert v == 0.0, f"Failed query must score 0.0 on {k}"

    def _rec(qid, hit, rank):
        return {
            "question_id": qid,
            "repository": "r",
            "category": "c",
            "query_style": "s",
            "answerable": True,
            "raw_question": "?",
            "first_relevant_rank": rank,
            "metrics": {
                "hit_rate_at_1": 1.0 if rank == 1 else 0.0,
                "hit_rate_at_3": 1.0 if rank and rank <= 3 else 0.0,
                "hit_rate_at_5": 1.0 if hit else 0.0,
                "precision_at_5": 0.2 if hit else 0.0,
                "span_recall_at_5": 1.0 if hit else 0.0,
                "mrr_at_5": (1.0 / rank) if rank else 0.0,
                "ndcg_at_5": 0.5 if hit else 0.0,
            },
        }

    ea = build_chunking_error_analysis(
        {
            "fixed_window_v0": [_rec("q1", False, None), _rec("q2", True, 2), _rec("q3", False, None)],
            "existing_chunker_v0": [_rec("q1", True, 1), _rec("q2", False, None), _rec("q3", False, None)],
            "code_aware_v1": [_rec("q1", True, 1), _rec("q2", True, 1), _rec("q3", False, None)],
        }
    )
    v1_vs_v0 = ea["pairwise_transitions"]["code_aware_v1_vs_existing_chunker_v0"]
    assert v1_vs_v0["gained_count"] == 1
    assert v1_vs_v0["regressed_count"] == 0
    assert v1_vs_v0["both_hit_count"] == 1
    assert ea["missed_by_all_strategies_count"] == 1


def test_day1_and_day2_files_preserved():
    expected_existing = [
        "experiments/results/baseline_summary.json",
        "experiments/results/baseline_evaluation.json",
        "experiments/embedding_experiments/results/embedding_summary.json",
        "experiments/embedding_experiments/results/embedding_comparison.json",
        "reports/day1/DAY1_FINAL_REPORT.md",
        "reports/day2/DAY2_TASK1_REPORT.md",
    ]
    for rel in expected_existing:
        assert os.path.exists(os.path.join(PROJECT_ROOT, rel)), f"Missing preserved artifact: {rel}"
