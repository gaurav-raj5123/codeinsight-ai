"""Verification tests for Day 2 embedding comparison configs and runner components."""

import json
import os
import sys

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.embedding_experiments.run_embedding_comparison import (
    CONFIG_FILE,
    DirectONNXEmbedder,
    build_error_analysis,
    make_collection_name,
    run_preflight_checks,
)


def test_embedding_models_config_complete():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["distance_metric"] == "l2"
    assert cfg["top_k"] == 5

    ids = {m["experiment_id"]: m for m in cfg["models"]}
    assert "E0_v0_minilm_l6_v2" in ids
    assert "E1a_bge_small_en_v15_no_prefix" in ids
    assert "E1b_bge_small_en_v15_instruct" in ids
    assert "E2_e5_small_v2" in ids
    assert "E3_minilm_l12_v2" in ids

    assert ids["E1a_bge_small_en_v15_no_prefix"]["pooling"] == "cls"
    assert ids["E2_e5_small_v2"]["document_prefix"] == "passage: "
    assert ids["E2_e5_small_v2"]["query_prefix"] == "query: "
    assert ids["E3_minilm_l12_v2"]["max_seq_length"] == 128


def test_preflight_and_collection_naming():
    repos = json.load(open(os.path.join(PROJECT_ROOT, "experiments/datasets/repositories.json"), encoding="utf-8"))
    questions = json.load(open(os.path.join(PROJECT_ROOT, "experiments/datasets/evaluation_questions.json"), encoding="utf-8"))
    gt = json.load(open(os.path.join(PROJECT_ROOT, "experiments/datasets/ground_truth.json"), encoding="utf-8"))
    stats, commits = run_preflight_checks(repos, questions, gt)
    assert stats["total_questions"] == 60
    assert stats["answerable_questions"] == 54
    assert all(v["verified"] for v in commits.values())

    name = make_collection_name("E1a_bge_small_en_v15_no_prefix", "express", "123456")
    assert 3 <= len(name) <= 63


def test_onnx_embedder_smoke_and_normalization():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    for m_cfg in cfg["models"]:
        if m_cfg["experiment_id"] == "E1b_bge_small_en_v15_instruct":
            # Prefix differences only affect inputs; test full embedder on E0, E1a, E2, E3 (and prefix on E1b)
            pass
        emb = DirectONNXEmbedder(m_cfg)
        w = emb.warmup()
        assert w >= 0.0
        vecs, stats = emb.encode_texts(["def f(x):\n    return x + 1"], prefix=emb.document_prefix)
        assert vecs.shape == (1, 384)
        assert np.all(np.isfinite(vecs))
        assert abs(stats["norm_stats"]["mean"] - 1.0) < 1e-4

        q_vec, q_dur, q_norm = emb.encode_single_query("What does f return?")
        assert q_vec.shape == (384,)
        assert q_dur >= 0.0
        assert abs(q_norm - 1.0) < 1e-4


def test_error_analysis_helper():
    recs = {
        "E0_v0_minilm_l6_v2": [
            {
                "question_id": "q1",
                "repository": "r",
                "category": "c",
                "query_style": "s",
                "answerable": True,
                "raw_question": "?",
                "first_relevant_rank": None,
                "metrics": {"hit_rate_at_5": 0.0, "span_recall_at_5": 0.0},
            },
            {
                "question_id": "q2",
                "repository": "r",
                "category": "c",
                "query_style": "s",
                "answerable": True,
                "raw_question": "?",
                "first_relevant_rank": 1,
                "metrics": {"hit_rate_at_5": 1.0, "span_recall_at_5": 1.0},
            },
        ],
        "E1": [
            {
                "question_id": "q1",
                "repository": "r",
                "category": "c",
                "query_style": "s",
                "answerable": True,
                "raw_question": "?",
                "first_relevant_rank": 2,
                "metrics": {"hit_rate_at_5": 1.0, "span_recall_at_5": 1.0},
            },
            {
                "question_id": "q2",
                "repository": "r",
                "category": "c",
                "query_style": "s",
                "answerable": True,
                "raw_question": "?",
                "first_relevant_rank": None,
                "metrics": {"hit_rate_at_5": 0.0, "span_recall_at_5": 0.0},
            },
        ],
    }
    ea = build_error_analysis(recs)
    assert ea["transitions_vs_baseline"]["E1"]["solved_by_model_missed_by_v0_count"] == 1
    assert ea["transitions_vs_baseline"]["E1"]["regressed_by_model_solved_by_v0_count"] == 1
    assert ea["missed_by_all_models_count"] == 0


def test_day2_experiment_results_integrity():
    """Verify completed Day 2 output artifacts exist, have zero failures, and match V0 baseline."""
    summary_path = os.path.join(PROJECT_ROOT, "experiments", "embedding_experiments", "results", "embedding_summary.json")
    comp_path = os.path.join(PROJECT_ROOT, "experiments", "embedding_experiments", "results", "embedding_comparison.json")
    assert os.path.exists(summary_path), "Missing embedding_summary.json"
    assert os.path.exists(comp_path), "Missing embedding_comparison.json"

    with open(summary_path, "r", encoding="utf-8") as f:
        s = json.load(f)

    assert s["suite_metadata"]["status"] == "VALID_COMPLETED"
    expected_ids = {
        "E0_v0_minilm_l6_v2",
        "E1a_bge_small_en_v15_no_prefix",
        "E1b_bge_small_en_v15_instruct",
        "E2_e5_small_v2",
        "E3_minilm_l12_v2",
    }
    assert set(s["models"].keys()) == expected_ids
    for eid in expected_ids:
        m = s["models"][eid]
        assert m["experiment_metadata"]["failed_queries"] == 0
        assert m["experiment_metadata"]["total_queries"] == 60
        assert abs(m["experiment_metadata"]["doc_vector_norm_verification"]["mean"] - 1.0) < 1e-4

    # E0 must reproduce Day 1 V0 metrics
    e0_metrics = s["models"]["E0_v0_minilm_l6_v2"]["overall_retrieval_metrics"]
    assert e0_metrics["mrr_at_5"] == 0.4534
    assert e0_metrics["ndcg_at_5"] == 0.3339


def test_runner_failure_accounting_on_query_exception():
    """Simulate a query exception in run_single_model_experiment to verify zeroed metrics and failure count."""
    from unittest.mock import patch
    from evaluation.retrieval_metrics import build_chunks_by_file_index
    from experiments.embedding_experiments.run_embedding_comparison import run_single_model_experiment

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        suite_cfg = json.load(f)
    m_cfg = suite_cfg["models"][0]
    repos_data = [{"name": "pallets/flask", "commit_sha": "abc1234"}]
    questions_data = [{
        "question_id": "q_fail_test",
        "repository": "pallets/flask",
        "category": "function_identification",
        "query_style": "identifier",
        "question": "Where is Flask?",
        "conversation_id": None,
        "turn_number": 1,
    }]
    gt_map = {
        "q_fail_test": {
            "question_id": "q_fail_test",
            "answerable": True,
            "evidence": [{"file_path": "a.py", "start_line": 1, "end_line": 10}],
        }
    }
    corpus_chunks = {"pallets/flask": [{"content": "class Flask: pass", "metadata": {"file_path": "a.py", "start_line": 1, "end_line": 5}}]}
    chunks_idx = {"pallets/flask": build_chunks_by_file_index(corpus_chunks["pallets/flask"])}

    class BrokenCollection:
        def add(self, **kwargs):
            return None
        def query(self, **kwargs):
            raise RuntimeError("Simulated Chroma failure")

    class BrokenClient:
        def delete_collection(self, **kwargs):
            pass
        def create_collection(self, **kwargs):
            return BrokenCollection()

    with patch("experiments.embedding_experiments.run_embedding_comparison.DirectONNXEmbedder.encode_single_query", side_effect=RuntimeError("Simulated query ONNX failure")):
        summary, recs, _ = run_single_model_experiment(
            model_cfg=m_cfg,
            suite_cfg=suite_cfg,
            repos_data=repos_data,
            questions_data=questions_data,
            gt_map=gt_map,
            corpus_chunks=corpus_chunks,
            corpus_chunks_by_file=chunks_idx,
            dataset_fingerprints={},
            commit_verifications={},
            chroma_client=BrokenClient(),
            run_id="test_run",
            run_short="test",
            env_metadata={},
        )

    assert summary["experiment_metadata"]["status"] == "INVALID_FAILED"
    assert summary["experiment_metadata"]["failed_queries"] == 1
    assert recs[0]["failed"] is True
    assert "RuntimeError" in recs[0]["error"]
    assert all(v == 0.0 for v in recs[0]["metrics"].values())


if __name__ == "__main__":
    test_embedding_models_config_complete()
    test_preflight_and_collection_naming()
    test_onnx_embedder_smoke_and_normalization()
    test_error_analysis_helper()
    test_day2_experiment_results_integrity()
    test_runner_failure_accounting_on_query_exception()
    print("ALL EMBEDDING RUNNER TESTS PASSED!")
