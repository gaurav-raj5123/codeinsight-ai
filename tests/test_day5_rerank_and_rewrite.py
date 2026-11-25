"""Unit and contract tests for Day 5 ONNX cross-encoder reranker and leakage-free query rewriter."""

import inspect
import json
import math
import os
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import retrieval.query_rewriter as qr
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.reranker import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_REVISION,
    HybridRerankRetriever,
    ONNXCrossEncoderReranker,
    stable_sigmoid,
)


@pytest.fixture(scope="module")
def cross_encoder():
    reranker = ONNXCrossEncoderReranker(
        model_name=DEFAULT_CROSS_ENCODER_MODEL,
        revision=DEFAULT_CROSS_ENCODER_REVISION,
        batch_size=8,
    )
    reranker.warmup()
    return reranker


def test_onnx_cross_encoder_pair_tokenization_and_segment_ids(cross_encoder):
    info = cross_encoder.get_model_info()
    assert info["cpu_providers"] == ["CPUExecutionProvider"]
    assert info["supports_token_type_ids"] is True
    assert info["max_length"] == 512
    assert info["special_tokens_per_pair"] == 3
    assert info["model_size_mb"] > 10.0

    # Verify native pair [CLS] query [SEP] candidate [SEP] and exact 0/1 segment IDs
    enc, meta = cross_encoder.encode_pair(
        "Where is Flask defined?",
        "class Flask(App):\n    pass",
    )
    tokens = enc["tokens"]
    type_ids = enc["token_type_ids"]
    assert tokens[0] == "[CLS]"
    sep_positions = [idx for idx, t in enumerate(tokens) if t == "[SEP]"]
    assert len(sep_positions) == 2
    first_sep, second_sep = sep_positions

    # Query side + [CLS] + first [SEP] MUST have segment ID 0
    assert all(tid == 0 for tid in type_ids[: first_sep + 1])
    # Candidate side + trailing [SEP] MUST have segment ID 1
    assert all(tid == 1 for tid in type_ids[first_sep + 1 : second_sep + 1])
    assert meta["query_truncated"] is False
    assert meta["candidate_truncated"] is False
    assert meta["pair_truncated"] is False


def test_only_second_and_oversized_query_truncation_telemetry(cross_encoder):
    # Case 1: Short query + >600-token candidate -> truncates ONLY candidate
    long_doc = " ".join(f"token_{i}" for i in range(600))
    logits_d, meta_d = cross_encoder.score_pairs("Where is dispatch_request?", [long_doc])
    assert len(logits_d) == 1
    assert math.isfinite(logits_d[0])
    assert meta_d["query_truncated"] is False
    assert meta_d["candidate_inputs_truncated_count"] == 1
    assert meta_d["pairs_truncated_count"] == 1

    # Case 2: Exceptionally long query (>550 tokens) MUST NOT crash & preserves segment 1 candidate room
    long_query = " ".join(["flask"] * 560)
    enc_lq, enc_meta = cross_encoder.encode_pair(long_query, "def hello_world(): return 42")
    assert enc_meta["final_pair_tokens"] <= 512
    assert enc_meta["query_truncated"] is True
    assert enc_meta["pair_truncated"] is True
    # Confirm candidate tokens survived with segment ID 1!
    assert 1 in enc_lq["token_type_ids"]
    logits_q, meta_q = cross_encoder.score_pairs(long_query, ["def hello_world(): return 42"])
    assert len(logits_q) == 1
    assert meta_q["query_truncated"] is True
    assert meta_q["pairs_truncated_count"] == 1


def test_cross_encoder_scoring_semantics_tie_breaking_and_fallback(cross_encoder):
    assert abs(stable_sigmoid(0.0) - 0.5) < 1e-9
    assert stable_sigmoid(1000.0) <= 1.0
    assert stable_sigmoid(-1000.0) >= 0.0

    # Semantic relevance check + tie breaking + record fields
    cands = [
        {"id": "irrel", "content": "function unrelatedTaxHelper(amount) { return amount * 0.1; }", "metadata": {}, "rrf_rank": 1},
        {"id": "rel", "content": "class Flask(App):\n    def dispatch_request(self): return self.view_functions()", "metadata": {}, "rrf_rank": 2},
    ]
    top, diag = cross_encoder.rerank("How does dispatch_request work in Flask?", cands, top_k=2)
    assert top[0]["id"] == "rel"
    assert top[0]["reranker_raw_logit"] > top[1]["reranker_raw_logit"]
    assert top[0]["reranker_score_interpretation"] == "raw_unnormalized_relevance_logit"
    assert top[0]["original_retrieval_rank"] == 2
    assert top[0]["rerank_rank"] == 1

    # Verify deterministic tie-breaking when raw logits are forced equal
    orig_score_fn = cross_encoder.score_pairs
    cross_encoder.score_pairs = lambda q, ts: ([1.5] * len(ts), {
        "pairs_count": len(ts), "pairs_truncated_count": 0, "query_truncated": False,
        "candidate_inputs_truncated_count": 0, "truncation_errors_recovered": 0,
        "tokenization_ms": 0.1, "inference_ms": 0.1, "total_scoring_ms": 0.2,
    })
    try:
        tied, _ = cross_encoder.rerank("q", [
            {"id": "b", "content": "x", "metadata": {}, "rrf_rank": 3},
            {"id": "a", "content": "y", "metadata": {}, "rrf_rank": 1},
        ], top_k=2)
        assert [c["id"] for c in tied] == ["a", "b"]
    finally:
        cross_encoder.score_pairs = orig_score_fn

    # Verify HybridRerankRetriever graceful fallback to RRF on reranker failure
    bm25 = BM25Retriever(["c1", "c2"], ["alpha code", "beta code"], [{"file_path": "a.py"}, {"file_path": "b.py"}])
    hyb = HybridRetriever(
        dense_search_fn=lambda q, n: [{"id": "c1", "content": "alpha code", "metadata": {"file_path": "a.py"}, "distance": 0.2}],
        bm25_retriever=bm25,
    )

    class _ExplodingReranker:
        def rerank(self, *args, **kwargs):
            raise RuntimeError("Simulated ONNX failure")

    hr = HybridRerankRetriever(hyb, _ExplodingReranker(), candidate_pool_mode="candidate_union")
    res_fb, diag_fb = hr.retrieve("alpha", top_k=5, fallback_to_rrf_on_error=True, return_diagnostics=True)
    assert diag_fb["rerank_failed"] is True
    assert diag_fb["fallback_to_rrf_used"] is True
    assert res_fb[0]["id"] == "c1"


def test_query_rewriter_leakage_isolation_and_deterministic_gate():
    # Static isolation verification: production rewriter must never touch dataset/ground-truth artifacts
    src = inspect.getsource(qr)
    for forbidden in (
        "expected_standalone_query",
        "ground_truth",
        "evidence_spans",
        "evaluation_questions",
        "answerable",
    ):
        assert forbidden not in src, f"Forbidden dataset artifact reference {forbidden!r} in query_rewriter.py"

    # Standalone without history -> gate skips LLM
    r1 = qr.rewrite_conversational_query(
        "Where is Flask class defined?",
        previous_context=None,
        llm_call_fn=lambda p: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )
    assert r1["rewritten"] is False
    assert r1["gemini_calls"] == 0
    assert r1["rewrite_latency_ms"] == 0.0
    assert r1["retrieval_query"] == "Where is Flask class defined?"

    # Standalone WITH history -> gate still skips LLM!
    r2 = qr.rewrite_conversational_query(
        "Where is the Request wrapper class defined in Flask?",
        previous_context=[{"role": "user", "content": "Where is Blueprint defined?"}],
        llm_call_fn=lambda p: (_ for _ in ()).throw(AssertionError("LLM must not be called for standalone query")),
    )
    assert r2["rewritten"] is False
    assert r2["gemini_calls"] == 0

    # Verify exact benchmark dataset gating: 48 None -> False, 12 Turn-2 -> True!
    qs = json.load(open(os.path.join(PROJECT_ROOT, "experiments/datasets/evaluation_questions.json"), encoding="utf-8"))
    assert len(qs) == 60
    conv_triggered = []
    standalone_triggered = []
    for q in qs:
        needs, _ = qr.needs_conversational_rewrite(q["question"], q.get("previous_context"))
        if q.get("turn_number") == 2:
            if needs:
                conv_triggered.append(q["question_id"])
        else:
            if needs:
                standalone_triggered.append(q["question_id"])
    assert len(conv_triggered) == 12
    assert standalone_triggered == []


def test_query_rewriter_retry_fallback_and_identifier_preservation():
    # Transient 429 retry recovers cleanly
    calls = {"n": 0}

    def _flaky_llm(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return "⚠️ Error: 429 quota transient", {"success": False, "error": "429 rate limit"}
        return "How does Flask.make_config initialize default configuration settings?", {"success": True, "error": None}

    ok = qr.rewrite_conversational_query(
        "How does it configure default settings?",
        previous_context="We looked at Flask.__init__ and make_config.",
        repository_name="pallets/flask",
        llm_call_fn=_flaky_llm,
        max_retries=1,
        retry_delay_seconds=0.01,
    )
    assert calls["n"] == 2
    assert ok["rewritten"] is True
    assert ok["rewrite_failed"] is False
    assert "make_config" in ok["retrieval_query"]

    # Hard failure preserves original query cleanly
    fail = qr.rewrite_conversational_query(
        "How does it configure default settings?",
        previous_context="Prior turn about Flask",
        llm_call_fn=lambda p: ("⚠️ Error calling Gemini API: fail", {"success": False, "error": "bad"}),
        max_retries=0,
    )
    assert fail["rewritten"] is False
    assert fail["rewrite_failed"] is True
    assert fail["retrieval_query"] == "How does it configure default settings?"


def test_embedder_hybrid_rerank_end_to_end_and_fallbacks(tmp_path, monkeypatch):
    import chromadb
    import embedder
    import llm
    from chunker import chunk_code
    from chunker_v1 import get_embedding_tokenizer

    db_dir = str(tmp_path / "chroma_day5_test")
    client = chromadb.PersistentClient(path=db_dir)
    monkeypatch.setattr(embedder, "get_chroma_client", lambda: client)
    embedder.clear_bm25_cache()

    tok, _ = get_embedding_tokenizer()
    files = [
        {
            "file_path": "src/flask/app.py",
            "extension": ".py",
            "content": (
                "class Flask:\n"
                "    def dispatch_request(self):\n"
                "        req = request_context_stack.top.request\n"
                "        return self.view_functions[req.endpoint]()\n"
            ),
        },
        {
            "file_path": "src/billing/tax.py",
            "extension": ".py",
            "content": (
                "def calculate_tax(subtotal, rate):\n"
                "    return subtotal * rate\n"
            ),
        },
        {
            "file_path": "src/flask/config.py",
            "extension": ".py",
            "content": (
                "class Config(dict):\n"
                "    def from_object(self, obj):\n"
                "        return obj\n"
            ),
        },
    ]
    all_chunks = []
    for f in files:
        all_chunks.extend(chunk_code(f, strategy="code_aware_v1", repository="repo_d5", tokenizer=tok))

    count, idx_meta = embedder.index_chunks(all_chunks, "repo_d5", return_metadata=True)
    assert count == len(all_chunks)
    assert "corpus_fingerprint" in idx_meta

    # Default call still uses V0 backward-compatible dense mode
    dense_res, dense_meta = embedder.search_similar(
        "Where is dispatch_request implemented?", "repo_d5", n_results=3, return_metadata=True
    )
    assert dense_meta["retrieval_mode"] == "dense"
    assert len(dense_res) == 3

    # hybrid_rerank mode runs ONNX cross-encoder and attaches raw logit + exploratory sigmoid
    rerank_res, rerank_meta = embedder.search_similar(
        "Where is dispatch_request implemented?",
        "repo_d5",
        n_results=3,
        return_metadata=True,
        retrieval_mode="hybrid_rerank",
        candidate_pool_mode="rrf_top_20",
    )
    assert len(rerank_res) == 3
    assert rerank_res[0]["metadata"]["file_path"] == "src/flask/app.py"
    assert rerank_meta["retrieval_mode"] == "hybrid_rerank"
    assert rerank_meta["rerank_failed"] is False
    assert rerank_meta["fallback_to_rrf_used"] is False
    assert "reranker_raw_logit" in rerank_res[0]
    assert rerank_res[0]["reranker_score_interpretation"] == "raw_unnormalized_relevance_logit"

    # Simulated cross-encoder failure falls back cleanly to Hybrid RRF without crashing
    monkeypatch.setattr(
        embedder,
        "load_cross_encoder_reranker",
        lambda: (_ for _ in ()).throw(RuntimeError("simulated onnx session error")),
    )
    fb_res, fb_meta = embedder.search_similar(
        "Where is dispatch_request implemented?",
        "repo_d5",
        n_results=3,
        return_metadata=True,
        retrieval_mode="hybrid_rerank",
    )
    assert len(fb_res) == 3
    assert fb_meta["rerank_failed"] is True
    assert fb_meta["fallback_to_rrf_used"] is True
    assert "fallback" in fb_meta["distance_metric"]

    # Optional conversational rewrite gate: standalone bypasses LLM, follow-up calls llm.ask_gemini cleanly
    llm_calls = []

    def _fake_ask(prompt_text, return_metadata=False):
        llm_calls.append(prompt_text)
        txt = "How does Flask Config.from_object load configuration defaults?"
        return (txt, {"success": True, "error": None}) if return_metadata else txt

    monkeypatch.setattr(llm, "ask_gemini", _fake_ask)
    standalone_res, standalone_meta = embedder.search_similar(
        "Where is Flask Config defined?",
        "repo_d5",
        n_results=2,
        return_metadata=True,
        retrieval_mode="hybrid_rrf",
        enable_query_rewrite=True,
        chat_history=[{"role": "user", "content": "Hi"}],
    )
    assert len(llm_calls) == 0
    assert standalone_meta["query_rewritten"] is False
    assert len(standalone_res) == 2

    follow_res, follow_meta = embedder.search_similar(
        "How does it load settings from an object?",
        "repo_d5",
        n_results=2,
        return_metadata=True,
        retrieval_mode="hybrid_rrf",
        enable_query_rewrite=True,
        chat_history=[
            {"role": "user", "content": "Where is Flask Config defined?"},
            {"role": "assistant", "content": "In src/flask/config.py as class Config."},
        ],
    )
    assert len(llm_calls) == 1
    assert follow_meta["query_rewritten"] is True
    assert "Config.from_object" in follow_meta["effective_query"]
    assert len(follow_res) == 2

    # Invalid candidate_pool_mode fails cleanly
    with pytest.raises(ValueError, match="Unsupported candidate_pool_mode"):
        embedder.search_similar("x", "repo_d5", candidate_pool_mode="invalid_pool")


def test_app_integration_and_historical_artifacts_preserved():
    import embedder
    import llm

    assert embedder.SUPPORTED_RETRIEVAL_MODES[0] == "hybrid_rerank"
    assert embedder.DEFAULT_RETRIEVAL_MODE == "dense"
    assert inspect.isfunction(llm.build_prompt)
    assert inspect.isfunction(llm.ask_gemini)

    app_src = open(os.path.join(PROJECT_ROOT, "app.py"), encoding="utf-8").read()
    assert "hybrid_rerank" in app_src
    assert "enable_query_rewrite" in app_src
    assert "Enable Conversational Query Rewriting" in app_src
    assert "chat_history=st.session_state.messages[:-1]" in app_src

    preserved = [
        "experiments/results/baseline_summary.json",
        "experiments/embedding_experiments/results/embedding_summary.json",
        "experiments/chunking_experiments/results/chunking_summary.json",
        "experiments/day4_hybrid/results/hybrid_summary.json",
        "reports/day1/DAY1_FINAL_REPORT.md",
        "reports/day2/DAY2_TASK1_REPORT.md",
        "reports/day3/DAY3_CHUNKING_ANALYSIS.md",
        "reports/day4/DAY4_HYBRID_REPORT.md",
    ]
    for rel in preserved:
        assert os.path.exists(os.path.join(PROJECT_ROOT, rel)), f"Missing preserved artifact: {rel}"
