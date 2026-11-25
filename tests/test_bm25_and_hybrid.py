"""Comprehensive tests for Day 4 code-aware BM25, header hygiene, RRF, and benchmark runner."""

import math
import os
import sys

import chromadb
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.retrieval_metrics import evaluate_retrieved_chunks
from experiments.day4_hybrid.run_hybrid_comparison import (
    calc_extended_distribution,
    compute_paired_comparison,
    paired_bootstrap_ci,
)
from retrieval.bm25_retriever import (
    BM25Retriever,
    build_bm25_from_collection,
    strip_v1_header_labels,
    tokenize_code,
)
from retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion


def test_v1_header_label_hygiene_preserves_values_and_ignores_body():
    chunk_text = (
        "File: src/flask/wrappers.py | Language: python | Type: method\n"
        "Class: Request | Symbol: get_json | Part: 2/3\n"
        "Signature: def get_json(self, force=False):\n"
        "    # Type: optional json body\n"
        "    return self.on_json_loading_failed()\n"
    )
    stripped = strip_v1_header_labels(chunk_text)
    header_part, body_part = stripped.split("# Type: optional", 1)
    for label in ("File:", "Language:", "Type:", "Class:", "Symbol:", "Part:", "Signature:"):
        assert label not in header_part
    # Field values preserved cleanly
    assert "src/flask/wrappers.py" in header_part
    assert "python" in header_part
    assert "method" in header_part
    assert "Request" in header_part
    assert "get_json" in header_part
    # Body comments NOT mistakenly stripped
    assert "# Type: optional json body" in stripped

    # Generic header-label tokens must NOT pollute corpus matching
    toks = tokenize_code(chunk_text)
    for generic in ("file", "language", "class", "symbol", "part", "signature"):
        assert generic not in toks


def test_code_aware_tokenizer_identifiers_paths_and_short_tokens():
    sample = (
        "handle_request createApplication HTTPResponseHandler FMT_STRING "
        "res.cookie app.engine Config.from_object fmt::format include/fmt/os.h "
        "g ip os fg bg"
    )
    toks = set(tokenize_code(sample, strip_v1_labels=False))
    expected = {
        "handle_request",
        "handle",
        "request",
        "createapplication",
        "create",
        "application",
        "httpresponsehandler",
        "http",
        "response",
        "handler",
        "fmt_string",
        "fmt",
        "string",
        "res.cookie",
        "res",
        "cookie",
        "app.engine",
        "app",
        "engine",
        "config.from_object",
        "config",
        "from_object",
        "from",
        "object",
        "fmt::format",
        "format",
        "include/fmt/os.h",
        "include",
        "os.h",
        "g",
        "ip",
        "os",
        "fg",
        "bg",
    }
    assert expected.issubset(toks)
    # Normal single identifier must not be duplicated within one token emission
    assert tokenize_code("flask", strip_v1_labels=False) == ["flask"]


def test_bm25_zero_overlap_exclusion_no_padding_and_negative_score_safety():
    ids = ["c1", "c2", "c3", "c4"]
    docs = [
        "File: lib/response.js | Language: javascript | Type: method\nSymbol: res.cookie\nfunction cookie() {}",
        "File: lib/request.js | Language: javascript | Type: getter\nSymbol: req.ip\nreturn trustProxy;",
        "File: include/fmt/os.h | Language: cpp | Type: class\nSymbol: file\nclass file { int fd; };",
        "File: src/flask/globals.py | Language: python | Type: module_context\nSymbol: g\ng = LocalProxy()",
    ]
    metas = [{"file_path": f"f{i}", "start_line": 1, "end_line": 2} for i in range(4)]

    bm25 = BM25Retriever(ids, docs, metas, idf_mode="lucene")
    # 1. Generic word 'class' matches ONLY c3 (has real 'class' keyword/token in body), NOT c1/c2/c4 headers
    res_class = bm25.search("class", top_k=10)
    assert [r["id"] for r in res_class] == ["c3"]

    # 2. Zero lexical match returns [] without padding
    assert bm25.search("nonexistent_symbol_xyz", top_k=10) == []
    assert bm25.search("   ", top_k=10) == []

    # 3. Fewer than K matches returns exact subset without padding
    res_cookie = bm25.search("res.cookie", top_k=5)
    assert len(res_cookie) == 1
    assert res_cookie[0]["id"] == "c1"
    assert res_cookie[0]["bm25_score"] > 0

    # 4. Verify valid zero/negative scores under raw_rsj when df > N/2 STILL pass eligibility via lexical overlap
    rsj_bm25 = BM25Retriever(
        ["d1", "d2", "d3"],
        ["alpha common", "beta common", "gamma other"],
        [{}, {}, {}],
        idf_mode="raw_rsj",
    )
    # 'common' appears in 2/3 docs -> RSJ log((3 - 2 + 0.5)/(2 + 0.5)) = log(0.6) < 0
    assert rsj_bm25.idf["common"] < 0
    rsj_hits = rsj_bm25.search("common", top_k=5)
    # Zero-overlap d3 ('gamma other') MUST be excluded; negative-score d1 & d2 MUST still return!
    assert {r["id"] for r in rsj_hits} == {"d1", "d2"}
    assert all(r["bm25_score"] < 0 for r in rsj_hits)


def test_rrf_fusion_empty_bm25_fallback_and_union_not_truncated():
    dense = [
        {"id": "d1", "content": "a", "metadata": {"file_path": "a.py", "start_line": 1, "end_line": 5}, "distance": 0.4},
        {"id": "d2", "content": "b", "metadata": {"file_path": "b.py", "start_line": 1, "end_line": 5}, "distance": 0.5},
        {"id": "d3", "content": "c", "metadata": {"file_path": "c.py", "start_line": 1, "end_line": 5}, "distance": 0.6},
    ]
    bm25_with_match = [
        {"id": "d3", "content": "c", "metadata": {"file_path": "c.py", "start_line": 1, "end_line": 5}, "bm25_score": 8.0},
        {"id": "b_only", "content": "x", "metadata": {"file_path": "x.py", "start_line": 1, "end_line": 5}, "bm25_score": 7.0},
    ]

    fused, stats = reciprocal_rank_fusion(dense, bm25_with_match, rrf_k=60, top_k=5)
    assert stats["overlap_candidates_count"] == 1
    assert stats["union_candidates_count"] == 4
    # d3 appeared at rank 3 in dense and rank 1 in bm25: 1/63 + 1/61 > d1's 1/61 alone
    assert fused[0]["id"] == "d3"
    assert fused[0]["retrieval_sources"] == ["dense", "bm25"]
    expected_d3 = round(1.0 / 63.0 + 1.0 / 61.0, 8)
    assert fused[0]["rrf_score"] == expected_d3

    # Empty BM25 fallback must preserve Dense rankings gracefully
    fused_empty, stats_empty = reciprocal_rank_fusion(dense, [], rrf_k=60, top_k=5)
    assert stats_empty["bm25_fallback_to_dense_only"] is True
    assert [c["id"] for c in fused_empty] == ["d1", "d2", "d3"]

    # Union > 20 items MUST NOT truncate candidate #25 when measuring candidate_union_hit_rate
    dense_20 = [
        {"id": f"dn_{i}", "content": "", "metadata": {"file_path": "miss.py", "start_line": 1, "end_line": 2}, "distance": 0.5}
        for i in range(20)
    ]
    bm25_20 = [
        {"id": f"bm_{i}", "content": "", "metadata": {"file_path": "target.py" if i == 15 else "miss.py", "start_line": 10, "end_line": 20}, "bm25_score": 2.0}
        for i in range(20)
    ]
    bm_idx = BM25Retriever(["dummy"], ["dummy"], [{}])
    # Stub bm25.search for exact 40-union control test
    bm_idx.search = lambda q, top_k=20, return_metadata=False: (
        (bm25_20[:top_k], {"eligible_overlap_docs": 20, "returned_count": 20, "query_tokens": [q]})
        if return_metadata
        else bm25_20[:top_k]
    )
    hr = HybridRetriever(dense_search_fn=lambda q, n: dense_20[:n], bm25_retriever=bm_idx, candidate_k=20, rrf_k=60)
    _, diag = hr.retrieve("q", top_k=5, return_diagnostics=True)
    assert len(diag["candidate_union"]) == 40
    # Passing top_k=20 would miss bm_15 at union index 35! Passing len(union) captures it accurately
    spans = [{"file_path": "target.py", "start_line": 12, "end_line": 15}]
    assert evaluate_retrieved_chunks(diag["candidate_union"], spans, top_k=20)["first_relevant_rank"] is None
    assert evaluate_retrieved_chunks(diag["candidate_union"], spans, top_k=len(diag["candidate_union"]))["first_relevant_rank"] == 36


def test_chroma_collection_bm25_hydration_and_paired_helpers():
    client = chromadb.Client()
    col = client.get_or_create_collection(name="day4_test_hydration")
    ids = ["v1_a", "v1_b"]
    docs = [
        "File: lib/app.js | Language: javascript | Type: function\nSymbol: createApplication\nfunction createApplication() {}",
        "File: lib/view.js | Language: javascript | Type: function\nSymbol: View\nfunction View() {}",
    ]
    metas = [
        {"file_path": "lib/app.js", "start_line": 1, "end_line": 2, "chunk_id": "v1_a"},
        {"file_path": "lib/view.js", "start_line": 1, "end_line": 2, "chunk_id": "v1_b"},
    ]
    col.add(embeddings=[[0.1] * 16, [0.2] * 16], documents=docs, metadatas=metas, ids=ids)

    bm25, fp1 = build_bm25_from_collection(col)
    direct = BM25Retriever(ids, docs, metas)
    assert fp1 == direct.corpus_fingerprint
    assert [r["id"] for r in bm25.search("createApplication", top_k=5)] == ["v1_a"]
    client.delete_collection(name="day4_test_hydration")

    # Bootstrap CI determinism & paired stats check
    ci1 = paired_bootstrap_ci([1.0, 0.0, -1.0, 1.0, 0.0], num_resamples=500, seed=42)
    ci2 = paired_bootstrap_ci([1.0, 0.0, -1.0, 1.0, 0.0], num_resamples=500, seed=42)
    assert ci1 == ci2
    ext = calc_extended_distribution([1.0, 2.0, 3.0, 4.0])
    assert ext["median"] == 2.5
    assert ext["iqr"] >= 0.0

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
                "hit_rate_at_5": 1.0 if hit else 0.0,
                "mrr_at_5": (1.0 / rank) if rank else 0.0,
                "span_recall_at_5": 1.0 if hit else 0.0,
            },
        }

    pc = compute_paired_comparison(
        [_rec("q1", False, None), _rec("q2", True, 3)],
        [_rec("q1", True, 1), _rec("q2", True, 1)],
    )
    assert pc["hit_at_5_quadrants"]["dense_miss_hybrid_hit"] == 1
    assert pc["mrr_and_span_transitions"]["hybrid_improved_count"] == 2


def test_embedder_chroma_persistent_hydration_restart_switch_and_reindex(tmp_path, monkeypatch):
    """Verify Override #7: restart persistence, Chroma hydration, re-index invalidation, repo switch, zero pickles."""
    import embedder
    from chunker import chunk_code
    from chunker_v1 import get_embedding_tokenizer

    db_dir = str(tmp_path / "chroma_persist_test")
    monkeypatch.setattr(embedder, "get_chroma_client", lambda: chromadb.PersistentClient(path=db_dir))
    embedder.clear_bm25_cache()

    tok, _ = get_embedding_tokenizer()
    chunks_a1 = chunk_code(
        {
            "file_path": "src/alpha.py",
            "extension": ".py",
            "content": "def legacy_foo(x):\n    return x + 1\n",
        },
        strategy="code_aware_v1",
        repository="repo_alpha",
        tokenizer=tok,
    )
    chunks_b = chunk_code(
        {
            "file_path": "src/beta.py",
            "extension": ".py",
            "content": "def beta_worker(y):\n    return y * 2\n",
        },
        strategy="code_aware_v1",
        repository="repo_beta",
        tokenizer=tok,
    )

    count_a1, idx_meta_a1 = embedder.index_chunks(chunks_a1, "repo_alpha", return_metadata=True)
    count_b, idx_meta_b = embedder.index_chunks(chunks_b, "repo_beta", return_metadata=True)
    assert count_a1 == len(chunks_a1)
    assert count_b == len(chunks_b)
    assert "corpus_fingerprint" in idx_meta_a1

    # Ensure no pickle files created in db_dir or workspace
    for root, _, files in os.walk(db_dir):
        assert not any(f.endswith((".pkl", ".pickle")) for f in files)

    # First query hydrates from Chroma (cache_hit=False), second reuses cached index (cache_hit=True)
    res1, rmeta1 = embedder.search_similar("legacy_foo", "repo_alpha", n_results=5, return_metadata=True, retrieval_mode="hybrid_rrf")
    assert rmeta1["bm25_cache_hit"] is False
    assert rmeta1["bm25_corpus_fingerprint"] == idx_meta_a1["corpus_fingerprint"]
    assert res1[0]["id"] == chunks_a1[0]["metadata"]["chunk_id"]

    res2, rmeta2 = embedder.search_similar("legacy_foo", "repo_alpha", n_results=5, return_metadata=True, retrieval_mode="hybrid_rrf")
    assert rmeta2["bm25_cache_hit"] is True

    # Switching repositories isolates results and fingerprints
    res_b, rmeta_b = embedder.search_similar("beta_worker", "repo_beta", n_results=5, return_metadata=True, retrieval_mode="bm25")
    assert [c["id"] for c in res_b] == [chunks_b[0]["metadata"]["chunk_id"]]
    assert rmeta_b["bm25_corpus_fingerprint"] == idx_meta_b["corpus_fingerprint"]
    assert embedder.search_similar("legacy_foo", "repo_beta", n_results=5, retrieval_mode="bm25") == []

    # Empty BM25 match falls back gracefully to dense rankings inside hybrid_rrf
    res_fallback, rmeta_fb = embedder.search_similar("totally_unseen_lexical_token_xyz", "repo_alpha", n_results=5, return_metadata=True, retrieval_mode="hybrid_rrf")
    assert rmeta_fb["bm25_fallback_to_dense_only"] is True
    assert len(res_fallback) == 1

    # Simulated application restart: clear memory cache and open fresh PersistentClient
    embedder.clear_bm25_cache()
    monkeypatch.setattr(embedder, "get_chroma_client", lambda: chromadb.PersistentClient(path=db_dir))
    res_restart, rmeta_restart = embedder.search_similar("legacy_foo", "repo_alpha", n_results=5, return_metadata=True, retrieval_mode="hybrid_rrf")
    assert rmeta_restart["bm25_cache_hit"] is False
    assert rmeta_restart["bm25_corpus_fingerprint"] == idx_meta_a1["corpus_fingerprint"]
    assert res_restart[0]["id"] == chunks_a1[0]["metadata"]["chunk_id"]

    # Re-indexing repo_alpha invalidates prior lexical index cleanly (no stale chunks!)
    chunks_a2 = chunk_code(
        {
            "file_path": "src/alpha.py",
            "extension": ".py",
            "content": "def modern_bar(x):\n    return x * 10\n",
        },
        strategy="code_aware_v1",
        repository="repo_alpha",
        tokenizer=tok,
    )
    _, idx_meta_a2 = embedder.index_chunks(chunks_a2, "repo_alpha", return_metadata=True)
    assert idx_meta_a2["corpus_fingerprint"] != idx_meta_a1["corpus_fingerprint"]
    stale_res, rmeta_stale = embedder.search_similar("legacy_foo", "repo_alpha", n_results=5, return_metadata=True, retrieval_mode="bm25")
    assert stale_res == []
    assert rmeta_stale["bm25_cache_hit"] is False
    assert rmeta_stale["bm25_corpus_fingerprint"] == idx_meta_a2["corpus_fingerprint"]
    res_new, rmeta_new = embedder.search_similar("modern_bar", "repo_alpha", n_results=5, return_metadata=True, retrieval_mode="hybrid_rrf")
    assert rmeta_new["bm25_cache_hit"] is True
    assert res_new[0]["id"] == chunks_a2[0]["metadata"]["chunk_id"]

    # Unknown retrieval mode must fail loudly
    with pytest.raises(ValueError, match="Unsupported retrieval_mode"):
        embedder.search_similar("x", "repo_alpha", retrieval_mode="unknown_mode")


def test_day1_day2_day3_artifacts_preserved():
    expected = [
        "experiments/results/baseline_summary.json",
        "experiments/embedding_experiments/results/embedding_summary.json",
        "experiments/chunking_experiments/results/chunking_summary.json",
        "reports/day1/DAY1_FINAL_REPORT.md",
        "reports/day2/DAY2_TASK1_REPORT.md",
        "reports/day3/DAY3_CHUNKING_AUDIT.md",
        "reports/day3/DAY3_CHUNKING_ANALYSIS.md",
    ]
    for rel in expected:
        assert os.path.exists(os.path.join(PROJECT_ROOT, rel)), f"Missing preserved file: {rel}"

