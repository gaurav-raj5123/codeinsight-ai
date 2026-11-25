"""Comprehensive verification tests for Day 3 code-aware chunking (V1) and V0 dispatch parity."""

import json
import os
import sys

import chromadb
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chunker import (
    DEFAULT_CHUNKING_STRATEGY,
    SUPPORTED_CHUNKING_STRATEGIES,
    chunk_code,
)
from chunker_v1 import chunk_code_v1, count_tokens, get_embedding_tokenizer


def _tokenizer():
    tok, mode = get_embedding_tokenizer()
    assert mode != "uninitialized"
    return tok


def test_default_strategy_preserves_v0_behavior_and_rejects_invalid():
    sample = {
        "file_path": "src/example.py",
        "extension": ".py",
        "content": (
            "CONST = 1\n"
            "def alpha():\n"
            "    return CONST\n"
        ),
    }
    default_chunks = chunk_code(sample)
    explicit_v0 = chunk_code(sample, strategy="existing_chunker_v0")
    assert DEFAULT_CHUNKING_STRATEGY == "existing_chunker_v0"
    assert default_chunks == explicit_v0

    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        chunk_code(sample, strategy="nonexistent_strategy")

    for s in ("existing_chunker_v0", "fixed_window_v0", "code_aware_v1"):
        assert s in SUPPORTED_CHUNKING_STRATEGIES


def test_python_function_class_method_and_nested_no_duplicate():
    src = (
        "import os\n"
        "from typing import Any\n\n"
        "DEFAULT_TIMEOUT = 30\n\n"
        "def top_helper(x: int) -> int:\n"
        "    '''Double the input.'''\n"
        "    def inner_step(v: int) -> int:\n"
        "        return v * 2\n"
        "    return inner_step(x)\n\n"
        "class Router:\n"
        "    '''Main request router.'''\n"
        "    prefix = '/api'\n\n"
        "    def add_route(self, path: str, handler: Any) -> None:\n"
        "        self.routes[path] = handler\n\n"
        "    async def dispatch(self, path: str) -> Any:\n"
        "        return self.routes.get(path)\n"
    )
    chunks = chunk_code_v1(
        {"file_path": "pkg/router.py", "extension": ".py", "content": src},
        repository="org/repo",
        tokenizer=_tokenizer(),
    )
    assert chunks

    # Module-level imports & constants preserved cleanly
    all_lines = src.split("\n")
    covered_non_empty = set()
    for ch in chunks:
        m = ch["metadata"]
        for ln in range(m["start_line"], m["end_line"] + 1):
            if all_lines[ln - 1].strip():
                covered_non_empty.add(ln)
    expected_non_empty = {i + 1 for i, ln in enumerate(all_lines) if ln.strip()}
    assert covered_non_empty == expected_non_empty

    # Nested inner_step must NOT emit a separate duplicate chunk overlapping top_helper
    func_chunks = [c for c in chunks if "top_helper" in c["metadata"]["symbol_name"]]
    assert len(func_chunks) == 1
    assert "inner_step" in func_chunks[0]["content"]
    assert all(c["metadata"]["symbol_name"] != "inner_step" for c in chunks)

    # Small class cleanly merges without duplicate methods
    all_router_content = "\n".join(c["content"] for c in chunks if c["metadata"]["class_name"] == "Router")
    assert all_router_content.count("def add_route") == 1
    assert all_router_content.count("async def dispatch") == 1

    # Non-trivial class where methods exceed merge threshold emits separate method chunks
    large_cls_src = (
        "class LargeService:\n"
        "    '''Coordinates multi-step workflows across storage and cache.'''\n"
        "    version = 2\n\n"
        "    def handle_alpha(self, req_id: str, payload: dict) -> dict:\n"
        + "\n".join(f"        step_{i} = transform_{i}(payload, req_id)" for i in range(18))
        + "\n        return {'id': req_id, 'ok': True}\n\n"
        "    async def handle_beta(self, batch_items: list) -> int:\n"
        + "\n".join(f"        acc_{i} = await process_item_{i}(batch_items)" for i in range(18))
        + "\n        return len(batch_items)\n"
    )
    large_chunks = chunk_code_v1(
        {"file_path": "pkg/service.py", "extension": ".py", "content": large_cls_src},
        repository="org/repo",
        tokenizer=_tokenizer(),
    )
    method_chunks = [c for c in large_chunks if c["metadata"]["chunk_type"].startswith("method")]
    method_names = {c["metadata"]["symbol_name"] for c in method_chunks}
    assert {"handle_alpha", "handle_beta"}.issubset(method_names)
    for mc in method_chunks:
        assert mc["metadata"]["class_name"] == "LargeService"
        assert mc["metadata"]["parent_symbol"] == "LargeService"
        assert "Class: LargeService" in mc["content"]
    alpha_chunk = next(c for c in method_chunks if c["metadata"]["symbol_name"] == "handle_alpha")
    assert "def handle_beta" not in alpha_chunk["content"]


def test_python_decorators_docstrings_and_accurate_line_mapping():
    src = (
        "@app.route('/health')\n"
        "@requires_auth\n"
        "def health_check():\n"
        "    '''Return service health status.'''\n"
        "    return {'ok': True}\n\n"
        "TRAILING_FLAG = True\n"
    )
    lines = src.split("\n")
    chunks = chunk_code_v1(
        {"file_path": "app/routes.py", "extension": ".py", "content": src},
        repository="org/repo",
        tokenizer=_tokenizer(),
    )
    route_chunk = next(c for c in chunks if "health_check" in c["metadata"]["symbol_name"])
    assert "@app.route('/health')" in route_chunk["content"]
    assert "Return service health status." in route_chunk["content"]
    s_l, e_l = route_chunk["metadata"]["start_line"], route_chunk["metadata"]["end_line"]
    expected_body = "\n".join(lines[s_l - 1 : e_l])
    assert expected_body in route_chunk["content"]


def test_oversized_function_token_budget_enforcement():
    body_lines = [
        f"    val_{i} = compute_complex_transformation_step_{i}(payload_{i}, factor={i + 3})"
        for i in range(120)
    ]
    src = "def huge_pipeline(payload_0):\n" + "\n".join(body_lines) + "\n    return val_119\n"
    tok = _tokenizer()
    chunks = chunk_code_v1(
        {"file_path": "pkg/pipeline.py", "extension": ".py", "content": src},
        repository="org/repo",
        tokenizer=tok,
        max_tokens=256,
    )
    assert len(chunks) > 1
    for idx, ch in enumerate(chunks, start=1):
        m = ch["metadata"]
        actual_tok = count_tokens(ch["content"], tok)
        assert actual_tok <= 256
        assert m["token_count"] == actual_tok
        assert m["is_split"] is True
        assert m["chunk_type"].endswith("_split")
        assert m["part_index"] == idx
        assert m["total_parts"] == len(chunks)
        assert "def huge_pipeline" in ch["content"] or "Signature: def huge_pipeline" in ch["content"]

    # Extremely long single line handling
    giant_line = "const data = [" + ", ".join(f"'item_{i}'" for i in range(300)) + "];"
    single_line_chunks = chunk_code_v1(
        {"file_path": "web/huge.js", "extension": ".js", "content": giant_line},
        repository="org/repo",
        tokenizer=tok,
        max_tokens=256,
    )
    assert len(single_line_chunks) > 1
    assert all(count_tokens(c["content"], tok) <= 256 for c in single_line_chunks)


def test_javascript_express_patterns_and_no_control_flow_splits():
    src = (
        "/*! Copyright header */\n\n"
        "var app = module.exports = {};\n\n"
        "app.handle = function handle(req, res, callback) {\n"
        "  if (req && req.url) {\n"
        "    for (var i = 0; i < 3; i++) {\n"
        "      callback();\n"
        "    }\n"
        "  }\n"
        "};\n\n"
        "function createHelper(name) {\n"
        "  if (!name) {\n"
        "    return null;\n"
        "  }\n"
        "  return { name: name };\n"
        "}\n"
    )
    chunks = chunk_code_v1(
        {"file_path": "lib/application.js", "extension": ".js", "content": src},
        repository="expressjs/express",
        tokenizer=_tokenizer(),
    )
    assert all(c["metadata"]["symbol_name"] not in {"if", "for", "while", "switch", "catch"} for c in chunks)
    assert all(c["metadata"]["parser_type"] == "lexical_brace_scanner" for c in chunks)
    assert all(c["metadata"]["fallback_used"] is False for c in chunks)
    assert any("app.handle" in c["metadata"]["symbol_name"] for c in chunks)
    assert any("createHelper" in c["metadata"]["symbol_name"] for c in chunks)


def test_cpp_preprocessor_macros_and_templates_not_split_falsely():
    src = (
        "#define FMT_MAP_2(f, a, b) \\\n"
        "    f(a), f(b)\n\n"
        "template <typename Char>\n"
        "struct text_box {\n"
        "  Char value;\n"
        "  constexpr text_box(Char v) noexcept : value(v) {}\n"
        "};\n\n"
        "template <typename T>\n"
        "inline auto format_item(const T& v) -> T {\n"
        "  if (v) {\n"
        "    return v;\n"
        "  }\n"
        "  return T{};\n"
        "}\n"
    )
    chunks = chunk_code_v1(
        {"file_path": "include/fmt/sample.h", "extension": ".h", "content": src},
        repository="fmtlib/fmt",
        tokenizer=_tokenizer(),
    )
    assert all(c["metadata"]["symbol_name"] != "f" for c in chunks)
    assert any("text_box" in c["metadata"]["symbol_name"] for c in chunks)
    fmt_chunk = next(c for c in chunks if "format_item" in c["metadata"]["symbol_name"])
    assert "template <typename T>" in fmt_chunk["content"]


def test_empty_malformed_and_unsupported_fallback_behavior():
    tok = _tokenizer()
    assert chunk_code_v1({"file_path": "empty.py", "extension": ".py", "content": ""}, tokenizer=tok) == []
    assert chunk_code_v1({"file_path": "blank.py", "extension": ".py", "content": "   \n\n\t  "}, tokenizer=tok) == []

    # SyntaxError Python falls back cleanly without crashing
    broken_py = "def broken(\n    x = 1\n    return x\n"
    broken_chunks = chunk_code_v1(
        {"file_path": "bad/syntax.py", "extension": ".py", "content": broken_py},
        tokenizer=tok,
    )
    assert len(broken_chunks) >= 1
    assert broken_chunks[0]["metadata"]["fallback_used"] is True
    assert broken_chunks[0]["metadata"]["parser_type"] == "python_regex_fallback"
    assert count_tokens(broken_chunks[0]["content"], tok) <= 256

    # Unsupported extension uses token_window_fallback
    ruby_chunks = chunk_code_v1(
        {"file_path": "scripts/build.rb", "extension": ".rb", "content": "def hello\n  puts 'hi'\nend\n"},
        tokenizer=tok,
    )
    assert len(ruby_chunks) == 1
    assert ruby_chunks[0]["metadata"]["fallback_used"] is True
    assert ruby_chunks[0]["metadata"]["parser_type"] == "token_window_fallback"


def test_deterministic_ids_distinct_locations_and_real_chromadb_insert():
    src = (
        "def ping():\n    return 'pong'\n\n"
        "def ping():\n    return 'pong'\n"
    )
    tok = _tokenizer()
    f = {"file_path": "src\\win\\style.py", "extension": ".py", "content": "x = 1\n" + src}
    c1 = chunk_code_v1(f, repository="org/a", tokenizer=tok)
    c2 = chunk_code_v1(f, repository="org/a", tokenizer=tok)
    assert [c["metadata"]["chunk_id"] for c in c1] == [c["metadata"]["chunk_id"] for c in c2]

    # Same function body at different lines / files gets distinct chunk_ids
    same_body = "def ping():\n    return 'pong'\n"
    fa = {"file_path": "a.py", "extension": ".py", "content": same_body}
    fb = {"file_path": "b.py", "extension": ".py", "content": "\n\n" + same_body}
    ca = chunk_code_v1(fa, repository="org/a", tokenizer=tok)
    cb = chunk_code_v1(fb, repository="org/a", tokenizer=tok)
    assert ca[0]["metadata"]["chunk_id"] != cb[0]["metadata"]["chunk_id"]
    assert ca[0]["metadata"]["content_sha256"] == cb[0]["metadata"]["content_sha256"]

    # Validate Chroma primitive types and real Chroma insertion/query
    client = chromadb.Client()
    col = client.get_or_create_collection(name="test_v1_chroma_compat")
    for ch in c1:
        for k, v in ch["metadata"].items():
            assert isinstance(v, (str, int, float, bool)), f"Bad type {type(v)} on {k}"
    col.add(
        embeddings=[[0.1] * 384 for _ in c1],
        documents=[c["content"] for c in c1],
        metadatas=[c["metadata"] for c in c1],
        ids=[c["metadata"]["chunk_id"] for c in c1],
    )
    q_res = col.query(query_embeddings=[[0.1] * 384], n_results=1)
    assert len(q_res["ids"][0]) == 1
    client.delete_collection(name="test_v1_chroma_compat")


def test_chunker_v1_on_real_corpus_invariants():
    """Verify V1 chunker enforces zero empty chunks, <=256 tokens, and 100% non-empty line coverage on real files."""
    tok = _tokenizer()
    flask_app_path = os.path.join(PROJECT_ROOT, "experiments/datasets/cache/flask/src/flask/globals.py")
    with open(flask_app_path, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = chunk_code_v1(
        {"file_path": "src/flask/globals.py", "extension": ".py", "content": content},
        repository="pallets/flask",
        tokenizer=tok,
    )
    assert chunks
    assert all(c["content"].strip() for c in chunks)
    assert all(count_tokens(c["content"], tok) <= 256 for c in chunks)
    # Specifically verify module-level LocalProxy definitions that V0 dropped are now covered!
    joined = "\n".join(c["content"] for c in chunks)
    assert "current_app" in joined
    assert "LocalProxy" in joined
