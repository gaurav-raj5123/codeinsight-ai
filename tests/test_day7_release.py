"""Day 7 Release Hardening, Chart Generation & Headless UI Smoke Tests."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm  # noqa: E402
from experiments.generate_final_charts import (  # noqa: E402
    ASSETS_DIR,
    generate_all_charts,
    load_day1_to_day6_summaries,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
EXPECTED_CHART_NAMES = [
    "v0_to_v3_retrieval_progression.png",
    "stage1_ceiling_vs_stage2_reranking.png",
    "latency_vs_quality_tradeoff.png",
    "day6_grounding_reliability.png",
]


def test_chart_generator_reads_saved_jsons_and_renders_valid_pngs(tmp_path: Path) -> None:
    summaries = load_day1_to_day6_summaries()
    assert set(summaries.keys()) == {"day1", "day2", "day3", "day4", "day5", "day6"}

    rendered = generate_all_charts(tmp_path)
    assert [p.name for p in rendered] == EXPECTED_CHART_NAMES
    for chart_path in rendered:
        assert chart_path.exists()
        raw = chart_path.read_bytes()
        assert raw.startswith(PNG_MAGIC)
        assert len(raw) > 50_000

    for chart_name in EXPECTED_CHART_NAMES:
        committed = ASSETS_DIR / chart_name
        assert committed.exists(), f"Missing committed release asset: {committed}"
        raw = committed.read_bytes()
        assert raw.startswith(PNG_MAGIC)
        assert len(raw) > 50_000


def test_streamlit_app_headless_smoke_and_defaults() -> None:
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not at.exception, f"AppTest raised exception(s): {at.exception}"

    assert at.session_state["chunking_strategy"] == "code_aware_v1"
    assert at.session_state["retrieval_mode"] == "hybrid_rerank"
    assert at.session_state["enable_query_rewrite"] is False
    assert at.session_state["grounded_generation"] is True
    assert at.session_state["grounded_model"] == "gemini-3.5-flash-lite"


def test_requirements_v0_anchors_and_secret_hygiene() -> None:
    req_lines = [
        line.strip().lower()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    pkg_names = [line.split(">=")[0].split("==")[0].split("<")[0].strip() for line in req_lines]
    assert "torch" not in pkg_names
    assert "sentence-transformers" not in pkg_names
    for required_pkg in (
        "streamlit",
        "chromadb",
        "onnxruntime",
        "tokenizers",
        "huggingface-hub",
        "google-genai",
        "pydantic",
        "matplotlib",
        "pytest",
    ):
        assert required_pkg in pkg_names

    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "AIza" not in env_example

    sig = inspect.signature(llm.build_prompt)
    assert list(sig.parameters.keys()) == ["question", "chunks", "directory_tree", "chat_history"]
    assert callable(llm.ask_gemini)
