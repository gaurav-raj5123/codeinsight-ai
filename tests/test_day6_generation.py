"""Deterministic offline unit and contract tests for Day 6 Grounded Generation (zero live API calls)."""

import json
import os
import sys
from unittest.mock import patch

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import llm
import telemetry
from experiments.day6_generation.run_day6_benchmark import (
    run_g4_offline_stage,
    run_generation_stage,
)
from generation.citations import (
    build_evidence_map,
    detect_unretrieved_file_mentions,
    validate_and_render_citations,
)
from generation.evaluation import (
    aggregate_layer1_metrics,
    detect_unstructured_abstention,
    evaluate_layer1_record,
    judge_single_candidate,
)
from generation.grounded_generator import (
    apply_g4_postprocessing,
    call_gemini_resilient,
    generate_grounded_answer,
    parse_structured_g3_response,
    run_generation_experiment_item,
)
from generation.prompts import (
    CONVERSATION_HISTORY_ISOLATION_NOTICE,
    DIRECTORY_TREE_ISOLATION_NOTICE,
    build_g0_prompt,
    build_g1_prompt,
    build_g2_prompt,
    build_g3_prompt,
    strip_redundant_v1_header_for_prompt,
)


def _sample_chunks():
    return [
        {
            "chunk_id": "v1_aaa111",
            "content": (
                "File: src/flask/app.py | Language: python | Type: method\n"
                "Class: Flask | Symbol: dispatch_request\n"
                "Signature: def dispatch_request(self):\n"
                "def dispatch_request(self):\n"
                "    return self.view_functions[rule.endpoint]()"
            ),
            "metadata": {
                "file_path": "src/flask/app.py",
                "symbol_name": "dispatch_request",
                "class_name": "Flask",
                "start_line": 1490,
                "end_line": 1525,
                "chunk_type": "method",
            },
        },
        {
            "chunk_id": "v1_bbb222",
            "content": (
                "File: src/flask/config.py | Language: python | Type: method\n"
                "Class: Config | Symbol: from_object\n"
                "def from_object(self, obj):\n"
                "    self[key] = getattr(obj, key)"
            ),
            "metadata": {
                "file_path": "src/flask/config.py",
                "symbol_name": "from_object",
                "class_name": "Config",
                "start_line": 240,
                "end_line": 265,
                "chunk_type": "method",
            },
        },
    ]


def test_prompts_v1_header_hygiene_tree_isolation_and_g0_compat():
    chunks = _sample_chunks()
    original_content = chunks[0]["content"]
    tree = "📂 flask/\n  📄 sessions.py\n  📄 cli.py"

    # Strip removes only duplicate File:/Class: lines for prompt display without mutating chunk
    stripped = strip_redundant_v1_header_for_prompt(original_content)
    assert not stripped.startswith("File: src/flask/app.py |")
    assert "Signature: def dispatch_request(self):" in stripped
    assert "return self.view_functions" in stripped
    assert chunks[0]["content"] == original_content

    # G0 still matches llm.build_prompt() cleanly
    g0 = build_g0_prompt("Where is dispatch?", chunks, directory_tree=tree)
    assert g0 == llm.build_prompt("Where is dispatch?", chunks, tree)

    # G1, G2, G3 contain verbatim Directory Tree Isolation Notice and formatted [EVIDENCE_1] blocks
    for builder in (build_g1_prompt, build_g2_prompt, build_g3_prompt):
        p = builder("Where is dispatch?", chunks, directory_tree=tree)
        assert DIRECTORY_TREE_ISOLATION_NOTICE in p
        assert "[EVIDENCE_1]" in p
        assert "File: src/flask/app.py" in p
        assert "Lines: 1490–1525" in p
        assert "File: src/flask/app.py | Language: python" not in p  # no duplicated header


def test_conversation_history_isolation_prevents_assistant_override():
    chunks = _sample_chunks()
    poisoned_history = [
        {"role": "user", "content": "How does Flask connect to PostgreSQL pools?"},
        {
            "role": "assistant",
            "content": "Flask uses src/flask/postgres_pool.py create_pg_pool() internally.",
        },
    ]
    p3 = build_g3_prompt(
        "What parameters does that pooling helper accept?",
        chunks,
        chat_history=poisoned_history,
    )
    assert CONVERSATION_HISTORY_ISOLATION_NOTICE in p3
    assert "Prior Assistant (UNVERIFIED): Flask uses src/flask/postgres_pool.py" in p3
    # Ensure history section appears strictly separate from RETRIEVED CODE EVIDENCE BLOCKS
    hist_idx = p3.index("CONVERSATION HISTORY")
    ev_idx = p3.index("RETRIEVED CODE EVIDENCE BLOCKS")
    assert hist_idx < ev_idx

    # Simulated follow-up answer ignoring poisoned history and abstaining cleanly
    mock_g3_json = json.dumps(
        {
            "sufficient": False,
            "confidence_tier": "insufficient",
            "reason": "The retrieved blocks cover dispatch_request and Config.from_object, not PostgreSQL pooling.",
            "missing_information": ["Any PostgreSQL connection pooling implementation"],
            "answer": "I couldn't establish any PostgreSQL pooling parameters from the retrieved code.",
            "citations": [],
            "abstained": True,
        }
    )
    res = generate_grounded_answer(
        "What parameters does that pooling helper accept?",
        chunks,
        chat_history=poisoned_history,
        llm_call_fn=lambda prompt: mock_g3_json,
    )
    assert res["g4"]["abstained"] is True
    assert res["g4"]["confidence_tier"] == "insufficient"
    assert res["g4"]["citation_validation_report"]["valid_citation_ids"] == []


def test_citation_validation_edge_cases_and_no_fabricated_links():
    chunks = _sample_chunks()
    # Mix valid, duplicate, multi-ID bracket, invalid ID, and unretrieved file path
    raw_ans = (
        "Routing uses `dispatch_request` [EVIDENCE_1, EVIDENCE_1]. "
        "Settings use `from_object` [EVIDENCE_2]. "
        "Fake helper in src/flask/magic.py [EVIDENCE_99] [EVIDENCE_0]."
    )
    cit = validate_and_render_citations(
        raw_answer=raw_ans,
        declared_citations=["EVIDENCE_1", "EVIDENCE_2", "EVIDENCE_1", "EVIDENCE_99", "not_an_id.py"],
        retrieved_chunks=chunks,
        abstained=False,
    )
    assert cit["valid_citation_ids"] == ["EVIDENCE_1", "EVIDENCE_2"]
    assert cit["duplicate_citation_ids"] == ["EVIDENCE_1"]
    assert set(cit["invalid_citation_ids"]) == {"EVIDENCE_99", "not_an_id.py", "EVIDENCE_0"}
    assert cit["has_invalid_citations"] is True
    assert cit["validation_status"] == "invalid_citations_detected"
    assert "[EVIDENCE_1 (src/flask/app.py:1490–1525)]" in cit["rendered_answer"]
    assert "[EVIDENCE_2 (src/flask/config.py:240–265)]" in cit["rendered_answer"]
    # Invalid citations MUST NOT become file links!
    assert "[UNVERIFIED_CITATION: EVIDENCE_99]" in cit["rendered_answer"]
    assert "[UNVERIFIED_CITATION: EVIDENCE_0]" in cit["rendered_answer"]
    assert "src/flask/magic.py" in cit["unretrieved_file_mentions"]

    # Missing file_path and invalid line range metadata handling
    bad_meta_chunks = [
        {"chunk_id": "c1", "content": "def x(): pass", "metadata": {"file_path": "", "start_line": 10, "end_line": 2}},
        {"chunk_id": "c2", "content": "def y(): pass", "metadata": {"file_path": "lib/y.py", "start_line": 50, "end_line": 10}},
    ]
    emap = build_evidence_map(bad_meta_chunks)
    assert emap["EVIDENCE_1"]["has_valid_file"] is False
    assert "missing_file_path" in emap["EVIDENCE_1"]["metadata_warnings"]
    assert emap["EVIDENCE_2"]["has_valid_file"] is True
    assert emap["EVIDENCE_2"]["has_valid_lines"] is False
    assert emap["EVIDENCE_2"]["rendered_ref"] == "`lib/y.py`"

    cit_bad = validate_and_render_citations(
        "Calls [EVIDENCE_1] and [EVIDENCE_2].",
        ["EVIDENCE_1", "EVIDENCE_2"],
        bad_meta_chunks,
    )
    assert cit_bad["valid_citation_ids"] == ["EVIDENCE_2"]
    assert "EVIDENCE_1" in cit_bad["invalid_citation_ids"]
    assert "[EVIDENCE_2 (lib/y.py)]" in cit_bad["rendered_answer"]
    assert "EVIDENCE_2:invalid_or_missing_line_range" in cit_bad["metadata_warnings"]


def test_g3_structured_variants_and_parse_failure_preservation():
    chunks = _sample_chunks()

    # 1. Partial evidence response
    partial_raw = json.dumps(
        {
            "sufficient": True,
            "confidence_tier": "partial",
            "reason": "Evidence shows dispatch_request lookup, not error handling wrappers.",
            "missing_information": ["Exception handling around full_dispatch_request"],
            "answer": "Matched endpoint execution occurs in `dispatch_request` [EVIDENCE_1], while full error handling wrappers are not in the retrieved blocks.",
            "citations": ["EVIDENCE_1"],
            "abstained": False,
        }
    )
    g3_partial = run_generation_experiment_item(
        "G3",
        "Walk through full dispatch and exception handling.",
        chunks,
        llm_call_fn=lambda _: partial_raw,
    )
    g4_partial = apply_g4_postprocessing(g3_partial, chunks)
    assert g4_partial["confidence_tier"] == "partial"
    assert g4_partial["sufficient"] is True
    assert g4_partial["abstained"] is False
    assert g4_partial["citation_validation_report"]["valid_citation_ids"] == ["EVIDENCE_1"]

    # 2. Malformed JSON & missing keys -> do NOT crash or fabricate fields!
    malformed = parse_structured_g3_response("Here is my prose answer without JSON")
    assert malformed["parse_success"] is False
    assert malformed["parsed"] is None

    incomplete_json = parse_structured_g3_response(json.dumps({"sufficient": True, "answer": "missing rest"}))
    assert incomplete_json["parse_success"] is False
    assert "Missing required schema keys" in incomplete_json["parse_error"]

    g3_bad = run_generation_experiment_item(
        "G3",
        "Where is dispatch?",
        chunks,
        llm_call_fn=lambda _: "{broken json",
    )
    g4_bad = apply_g4_postprocessing(g3_bad, chunks)
    assert g4_bad["validation_status"] == "parse_failure"
    assert g4_bad["raw_g3_response_preserved"] == "{broken json"

    # 3. Empty evidence triggers clean insufficient schema & zero citations
    empty_res = generate_grounded_answer(
        "Where is dispatch?",
        [],
        llm_call_fn=lambda _: json.dumps(
            {
                "sufficient": False,
                "confidence_tier": "insufficient",
                "reason": "No code blocks retrieved.",
                "missing_information": ["Relevant implementation files"],
                "answer": "No relevant code blocks were retrieved.",
                "citations": [],
                "abstained": True,
            }
        ),
    )
    assert empty_res["g4"]["abstained"] is True
    assert empty_res["g4"]["citation_validation_report"]["validation_status"] == "valid_abstention_no_citations"


def test_api_resilience_retry_timeout_and_secret_sanitization():
    # Transient 429 & 503 recover cleanly on 3rd attempt
    attempts = {"n": 0}

    def _flaky_transient(prompt):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota")
        if attempts["n"] == 2:
            raise RuntimeError("503 UNAVAILABLE overloaded")
        return "Recovered answer"

    txt, meta = call_gemini_resilient(
        "prompt",
        max_retries=2,
        base_backoff_seconds=0.01,
        timeout_seconds=5.0,
        llm_call_fn=_flaky_transient,
    )
    assert txt == "Recovered answer"
    assert meta["success"] is True
    assert meta["retry_count"] == 2
    assert meta["api_calls_made"] == 3

    # Non-retryable 400 / API key error must NOT burn retries AND must sanitize AIza secret
    bad_attempts = {"n": 0}

    def _fatal_err(prompt):
        bad_attempts["n"] += 1
        raise RuntimeError("400 INVALID_ARGUMENT key=AIzaSySecretLeaked123456789")

    err_txt, err_meta = call_gemini_resilient(
        "prompt",
        max_retries=2,
        base_backoff_seconds=0.01,
        llm_call_fn=_fatal_err,
    )
    assert bad_attempts["n"] == 1
    assert err_meta["success"] is False
    assert "AIzaSy" not in err_txt
    assert "AIzaSy" not in err_meta["error"]
    assert "[REDACTED_API_KEY]" in err_meta["error"]

    # Retry exhaustion on TimeoutError returns failure cleanly
    to_attempts = {"n": 0}

    def _timeout_err(prompt):
        to_attempts["n"] += 1
        raise TimeoutError("504 DEADLINE_EXCEEDED")

    t_txt, t_meta = call_gemini_resilient(
        "prompt",
        max_retries=1,
        base_backoff_seconds=0.01,
        llm_call_fn=_timeout_err,
    )
    assert to_attempts["n"] == 2
    assert t_meta["success"] is False
    assert t_meta["timeout_occurred"] is True

    # Daily quota exhaustion (`GenerateRequestsPerDay*`) must NEVER be retried
    daily_attempts = {"n": 0}

    def _daily_quota_err(prompt):
        daily_attempts["n"] += 1
        raise RuntimeError(
            "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20"
        )

    d_txt, d_meta = call_gemini_resilient(
        "prompt",
        max_retries=2,
        base_backoff_seconds=0.01,
        llm_call_fn=_daily_quota_err,
    )
    assert daily_attempts["n"] == 1
    assert d_meta["success"] is False
    assert d_meta["retry_count"] == 0


def test_checkpoint_and_resume_behavior_with_simulated_partial_failure(tmp_path):
    chunks = _sample_chunks()
    frozen_mock = {
        "repository_trees": {"pallets/flask": "📂 flask/\n  📄 app.py"},
        "records": [
            {
                "question_id": "q1",
                "question": "Q1?",
                "repository": "pallets/flask",
                "commit_sha": "abc",
                "cohort": "cohort_a_hit",
                "answerable": True,
                "turn_number": 1,
                "previous_context": None,
                "retrieved_chunk_ids": ["v1_aaa111", "v1_bbb222"],
                "retrieved_chunks": chunks,
            },
            {
                "question_id": "q2",
                "question": "Q2?",
                "repository": "pallets/flask",
                "commit_sha": "abc",
                "cohort": "cohort_c_unanswerable",
                "answerable": False,
                "turn_number": 1,
                "previous_context": None,
                "retrieved_chunk_ids": ["v1_aaa111", "v1_bbb222"],
                "retrieved_chunks": chunks,
            },
        ],
    }
    ckpt_file = tmp_path / "g3_ckpt.json"

    calls_run1 = []

    def _run1_llm(prompt):
        calls_run1.append(prompt)
        if "Q1?" in prompt:
            return json.dumps(
                {
                    "sufficient": True,
                    "confidence_tier": "strong",
                    "reason": "Direct match.",
                    "missing_information": [],
                    "answer": "Found [EVIDENCE_1].",
                    "citations": ["EVIDENCE_1"],
                    "abstained": False,
                }
            )
        raise RuntimeError("503 UNAVAILABLE on q2")

    out1 = run_generation_stage(
        "G3",
        frozen_mock,
        checkpoint_path=ckpt_file,
        max_retries=0,
        pacing_seconds=0.0,
        llm_call_fn=_run1_llm,
    )
    assert out1[0]["api_success"] is True
    assert out1[1]["api_success"] is False
    assert os.path.exists(ckpt_file)

    # Resume: q1 MUST be skipped (0 calls), ONLY q2 retried!
    calls_run2 = []

    def _run2_llm(prompt):
        calls_run2.append(prompt)
        assert "Q2?" in prompt, "Completed q1 must NOT be re-invoked on resume!"
        return json.dumps(
            {
                "sufficient": False,
                "confidence_tier": "insufficient",
                "reason": "Out of scope.",
                "missing_information": ["Unimplemented feature"],
                "answer": "Cannot establish from retrieved blocks.",
                "citations": [],
                "abstained": True,
            }
        )

    out2 = run_generation_stage(
        "G3",
        frozen_mock,
        checkpoint_path=ckpt_file,
        max_retries=0,
        pacing_seconds=0.0,
        llm_call_fn=_run2_llm,
    )
    assert len(calls_run2) == 1
    assert out2[0]["api_success"] is True
    assert out2[1]["api_success"] is True

    # Offline G4 makes zero API calls and preserves G3 raw text
    g4_out, cit_sum = run_g4_offline_stage(frozen_mock, out2)
    assert all(r["api_calls_made"] == 0 for r in g4_out)
    assert g4_out[0]["valid_citation_ids"] if "valid_citation_ids" in g4_out[0] else g4_out[0]["citation_validation_report"]["valid_citation_ids"] == ["EVIDENCE_1"]
    assert g4_out[1]["abstained"] is True
    assert cit_sum["additional_live_api_calls"] == 0


def test_layer1_cohort_aggregation_and_judge_and_telemetry_compat(tmp_path):
    chunks = _sample_chunks()
    assert detect_unstructured_abstention("I could not establish this from the provided code.") is True
    assert detect_unstructured_abstention("The Flask class implements wsgi_app cleanly.") is False

    # Failed query MUST remain in cohort denominator!
    items = [
        {
            "question_id": "a1",
            "repository": "pallets/flask",
            "cohort": "cohort_a_hit",
            "answerable": True,
            "turn_number": 1,
            "question": "Q a1",
            "retrieved_chunks": chunks,
        },
        {
            "question_id": "b1",
            "repository": "pallets/flask",
            "cohort": "cohort_b_miss",
            "answerable": True,
            "turn_number": 1,
            "question": "Q b1",
            "retrieved_chunks": chunks,
        },
        {
            "question_id": "c1",
            "repository": "pallets/flask",
            "cohort": "cohort_c_unanswerable",
            "answerable": False,
            "turn_number": 1,
            "question": "Q c1",
            "retrieved_chunks": chunks,
        },
    ]
    g3_rows = [
        run_generation_experiment_item(
            "G3",
            "Q a1",
            chunks,
            llm_call_fn=lambda _: json.dumps(
                {
                    "sufficient": True,
                    "confidence_tier": "strong",
                    "reason": "Direct",
                    "missing_information": [],
                    "answer": "Uses [EVIDENCE_1].",
                    "citations": ["EVIDENCE_1"],
                    "abstained": False,
                }
            ),
        ),
        run_generation_experiment_item(
            "G3",
            "Q b1",
            chunks,
            max_retries=0,
            llm_call_fn=lambda _: (_ for _ in ()).throw(RuntimeError("400 INVALID_ARGUMENT")),
        ),
        run_generation_experiment_item(
            "G3",
            "Q c1",
            chunks,
            llm_call_fn=lambda _: json.dumps(
                {
                    "sufficient": False,
                    "confidence_tier": "insufficient",
                    "reason": "Missing",
                    "missing_information": ["Feature"],
                    "answer": "Not established.",
                    "citations": [],
                    "abstained": True,
                }
            ),
        ),
    ]
    l1_rows = [evaluate_layer1_record(it, g3) for it, g3 in zip(items, g3_rows)]
    agg = aggregate_layer1_metrics(l1_rows)
    assert agg["total_questions"] == 3
    assert agg["api_failure_count"] == 1
    assert agg["cohorts"]["cohort_b_miss"]["count"] == 1
    assert agg["cohorts"]["cohort_b_miss"]["failed_or_unparsed_count"] == 1
    assert agg["cohorts"]["cohort_b_miss"]["abstention_rate"] == 0.0  # denominator includes failed item!
    assert agg["cohorts"]["cohort_c_unanswerable"]["abstention_rate"] == 1.0

    # Mock blinded judge call
    verdict = judge_single_candidate(
        items[0],
        "Uses [EVIDENCE_1].",
        "G4",
        llm_call_fn=lambda _: json.dumps(
            {
                "correctness": "correct",
                "faithfulness": "supported",
                "citation_relevance": "relevant",
                "answer_usefulness": "high",
                "abstention_correctness": "appropriate_answer",
                "parametric_leakage": False,
                "unsupported_claims": [],
                "rationale": "Backed by EVIDENCE_1.",
            }
        ),
    )
    assert verdict["parse_success"] is True
    assert verdict["verdict"]["parametric_leakage"] is False

    # Telemetry backward compatibility + extended Day 6 fields + corruption safety
    q_legacy = telemetry.create_query_record(
        query_text="Legacy",
        repo_name="flask",
        top_k_requested=5,
        actual_results_returned=1,
        retrieval_latency_seconds=0.1,
        retrieved_chunks=chunks[:1],
    )
    assert q_legacy["generation"]["model"] == "gemini-3.5-flash"

    q_day6 = telemetry.create_query_record(
        query_text="Grounded",
        repo_name="flask",
        top_k_requested=5,
        actual_results_returned=1,
        retrieval_latency_seconds=0.1,
        retrieved_chunks=chunks[:1],
        generation_details={
            "experiment": "G4",
            "prompt_version": "g3_full_structured_v1",
            "evidence_sufficiency": True,
            "confidence_tier": "strong",
            "abstention_status": False,
            "valid_citation_ids": ["EVIDENCE_1"],
            "invalid_citation_ids": [],
            "validation_status": "valid",
            "raw_response": "secret AIzaSyLeakedTest999",
        },
    )
    assert q_day6["generation"]["validation_status"] == "valid"
    assert "AIzaSy" not in q_day6["generation"]["raw_response"]
    assert "[REDACTED_API_KEY]" in q_day6["generation"]["raw_response"]
    assert detect_unretrieved_file_mentions("Nothing missing.", chunks) == []

    # Telemetry file corruption raises RuntimeError cleanly without erasing data
    bad_file = tmp_path / "corrupted_telemetry.json"
    bad_file.write_text("{broken json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Failed to read existing baseline results"):
        telemetry.save_baseline_run(telemetry.create_baseline_run_record(repo_name="r"), filepath=str(bad_file))
    assert bad_file.read_text(encoding="utf-8") == "{broken json"


def test_day6_ui_wiring_and_day1_to_day5_preservation():
    app_src = open(os.path.join(PROJECT_ROOT, "app.py"), encoding="utf-8").read()
    assert "grounded_generation" in app_src
    assert "Verified Grounded Citations & Abstention" in app_src
    assert "generation.generate_grounded_answer" in app_src
    assert "_render_grounding_banners" in app_src
    assert "_render_retrieved_chunks_panel" in app_src
    assert "Cited Source Navigation" in app_src
    assert "Telemetry save skipped safely" in app_src

    # Verify original llm.py anchors unchanged cleanly
    assert callable(llm.build_prompt)
    assert callable(llm.ask_gemini)

    # Verify Day 1-5 artifacts & frozen evidence remain intact
    for rel in (
        "reports/day1/DAY1_FINAL_REPORT.md",
        "reports/day2/DAY2_TASK1_REPORT.md",
        "reports/day3/DAY3_CHUNKING_ANALYSIS.md",
        "reports/day4/DAY4_HYBRID_REPORT.md",
        "reports/day5/DAY5_RERANK_REWRITE_REPORT.md",
        "experiments/day6_generation/frozen_retrieval_evidence.json",
        "experiments/day6_generation/ablation_question_ids.json",
        "experiments/day6_generation/judge_question_ids.json",
    ):
        assert os.path.exists(os.path.join(PROJECT_ROOT, rel)), f"Missing required artifact: {rel}"
