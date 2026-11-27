"""Day 6 Evaluation Package: Layer 1 Deterministic Checks & Layer 2 Blinded LLM Judge (Sections 16, 17, 19, 20)."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel

from generation.citations import (
    detect_unretrieved_file_mentions,
    detect_unretrieved_line_mentions,
    validate_and_render_citations,
)
from generation.grounded_generator import (
    DEFAULT_GEMINI_MODEL,
    call_gemini_resilient,
    extract_json_object,
)
from generation.prompts import JUDGE_PROMPT_VERSION, build_blinded_judge_prompt, format_evidence_blocks

_UNSTRUCTURED_ABSTENTION_PATTERNS = (
    re.compile(r"\bcould\s*n[o']?t\s+establish\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+(?:establish|determine|verify|confirm|answer|find)\b", re.IGNORECASE),
    re.compile(r"\bcan\s*not\s+(?:establish|determine|verify|confirm)\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+not\s+(?:contain|provide|show|include|establish|appear)\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+(?:contain|provide|show|include|establish)\b", re.IGNORECASE),
    re.compile(r"\binsufficient\s+(?:evidence|context|information)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:enough|sufficient)\s+(?:information|context|evidence)\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:evidence|implementation|definition)\b.*?\b(?:retrieved|provided)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:shown|present|found|included)\s+in\s+the\s+(?:provided|retrieved)\b", re.IGNORECASE),
    re.compile(r"\bunable\s+to\s+(?:determine|establish|verify|confirm)\b", re.IGNORECASE),
)


class JudgeVerdictSchema(BaseModel):
    correctness: Literal["correct", "partially_correct", "incorrect"]
    faithfulness: Literal["supported", "partially_supported", "unsupported", "contradicted"]
    citation_relevance: Literal[
        "relevant", "partially_relevant", "irrelevant", "no_citations", "not_applicable"
    ]
    answer_usefulness: Literal["high", "medium", "low"]
    abstention_correctness: Literal[
        "appropriate_answer", "appropriate_abstention", "false_refusal", "missed_abstention"
    ]
    parametric_leakage: bool
    unsupported_claims: list[str]
    rationale: str


def detect_unstructured_abstention(text: str) -> bool:
    """Heuristic phrase detector for free-text experiments (`G0`/`G1`/`G2`) lacking an `abstained` flag."""
    if not isinstance(text, str) or not text.strip():
        return False
    # Inspect full text conservatively: e.g. answers starting with or stating inability to establish
    return any(p.search(text) for p in _UNSTRUCTURED_ABSTENTION_PATTERNS)


def _compute_percentiles(vals: Sequence[float]) -> dict[str, float | None]:
    clean = sorted(float(v) for v in vals if v is not None)
    n = len(clean)
    if n == 0:
        return {"count": 0, "mean": None, "median_p50": None, "p95": None, "max": None}
    mean = sum(clean) / n
    if n % 2 == 1:
        p50 = clean[n // 2]
    else:
        p50 = 0.5 * (clean[n // 2 - 1] + clean[n // 2])
    idx95 = max(0, min(n - 1, math.ceil(0.95 * n) - 1))
    return {
        "count": n,
        "mean": round(mean, 4),
        "median_p50": round(p50, 4),
        "p95": round(clean[idx95], 4),
        "max": round(clean[-1], 4),
    }


def evaluate_layer1_record(
    frozen_item: dict[str, Any],
    exp_record: dict[str, Any],
    g4_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute Layer 1 deterministic checks cleanly with ZERO Gemini calls."""
    exp_id = exp_record.get("experiment", "")
    retrieved = frozen_item.get("retrieved_chunks", [])
    cohort = frozen_item["cohort"]
    api_success = bool(exp_record.get("api_success", False))

    if exp_id in ("G3", "G4"):
        raw_text = str(exp_record.get("raw_response", ""))
        parse_ok = bool(exp_record.get("parse_success", False))
        parsed = exp_record.get("parsed_response") if isinstance(exp_record.get("parsed_response"), dict) else {}
        g4_data = g4_record or {}
        if parse_ok and api_success:
            abstained = bool(parsed.get("abstained", False))
            sufficient = bool(parsed.get("sufficient", False))
            tier = parsed.get("confidence_tier")
            answer_text = str(parsed.get("answer", ""))
            cit_rep = g4_data.get("citation_validation_report") or validate_and_render_citations(
                answer_text, parsed.get("citations", []), retrieved, abstained=abstained
            )
        else:
            abstained = None
            sufficient = None
            tier = None
            answer_text = raw_text
            cit_rep = {
                "all_cited_ids_raw": [],
                "valid_citation_ids": [],
                "invalid_citation_ids": [],
                "duplicate_citation_ids": [],
                "metadata_warnings": [],
                "has_valid_citations": False,
                "has_invalid_citations": False,
                "citation_validity_rate": None,
                "unretrieved_file_mentions": [],
                "validation_status": "parse_failure" if api_success else "api_failure",
            }
    else:
        raw_text = str(exp_record.get("raw_response", ""))
        parse_ok = api_success
        abstained = detect_unstructured_abstention(raw_text) if api_success else None
        sufficient = (not abstained) if abstained is not None else None
        tier = None
        answer_text = raw_text
        if exp_id == "G2" and api_success:
            cit_rep = exp_record.get("citation_check") or validate_and_render_citations(
                answer_text, [], retrieved, abstained=bool(abstained)
            )
        else:
            # G0 / G1: no `[EVIDENCE_i]` citations expected; only check unretrieved file/line fabrication heuristics
            cit_rep = {
                "all_cited_ids_raw": [],
                "valid_citation_ids": [],
                "invalid_citation_ids": [],
                "duplicate_citation_ids": [],
                "metadata_warnings": [],
                "has_valid_citations": False,
                "has_invalid_citations": False,
                "citation_validity_rate": None,
                "unretrieved_file_mentions": detect_unretrieved_file_mentions(raw_text, retrieved) if api_success else [],
                "validation_status": "unstructured_text" if api_success else "api_failure",
            }

    unretrieved_files = (
        cit_rep.get("unretrieved_file_mentions")
        if api_success
        else []
    )
    unretrieved_lines = (
        detect_unretrieved_line_mentions(answer_text, retrieved) if (api_success and parse_ok) else []
    )

    return {
        "question_id": frozen_item["question_id"],
        "repository": frozen_item["repository"],
        "cohort": cohort,
        "answerable": frozen_item["answerable"],
        "turn_number": frozen_item["turn_number"],
        "experiment": "G4" if g4_record is not None else exp_id,
        "api_success": api_success,
        "parse_success": parse_ok,
        "abstained": abstained,
        "abstention_detection_mode": "structured_field" if exp_id in ("G3", "G4") else "heuristic_text_match",
        "sufficient": sufficient,
        "confidence_tier": tier,
        "generation_latency_seconds": exp_record.get("generation_latency_seconds"),
        "postprocessing_latency_ms": (g4_record or {}).get("postprocessing_latency_ms", 0.0),
        "retry_count": exp_record.get("retry_count", 0),
        "citation_count_valid": len(cit_rep.get("valid_citation_ids", [])),
        "citation_count_invalid": len(cit_rep.get("invalid_citation_ids", [])),
        "duplicate_citation_count": len(cit_rep.get("duplicate_citation_ids", [])),
        "has_valid_citations": bool(cit_rep.get("has_valid_citations", False)),
        "has_invalid_citations": bool(cit_rep.get("has_invalid_citations", False)),
        "citation_validity_rate": cit_rep.get("citation_validity_rate"),
        "unretrieved_file_mentions": unretrieved_files or [],
        "unretrieved_line_mentions": unretrieved_lines,
        "validation_status": cit_rep.get("validation_status"),
    }


def aggregate_layer1_metrics(layer1_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Layer 1 deterministic checks strictly preserving failed items in denominators."""
    n = len(layer1_rows)
    if n == 0:
        return {"total_questions": 0}

    completed = [r for r in layer1_rows if r["api_success"] and r["parse_success"]]
    api_ok = [r for r in layer1_rows if r["api_success"]]

    def _cohort_stats(coh: str) -> dict[str, Any]:
        sub = [r for r in layer1_rows if r["cohort"] == coh]
        cn = len(sub)
        if cn == 0:
            return {"count": 0}
        abstained_cnt = sum(1 for r in sub if r["abstained"] is True)
        answered_cnt = sum(1 for r in sub if r["abstained"] is False)
        failed_or_unparsed = cn - abstained_cnt - answered_cnt
        unret_file_when_ans = sum(
            1 for r in sub if r["abstained"] is False and len(r["unretrieved_file_mentions"]) > 0
        )
        return {
            "count": cn,
            "completed_valid_count": abstained_cnt + answered_cnt,
            "failed_or_unparsed_count": failed_or_unparsed,
            "abstained_count": abstained_cnt,
            "substantive_answered_count": answered_cnt,
            "abstention_rate": round(abstained_cnt / cn, 4),
            "substantive_answer_rate": round(answered_cnt / cn, 4),
            "answered_with_unretrieved_file_mention_count": unret_file_when_ans,
            "answered_with_unretrieved_file_mention_rate": round(unret_file_when_ans / cn, 4),
        }

    coh_a = _cohort_stats("cohort_a_hit")
    coh_b = _cohort_stats("cohort_b_miss")
    coh_c = _cohort_stats("cohort_c_unanswerable")

    total_valid_cits = sum(r["citation_count_valid"] for r in completed)
    total_invalid_cits = sum(r["citation_count_invalid"] for r in completed)
    total_cits = total_valid_cits + total_invalid_cits
    non_abstained_completed = [r for r in completed if r["abstained"] is False]
    cited_among_answered = sum(1 for r in non_abstained_completed if r["has_valid_citations"])

    lat_s = [r["generation_latency_seconds"] for r in api_ok if r["generation_latency_seconds"] is not None]
    post_ms = [r["postprocessing_latency_ms"] for r in layer1_rows if r["postprocessing_latency_ms"] is not None]

    return {
        "total_questions": n,
        "api_success_count": len(api_ok),
        "api_failure_count": n - len(api_ok),
        "parse_success_count": len(completed),
        "parse_failure_count": len(api_ok) - len(completed),
        "json_schema_validity_rate": round(len(completed) / len(api_ok), 4) if api_ok else None,
        "cohorts": {
            "cohort_a_hit": coh_a,
            "cohort_b_miss": coh_b,
            "cohort_c_unanswerable": coh_c,
        },
        "summary_rates": {
            "cohort_a_false_refusal_rate": coh_a.get("abstention_rate"),
            "cohort_a_substantive_answer_rate": coh_a.get("substantive_answer_rate"),
            "cohort_b_abstention_rate": coh_b.get("abstention_rate"),
            "cohort_b_non_abstained_potential_leakage_rate": coh_b.get("substantive_answer_rate"),
            "cohort_c_true_abstention_rate": coh_c.get("abstention_rate"),
            "cohort_c_unsupported_answer_rate": coh_c.get("substantive_answer_rate"),
        },
        "citations": {
            "total_valid_citations": total_valid_cits,
            "total_invalid_citations": total_invalid_cits,
            "citation_validity_rate_token_level": round(total_valid_cits / total_cits, 4)
            if total_cits > 0
            else None,
            "queries_with_invalid_citations": sum(1 for r in completed if r["has_invalid_citations"]),
            "queries_with_duplicate_citations": sum(1 for r in completed if r["duplicate_citation_count"] > 0),
            "answered_with_valid_citation_presence": round(
                cited_among_answered / len(non_abstained_completed), 4
            )
            if non_abstained_completed
            else None,
            "mean_valid_citations_per_answered_query": round(
                total_valid_cits / len(non_abstained_completed), 4
            )
            if non_abstained_completed
            else None,
            "answered_with_unretrieved_file_mentions": sum(
                1 for r in non_abstained_completed if len(r["unretrieved_file_mentions"]) > 0
            ),
            "answered_with_unretrieved_line_mentions": sum(
                1 for r in non_abstained_completed if len(r["unretrieved_line_mentions"]) > 0
            ),
        },
        "latency_seconds": _compute_percentiles(lat_s),
        "postprocessing_latency_ms": _compute_percentiles(post_ms),
    }


def blind_evaluation_id(question_id: str, variant_key: str) -> str:
    """Deterministic anonymized key preventing experiment-label leakage into judge prompts."""
    digest = hashlib.sha256(f"day6_judge|{question_id}|{variant_key}".encode("utf-8")).hexdigest()[:10]
    return f"cand_{digest}"


def judge_single_candidate(
    frozen_item: dict[str, Any],
    candidate_answer: str,
    variant_label: str,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout_seconds: float = 45.0,
    max_retries: int = 2,
    base_backoff_seconds: float = 2.0,
    llm_call_fn: Callable[[str], tuple[str, dict[str, Any]] | str] | None = None,
) -> dict[str, Any]:
    """Run blinded structured LLM judge verdict for a single `(question, answer)` pair."""
    evidence_text = format_evidence_blocks(frozen_item["retrieved_chunks"])
    prompt = build_blinded_judge_prompt(
        question=frozen_item["question"],
        evidence_blocks_text=evidence_text,
        candidate_answer=candidate_answer,
        previous_context=frozen_item.get("previous_context"),
        repository_name=frozen_item["repository"],
    )
    raw_text, call_meta = call_gemini_resilient(
        prompt=prompt,
        model=model,
        response_schema=JudgeVerdictSchema,
        temperature=0.0,
        thinking_budget=0,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        base_backoff_seconds=base_backoff_seconds,
        llm_call_fn=llm_call_fn,
    )
    parsed_verdict = None
    parse_error = None
    if call_meta["success"]:
        obj, err = extract_json_object(raw_text)
        if obj is not None:
            try:
                parsed_verdict = JudgeVerdictSchema(**obj).model_dump()
            except Exception as exc:
                parse_error = f"Schema validation error: {exc}"
        else:
            parse_error = err

    return {
        "question_id": frozen_item["question_id"],
        "repository": frozen_item["repository"],
        "cohort": frozen_item["cohort"],
        "variant": variant_label,
        "blind_id": blind_evaluation_id(frozen_item["question_id"], variant_label),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_model": model,
        "api_success": bool(call_meta["success"]),
        "parse_success": parsed_verdict is not None,
        "error": call_meta["error"] or parse_error,
        "retry_count": call_meta["retry_count"],
        "api_calls_made": call_meta["api_calls_made"],
        "judge_latency_seconds": call_meta["latency_seconds"],
        "raw_response": raw_text,
        "verdict": parsed_verdict,
    }


def aggregate_judge_results(verdicts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Layer 2 blinded LLM judge verdicts per variant (clearly labeled approximate)."""
    n = len(verdicts)
    valid = [v for v in verdicts if v.get("api_success") and isinstance(v.get("verdict"), dict)]
    if n == 0:
        return {"total": 0}

    vn = len(valid)
    def _share(pred: Callable[[dict[str, Any]], bool], subset: list[dict[str, Any]] | None = None) -> float | None:
        pool = valid if subset is None else [v for v in subset if isinstance(v.get("verdict"), dict)]
        if not pool:
            return None
        return round(sum(1 for v in pool if pred(v["verdict"])) / len(pool), 4)

    coh_a = [v for v in valid if v["cohort"] == "cohort_a_hit"]
    coh_b = [v for v in valid if v["cohort"] == "cohort_b_miss"]
    coh_c = [v for v in valid if v["cohort"] == "cohort_c_unanswerable"]
    ans_all = coh_a + coh_b

    # Citation precision on items that cited something (exclude 'not_applicable' & 'no_citations')
    cited_items = [
        v for v in valid if v["verdict"]["citation_relevance"] not in ("not_applicable", "no_citations")
    ]
    non_abstaining = [
        v for v in valid if v["verdict"]["abstention_correctness"] in ("appropriate_answer", "missed_abstention")
    ]

    return {
        "note": "Approximate LLM-judge evaluation (same Gemini model family bias possible).",
        "total_items": n,
        "valid_verdicts": vn,
        "correctness_strict_rate": _share(lambda d: d["correctness"] == "correct"),
        "correctness_lenient_rate": _share(lambda d: d["correctness"] in ("correct", "partially_correct")),
        "faithfulness_supported_strict_rate": _share(lambda d: d["faithfulness"] == "supported"),
        "faithfulness_supported_or_partial_rate": _share(
            lambda d: d["faithfulness"] in ("supported", "partially_supported")
        ),
        "faithfulness_unsupported_or_contradicted_rate": _share(
            lambda d: d["faithfulness"] in ("unsupported", "contradicted")
        ),
        "citation_precision_relevant_rate": _share(
            lambda d: d["citation_relevance"] == "relevant", cited_items
        ),
        "citation_precision_lenient_rate": _share(
            lambda d: d["citation_relevance"] in ("relevant", "partially_relevant"), cited_items
        ),
        "citation_coverage_rate_among_answered": _share(
            lambda d: d["citation_relevance"] != "no_citations", non_abstaining
        ),
        "usefulness_high_or_med_rate": _share(lambda d: d["answer_usefulness"] in ("high", "medium")),
        "abstention_accuracy": _share(
            lambda d: d["abstention_correctness"] in ("appropriate_answer", "appropriate_abstention")
        ),
        "parametric_leakage_overall_rate": _share(lambda d: bool(d["parametric_leakage"])),
        "by_cohort": {
            "cohort_a_hit": {
                "count": len(coh_a),
                "correctness_strict": _share(lambda d: d["correctness"] == "correct", coh_a),
                "faithfulness_supported": _share(lambda d: d["faithfulness"] == "supported", coh_a),
                "false_refusal_rate": _share(lambda d: d["abstention_correctness"] == "false_refusal", coh_a),
                "substantive_answer_rate": _share(
                    lambda d: d["abstention_correctness"] != "false_refusal", coh_a
                ),
            },
            "cohort_b_miss": {
                "count": len(coh_b),
                "appropriate_abstention_rate": _share(
                    lambda d: d["abstention_correctness"] == "appropriate_abstention", coh_b
                ),
                "parametric_leakage_rate": _share(lambda d: bool(d["parametric_leakage"]), coh_b),
                "missed_abstention_rate": _share(
                    lambda d: d["abstention_correctness"] == "missed_abstention", coh_b
                ),
            },
            "cohort_c_unanswerable": {
                "count": len(coh_c),
                "true_abstention_rate": _share(
                    lambda d: d["abstention_correctness"] == "appropriate_abstention", coh_c
                ),
                "parametric_leakage_rate": _share(lambda d: bool(d["parametric_leakage"]), coh_c),
            },
            "answerable_combined": {
                "count": len(ans_all),
                "substantive_coverage": _share(
                    lambda d: d["abstention_correctness"] in ("appropriate_answer", "missed_abstention"),
                    ans_all,
                ),
            },
        },
    }
