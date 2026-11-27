"""Resilient Grounded Generation & G4 Post-Processing (Sections 7, 11, 12, 14).

Implements:
  - Single-call G3 structured generation using `google.genai` `GenerateContentConfig`
    (`response_mime_type="application/json"`, `response_schema`, `thinking_budget=0`,
    45s timeout guard + bounded transient backoff for 429/503/504/timeouts).
  - Safe JSON/schema parser that preserves raw output and flags parsing failures cleanly.
  - G4 offline deterministic citation validator & renderer (ZERO live Gemini calls).
  - Runner dispatch across G0, G1, G2, and G3/G4 preserving identical interfaces.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Literal, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel

from generation.citations import validate_and_render_citations
from generation.prompts import (
    G0_PROMPT_VERSION,
    G1_PROMPT_VERSION,
    G2_PROMPT_VERSION,
    G3_PROMPT_VERSION,
    build_g0_prompt,
    build_g1_prompt,
    build_g2_prompt,
    build_g3_prompt,
)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"

_RETRYABLE_ERROR_MARKERS = (
    "429",
    "503",
    "504",
    "rate",
    "quota",
    "resource_exhausted",
    "unavailable",
    "deadline_exceeded",
    "timed out",
    "timeout",
    "temporarily",
    "connection reset",
    "connection aborted",
)

_NON_RETRYABLE_MARKERS = (
    "api_key_invalid",
    "invalid_argument",
    "permission_denied",
    "not_found",
    "gemini_api_key is not set",
    "generaterequestsperday",
    "requests per day",
    "per day limit exceeded",
    "daily limit",
    "400 ",
    "401 ",
    "403 ",
    "404 ",
)


def is_daily_quota_error(error_msg: str) -> bool:
    """Detect non-recoverable daily quota exhaustion (`GenerateRequestsPerDay*`) so callers halt cleanly."""
    low = (error_msg or "").lower()
    return (
        "generaterequestsperday" in low
        or "requests per day" in low
        or "per day limit exceeded" in low
        or "daily limit" in low
    )


class GroundedGenerationSchema(BaseModel):
    sufficient: bool
    confidence_tier: Literal["strong", "partial", "insufficient"]
    reason: str
    missing_information: list[str]
    answer: str
    citations: list[str]
    abstained: bool


def sanitize_secret(text: str) -> str:
    return re.sub(r"AIza[0-9A-Za-z-_]+", "[REDACTED_API_KEY]", str(text or ""))


def is_retryable_gemini_error(error_msg: str) -> bool:
    low = (error_msg or "").lower()
    if any(m in low for m in _NON_RETRYABLE_MARKERS):
        return False
    return any(m in low for m in _RETRYABLE_ERROR_MARKERS)


def extract_json_object(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a JSON object cleanly, stripping optional markdown code fences."""
    txt = (raw_text or "").strip()
    if not txt:
        return None, "Empty response text"

    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s*```$", "", txt).strip()

    try:
        parsed = json.loads(txt)
        if isinstance(parsed, dict):
            return parsed, None
        return None, f"Expected JSON object dict, got {type(parsed).__name__}"
    except Exception:
        # Fallback: locate outermost `{ ... }` span cleanly
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(txt[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed, None
            except Exception as exc:
                return None, f"JSON decode error: {exc}"
        return None, "Could not parse JSON object from response"


def parse_structured_g3_response(raw_text: str) -> dict[str, Any]:
    """Strictly validate all 7 required G3 JSON schema fields without fabricating missing fields."""
    obj, json_err = extract_json_object(raw_text)
    if obj is None:
        return {
            "parse_success": False,
            "parse_error": json_err,
            "parsed": None,
            "safe_fallback_answer": (
                "⚠️ Structured grounded response parsing failed; raw model output was preserved safely "
                "without fabricating structured sufficiency metadata."
            ),
        }

    required_keys = (
        "sufficient",
        "confidence_tier",
        "reason",
        "missing_information",
        "answer",
        "citations",
        "abstained",
    )
    missing_keys = [k for k in required_keys if k not in obj]
    if missing_keys:
        return {
            "parse_success": False,
            "parse_error": f"Missing required schema keys: {missing_keys}",
            "parsed": None,
            "safe_fallback_answer": (
                "⚠️ Structured grounded response missed required fields; raw model output was preserved."
            ),
        }

    tier = str(obj.get("confidence_tier", "")).strip().lower()
    if tier not in ("strong", "partial", "insufficient"):
        return {
            "parse_success": False,
            "parse_error": f"Invalid confidence_tier {obj.get('confidence_tier')!r}",
            "parsed": None,
            "safe_fallback_answer": (
                "⚠️ Structured grounded response had an invalid confidence_tier; raw output preserved."
            ),
        }

    if not isinstance(obj.get("sufficient"), bool) or not isinstance(obj.get("abstained"), bool):
        return {
            "parse_success": False,
            "parse_error": "Fields 'sufficient' and 'abstained' must be booleans",
            "parsed": None,
            "safe_fallback_answer": (
                "⚠️ Structured grounded response had invalid boolean flags; raw output preserved."
            ),
        }

    if not isinstance(obj.get("missing_information"), list) or not isinstance(obj.get("citations"), list):
        return {
            "parse_success": False,
            "parse_error": "Fields 'missing_information' and 'citations' must be lists",
            "parsed": None,
            "safe_fallback_answer": (
                "⚠️ Structured grounded response had invalid list fields; raw output preserved."
            ),
        }

    normalized = {
        "sufficient": bool(obj["sufficient"]),
        "confidence_tier": tier,
        "reason": str(obj.get("reason") or "").strip(),
        "missing_information": [str(x).strip() for x in obj["missing_information"] if str(x).strip()],
        "answer": str(obj.get("answer") or "").strip(),
        "citations": [str(x).strip() for x in obj["citations"] if str(x).strip()],
        "abstained": bool(obj["abstained"]),
    }
    return {
        "parse_success": True,
        "parse_error": None,
        "parsed": normalized,
        "safe_fallback_answer": None,
    }


def call_gemini_resilient(
    prompt: str,
    model: str = DEFAULT_GEMINI_MODEL,
    response_schema: Any = None,
    temperature: float | None = 0.0,
    thinking_budget: int | None = 0,
    timeout_seconds: float = 45.0,
    max_retries: int = 2,
    base_backoff_seconds: float = 2.0,
    llm_call_fn: Callable[[str], tuple[str, dict[str, Any]] | str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call Gemini with bounded HTTP + wall-clock timeouts and transient error retries.

    Preserves `AIza...` secret sanitization cleanly.
    """
    t_total0 = time.perf_counter()

    if llm_call_fn is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            err_msg = "⚠️ Error: GEMINI_API_KEY is not set. Please add it to your .env file."
            return err_msg, {
                "model": model,
                "success": False,
                "error": "GEMINI_API_KEY is not set",
                "retry_count": 0,
                "api_calls_made": 0,
                "timeout_occurred": False,
                "latency_seconds": 0.0,
                "prompt_length_chars": len(prompt),
                "response_length_chars": len(err_msg),
            }

    def _single_attempt() -> str:
        if llm_call_fn is not None:
            out = llm_call_fn(prompt)
            if isinstance(out, tuple) and len(out) == 2:
                raw_txt, meta = out
                if isinstance(meta, dict) and meta.get("success") is False:
                    raise RuntimeError(str(meta.get("error") or raw_txt))
                return str(raw_txt or "")
            s = str(out or "")
            if s.startswith("⚠️ Error"):
                raise RuntimeError(s)
            return s

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        cfg_kwargs: dict[str, Any] = {
            "http_options": types.HttpOptions(timeout=max(1000, int(timeout_seconds * 1000)))
        }
        if temperature is not None:
            cfg_kwargs["temperature"] = temperature
        if thinking_budget is not None and "lite" not in model.lower():
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
        if response_schema is not None:
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_schema"] = response_schema

        cfg = types.GenerateContentConfig(**cfg_kwargs)
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        return resp.text or ""

    def _invoke_with_wall_timeout() -> str:
        if timeout_seconds <= 0:
            return _single_attempt()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_single_attempt)
        try:
            txt = future.result(timeout=timeout_seconds + 3.0)
            executor.shutdown(wait=False)
            return txt
        except FuturesTimeoutError as exc:
            executor.shutdown(wait=False)
            raise TimeoutError(f"Gemini call timed out after {timeout_seconds:.1f}s") from exc

    api_calls = 0
    retry_count = 0
    last_error = ""
    timeout_hit = False

    for attempt in range(max_retries + 1):
        api_calls += 1
        try:
            raw_text = _invoke_with_wall_timeout()
            total_lat = round(time.perf_counter() - t_total0, 4)
            return raw_text, {
                "model": model,
                "success": True,
                "error": None,
                "retry_count": retry_count,
                "api_calls_made": api_calls,
                "timeout_occurred": False,
                "latency_seconds": total_lat,
                "prompt_length_chars": len(prompt),
                "response_length_chars": len(raw_text),
            }
        except Exception as exc:
            last_error = sanitize_secret(str(exc))
            if "timed out" in last_error.lower() or isinstance(exc, TimeoutError):
                timeout_hit = True
            if attempt < max_retries and is_retryable_gemini_error(last_error):
                retry_count += 1
                time.sleep(base_backoff_seconds * (2**attempt))
                continue
            break

    total_lat = round(time.perf_counter() - t_total0, 4)
    err_msg = f"⚠️ Error calling Gemini API: {last_error}"
    return err_msg, {
        "model": model,
        "success": False,
        "error": last_error,
        "retry_count": retry_count,
        "api_calls_made": api_calls,
        "timeout_occurred": timeout_hit,
        "latency_seconds": total_lat,
        "prompt_length_chars": len(prompt),
        "response_length_chars": len(err_msg),
    }


def apply_g4_postprocessing(
    g3_record: dict[str, Any],
    retrieved_chunks: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """G4: Pure deterministic offline citation validation & rendering on saved G3 output.

    Zero live Gemini calls. Preserves `raw_response` and `parsed_response` intact.
    """
    t0 = time.perf_counter()
    raw_response = str(g3_record.get("raw_response", ""))
    parsed = g3_record.get("parsed_response")
    api_success = bool(g3_record.get("api_success", False))

    if not api_success:
        val_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        return {
            "experiment": "G4",
            "prompt_version": g3_record.get("prompt_version", G3_PROMPT_VERSION),
            "postprocessing_latency_ms": val_ms,
            "generation_success": False,
            "parse_success": False,
            "validation_status": "upstream_generation_failed",
            "sufficient": None,
            "confidence_tier": None,
            "abstained": None,
            "reason": "",
            "missing_information": [],
            "rendered_answer": raw_response,
            "raw_g3_response_preserved": raw_response,
            "parsed_g3_preserved": parsed,
            "citation_validation_report": {
                "all_cited_ids_raw": [],
                "valid_citation_ids": [],
                "invalid_citation_ids": [],
                "duplicate_citation_ids": [],
                "metadata_warnings": [],
                "resolved_sources": [],
                "has_valid_citations": False,
                "has_invalid_citations": False,
                "citation_validity_rate": None,
                "unretrieved_file_mentions": [],
                "validation_status": "upstream_generation_failed",
            },
        }

    if not g3_record.get("parse_success") or not isinstance(parsed, dict):
        # Preserve raw output safely; flag as parse failure without inventing answers
        cit_report = validate_and_render_citations(
            raw_answer="",
            declared_citations=[],
            retrieved_chunks=retrieved_chunks,
            abstained=True,
        )
        val_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        return {
            "experiment": "G4",
            "prompt_version": g3_record.get("prompt_version", G3_PROMPT_VERSION),
            "postprocessing_latency_ms": val_ms,
            "generation_success": True,
            "parse_success": False,
            "parse_error": g3_record.get("parse_error"),
            "validation_status": "parse_failure",
            "sufficient": False,
            "confidence_tier": "insufficient",
            "abstained": True,
            "reason": "Structured JSON parsing failed",
            "missing_information": [],
            "rendered_answer": g3_record.get("safe_fallback_answer")
            or "⚠️ Unable to parse structured grounded response; raw response preserved.",
            "raw_g3_response_preserved": raw_response,
            "parsed_g3_preserved": parsed,
            "citation_validation_report": {
                **cit_report,
                "validation_status": "parse_failure",
            },
        }

    cit = validate_and_render_citations(
        raw_answer=parsed.get("answer", ""),
        declared_citations=parsed.get("citations", []),
        retrieved_chunks=retrieved_chunks,
        abstained=bool(parsed.get("abstained", False)),
    )
    val_ms = round((time.perf_counter() - t0) * 1000.0, 4)

    cit_summary = {
        "all_cited_ids_raw": cit["all_cited_ids_raw"],
        "declared_citation_ids": cit["declared_citation_ids"],
        "inline_citation_ids": cit["inline_citation_ids"],
        "valid_citation_ids": cit["valid_citation_ids"],
        "invalid_citation_ids": cit["invalid_citation_ids"],
        "duplicate_citation_ids": cit["duplicate_citation_ids"],
        "metadata_warnings": cit["metadata_warnings"],
        "resolved_sources": cit["resolved_sources"],
        "has_valid_citations": cit["has_valid_citations"],
        "has_invalid_citations": cit["has_invalid_citations"],
        "citation_validity_rate": cit["citation_validity_rate"],
        "unretrieved_file_mentions": cit["unretrieved_file_mentions"],
        "validation_status": cit["validation_status"],
    }

    return {
        "experiment": "G4",
        "prompt_version": g3_record.get("prompt_version", G3_PROMPT_VERSION),
        "postprocessing_latency_ms": val_ms,
        "generation_success": True,
        "parse_success": True,
        "parse_error": None,
        "validation_status": cit["validation_status"],
        "sufficient": bool(parsed["sufficient"]),
        "confidence_tier": parsed["confidence_tier"],
        "abstained": bool(parsed["abstained"]),
        "reason": parsed["reason"],
        "missing_information": parsed["missing_information"],
        "rendered_body": cit["rendered_body"],
        "rendered_answer": cit["rendered_answer"],
        "raw_g3_response_preserved": raw_response,
        "parsed_g3_preserved": parsed,
        "citation_validation_report": cit_summary,
    }


def run_generation_experiment_item(
    experiment_id: Literal["G0", "G1", "G2", "G3"],
    question: str,
    retrieved_chunks: Sequence[dict[str, Any]],
    directory_tree: str = "",
    chat_history: Sequence[dict[str, Any]] | None = None,
    previous_context: str | None = None,
    repository_name: str = "",
    model: str = DEFAULT_GEMINI_MODEL,
    timeout_seconds: float = 45.0,
    max_retries: int = 2,
    base_backoff_seconds: float = 2.0,
    llm_call_fn: Callable[[str], tuple[str, dict[str, Any]] | str] | None = None,
) -> dict[str, Any]:
    """Execute one generation experiment item (`G0`, `G1`, `G2`, or `G3`) cleanly with full telemetry."""
    if experiment_id == "G0":
        prompt_version = G0_PROMPT_VERSION
        prompt = build_g0_prompt(
            question=question,
            chunks=retrieved_chunks,
            directory_tree=directory_tree,
            chat_history=chat_history,
            previous_context=previous_context,
        )
        schema = None
        temp = None
        think_budget = None
    elif experiment_id == "G1":
        prompt_version = G1_PROMPT_VERSION
        prompt = build_g1_prompt(
            question=question,
            chunks=retrieved_chunks,
            directory_tree=directory_tree,
            chat_history=chat_history,
            previous_context=previous_context,
            repository_name=repository_name,
        )
        schema = None
        temp = 0.0
        think_budget = 0
    elif experiment_id == "G2":
        prompt_version = G2_PROMPT_VERSION
        prompt = build_g2_prompt(
            question=question,
            chunks=retrieved_chunks,
            directory_tree=directory_tree,
            chat_history=chat_history,
            previous_context=previous_context,
            repository_name=repository_name,
        )
        schema = None
        temp = 0.0
        think_budget = 0
    elif experiment_id == "G3":
        prompt_version = G3_PROMPT_VERSION
        prompt = build_g3_prompt(
            question=question,
            chunks=retrieved_chunks,
            directory_tree=directory_tree,
            chat_history=chat_history,
            previous_context=previous_context,
            repository_name=repository_name,
        )
        schema = GroundedGenerationSchema
        temp = 0.0
        think_budget = 0
    else:
        raise ValueError(f"Unsupported experiment_id {experiment_id!r}")

    if model == DEFAULT_GEMINI_MODEL and experiment_id == "G0" and llm_call_fn is None:
        # Preserve V0 ask_gemini path inside resilient timeout/retry wrapper for baseline control parity
        import llm

        def _g0_call(p: str) -> tuple[str, dict[str, Any]]:
            return llm.ask_gemini(p, return_metadata=True)

        raw_text, call_meta = call_gemini_resilient(
            prompt=prompt,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            base_backoff_seconds=base_backoff_seconds,
            llm_call_fn=_g0_call,
        )
    else:
        raw_text, call_meta = call_gemini_resilient(
            prompt=prompt,
            model=model,
            response_schema=schema,
            temperature=temp,
            thinking_budget=think_budget,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            base_backoff_seconds=base_backoff_seconds,
            llm_call_fn=llm_call_fn,
        )

    out: dict[str, Any] = {
        "experiment": experiment_id,
        "prompt_version": prompt_version,
        "model": call_meta["model"],
        "api_success": bool(call_meta["success"]),
        "error": call_meta["error"],
        "retry_count": call_meta["retry_count"],
        "api_calls_made": call_meta["api_calls_made"],
        "timeout_occurred": call_meta["timeout_occurred"],
        "generation_latency_seconds": call_meta["latency_seconds"],
        "prompt_length_chars": call_meta["prompt_length_chars"],
        "raw_response": raw_text,
    }

    if experiment_id == "G3":
        if call_meta["success"]:
            parsed_info = parse_structured_g3_response(raw_text)
        else:
            parsed_info = {
                "parse_success": False,
                "parse_error": call_meta["error"],
                "parsed": None,
                "safe_fallback_answer": raw_text,
            }
        out["parse_success"] = parsed_info["parse_success"]
        out["parse_error"] = parsed_info["parse_error"]
        out["parsed_response"] = parsed_info["parsed"]
        out["safe_fallback_answer"] = parsed_info["safe_fallback_answer"]
    else:
        out["parsed_response"] = None
        if experiment_id == "G2" and call_meta["success"]:
            out["citation_check"] = validate_and_render_citations(
                raw_answer=raw_text,
                declared_citations=[],
                retrieved_chunks=retrieved_chunks,
                abstained=False,
            )
    return out


def generate_grounded_answer(
    question: str,
    retrieved_chunks: Sequence[dict[str, Any]],
    directory_tree: str = "",
    chat_history: Sequence[dict[str, Any]] | None = None,
    repository_name: str = "",
    model: str = DEFAULT_GEMINI_MODEL,
    timeout_seconds: float = 45.0,
    max_retries: int = 2,
    llm_call_fn: Callable[[str], tuple[str, dict[str, Any]] | str] | None = None,
) -> dict[str, Any]:
    """Production single-call Day 6 pipeline: executes G3 + offline G4 validation/rendering."""
    g3_res = run_generation_experiment_item(
        experiment_id="G3",
        question=question,
        retrieved_chunks=retrieved_chunks,
        directory_tree=directory_tree,
        chat_history=chat_history,
        repository_name=repository_name,
        model=model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        llm_call_fn=llm_call_fn,
    )
    g4_res = apply_g4_postprocessing(g3_res, retrieved_chunks)
    return {
        "g3": g3_res,
        "g4": g4_res,
        "rendered_answer": g4_res["rendered_answer"],
        "api_success": g3_res["api_success"],
        "error": g3_res["error"],
        "generation_latency_seconds": g3_res["generation_latency_seconds"],
        "postprocessing_latency_ms": g4_res["postprocessing_latency_ms"],
        "retry_count": g3_res["retry_count"],
        "prompt_version": g3_res["prompt_version"],
        "model": g3_res["model"],
        "prompt_length_chars": g3_res["prompt_length_chars"],
    }
