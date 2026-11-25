"""Leakage-Free Conversational Query Rewriter (Day 5).

Strict data-leakage & architectural boundaries:
  - Accepts ONLY: original user query, previous conversation context
    (text or prior user/assistant turns), and optional repository name.
  - Strictly separated from evaluation datasets, reference rewrites,
    evidence line spans, chunk IDs, or benchmark labels.
  - Uses a deterministic gate so standalone queries pass through unchanged with
    zero Gemini API calls (`rewritten=False`, `rewrite_latency_ms=0.0`).
  - Uses the existing Gemini client in `llm.ask_gemini` rather than creating
    a competing SDK client, and falls back safely to the original query on errors.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

# Whole-word pronouns & definite/demonstrative context references indicating
# a follow-up query depends on earlier conversation turns.
_ANAPHORIC_TOKEN_RE = re.compile(
    r"\b(?:it|its|itself|they|them|their|theirs|this|these|those|such|"
    r"former|latter|aforementioned|above|previous)\b",
    re.IGNORECASE,
)
_DEMONSTRATIVE_THAT_RE = re.compile(
    r"(?:\b(?:does|did|is|was|will|can|should|would|to|into|during|for|from|with|by|in|inside|using|via)\s+that\b|"
    r"\bthat\s+(?:[A-Za-z_]+\s+){0,2}(?:method|function|class|struct|enum|helper|macro|syntax|"
    r"instance|object|handler|app|application|blueprint|router|request|"
    r"response|stream|view|style|trait|setting|settings|config|configuration|"
    r"rule|rules|route|routes|context|contexts|process|mechanism|approach|"
    r"feature|behavior|error|exception|parameter|argument|value|file|module)\b)",
    re.IGNORECASE,
)
_ELLIPTICAL_PREFIX_RE = re.compile(
    r"^\s*(?:and\s+|also\s+|what about\b|how about\b|why does\b|why is\b|"
    r"explain\s+(?:more|that|it|why|how)|elaborate\b|what else\b)",
    re.IGNORECASE,
)

_TRANSIENT_ERR_MARKERS = ("429", "503", "rate", "quota", "timeout", "temporarily", "unavailable")


def format_previous_context(
    previous_context: str | Sequence[dict[str, str]] | None,
    max_chars: int = 1200,
) -> str:
    """Format prior conversation turns or context string cleanly for the rewriter."""
    if previous_context is None:
        return ""
    if isinstance(previous_context, str):
        return previous_context.strip()[:max_chars]

    parts: list[str] = []
    # Keep at most last 4 messages (2 prior Q&A turns) for crisp disambiguation
    for msg in list(previous_context)[-4:]:
        if not isinstance(msg, dict):
            continue
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = str(msg.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content[:400]}")
    return "\n".join(parts)[:max_chars]


def needs_conversational_rewrite(
    query: str,
    previous_context: str | Sequence[dict[str, str]] | None = None,
) -> tuple[bool, str]:
    """Deterministic gate: rewrite ONLY when meaningful prior context exists AND query depends on it."""
    q = (query or "").strip()
    if not q:
        return False, "empty_query"

    ctx_str = format_previous_context(previous_context)
    if not ctx_str:
        return False, "no_previous_context"

    if _ANAPHORIC_TOKEN_RE.search(q):
        return True, "anaphoric_pronoun_or_demonstrative"
    if _DEMONSTRATIVE_THAT_RE.search(q):
        return True, "demonstrative_that_phrase"
    if _ELLIPTICAL_PREFIX_RE.search(q) or len(q.split()) <= 4:
        return True, "elliptical_followup"

    return False, "standalone_with_history"


def _build_rewrite_prompt(
    query: str,
    formatted_context: str,
    repository_name: str = "",
) -> str:
    repo_line = f"Repository: {repository_name}\n" if repository_name else ""
    return (
        "You are a precise code search query rewriter.\n"
        "Rewrite the follow-up user query into a single self-contained standalone search query "
        "by resolving ambiguous pronouns and references (such as 'it', 'that method', 'those settings') "
        "using ONLY the provided previous conversation context.\n\n"
        "Strict rules:\n"
        "- Preserve ALL function names, class names, file paths, variable names, API identifiers, "
        "error messages, programming language, and the original user intent.\n"
        "- Do NOT invent or speculate file paths, symbols, or technical facts not present in the inputs.\n"
        "- Output ONLY the rewritten search query on a single plain text line with no quotes, "
        "markdown formatting, or explanation.\n\n"
        f"{repo_line}"
        f"Previous Conversation Context:\n{formatted_context}\n\n"
        f"Follow-Up User Query: {query}\n\n"
        "Standalone Search Query:"
    )


def _sanitize_rewritten_output(raw_text: str) -> str:
    cleaned = (raw_text or "").strip()
    # Remove accidental markdown code fences or outer quotes
    cleaned = re.sub(r"^```(?:text)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if lines:
        cleaned = lines[0]
    for prefix in ("Standalone Search Query:", "Rewritten Query:", "Query:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
    cleaned = cleaned.strip().strip('"').strip("'").strip("`").strip()
    return cleaned


def _default_llm_caller(prompt: str) -> tuple[str, dict[str, Any]]:
    import llm

    return llm.ask_gemini(prompt, return_metadata=True)


def rewrite_conversational_query(
    query: str,
    previous_context: str | Sequence[dict[str, str]] | None = None,
    repository_name: str = "",
    llm_call_fn: Callable[[str], tuple[str, dict[str, Any]] | str] | None = None,
    max_retries: int = 1,
    retry_delay_seconds: float = 1.5,
    timeout_seconds: float | None = 30.0,
) -> dict[str, Any]:
    """Rewrite a context-dependent follow-up query via `llm.ask_gemini`, or pass through unchanged."""
    orig_query = (query or "").strip()
    should_rewrite, gate_reason = needs_conversational_rewrite(orig_query, previous_context)

    if not should_rewrite:
        return {
            "original_query": query,
            "retrieval_query": query,
            "rewritten": False,
            "gate_triggered": False,
            "gate_reason": gate_reason,
            "rewrite_failed": False,
            "error": None,
            "rewrite_latency_ms": 0.0,
            "gemini_calls": 0,
        }

    formatted_ctx = format_previous_context(previous_context)
    prompt = _build_rewrite_prompt(orig_query, formatted_ctx, repository_name=repository_name)
    caller = llm_call_fn if llm_call_fn is not None else _default_llm_caller

    t0 = time.perf_counter()
    gemini_calls = 0
    last_err: str | None = None
    candidate_query = ""

    def _invoke_with_timeout() -> Any:
        if timeout_seconds is None or timeout_seconds <= 0:
            return caller(prompt)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(caller, prompt)
        try:
            res = future.result(timeout=timeout_seconds)
            executor.shutdown(wait=False)
            return res
        except FuturesTimeoutError as exc:
            executor.shutdown(wait=False)
            raise TimeoutError(f"Gemini rewrite timed out after {timeout_seconds:.1f}s") from exc

    for attempt in range(max_retries + 1):
        gemini_calls += 1
        try:
            raw_res = _invoke_with_timeout()
            if isinstance(raw_res, tuple) and len(raw_res) == 2:
                raw_text, meta = raw_res
                success = bool(meta.get("success", True))
                err_str = meta.get("error")
            else:
                raw_text = str(raw_res or "")
                success = not raw_text.startswith("⚠️ Error")
                err_str = raw_text if not success else None

            if success:
                cleaned = _sanitize_rewritten_output(raw_text)
                if cleaned and len(cleaned) >= 5 and len(cleaned) <= max(400, len(orig_query) * 5):
                    candidate_query = cleaned
                    last_err = None
                    break
                last_err = f"Invalid rewritten query output: {raw_text!r}"
            else:
                last_err = str(err_str or "Unknown Gemini rewrite error")
        except Exception as exc:
            last_err = str(exc)

        is_transient = any(tok in (last_err or "").lower() for tok in _TRANSIENT_ERR_MARKERS)
        if attempt < max_retries and is_transient:
            time.sleep(retry_delay_seconds * (2**attempt))
            continue
        break

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    if last_err is None and candidate_query:
        return {
            "original_query": query,
            "retrieval_query": candidate_query,
            "rewritten": candidate_query != orig_query,
            "gate_triggered": True,
            "gate_reason": gate_reason,
            "rewrite_failed": False,
            "error": None,
            "rewrite_latency_ms": elapsed_ms,
            "gemini_calls": gemini_calls,
        }

    logger.warning("Conversational query rewrite failed (%s); preserving original query.", last_err)
    return {
        "original_query": query,
        "retrieval_query": query,
        "rewritten": False,
        "gate_triggered": True,
        "gate_reason": gate_reason,
        "rewrite_failed": True,
        "error": last_err,
        "rewrite_latency_ms": elapsed_ms,
        "gemini_calls": gemini_calls,
    }
