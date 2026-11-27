"""Versioned Grounded Prompts & Evidence Formatting for Day 6 (Sections 9, 10, 15).

Guarantees:
  - Preserves V0 `llm.build_prompt()` cleanly for G0 baseline comparisons.
  - Formats retrieved code chunks into explicit deterministic `[EVIDENCE_i]` blocks
    without duplicating embedded `code_aware_v1` leading metadata headers and
    without mutating stored chunk content.
  - Strictly isolates `directory_tree` and `chat_history` as non-evidence context.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import llm

G0_PROMPT_VERSION = "g0_baseline_v0"
G1_PROMPT_VERSION = "g1_grounded_only_v1"
G2_PROMPT_VERSION = "g2_grounded_citations_v1"
G3_PROMPT_VERSION = "g3_full_structured_v1"
JUDGE_PROMPT_VERSION = "judge_blinded_rubric_v1"

DIRECTORY_TREE_ISOLATION_NOTICE = (
    "The directory tree is provided only for high-level repository navigation. "
    "It is not implementation evidence. Do not infer file behavior, function behavior, "
    "APIs, or implementation details from filenames or directory structure. "
    "Do not cite a file merely because it appears in the directory tree. "
    "Implementation claims must be supported by retrieved evidence blocks."
)

CONVERSATION_HISTORY_ISOLATION_NOTICE = (
    "CONVERSATION HISTORY (UNVERIFIED REFERENCE ONLY — NOT REPOSITORY EVIDENCE):\n"
    "Use previous conversation turns strictly to disambiguate pronouns or references "
    "(such as 'it', 'that method', or 'those settings'). Previous Assistant responses may "
    "contain errors or ungrounded statements and are NOT repository evidence. Never treat "
    "any prior Assistant claim as factual proof; every repository claim must be grounded "
    "solely in the retrieved [EVIDENCE_i] code blocks below."
)

EVIDENCE_RULES: tuple[str, ...] = (
    "1. Retrieved repository code in [EVIDENCE_i] blocks is your ONLY factual source of truth.",
    "2. Do not invent functions, classes, methods, variables, filenames, APIs, configuration keys, defaults, or behaviors.",
    "3. Do not assume an implementation exists merely because you recognize a public repository (e.g., Flask, Express, fmt).",
    "4. Do not use pretrained memory about the repository as evidence.",
    "5. If the retrieved evidence does not establish a claim, explicitly state the limitation.",
    "6. Explicitly distinguish code behavior directly observed in retrieved blocks from any cautious inference.",
    "7. If retrieved evidence only supports part of the question, answer ONLY the supported portion and qualify what is missing.",
    "8. If retrieved evidence blocks conflict or appear ambiguous across files/parts, acknowledge the conflict openly.",
    "9. Do not make implementation claims about files whose code was not retrieved.",
    "10. Do not claim a feature or symbol does not exist anywhere in the repository merely because it was not retrieved.",
    "11. Missing retrieved evidence means 'not established by the retrieved code context', NOT 'absent from the repository'.",
    "12. Avoid unsupported causal explanations not backed by the retrieved code.",
    "13. Keep answers concise, clear, and technically accurate.",
    "14. Cite factual repository implementation claims directly from the retrieved evidence.",
)

_V1_FILE_LINE_RE = re.compile(r"^File:\s+.+?\s*\|\s*Language:\s+.+?\s*\|\s*Type:\s+.+$")
_V1_DETAIL_LINE_RE = re.compile(r"^(?:Class|Parent|Symbol|Part):\s+.+")


def strip_redundant_v1_header_for_prompt(content: str) -> str:
    """Return a prompt-display copy of chunk content without duplicate V1 header lines.

    Does NOT mutate caller data structures. Preserves `Signature:` lines and every
    line of source code/docstrings untouched. If `content` does not start with a
    `code_aware_v1` `File: ... | Language: ... | Type: ...` header, returns it as-is.
    """
    if not isinstance(content, str) or not content:
        return ""
    lines = content.splitlines()
    idx = 0
    if lines and _V1_FILE_LINE_RE.match(lines[0].strip()):
        idx = 1
        if idx < len(lines) and _V1_DETAIL_LINE_RE.match(lines[idx].strip()):
            idx += 1
    if idx == 0:
        return content
    remainder = "\n".join(lines[idx:]).lstrip("\n")
    return remainder if remainder else content


def normalize_chat_history(
    chat_history: Sequence[dict[str, Any]] | None = None,
    previous_context: str | None = None,
) -> list[dict[str, str]]:
    """Normalize conversation history safely, retaining at most the last 5 Q&A pairs (10 msgs)."""
    cleaned: list[dict[str, str]] = []
    if chat_history:
        for msg in list(chat_history)[-10:]:
            if not isinstance(msg, dict):
                continue
            role = "assistant" if msg.get("role") == "assistant" else "user"
            txt = str(msg.get("content", "")).strip()
            if txt:
                cleaned.append({"role": role, "content": txt})
    elif previous_context and str(previous_context).strip():
        cleaned.append({"role": "user", "content": str(previous_context).strip()})
    return cleaned


def _chunk_meta(chunk: dict[str, Any]) -> dict[str, Any]:
    meta = chunk.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _format_symbol_label(chunk: dict[str, Any], meta: dict[str, Any]) -> str:
    sym = str(meta.get("symbol_name") or chunk.get("symbol_name") or "").strip()
    cls = str(meta.get("class_name") or "").strip()
    if cls and sym and cls != sym and not sym.startswith(f"{cls}."):
        return f"{cls}.{sym}"
    if sym:
        return sym
    if cls:
        return cls
    return "(module / top-level)"


def _format_lines_label(chunk: dict[str, Any], meta: dict[str, Any]) -> str:
    s = meta.get("start_line", chunk.get("start_line"))
    e = meta.get("end_line", chunk.get("end_line"))
    if isinstance(s, int) and not isinstance(s, bool) and isinstance(e, int) and not isinstance(e, bool) and s > 0 and e >= s:
        return f"{s}–{e}"
    return "unknown"


def format_evidence_blocks(retrieved_chunks: Sequence[dict[str, Any]]) -> str:
    """Format retrieved chunks into deterministic `[EVIDENCE_i]` prompt blocks without duplicating headers."""
    if not retrieved_chunks:
        return "No code evidence blocks were retrieved for this question."

    blocks: list[str] = []
    for idx, ch in enumerate(retrieved_chunks, start=1):
        meta = _chunk_meta(ch)
        file_path = str(meta.get("file_path") or ch.get("file_path") or "unknown").strip()
        symbol_label = _format_symbol_label(ch, meta)
        lines_label = _format_lines_label(ch, meta)
        part_idx = meta.get("part_index")
        total_parts = meta.get("total_parts")
        part_suffix = (
            f" (Part {part_idx}/{total_parts})"
            if isinstance(part_idx, int) and isinstance(total_parts, int) and total_parts > 1
            else ""
        )
        raw_content = str(ch.get("content", ""))
        prompt_content = strip_redundant_v1_header_for_prompt(raw_content)

        blocks.append(
            "\n".join(
                [
                    f"[EVIDENCE_{idx}]",
                    f"File: {file_path}",
                    f"Symbol: {symbol_label}{part_suffix}",
                    f"Lines: {lines_label}",
                    "Content:",
                    prompt_content,
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_isolated_sections(
    directory_tree: str,
    chat_history: Sequence[dict[str, Any]] | None,
    previous_context: str | None,
) -> list[str]:
    parts: list[str] = []
    norm_hist = normalize_chat_history(chat_history=chat_history, previous_context=previous_context)
    if norm_hist:
        parts.append(CONVERSATION_HISTORY_ISOLATION_NOTICE)
        for item in norm_hist:
            label = "Prior User" if item["role"] == "user" else "Prior Assistant (UNVERIFIED)"
            parts.append(f"{label}: {item['content']}")
        parts.append("(End of unverified conversation history)\n")

    if directory_tree and directory_tree.strip():
        parts.append(
            "REPOSITORY DIRECTORY TREE (NAVIGATION ONLY — NOT IMPLEMENTATION EVIDENCE):\n"
            f"{DIRECTORY_TREE_ISOLATION_NOTICE}\n\n"
            f"{directory_tree.strip()}\n"
        )
    return parts


def build_g0_prompt(
    question: str,
    chunks: Sequence[dict[str, Any]],
    directory_tree: str = "",
    chat_history: Sequence[dict[str, Any]] | None = None,
    previous_context: str | None = None,
) -> str:
    """Build exact V0 baseline prompt via `llm.build_prompt()`."""
    norm_hist = normalize_chat_history(chat_history=chat_history, previous_context=previous_context)
    return llm.build_prompt(
        question=question,
        chunks=list(chunks),
        directory_tree=directory_tree or "",
        chat_history=norm_hist or None,
    )


def build_g1_prompt(
    question: str,
    chunks: Sequence[dict[str, Any]],
    directory_tree: str = "",
    chat_history: Sequence[dict[str, Any]] | None = None,
    previous_context: str | None = None,
    repository_name: str = "",
) -> str:
    """G1: Grounded generation instructions + isolated context + `[EVIDENCE_i]` blocks (no citation syntax or JSON schema)."""
    repo_line = f"Target Repository: {repository_name}\n" if repository_name else ""
    rules_text = "\n".join(EVIDENCE_RULES)
    evidence_text = format_evidence_blocks(chunks)
    sections = [
        (
            "You are a trustworthy codebase analysis assistant.\n"
            f"{repo_line}"
            "Answer ONLY from the retrieved code evidence blocks provided below.\n\n"
            f"STRICT EVIDENCE RULES:\n{rules_text}\n\n"
            "SUFFICIENCY BEHAVIOR:\n"
            "- If the retrieved blocks directly establish the answer, answer clearly and concisely.\n"
            "- If the retrieved blocks only partially support the question, answer ONLY the supported portion and explicitly explain what is missing.\n"
            "- If the retrieved blocks do not contain enough evidence to establish the requested behavior, abstain honestly "
            "(explain what was inspected and state that the retrieved code does not establish the behavior, without claiming it does not exist anywhere in the repo)."
        ),
        *_format_isolated_sections(directory_tree, chat_history, previous_context),
        f"RETRIEVED CODE EVIDENCE BLOCKS:\n{evidence_text}",
        f"CURRENT QUESTION: {question}\n\nProvide a concise, strictly grounded answer in Markdown:",
    ]
    return "\n\n".join(s for s in sections if s)


def build_g2_prompt(
    question: str,
    chunks: Sequence[dict[str, Any]],
    directory_tree: str = "",
    chat_history: Sequence[dict[str, Any]] | None = None,
    previous_context: str | None = None,
    repository_name: str = "",
) -> str:
    """G2: G1 grounded rules PLUS strict `[EVIDENCE_i]` inline citations (without G3 JSON sufficiency schema)."""
    repo_line = f"Target Repository: {repository_name}\n" if repository_name else ""
    rules_text = "\n".join(EVIDENCE_RULES)
    evidence_text = format_evidence_blocks(chunks)
    sections = [
        (
            "You are a trustworthy codebase analysis assistant.\n"
            f"{repo_line}"
            "Answer ONLY from the retrieved [EVIDENCE_i] code blocks provided below.\n\n"
            f"STRICT EVIDENCE RULES:\n{rules_text}\n\n"
            "STRICT CITATION RULES:\n"
            "- Cite every factual repository claim inline using ONLY the exact evidence block handle such as [EVIDENCE_1] or [EVIDENCE_2].\n"
            "- Never invent evidence IDs (such as [EVIDENCE_99] when only 5 blocks exist).\n"
            "- Never invent or paraphrase file paths or line ranges outside the retrieved blocks.\n"
            "- If the retrieved evidence does not establish the answer, abstain clearly and cite at most the inspected blocks whose scope was negative/unrelated."
        ),
        *_format_isolated_sections(directory_tree, chat_history, previous_context),
        f"RETRIEVED CODE EVIDENCE BLOCKS:\n{evidence_text}",
        f"CURRENT QUESTION: {question}\n\nProvide a grounded answer with inline [EVIDENCE_i] citations:",
    ]
    return "\n\n".join(s for s in sections if s)


def build_g3_prompt(
    question: str,
    chunks: Sequence[dict[str, Any]],
    directory_tree: str = "",
    chat_history: Sequence[dict[str, Any]] | None = None,
    previous_context: str | None = None,
    repository_name: str = "",
) -> str:
    """G3: Single-call structured JSON grounded generation, sufficiency assessment, abstention, and `[EVIDENCE_i]` citations."""
    repo_line = f"Target Repository: {repository_name}\n" if repository_name else ""
    rules_text = "\n".join(EVIDENCE_RULES)
    evidence_text = format_evidence_blocks(chunks)
    valid_ids = [f"EVIDENCE_{i}" for i in range(1, len(chunks) + 1)]
    valid_ids_str = ", ".join(valid_ids) if valid_ids else "(none retrieved)"

    sections = [
        (
            "You are a trustworthy codebase analysis engine running in a SINGLE structured inference step.\n"
            f"{repo_line}"
            "Perform evidence sufficiency assessment, grounded answering/abstention, and exact evidence citation "
            "simultaneously from ONLY the retrieved code evidence blocks below.\n\n"
            f"STRICT EVIDENCE RULES:\n{rules_text}\n\n"
            "SUFFICIENCY & ABSTENTION RUBRIC (Judge strictly from code content, NEVER from similarity score alone):\n"
            "1. 'strong': Retrieved evidence directly establishes the requested behavior or definition. Set sufficient=true, abstained=false, missing_information=[].\n"
            "2. 'partial': Retrieved evidence establishes part of the question (e.g., an entrypoint, caller, interface, or one sub-component) while another detail is absent. "
            "Do NOT over-abstain: set sufficient=true, confidence_tier='partial', abstained=false, answer ONLY the supported portion with inline [EVIDENCE_i] citations, explicitly qualify the scope limitation inside 'answer', and list what could not be confirmed in 'missing_information'.\n"
            "3. 'insufficient': Retrieved evidence does NOT establish the core requested implementation/behavior (including unretrieved implementation details or out-of-scope features). "
            "Set sufficient=false, confidence_tier='insufficient', abstained=true. Do NOT use pretrained memory about public libraries to fill gaps. In 'answer', state clearly what cannot be established from the retrieved code (without claiming the feature is globally absent from the repo), describe what is missing in 'missing_information', and cite only actually relevant/inspected IDs if referenced.\n"
            "4. Empty retrieval: If no evidence blocks were provided, set sufficient=false, confidence_tier='insufficient', abstained=true, citations=[].\n\n"
            f"CITATION RULES:\n"
            f"- Allowed evidence IDs for this query: [{valid_ids_str}].\n"
            "- Every implementation claim in 'answer' MUST carry an inline [EVIDENCE_i] tag (e.g. [EVIDENCE_1]).\n"
            "- List all cited evidence IDs (without brackets, e.g. 'EVIDENCE_1') inside the JSON 'citations' array.\n"
            "- NEVER cite files or line numbers not present inside the allowed [EVIDENCE_i] blocks.\n\n"
            "OUTPUT FORMAT REQUIREMENT:\n"
            "Return ONLY a single valid JSON object matching keys: "
            "sufficient (bool), confidence_tier ('strong'|'partial'|'insufficient'), reason (str), "
            "missing_information (list[str]), answer (str), citations (list[str]), abstained (bool)."
        ),
        *_format_isolated_sections(directory_tree, chat_history, previous_context),
        f"RETRIEVED CODE EVIDENCE BLOCKS:\n{evidence_text}",
        f"CURRENT QUESTION: {question}",
    ]
    return "\n\n".join(s for s in sections if s)


def build_blinded_judge_prompt(
    question: str,
    evidence_blocks_text: str,
    candidate_answer: str,
    previous_context: str | None = None,
    repository_name: str = "",
) -> str:
    """Blinded structured evaluation rubric comparing any candidate response strictly against frozen evidence."""
    repo_line = f"Repository: {repository_name}\n" if repository_name else ""
    ctx_block = (
        f"PRIOR CONVERSATION TOPIC (reference only, NOT evidence): {previous_context}\n\n"
        if previous_context
        else ""
    )
    return (
        "You are an impartial, strict software engineering RAG evaluator.\n"
        f"{repo_line}"
        "Evaluate the CANDIDATE ANSWER using ONLY the QUESTION and the RETRIEVED CODE EVIDENCE BLOCKS below.\n"
        "CRITICAL RULE: Do NOT use your pretrained knowledge of Flask, Express, fmtlib, or any codebase as ground truth. "
        "Even if a statement in the Candidate Answer is actually true in the real upstream repository, if it is NOT supported by the retrieved blocks below, mark it UNSUPPORTED / PARAMETRIC LEAKAGE.\n\n"
        "Evaluate carefully across all rubric dimensions:\n"
        "1. correctness: 'correct' (accurately addresses question using retrieved evidence OR appropriately abstains when evidence is insufficient), 'partially_correct', or 'incorrect'.\n"
        "2. faithfulness: 'supported' (all factual repo claims directly backed by retrieved blocks), 'partially_supported', 'unsupported' (makes claims not backed by retrieved blocks), or 'contradicted'.\n"
        "3. citation_relevance: 'relevant' (cited sources support the claims), 'partially_relevant', 'irrelevant', 'no_citations' (made repository claims without source references/citations), or 'not_applicable' (pure abstention citing nothing).\n"
        "4. answer_usefulness: 'high', 'medium', or 'low'.\n"
        "5. abstention_correctness: 'appropriate_answer' (evidence supported answering and candidate answered), 'appropriate_abstention' (evidence did not support core answer and candidate abstained cleanly), 'false_refusal' (abstained despite evidence supporting a direct or partial answer), or 'missed_abstention' (answered speculatively or from prior memory despite insufficient retrieved evidence).\n"
        "6. parametric_leakage (bool): true if the candidate introduced repository-specific symbols, file behaviors, defaults, config keys, or implementation claims not present in the retrieved blocks below.\n"
        "7. unsupported_claims (list[str]): concrete repository claims made without retrieved backing (empty if none).\n"
        "8. rationale (str): 1–3 concise sentences explaining your verdict strictly from the blocks below.\n\n"
        f"{ctx_block}"
        f"QUESTION: {question}\n\n"
        f"RETRIEVED CODE EVIDENCE BLOCKS:\n{evidence_blocks_text}\n\n"
        f"CANDIDATE ANSWER:\n{candidate_answer}\n\n"
        "Return ONLY valid JSON matching the evaluation schema."
    )
