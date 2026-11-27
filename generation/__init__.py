"""Day 6 Trustworthy Grounded Generation Package for CodeInsight AI Assistant.

Modules:
  - prompts: Versioned G0/G1/G2/G3/Judge prompt builders with strict
    directory-tree and conversation-history isolation and V1 header hygiene.
  - citations: Deterministic `[EVIDENCE_i]` resolution, validation, fabrication
    checks, and readable path:line rendering.
  - grounded_generator: Single-call G3 structured generation, resilient Gemini
    invocation (timeout + bounded transient backoff), and offline G4 post-processing.
  - evaluation: Layer 1 deterministic checks and Layer 2 blinded LLM judge helpers.
"""

from generation.citations import (
    build_evidence_map,
    detect_unretrieved_file_mentions,
    extract_inline_evidence_ids,
    validate_and_render_citations,
)
from generation.evaluation import (
    JudgeVerdictSchema,
    aggregate_judge_results,
    aggregate_layer1_metrics,
    detect_unstructured_abstention,
    evaluate_layer1_record,
)
from generation.grounded_generator import (
    GroundedGenerationSchema,
    apply_g4_postprocessing,
    call_gemini_resilient,
    generate_grounded_answer,
    parse_structured_g3_response,
    run_generation_experiment_item,
)
from generation.prompts import (
    CONVERSATION_HISTORY_ISOLATION_NOTICE,
    DIRECTORY_TREE_ISOLATION_NOTICE,
    EVIDENCE_RULES,
    G0_PROMPT_VERSION,
    G1_PROMPT_VERSION,
    G2_PROMPT_VERSION,
    G3_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    build_blinded_judge_prompt,
    build_g0_prompt,
    build_g1_prompt,
    build_g2_prompt,
    build_g3_prompt,
    format_evidence_blocks,
    normalize_chat_history,
    strip_redundant_v1_header_for_prompt,
)

__all__ = [
    "CONVERSATION_HISTORY_ISOLATION_NOTICE",
    "DIRECTORY_TREE_ISOLATION_NOTICE",
    "EVIDENCE_RULES",
    "G0_PROMPT_VERSION",
    "G1_PROMPT_VERSION",
    "G2_PROMPT_VERSION",
    "G3_PROMPT_VERSION",
    "GroundedGenerationSchema",
    "JUDGE_PROMPT_VERSION",
    "JudgeVerdictSchema",
    "aggregate_judge_results",
    "aggregate_layer1_metrics",
    "apply_g4_postprocessing",
    "build_blinded_judge_prompt",
    "build_evidence_map",
    "build_g0_prompt",
    "build_g1_prompt",
    "build_g2_prompt",
    "build_g3_prompt",
    "call_gemini_resilient",
    "detect_unretrieved_file_mentions",
    "detect_unstructured_abstention",
    "evaluate_layer1_record",
    "extract_inline_evidence_ids",
    "format_evidence_blocks",
    "generate_grounded_answer",
    "normalize_chat_history",
    "parse_structured_g3_response",
    "run_generation_experiment_item",
    "strip_redundant_v1_header_for_prompt",
    "validate_and_render_citations",
]
