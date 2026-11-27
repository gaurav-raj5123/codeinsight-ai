# DAY 6 — INITIAL GENERATION & TRUSTWORTHINESS AUDIT (`DAY6_GENERATION_AUDIT.md`)

**Project:** CodeInsight AI Assistant (GitHub Codebase RAG)  
**Scope:** Pre-implementation technical audit of LLM generation, prompt assembly, history/directory-tree isolation, citations, telemetry, API resilience, and frozen Day 5 retrieval verification.  
**Verification Constraint:** All paths, module signatures, SDK types, and Day 5 cohort counts documented below were directly inspected prior to Day 6 implementation.

---

## 1. Existing Production & Retrieval Baseline Verification

### 1.1 Selected Day 5 Configuration & Corpus Artifacts
Inspection of [day5_summary.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day5_rerank_rewrite/results/day5_summary.json) and [DAY5_RERANK_REWRITE_REPORT.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day5/DAY5_RERANK_REWRITE_REPORT.md) confirms:
- **Final Selected Retrieval Configuration:** `D5-E3a_rerank_rrf20`
  - Chunking: `code_aware_v1` (`chunker_v1.py`, <= 256 WordPiece tokens, embedded leading metadata headers inside `chunk["content"]`).
  - Dense Embeddings: `DefaultEmbeddingFunction()` (`all-MiniLM-L6-v2` via ChromaDB ONNX Runtime, 384-d, squared $L_2$).
  - Stage-1 Hybrid: Dense@20 + Code-Aware Okapi BM25@20 fused via Reciprocal Rank Fusion (`rrf_k = 60`).
  - Stage-2 Reranking: CPU ONNX cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (rev `233902d25c44...`, `86.80 MB`) scoring fixed `RRF Top-20` -> Final Top-5 (`Hit@1=0.4630`, `Hit@5=0.7963`, `MRR@5=0.5861`, `NDCG@5=0.4366`).
  - Query Rewriting: Disabled by default (`enable_query_rewrite = False` in [app.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/app.py)), since `Q0` unrewritten Turn-2 `Hit@5` reached `0.9167` (`11/12`).
- **Exact Day 5 Evaluation Artifact Verified:** [D5-E3a_rerank_rrf20_evaluation.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day5_rerank_rewrite/results/per_experiment/D5-E3a_rerank_rrf20_evaluation.json) (`60` records, Top-5 results + raw logits + sigmoid scores).
- **Exact Day 5 ChromaDB Store Verified:** `experiments/day4_hybrid/chroma_day4_db/` containing:
  - `day4_v1_flask` (`1,352` chunks, fingerprint `7d996d0cf4d1...`)
  - `day4_v1_express` (`1,297` chunks, fingerprint `d92e0235ef6e...`)
  - `day4_v1_fmt` (`1,072` chunks, fingerprint `4bae980c35f2...`)
- **Hydration Pre-Check Result:** Direct verification across all `60 × 5 = 300` Top-5 chunk IDs in `D5-E3a` confirmed **`300 / 300` exact matches** (`0` missing IDs, `0` mismatched `file_path` / `start_line` / `end_line` fields). Cached repository trees under `experiments/datasets/cache/{flask,express,fmt}` match pinned commit SHAs (`49943b5a...`, `43fca52d...`, `3be6709f...`).

### 1.2 Verified Evaluation Cohorts (`N = 60`)
Direct cross-verification of `D5-E3a_rerank_rrf20_evaluation.json`, [evaluation_questions.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/datasets/evaluation_questions.json), and [ground_truth.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/datasets/ground_truth.json) confirms exact agreement with reviewer expectations (`0` discrepancies):

| Cohort | Definition | Count | Question IDs (B & C) |
| :--- | :--- | :---: | :--- |
| **Cohort A — Answerable + Retrieval Hit** | `answerable == true` and `hit_rate_at_5 == 1.0` | **`43`** | Remaining `43/54` answerable queries |
| **Cohort B — Answerable + Retrieval Miss** | `answerable == true` and `hit_rate_at_5 == 0.0` | **`11`** | `flask_002`, `flask_017`, `flask_018`, `express_008`, `express_011`, `express_013`, `express_018`, `fmt_003`, `fmt_004`, `fmt_013`, `fmt_016` |
| **Cohort C — Unanswerable** | `answerable == false` | **`6`** | `flask_019`, `flask_020`, `express_019`, `express_020`, `fmt_019`, `fmt_020` |

*(Wait-free note on Day 5 scores: Day 5 demonstrated wide logit overlap between Cohort A/B and Cohort C—e.g., `fmt_020` unanswerable scored `+2.4591` while `express_010` hit scored `-10.0492`—confirming reranker score alone cannot dictate sufficiency or abstention.)*

---

## 2. Audit of Existing Generation & Orchestration Layers

### 2.1 Gemini Model & SDK (`llm.py`)
- **Client & Model:** Uses `google-genai` SDK (`2.22.0`) instantiating `genai.Client(api_key=api_key)` inside `try/except` and invoking `client.models.generate_content(model="gemini-3.5-flash", contents=prompt)`.
- **Secret Hygiene:** Masks `AIza[0-9A-Za-z-_]+` substrings with `[REDACTED_API_KEY]` before returning or caching errors in `_LAST_GENERATION_METADATA` — covered explicitly by [test_baseline_instrumentation.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/tests/test_baseline_instrumentation.py).
- **Generation Configuration:** Uses bare SDK defaults (no `temperature=0.0`, no `GenerateContentConfig`, no structured JSON schema or `response_mime_type`).

### 2.2 Existing API Timeout & Retry Gaps (`llm.py` & Live Preflight)
- **Docstring vs. Implementation Mismatch:** [llm.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/llm.py#L105-L145) docstring mentions exponential backoff retries for rate limits, yet `ask_gemini()` executes a single un-bounded `generate_content()` call with **no timeout**, **no retry loop**, and **no transient backoff** for `429`, `503`, or `504` responses.
- **Empirical API Latency / Stalls Observed:** During Day 5 (`gemini-3.5-flash`), `4/12` calls failed with `503 UNAVAILABLE` or `30s` timeouts. During Day 6 preflight probing with `response_schema`, `gemini-3.5-flash` raised `504 DEADLINE_EXCEEDED` at `44.80s` (`HttpOptions(timeout=45000)`), and unbounded `generate_content()` hung >`90s` when no HTTP/executor timeout was enforced. Robust Day 6 benchmarking therefore strictly requires caller + HTTP timeouts (`~45s`), conservative pacing (`2–3s`), bounded transient retries (`429/503/504/timeout`), and atomic per-query checkpoint/resume persistence.

### 2.3 Prompt Construction & Trustworthiness Hazards (`llm.build_prompt`)
Auditing [llm.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/llm.py#L31-L100) reveals five critical groundedness gaps:
1. **Directory Tree Treated as Implementation Evidence:** `build_prompt()` injects `REPOSITORY STRUCTURE` (`5,552` chars for Flask, `5,063` for Express, `2,934` for fmt) and explicitly commands `"- For architecture questions, use the repository structure above"`. Models can hallucinate unretrieved file roles, symbols, or features purely from directory/file names (e.g., `sessions.py`, `db.py`, `chrono.h`).
2. **Un-isolated Conversation History:** Formats up to `chat_history[-10:]` as bare `User:` / `Assistant:` text without marking prior assistant statements as **unverified conversational context (not source evidence)**. Any earlier misstatement by the assistant can contaminate downstream answers.
3. **V1 Header Duplication & Lack of Evidence Handles:** V1 chunks (`chunker_v1.py`) already embed leading lines (`File: ... | Language: ... | Type: ...`, `Class: ... | Symbol: ... | Part: ...`, `Signature: ...`) inside `chunk["content"]`. `build_prompt()` prepends `--- 📄 File: {file_path} | Lines {start_line}-{end_line}` directly above `chunk["content"]`, duplicating `File:` metadata while offering **no `[EVIDENCE_i]` identifiers** for deterministic citation binding.
4. **No Single-Call Sufficiency / Abstention Structure:** Uses only a single generic rule (`"- If the context doesn't contain enough info, say so"`). There is no distinction between **`strong`**, **`partial`** (answer supported subset + state missing piece), and **`insufficient`** (abstain cleanly without parametric memory from pretraining on Flask/Express/fmt), nor protection against confusing *"not retrieved in Top-5"* with *"does not exist in the repo"*.
5. **Free-Form Unverifiable Citations:** Instructs `"Reference specific files and line numbers"`, allowing the model to paraphrase, shift, or hallucinate paths/line numbers without deterministic metadata verification.

### 2.4 Telemetry, Evaluation & UI Inspection (`telemetry.py`, `app.py`, `evaluation/`)
- **Telemetry ([telemetry.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/telemetry.py)):** Persists atomic JSON runs (`tempfile.mkstemp` + `os.fsync` + `os.replace`) and protects corrupt files via `RuntimeError`. However, `create_query_record()` only captures retrieval chunks/latency and basic generation metadata (`model`, `latency_seconds`, `success`, `error`, lengths); it lacks fields for prompt version, experiment ID, commit SHA, frozen evidence IDs, sufficiency, confidence tier, reason, missing info, abstention status, parsed JSON, valid/invalid citation IDs, validation status, retry count, and evaluation results. Note: [baseline_run.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/results/baseline_run.json) is currently empty (`[]`).
- **Streamlit UI ([app.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/app.py)):** Renders raw `response` markdown plus `📄 Retrieved Code Chunks` and `📊 Baseline Diagnostics` expanders. Passes `chat_history=st.session_state.messages[:-1]` (which excludes the current prompt cleanly). Note that in the message-replay loop (`lines 256-270`), historical chunks omit the rerank logit label present in live turn rendering (`lines 352-357`), and there is no structured rendering of resolved `[EVIDENCE_i]` citations, partial-evidence qualifications, or abstention notices.
- **Evaluation (`evaluation/`):** Only [retrieval_metrics.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/evaluation/retrieval_metrics.py) exists (Hit@K, Precision@5, Span Recall, MRR@5, NDCG@5, latency); there is no generation-layer deterministic validator, blinded LLM judge harness, or abstention/faithfulness reporting.
- **Test Compatibility Anchors:** Existing regression suites (`51` pytest cases + [test_baseline_instrumentation.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/tests/test_baseline_instrumentation.py) & [test_task4_corrections.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/tests/test_task4_corrections.py)) explicitly test:
  - `llm.build_prompt(question, chunks, directory_tree, chat_history=...)` exact V0 behavior and `QUESTION:` / `CONVERSATION HISTORY` markers;
  - `llm.ask_gemini(prompt, return_metadata=...)` exact signature, mockable `google.genai.Client`, `model="gemini-3.5-flash"`, error prefixes, and `AIza...` secret masking;
  - `telemetry.create_baseline_run_record()`, `create_query_record()`, and `save_baseline_run()` atomic/corruption guarantees;
  - `app.py` static compilation/import and default `embedder.DEFAULT_RETRIEVAL_MODE == "dense"` preservation.

---

## 3. Compatibility & Architectural Plan for Day 6

1. **Zero Changes to Historical Retrieval/Data:** Preserve [chunker.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/chunker.py), [chunker_v1.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/chunker_v1.py), `retrieval/*`, `experiments/datasets/*`, `./chroma_db`, and all Day 1–5 results.
2. **Preserve `llm.py` Anchors:** Leave `llm.build_prompt()` and `llm.ask_gemini()` signatures and default V0 prompt/return contracts intact; place all new Day 6 grounded generation, citation, prompt versioning, and evaluation logic cleanly inside a modular `generation/` package (`prompts.py`, `citations.py`, `grounded_generator.py`, `evaluation.py`).
3. **Freeze Exact Retrieval Context:** Build `experiments/day6_generation/frozen_retrieval_evidence.json` hydrating all 60 queries' Top-5 chunks from `day4_v1_{flask,express,fmt}` matched against `D5-E3a_rerank_rrf20_evaluation.json`, plus cached repo `directory_tree` and Turn-2 `previous_context`.
4. **Resilient API Execution:** Investigate fast/reliable `gemini-3.5-flash` JSON generation settings (e.g., `thinking_config` / `response_mime_type` / bounded timeout ~45s / 2–3s pacing / retry on 429/503/504/timeout) and persist per-query checkpoints incrementally after every API call.
