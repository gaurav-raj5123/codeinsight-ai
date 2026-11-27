# DAY 6 — TRUSTWORTHY GROUNDED GENERATION REPORT (`DAY6_GROUNDED_GENERATION_REPORT.md`)

**Project:** CodeInsight AI Assistant (GitHub Codebase RAG)  
**Pre-Implementation Audit:** [DAY6_GENERATION_AUDIT.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day6/DAY6_GENERATION_AUDIT.md)  
**Benchmark & Freeze Scripts:** [freeze_evidence.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day6_generation/freeze_evidence.py), [run_day6_benchmark.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day6_generation/run_day6_benchmark.py)  
**Summary Artifacts:** [day6_summary.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day6_generation/results/day6_summary.json), [layer1_evaluation.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day6_generation/results/layer1_evaluation.json), [layer2_blinded_judge.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day6_generation/results/layer2_blinded_judge.json), [citation_validation_report.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day6_generation/results/citation_validation_report.json)  
**Production Modules:** [generation/prompts.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/generation/prompts.py), [generation/citations.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/generation/citations.py), [generation/grounded_generator.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/generation/grounded_generator.py), [generation/evaluation.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/generation/evaluation.py), [telemetry.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/telemetry.py), [app.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/app.py)

---

## 1. Executive Summary

Day 6 upgraded **CodeInsight AI Assistant** from unverified prose generation (`llm.build_prompt` / `llm.ask_gemini`) to **evidence-grounded single-call structured generation (`G3`) paired with offline deterministic citation validation & rendering (`G4`)**, evaluated across identical frozen Day 5 retrieval context (`D5-E3a_rerank_rrf20`, `300/300` hydrated Top-5 chunks verified).

**Key Conclusions:**
1. **Zero Fabricated Citations or Unretrieved Paths (`G4`, `N = 60`):** G4 achieved **`100.0%` citation validity (`154/154` tokens across `100%` of answered queries, `0` invalid citations, `0` duplicates)**, **`100.0%` JSON schema validity (`60/60`)**, and **eliminated unretrieved file mentions (`2 -> 0`) and unretrieved line-range mentions (`9 -> 0`)** relative to `G0` — adding only **`0.33 ms` median (`1.19 ms` max)** deterministic offline post-processing latency (`0` additional LLM calls).
2. **Elimination of Blinded-Judge Parametric Leakage & Unsupported Unanswerable Answers:** On the full `N = 60` Layer 1 audit, **`G4` raised Cohort C true abstention from `83.3%` (`5/6`) to `100.0%` (`6/6`)** (preventing `G0`'s overclaim on `express_019`) and increased **Cohort B miss abstention from `36.4%` (`4/11`) to `54.5%` (`6/11`)** while answering the remaining `5/11` strictly as `partial` evidence from retrieved JSDoc/test context. On the stratified blinded Layer 2 Judge (`N = 24`), **`G4` achieved `100%` strict faithfulness (`24/24`, vs `95.83%` for `G0`) and `0.0%` parametric leakage (`0/24`, catching `G0`'s fabricated `lib/utils.js:51–60` citation in `express_015`)**.
3. **API Quota Finding & Resilient Execution:** Initial baseline runs on `gemini-3.5-flash` (routed to `gemini-3.6-flash`) hit a hard Free Tier **`20` RPD (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`)** quota after **`18/60` completed `G0` calls (`47` total attempts)** and exhibited `504 DEADLINE_EXCEEDED` at `44.8s` unless `thinking_budget=0` was set. Preserving that partial `gemini-3.5-flash` artifact separately ([results/gemini35flash_partial](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day6_generation/results/gemini35flash_partial/day6_summary.json)), the full controlled Day 6 study (`204 / 204` live calls: `G0=60`, `G3=60`, `Judge=48`, `G1=18`, `G2=18`) ran to `100%` completion with `0` errors on **`gemini-3.5-flash-lite`** (`P50 = 1.837s` for G3).

---

## 2. Original Architecture & Initial Audit Summary

Per [DAY6_GENERATION_AUDIT.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day6/DAY6_GENERATION_AUDIT.md), auditing [llm.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/llm.py), [telemetry.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/telemetry.py), [app.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/app.py), and [chunker_v1.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/chunker_v1.py) established five generation gaps while identifying `llm.build_prompt()`, `llm.ask_gemini()`, and `telemetry.save_baseline_run()` as non-negotiable backward-compatibility anchors:
1. `llm.build_prompt()` instructed `"- For architecture questions, use the repository structure above"`, encouraging file/API inference from the ASCII `directory_tree` without retrieved code.
2. Conversation history was injected as raw `User:`/`Assistant:` prose without marking prior assistant statements as unverified context.
3. Day 3 `code_aware_v1` chunks embed leading headers (`File: ... | Language: ... | Type: ...`), causing `llm.build_prompt()` to duplicate `File:` headers and lack deterministic `[EVIDENCE_i]` handles.
4. `llm.ask_gemini()` lacked configurable HTTP/caller timeouts, retry/backoff loops, or structured JSON schema outputs.
5. `telemetry.py` lacked citation, sufficiency tier, confidence, abstention, and prompt-version fields, and `app.py` had no verification UI or telemetry-error isolation.

---

## 3. Frozen Day 5 Evidence & Evaluation Cohorts (`N = 60`)

To keep retrieval 100% frozen across `G0–G4` and the Blinded Judge, [freeze_evidence.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day6_generation/freeze_evidence.py) hydrated all 60 queries' Top-5 chunks directly from persisted Day 4/5 ChromaDB collections (`experiments/day4_hybrid/chroma_day4_db/`: `day4_v1_flask`, `day4_v1_express`, `day4_v1_fmt`) matched against [D5-E3a_rerank_rrf20_evaluation.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day5_rerank_rewrite/results/per_experiment/D5-E3a_rerank_rrf20_evaluation.json):
- **Hydration Verification:** **`300 / 300`** chunks matched (`0` missing IDs, `0` `file_path` / `start_line` / `end_line` mismatches).
- **Non-Chunk Context Parity:** Exact pinned `commit_sha` values, deterministic cached `directory_tree` strings (`5,552` / `5,063` / `2,934` chars for Flask/Express/fmt), and Turn-2 `previous_context` (`12` queries) were frozen inside [frozen_retrieval_evidence.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day6_generation/frozen_retrieval_evidence.json) (`737.2 KB`) so `G0–G4` received identical non-chunk context.
- **Verified Cohort Distribution (`0` Discrepancies vs. Day 5):**
  - **Cohort A (`Answerable + Retrieval Hit`):** **`43 / 54`** (`flask=15`, `express=14`, `fmt=14`)
  - **Cohort B (`Answerable + Retrieval Miss`):** **`11 / 54`** (`flask=3`, `express=4`, `fmt=4`: `flask_002`, `flask_017`, `flask_018`, `express_008`, `express_011`, `express_013`, `express_018`, `fmt_003`, `fmt_004`, `fmt_013`, `fmt_016`)
  - **Cohort C (`Unanswerable`):** **`6`** (`flask_019`, `flask_020`, `express_019`, `express_020`, `fmt_019`, `fmt_020`)

---

## 4. Experiment Design, Subsets & API Resilience

### 4.1 Experiment Matrix (`G0–G4`) & Stratified Subsets

| Experiment | Sample Size | Live API Calls | Configuration |
| :--- | :---: | :---: | :--- |
| **`G0` Baseline** | `N = 60` *(subset `18` also aligned)* | `60` | Original `llm.build_prompt()` + plain prose generation (`g0_baseline_v0`) |
| **`G1` Grounded Only** | `N = 18` *(6/repo; 9A/6B/3C)* | `18` | 14 Evidence Rules + Tree & History Isolation notices + `[EVIDENCE_i]` input blocks, prose output (`g1_grounded_only_v1`) |
| **`G2` Grounded + Citations** | `N = 18` *(same 18 queries)* | `18` | G1 + mandatory inline `[EVIDENCE_i]` citations, no JSON schema/abstention flags (`g2_grounded_citations_v1`) |
| **`G3` Full Structured** | `N = 60` | `60` | Single-call JSON (`GroundedGenerationSchema`): `sufficient`, `confidence_tier`, `reason`, `missing_information`, `answer`, `citations`, `abstained` (`g3_full_structured_v1`) |
| **`G4` Offline Citation Val.** | `N = 60` | **`0`** | Pure Python deterministic validation, invalid-citation blocking, readable `path:start–end` rendering preserving raw G3 untouched |
| **Layer 2 Blinded Judge** | `N = 24` *(12A/6B/6C; 8/repo)* | `48` | Blinded `JudgeVerdictSchema` evaluation comparing `G0` vs. `G4` strictly against frozen evidence (`judge_blinded_rubric_v1`) |

- Locked subset manifests: [ablation_question_ids.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day6_generation/ablation_question_ids.json) (`N=18`, `4` Turn-2, `5` multi-span, all 5 categories) and [judge_question_ids.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day6_generation/judge_question_ids.json) (`N=24`, all `6` Cohort C queries + `6` Cohort B + `12` Cohort A).

### 4.2 API Budget, Failure Resilience & Checkpointing Telemetry

| Run Series | Stage Breakdown | Planned Live Calls | Actual Live Attempts | Succeeded | Failed | Retries |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Initial (`gemini-3.5-flash`)** *(archived partial)* | `G0` only (`27` reached) | `204` | `47` | `18` | `9` (`20 RPD` cap) | `20` |
| **Controlled Complete (`gemini-3.5-flash-lite`)** | `G0(60)+G3(60)+G4(0)+Judge(48)+G1(18)+G2(18)` | **`204`** | **`204`** | **`204`** | **`0`** | **`0`** |

- **Resilience Features ([grounded_generator.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/generation/grounded_generator.py), [run_day6_benchmark.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day6_generation/run_day6_benchmark.py)):**
  - `45.0s` HTTP + `ThreadPoolExecutor` timeout guard, `2.0s` pacing, bounded exponential backoff (`429/503/504/timeout`), immediate non-retryable abort on `GenerateRequestsPerDay*` daily quota exhaustion or `400 INVALID_ARGUMENT` (fixed automatically for `*-lite` models by omitting `ThinkingConfig(thinking_budget=0)`), and atomic per-query checkpointing (proven during host suspend/resume without losing progress on `G0` query `fmt_004`).

---

## 5. Architecture Implemented (`generation/`, `telemetry.py`, `app.py`)

- **Prompt & Context Isolation ([prompts.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/generation/prompts.py)):**
  - Enforces all **14 strict evidence rules** + verbatim **Directory Tree Isolation** + **History Isolation (`(UNVERIFIED)` assistant tags)**.
  - Uses `strip_redundant_v1_header_for_prompt()` to remove only duplicated leading `File: ... | Language: ...` / `Class:/Symbol:` headers from `code_aware_v1` chunks during prompt formatting (while keeping `Signature:` lines and leaving frozen chunks completely unmutated).
- **Single-Call Structured G3 + Sufficiency ([grounded_generator.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/generation/grounded_generator.py)):**
  - Assesses `strong`, `partial` (`sufficient=true, abstained=false`, answering supported subset + populating `missing_information`), or `insufficient` (`sufficient=false, abstained=true`) in a single Gemini call via Pydantic `response_schema`.
  - Parse failures preserve `raw_response` safely without crashing or fabricating structured answers (`parse_success=False`, `validation_status="parse_failure"`).
- **Deterministic G4 Citation System ([citations.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/generation/citations.py)):**
  - Resolves `[EVIDENCE_i]` tokens and array declarations deterministically to chunk metadata (`src/example.py:42–78`), deduplicates repeats, flags invalid IDs (`[UNVERIFIED_CITATION: ...]` instead of fake file paths), handles missing/invalid line numbers cleanly, and records unretrieved file/line mention heuristics.
- **Telemetry ([telemetry.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/telemetry.py)) & UI ([app.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/app.py)):**
  - Preserves existing `create_query_record()` / `save_baseline_run()` contracts cleanly while adding sanitized Day 6 provenance, sufficiency, citation, and validation telemetry wrapped in crash-safe `try/except` in `app.py`.
  - Sidebar toggle **"Verified Grounded Citations & Abstention (Recommended)"** (`default=True`), rendering `[EVIDENCE_i]`-linked chunk blocks, partial/insufficient notices, `📚 Cited Source Navigation`, and safe fallbacks without exposing experiment IDs.

---

## 6. Evaluation Results

*(Note: Unless explicitly marked `gemini-3.5-flash partial`, all comparative metrics below share identical `gemini-3.5-flash-lite` generation on frozen `D5-E3a` retrieval evidence. Heuristic text-abstention metrics on `G0/G1/G2` are explicitly distinguished from `G3/G4`'s structured `abstained` boolean; Layer 2 Judge metrics are explicitly approximate.)*

### 6.1 Master Experiment Comparison (Full-Sample vs. Same-Sample `N = 18` Ablation)

| Metric | `G0` (`N=60`) | `G3` (`N=60`) | `G4` (`N=60`) | `G0` (`N=18` Sub) | `G1` (`N=18` Sub) | `G2` (`N=18` Sub) | `G4` (`N=18` Sub) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **JSON Schema Validity Rate** | `—` | `1.0000` (`60/60`) | `1.0000` (`60/60`) | `—` | `—` | `—` | `1.0000` (`18/18`) |
| **Citation Validity (Token Level)** | `N/A` | `1.0000` (`154/154`) | **`1.0000`** (`154/154`) | `N/A` | `N/A` | `1.0000` (`48/48`) | **`1.0000`** (`41/41`) |
| **Answered Citation Presence** | `0.0000` *(unstructured)* | `1.0000` (`45/45`) | **`1.0000`** (`45/45`) | `0.0000` | `0.0000` | `1.0000` | **`1.0000`** |
| **Queries w/ Unretrieved File Mentions** | `2` | **`0`** | **`0`** | `0` | **`0`** | **`0`** | **`0`** |
| **Queries w/ Unretrieved Line Mentions** | `9` | **`0`** | **`0`** | `4` | **`0`** | **`0`** | **`0`** |
| **Cohort A Answer Rate (`Hit`)** | `0.9302` (`40/43`) | `0.9302` (`40/43`) | `0.9302` (`40/43`) | `0.8889` (`8/9`) | `0.5556` (`5/9`)\* | `0.7778` (`7/9`) | `0.8889` (`8/9`) |
| **Cohort A False Refusal Rate** | `0.0698` (`3/43`)\* | `0.0698` (`3/43`) | `0.0698` (`3/43`) | `0.1111` (`1/9`)\* | `0.4444` (`4/9`)\* | `0.2222` (`2/9`)\* | `0.1111` (`1/9`) |
| **Cohort B Abstention Rate (`Miss`)** | `0.3636` (`4/11`) | **`0.5455`** (`6/11`) | **`0.5455`** (`6/11`) | `0.3333` (`2/6`) | `0.6667` (`4/6`) | `0.5000` (`3/6`) | `0.5000` (`3/6`) |
| **Cohort B Non-Abstained Answer Rate** | `0.6364` (`7/11`) | `0.4545` (`5/11`)\*\* | `0.4545` (`5/11`)\*\* | `0.6667` | `0.3333` | `0.5000` | `0.5000` |
| **Cohort C True Abstention (`Unans`)** | `0.8333` (`5/6`) | **`1.0000`** (`6/6`) | **`1.0000`** (`6/6`) | `1.0000` (`3/3`) | `1.0000` (`3/3`) | `1.0000` (`3/3`) | **`1.0000`** (`3/3`) |
| **Median Latency (`s`)** | `1.8830` | `1.8369` | `1.8369` + `0.33ms` | `1.8830` | `1.8963` | `1.9377` | `1.9566` + `0.28ms` |
| **P95 Latency (`s`)** | `2.8393` | **`2.1810`** | **`2.1810`** + `0.53ms` | `2.8251` | `3.6090` | `3.2656` | **`2.3560`** + `1.19ms` |

*\*Note on Heuristic Free-Text Abstention (`G0`/`G1`/`G2`):* Manual inspection proved that all 3 `G0` Cohort A flags (`flask_001`, `flask_016`, `fmt_010`) and 3 of 4 `G1` Cohort A flags (`flask_001`, `express_010`, `fmt_009`) were actually substantive answers whose trailing limitation caveats matched regex phrases (`"does not contain"` / `"not fully listed"`). Using `G3`'s explicit structured `abstained` boolean eliminates this ambiguity (`0.1111` on `N=18`, `0.0698` on `N=60`).  
*\*\*Note on Cohort B Non-Abstained (`5/11` in `G4`):* All 5 queries (`flask_017`, `express_008`, `express_013`, `express_018`, `fmt_016`) emitted `confidence_tier = "partial"` (zero `"strong"` claims) using retrieved JSDoc/test/adjacent blocks.  
*Latency note:* `G0 (N=60)` raw mean (`48.7151s`, max `2,804.98s`) includes an OS host-sleep interval on `fmt_004`; excluding `fmt_004`, active `G0` latency (`n=59`) is `mean = 1.9988s`, `P50 = 1.8719s`, `P95 = 2.8393s`, `max = 4.2151s`. Initial partial `G0` on `gemini-3.5-flash` (`N=27` attempted / `18` completed) had `P50 = 10.8081s`, `mean = 11.5705s`, `6` unretrieved line mentions, and `1` failed Cohort C completion (`api_failure_count=9`, `cohort_c_true_abstention_rate=0.5000` over total denominator `2`).

- **Turn-2 Conversational Isolation (`N = 12`):** Under `G4`, `10/11` Cohort A Turn-2 queries answered accurately (`1` conservative abstention on `flask_006` where span 2 `app.py` was absent), and `1/1` Cohort B Turn-2 (`flask_002`) cleanly abstained despite Turn-1 history.

### 6.2 Layer 2 — Blinded Structured LLM Judge (`N = 24` Stratified Subset, `G0` vs. `G4`, Approximate)

| Blinded Judge Metric (`N = 24` Each, Same-Model Family) | `G0` Baseline (`N=24`) | `G4` Grounded + Validated (`N=24`) |
| :--- | :---: | :---: |
| **Correctness (Strict `correct`)** | `0.9167` (`22/24`) | **`0.9583`** (`23/24`) |
| **Correctness (Lenient `correct + partial`)** | `1.0000` (`24/24`) | `1.0000` (`24/24`) |
| **Faithfulness (Strict `supported`)** | `0.9583` (`23/24`) | **`1.0000`** (`24/24`) |
| **Unsupported / Contradicted Claim Rate** | `0.0000` | `0.0000` |
| **Parametric Leakage Rate (`True`)** | `0.0417` (`1/24` — `express_015`) | **`0.0000`** (`0/24`) |
| **Citation Precision (`relevant` among cited)** | `1.0000` (`18/18`) | **`1.0000`** (`19/19`) |
| **Citation Coverage (among answered)** | `0.9444` (`17/18` — `express_002` uncited) | **`1.0000`** |
| **Abstention Accuracy (Judge Rubric)** | `1.0000` (`24/24`)* | `1.0000` (`24/24`) |
| **Usefulness (`high + medium`)** | `1.0000` (`24/24`) | `1.0000` (`24/24`) |
| **Cohort A Correctness / Faithfulness (`n=12`)** | `0.8333 / 0.9167` (`false_refusal=0.0`) | **`0.9167 / 1.0000`** (`false_refusal=0.0`) |
| **Cohort B Appropriate Abstention / Leakage (`n=6`)** | `0.3333 / 0.0000` | **`0.6667 / 0.0000`** |
| **Cohort C True Abstention (`n=6`)** | `0.6667` (`express_019` answered non-existence) | **`1.0000`** (`6/6`) |
| **Answerable Coverage (`n=18`, substantive)** | `0.8889` (`16/18`) | `0.7222` (`13/18`, conservative on misses) |

*\*Judge nuance:* On `express_019` (`Cohort C`) and `flask_019`, `G0` asserted *"there is no built-in..."* which the judge marked `appropriate_answer` (`66.7%` abstention on C vs `100%` clean abstention for `G4`); on `flask_006`, Judge rated `G0` `partially_correct` (`appropriate_answer`) and `G4` `correct` (`appropriate_abstention`).

---

## 7. Layer 3 — Manual Audit (10 Representative Cases)

All 10 inspections use verbatim frozen evidence, saved `G0`/`G3`/`G4` outputs, and blinded judge verdicts:

1. **Strong Evidence & Multi-File Definition (`fmt_011`, `Cohort A`, `spans=2`):**
   - *Evidence:* `EVIDENCE_1..5` (`include/fmt/printf.h:589-605`, `606-609`, `610-615`, `562-578`, `616-627`).
   - *G0 vs G3/G4:* Both accurately traced `sprintf -> make_printf_args -> vsprintf -> detail::vprintf`. `G3` scored `confidence_tier="strong"`, `abstained=False`, `missing_information=[]`. `G4` mapped `[EVIDENCE_1]`, `[EVIDENCE_2]`, `[EVIDENCE_3]` cleanly (`include/fmt/printf.h:589–605`, `606–609`, `610–615`). Judge: both `correct / supported / relevant / leakage=False`.
2. **Conversational Follow-Up + Multi-File + Low Reranker Logit (`express_010`, `Cohort A`, `turn=2`, Day 5 logit `-10.05`):**
   - *Prev Context:* `In express_009, we saw how res.send formats and dispatches the response.`
   - *Evidence:* Includes `[EVIDENCE_3]` (`lib/express.js:37–63`), `[EVIDENCE_5]` (`lib/application.js:125–151`).
   - *Outcome:* Despite Day 5 reranker scoring `-10.05` raw logit, `G3/G4` accurately grounded `app(req, res, next) -> app.handle(req, res, next)` (`partial`, citing `[EVIDENCE_3]`, `[EVIDENCE_5]`), proving Day 5's finding that negative MS MARCO logit != insufficient evidence! Judge: `correct / supported / leakage=False`.
3. **Partial Evidence Without Over-Abstaining (`flask_001`, `Cohort A`):**
   - *Evidence:* `class Flask(App)` in `[EVIDENCE_2]` (`src/flask/app.py:108–126`) + `make_config` `[EVIDENCE_5]` (`src/flask/sansio/app.py:480–494`); full `__init__` parameter list absent.
   - *G3/G4:* Classified `confidence_tier="partial"`, `abstained=False`, listed `Complete method signature... of Flask.__init__` in `missing_information`, and answered the supported parts cleanly with verified citations (`src/flask/app.py:108–126`, `src/flask/sansio/app.py:480–494`).
4. **Partial Evidence on a Retrieval Miss (`express_018`, `Cohort B`, GT `lib/application.js:294-309` missed):**
   - *Evidence:* `[EVIDENCE_1]` (`lib/application.js:256–277` — JSDoc directly above `app.engine`).
   - *Outcome:* Rather than blind refusal or guessing body internals, `G3/G4` answered how `app.engine(ext, fn)` registers extensions from `[EVIDENCE_1]` (`partial`, `abstained=False`) and explicitly noted the missing function body (`lib/application.js:256–277` valid citation).
5. **Retrieval Miss + Directory-Tree Leakage Prevention (`express_011`, `Cohort B`, `spans=2`):**
   - *Evidence:* Only `test/app.render.js` and `examples/view-constructor/index.js` chunks; neither `lib/application.js` nor `lib/view.js` retrieved.
   - *G0 Failure Mode:* `G0` explicitly referenced `(as the contents of lib/application.js, lib/response.js, and lib/view.js are not included...)` — naming `lib/response.js` purely from the directory tree (`unretrieved_file_mentions` triggered in Layer 1)!
   - *G3/G4 Fix:* Cleanly set `sufficient=false`, `abstained=true`, stated retrieved test/example scope (`[EVIDENCE_1..5]` resolved to `test/app.render.js` & `examples/view-constructor/index.js`) with zero directory-tree filename synthesis.
6. **Turn-2 Conversational Follow-Up on Retrieval Miss (`flask_002`, `Cohort B`, `turn=2`):**
   - *Prev Context:* `flask_001` Flask constructor question. Retrieved chunks cover `Config.from_object`/`from_envvar` (`src/flask/config.py`) but missed `make_config`.
   - *Outcome:* Both `G0` and `G4` refused to fabricate `Flask.__init__` defaults (`G4` structured `abstained=True`, citing inspected `Config` methods). Judge: `appropriate_abstention / supported`.
7. **Citation Fabrication / Parametric Leakage Smoking Gun (`express_015`, `Cohort A`):**
   - *Evidence:* `[EVIDENCE_3]` (`lib/utils.js:130–161`), `[EVIDENCE_4]` (`lib/utils.js:40–50`), plus `test/` chunks (`exports.wetag` body at `51–60` NOT retrieved).
   - *G0 Hallucination:* Claimed `"Weak ETags: Generated using exports.wetag (found in lib/utils.js, lines 51–60)"` — **flagged by BOTH Layer 1 (`unretrieved_line_mentions: ['lines 51–60']`) AND the Blinded Judge (`faithfulness="partially_supported", parametric_leakage=True`)**!
   - *G3/G4 Fix:* Cited ONLY `[EVIDENCE_3] (lib/utils.js:130–161)` and `[EVIDENCE_4] (lib/utils.js:40–50)`, explicitly stating that `exports.etag`/`exports.wetag` implementation details are absent (`faithfulness="supported", leakage=False`). *(Boundary lesson: `G3` marked `abstained=True` conservatively despite giving the exact `compileETag` answer — illustrating occasional cautious `abstained` flag calibration noise.)*
8. **Conservative Over-Abstention Boundary Case (`flask_006`, `Cohort A`, `turn=2`, `spans=2`):**
   - *Evidence:* Retrieved `[EVIDENCE_3]` (`src/flask/sansio/scaffold.py:375–402`, docstring/signature of `add_url_rule`) and `[EVIDENCE_4]` (`app.py:385–397`), missing span 2 (`app.py:507–530`).
   - *G3/G4 Behavior:* Explained `Scaffold.add_url_rule`'s role from `[EVIDENCE_3]`, yet set `abstained=True, confidence_tier="insufficient"` because the `url_map` mutation body wasn't present — whereas `partial, abstained=False` would have yielded higher answer coverage on Cohort A (`false_refusal` on `flask_006` / `express_009` / `express_015`).
9. **Unanswerable Global Non-Existence Claim (`express_019`, `Cohort C`):**
   - *G0:* Asserted `"there is no built-in JWT (JSON Web Token) authentication verification middleware implemented in Express"` (global claim breaking Rule 10/11).
   - *G3/G4:* Scoped honestly (`"The retrieved evidence does not establish where or if a built-in JWT... is implemented in Express"`, `abstained=true`, `citations=[]`). Judge: `correct / appropriate_abstention / leakage=False`.
10. **Unanswerable Query w/ High Day 5 Cross-Encoder Logit (`fmt_020`, `Cohort C`, Day 5 logit `+2.4591` / sigmoid `0.9212`):**
    - *Evidence:* `compile.h:24-37` (`license_header`), `enum.h`, `chrono.h`, `fmt-c.h`.
    - *G3/G4:* Ignored the misleading `+2.459` lexical similarity match, classified `insufficient / abstained=true`, listed `JSON parsing/serialization/struct mapping` in `missing_information`, and refused cleanly with `0` spurious citations (`Judge: correct / supported / appropriate_abstention`).

---

## 8. Limitations & Mandatory Disclosures

1. **Limited Benchmark Sample Size:** `60` total queries (`43` Cohort A, `11` Cohort B, `6` Cohort C), `18` in `G1/G2` ablations, and `24` in Layer 2 evaluation limit statistical power; no formal significance claims are asserted across small cohort counts.
2. **Same-Family LLM Judge Bias:** Blinded Layer 2 evaluation used `gemini-3.5-flash-lite` (the same Gemini model family as the generator), introducing potential stylistic/rubric alignment bias; judge numbers are reported strictly as **approximate evaluation**.
3. **Model Substitution Due to Free Tier `20 RPD` Cap:** Because Google's Free Tier capped `gemini-3.5-flash` at `20` RPD after `18` completed `G0` items, the complete `204/204` controlled comparison ran uniformly on `gemini-3.5-flash-lite`. Capabilities and abstention aggressiveness may differ on larger Gemini Pro/Flash models.
4. **Citation Existence Does Not Prove Semantic Support:** Deterministic `G4` validation (`100%` ID/metadata resolution) guarantees citations exist in retrieved evidence, NOT that every cited chunk semantically supports the specific claim — semantic faithfulness still relies on Layer 2/3 audits.
5. **No Zero-Hallucination Guarantee:** Structured prompting cannot guarantee zero hallucinations or eliminate all pretrained familiarity with public libraries like Flask/Express/fmt.
6. **Retrieval Miss Bottleneck Remains:** When Top-5 omits core implementation (`Cohort B = 11/54`), even ideal grounded generation must abstain or give partial context — trustworthiness exposes retrieval gaps (`express_011`, `flask_002`, `fmt_003`) rather than masking them with memorized priors (`Cohort B` abstention reduces raw answerable coverage from `88.9%` to `72.2%` in `N=24` judge accounting).
7. **Text-Abstention Heuristics vs. Structured Flags:** Unstructured `G0/G1` abstention rates depend on regex caveat matching, over-counting abstentions when answers append disclaimers (`flask_001`, `fmt_009`).

---

## 9. Final Selected Configuration (`G3 + G4` Grounded Pipeline)

- **Selected Production Configuration:** **Single-call structured generation (`G3`, `g3_full_structured_v1`) + deterministic offline validation & citation rendering (`G4`)**, running on `gemini-3.5-flash-lite` in grounded UI mode (`45.0s` timeout, bounded transient retries, `thinking_budget` omitted cleanly on `*-lite`), integrated into [app.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/app.py) with original [llm.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/llm.py) (`gemini-3.5-flash` V0 anchor) preserved untouched for backward compatibility.
- **Selection Tradeoff Justification:** Rather than optimizing naive prose fluency or raw answerable coverage, **`G3 + G4`** delivers **`100%` deterministic citation validity (`154/154`), `0` fabricated file/line citations, `100%` Cohort C abstention (`6/6`), `100%` blinded-judge faithfulness (`24/24`), and `0%` blinded-judge parametric leakage**, at **lower P95 latency (`2.18s` vs `2.84s` for `G0`)** and **`0.33 ms` median offline post-processing overhead**. The modest partial-span over-abstention observed in `3/43` Cohort A queries (`flask_006`, `express_009`, `express_015` where summary text was still helpful) is an explicit target for fine-tuning `partial` vs `insufficient` threshold prompts cleanly in Day 7.
