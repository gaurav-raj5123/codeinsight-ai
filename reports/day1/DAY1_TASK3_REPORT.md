# DAY 1 — TASK 3 REPORT: REPOSITORY SELECTION & EVALUATION DATASET CREATION (REVISED)

## Executive Summary

- **Status**: Completed, Hardened & 100% Validated
- **Artifacts Created / Revised**:
  - [`experiments/datasets/repositories.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/datasets/repositories.json) (Single Source of Truth)
  - [`experiments/datasets/evaluation_questions.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/datasets/evaluation_questions.json) (60 questions: 22 identifier, 20 conceptual, 12 conversational, 6 unanswerable)
  - [`experiments/datasets/ground_truth.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/datasets/ground_truth.json) (71 verified evidence spans, 17 multi-file questions, strictly zero chunk IDs)
  - [`experiments/datasets/dataset_readme.md`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/datasets/dataset_readme.md) (Roadmap aligned to Days 2–7, relevance rule, schema documentation)
  - [`experiments/datasets/validate_dataset.py`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/datasets/validate_dataset.py) (Hardened validation suite with loader indexability check)
- **Review Corrections Implemented**:
  1. *Context-Dependent Follow-ups*: All 12 Turn-2 questions rewritten to naturally depend on Turn 1 via anaphora; all 12 are strictly distinct from their `expected_standalone_query`.
  2. *Genuine Multi-File Evidence*: Expanded ground truth to 17 multi-evidence questions (71 total spans) across all 10 `cross_file_dependencies` and key architecture questions.
  3. *Elimination of Leaked File Paths*: 0 out of 60 questions (0.0%) leak internal file paths; balanced with 20 conceptual/intent-based queries and 22 identifier queries.
  4. *Hardened Validation*: Integrated dynamic metadata loading from `repositories.json`, `repo_loader.get_source_files()` indexability check, multi-turn strictness check, and multi-file reporting.
  5. *Documentation Aligned*: Updated roadmap references to Day 5 for multi-turn query rewriting.
- **Strict Scope Boundaries**:
  - Application code (`app.py`, `chunker.py`, `embedder.py`, `llm.py`, `repo_loader.py`, `telemetry.py`) was **NOT modified**.
  - All 7 tests in [`test_baseline_instrumentation.py`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/test_baseline_instrumentation.py) pass.
  - Day 1 Task 4 baseline evaluation has **NOT been run**.

---

## 1. Selected Repositories Metadata (Single Source of Truth)

All repository metadata is loaded dynamically from [`repositories.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/datasets/repositories.json):

| Repository | GitHub URL | Primary Language | Additional Languages | Pinned Commit SHA | Indexed Source Files | Architectural Profile |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`pallets/flask`** | [pallets/flask](https://github.com/pallets/flask) | Python | None | `49943b5a94413a530761350f04e896f76e8d620e` | 83 | WSGI web framework; complex request/app context lifecycle (`AppContext`), decorator routing (`@app.route`), Blueprint registration state, and config mapping. |
| **`expressjs/express`** | [expressjs/express](https://github.com/expressjs/express) | JavaScript | None | `43fca52d28a422c8be48fd489cbd4f2d6152bab6` | 142 | Enterprise JavaScript web framework; prototypical inheritance (`mixin(app, proto)`), middleware execution pipelines (`app.use`), response content negotiation, and lazy router creation. |
| **`fmtlib/fmt`** | [fmtlib/fmt](https://github.com/fmtlib/fmt) | C++ | C, Python | `3be6709f575a609620f5dd82a9ab2fa91bef6a5e` | 28 | Modern C++ format library; template metaprogramming, compile-time string validation (`FMT_COMPILE`), range traits (`is_range`), ANSI terminal color formatting, and OS error abstraction. |

---

## 2. Loader Indexability Audit

The hardened validation suite imports `repo_loader.get_source_files(repo_dir)` to audit all files referenced in `ground_truth.json`:
- **`pallets/flask`**: 83 files indexed by loader. All annotated files (`src/flask/app.py`, `src/flask/sansio/app.py`, `src/flask/sansio/scaffold.py`, `src/flask/sansio/blueprints.py`, `src/flask/ctx.py`, `src/flask/sessions.py`, `src/flask/helpers.py`, `src/flask/config.py`, `src/flask/cli.py`, `src/flask/wrappers.py`) are fully indexed.
- **`expressjs/express`**: 142 files indexed by loader. All annotated files (`lib/express.js`, `lib/application.js`, `lib/request.js`, `lib/response.js`, `lib/utils.js`, `lib/view.js`) are fully indexed.
- **`fmtlib/fmt`**: 28 modular header and utility files indexed by loader. All annotated files (`include/fmt/color.h`, `include/fmt/ranges.h`, `include/fmt/os.h`, `include/fmt/args.h`, `include/fmt/printf.h`, `include/fmt/chrono.h`, `include/fmt/compile.h`, `include/fmt/enum.h`) are $\le 100\text{ KB}$ and match the extension whitelist.
- **Audit Result**: **0 files caught/rejected**. 100% of ground-truth evidence files are indexable by the V0 baseline loader.

---

## 3. Question Categories & Query Styles

The 60 questions are distributed evenly across categories and query styles:

| Category | `pallets/flask` | `expressjs/express` | `fmtlib/fmt` | Total Questions | Focus |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`function_identification`** | 4 | 4 | 4 | **12** | Definitions, constructor signatures, inheritance hierarchies. |
| **`implementation_understanding`** | 4 | 4 | 4 | **12** | Core algorithms, data conversions, internal mechanics. |
| **`cross_file_dependencies`** | 3 | 4 | 3 | **10** | Inter-module couplings, delegated registrations, protocol bindings. |
| **`architecture_flow`** | 4 | 3 | 3 | **10** | WSGI/HTTP request dispatch, exception handling, and render pipelines. |
| **`debugging_configuration`** | 3 | 3 | 4 | **10** | Cookie flags, CLI wrappers, proxy trust, compile-time macros. |
| **`unanswerable`** | 2 | 2 | 2 | **6** | Queries regarding absent/external features to test hallucination resistance. |
| **Total** | **20** | **20** | **20** | **60** | |

### Query Style Distribution
- **`conceptual`**: **20 questions** (33.3%) — behavior/intent-based without naming internal symbols.
- **`identifier`**: **22 questions** (36.7%) — targeted questions naming classes or functions without file paths.
- **`conversational`**: **12 questions** (20.0%) — context-dependent follow-up questions.
- **`unanswerable`**: **6 questions** (10.0%) — absent features with detailed justifications.
- **Leaked File Paths**: **0 of 60 (0.0%)** — 100% free of leaked directory paths or filenames in question text.

---

## 4. Key Review Corrections Before / After Examples

### A. Turn-2 Conversational Rewrites (Context-Dependent)
All 12 Turn-2 questions were rewritten so they naturally depend on Turn 1 via anaphora ("it", "that method", "those settings", "that stream") and cannot be cleanly retrieved without Turn-1 context:

1. **Flask (`flask_conv_01`)**:
   - *Before*: "How does the Flask application instantiate and configure its default configuration settings during initialization?"
   - *After*: "How does it configure default configuration settings during that initialization process?"
   - *Expected Standalone Query*: "How does the Flask application initialize and configure its default configuration settings during initialization?"
2. **Express (`express_conv_02`)**:
   - *Before*: "How does the router getter in lib/application.js lazily initialize the Router instance with application settings?" (previously identical to expected_standalone_query)
   - *After*: "How does that instance lazily initialize its underlying router with case-sensitive and strict routing settings?"
   - *Expected Standalone Query*: "How does the Express application lazily initialize its router with case-sensitive and strict routing settings?"
3. **fmt (`fmt_conv_01`)**:
   - *Before*: "How does the styled helper function in include/fmt/color.h bind an argument value to a text_style for formatted output?" (previously identical to expected_standalone_query)
   - *After*: "How does the styled helper function apply that style to an argument value for formatted output?"
   - *Expected Standalone Query*: "How does the styled helper function apply a text_style to an argument value for formatted output in fmt?"

### B. Multi-File Evidence Additions (Cross-File Spans)
Expanded from 1 span per question to 17 multi-evidence questions (71 total spans across files):

1. **Express ETag Coordination (`express_009`)**:
   - *Before*: Only `lib/response.js:126-190`
   - *After*:
     - Span 1: `lib/response.js:126-190` (`res.send` response body formatting and ETag assignment)
     - Span 2: `lib/utils.js:130-152` (`exports.compileETag` ETag function compiler)
2. **Express View Rendering (`express_011`)**:
   - *Before*: Only `lib/application.js:522-575`
   - *After*:
     - Span 1: `lib/application.js:522-575` (`app.render` view lookup and options merging)
     - Span 2: `lib/view.js:133-165` (`View.prototype.render` engine dispatch and file execution)
3. **Flask Blueprint Registration (`flask_011`)**:
   - *Before*: Only `src/flask/sansio/blueprints.py:273-315`
   - *After*:
     - Span 1: `src/flask/sansio/blueprints.py:273-315` (`Blueprint.register` deferred function replay)
     - Span 2: `src/flask/sansio/app.py:571-605` (`App.register_blueprint` application attachment)

### C. File-Path-Stripped Conceptual Questions
Eliminated file paths from all answerable questions and rephrased queries to describe intent/behavior:

1. **fmt RAII File Handles (`fmt_003`)**:
   - *Before*: "Where is the buffered_file class defined in include/fmt/os.h and how does it manage C file pointers with RAII?"
   - *After*: "How does the format library provide RAII management for C-style file streams?"
2. **Express Static File Streaming (`express_008`)**:
   - *Before*: "How does res.sendFile validate paths, check options, and stream files to the client?"
   - *After*: "How does Express stream static files from disk to the client while handling range headers and transmission errors?"
3. **Flask Thread Context Isolation (`flask_008`)**:
   - *Before*: "How does Flask create and manage the application context (AppContext) and push it onto the context stack?"
   - *After*: "How does the framework isolate request and application state across threads and push contexts onto the active stack?"

---

## 5. Terminal Output: Hardened Validation Suite

Execution of `python experiments/datasets/validate_dataset.py`:

```text
===========================================================================
           HARDENED EVALUATION BENCHMARK VALIDATION REPORT
===========================================================================
Total Questions Evaluated       : 60
Total Ground Truth Annotations  : 60
Answerable Questions            : 54
Unanswerable Questions          : 6
---------------------------------------------------------------------------
Query Style Breakdown:
  - conversational                 : 12 questions
  - unanswerable                   : 6 questions
  - conceptual                     : 20 questions
  - identifier                     : 22 questions
---------------------------------------------------------------------------
Multi-Turn Conversational Integrity:
  - Total Conversations           : 12
  - Turn-2 Follow-up Questions    : 12 (target: 10-15)
  - Context-Dependent Rewrites    : ALL 12 distinct from expected_standalone_query
---------------------------------------------------------------------------
Evidence & Multi-File Metrics:
  - Total Evidence Spans          : 71
  - Multi-Evidence Questions (>=2): 17 (target: >= 8-10)
  - Questions Leaking File Paths  : 0 of 60 (0.0%)
---------------------------------------------------------------------------
Loader Indexability & Ground Truth Alignment:
  - Loader Indexability Passed    : 100% of evidence files recognized by repo_loader
  - Zero Chunk IDs Verified       : 100% compliant (NO chunk IDs)
  - File Existence Verified       : 100% compliant
  - Line Range Validity Verified  : 100% compliant (1 <= start <= end <= max)
  - Symbol Presence Verified      : 100% compliant
===========================================================================

VALIDATION STATUS: PASSED (100% OF CHECKS SUCCESSFUL)
```

---

## 6. Confirmation of Constraints

1. **Application Code Untouched**:
   - `app.py`, `chunker.py`, `embedder.py`, `llm.py`, `repo_loader.py`, and `telemetry.py` were not altered.
2. **Instrumentation Tests Passing**:
   - All 7 tests in `test_baseline_instrumentation.py` pass without regression.
3. **No Baseline Benchmark Run**:
   - The Day 1 Task 4 baseline evaluation has **not** been run.
