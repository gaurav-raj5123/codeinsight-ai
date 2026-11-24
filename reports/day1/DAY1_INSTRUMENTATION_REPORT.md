# DAY 1 — TASK 2: BASELINE INSTRUMENTATION REPORT (FINAL)
**Project:** Codebase AI Assistant (RAG)  
**Phase:** Day 1 — Baseline Establishment  
**Report File:** `DAY1_INSTRUMENTATION_REPORT.md`  
**Execution Date:** November 2026  
**Status:** All Minor Corrections Applied & Validated  

---

## 1. Executive Summary

Following review feedback on Task 2, final targeted corrections were applied to the baseline instrumentation of the **Codebase AI Assistant**:
1. **Duplicate Conversation History Resolved:** `app.py` passes `st.session_state.messages[:-1]` to `llm.build_prompt()`, ensuring the current user prompt appears exclusively in the `QUESTION` section and is not duplicated in `CONVERSATION HISTORY`.
2. **Gemini Client Initialization Resilience:** `genai.Client(api_key=api_key)` was moved inside the existing `try` block in `ask_gemini()`, guaranteeing that any client initialization failure is caught, sanitized, recorded in generation metadata, and returned in the standard backward-compatible error format.
3. **Genuine Results vs. Synthetic Fixtures Disambiguated:** Synthetic/mocked example data previously logged during test development has been relocated to `experiments/results/baseline_run_example.json`. The primary experiment results file `experiments/results/baseline_run.json` is initialized as empty (`[]`) and strictly reserved for genuine application benchmark runs.
4. **Accurate Indexing Measurement:** Explicitly separated the timer around the actual `collection.add()` calls (`collection_add_duration_seconds`, measuring combined ONNX embedding generation and HNSW insertion) from the overall `index_chunks()` function execution timer (`total_index_function_duration_seconds`).
5. **Preserved Repository Loading Breakdown:** All three loading intervals from `repo_loader.py`—`download_duration_seconds`, `extraction_duration_seconds`, and `total_loading_duration_seconds`—are preserved as distinct fields in telemetry records.
6. **Safe Atomic Persistence & File Protection:** `telemetry.py` saves records via atomic temporary files (`tempfile.mkstemp` + `os.replace`). If an existing results file is corrupted, a `RuntimeError` is raised and the original file is preserved intact.
7. **Algorithmic Invariance Maintained:** No changes were made to chunking, embeddings (`all-MiniLM-L6-v2`), distance metric (`l2`), retrieval ranking, prompts, or Gemini generation configurations.

---

## 2. Modified Files & Exact Review Corrections

| File | Correction Applied |
| :--- | :--- |
| [app.py](file:///c:/Users/gaura/Downloads/RAG%20PROJECT/app.py) | 1. Passed `chat_history=st.session_state.messages[:-1]` to `llm.build_prompt()` so the current prompt appears only in the `QUESTION` block.<br>2. Updated sidebar indexing expander and query expander to display distinct loading and indexing intervals. |
| [llm.py](file:///c:/Users/gaura/Downloads/RAG%20PROJECT/llm.py) | 1. Moved `client = genai.Client(api_key=api_key)` inside the `try` block in `ask_gemini()` to catch client-initialization errors.<br>2. Broadened the API key redaction pattern (`r'AIza[0-9A-Za-z-_]+'`). |
| [embedder.py](file:///c:/Users/gaura/Downloads/RAG%20PROJECT/embedder.py) | Added an isolated timer strictly wrapping the batch `collection.add()` calls (`collection_add_duration_seconds`), while maintaining `total_index_function_duration_seconds` around the entire `index_chunks()` invocation. |
| [telemetry.py](file:///c:/Users/gaura/Downloads/RAG%20PROJECT/telemetry.py) | 1. Preserved `repo_download`, `repo_extraction`, and `repo_total_loading` distinctly.<br>2. Preserved `chromadb_add_combined` and `total_index_function` distinctly.<br>3. Implemented safe atomic persistence (`tempfile.mkstemp` + `os.fsync` + `os.replace`).<br>4. Prevented silent resets by raising `RuntimeError` on invalid/corrupted JSON, preserving the original file. |
| [baseline_run.json](file:///c:/Users/gaura/Downloads/RAG%20PROJECT/experiments/results/baseline_run.json) | Cleaned and initialized to `[]`, reserved exclusively for genuine application benchmark runs. |
| [baseline_run_example.json](file:///c:/Users/gaura/Downloads/RAG%20PROJECT/experiments/results/baseline_run_example.json) *(New)* | Dedicated fixture containing synthetic sample data for schema reference. |
| [test_baseline_instrumentation.py](file:///c:/Users/gaura/Downloads/RAG%20PROJECT/test_baseline_instrumentation.py) | Added test for conversation history exclusion (`test_conversation_history_excludes_current_question`) and client initialization failure, alongside mocked repository download and Gemini generation tests. |

---

## 3. Status of Baseline Benchmark Measurements

> **Explicit Confirmation:**  
> Genuine V0 baseline benchmark measurements have **not** been collected yet.  
> The baseline instrumentation, telemetry logging, and test fixtures are fully implemented and verified via automated smoke tests.  
> Synthetic/fixture data used during verification has been relocated to [experiments/results/baseline_run_example.json](file:///c:/Users/gaura/Downloads/RAG%20PROJECT/experiments/results/baseline_run_example.json).  
> The primary experiment log [experiments/results/baseline_run.json](file:///c:/Users/gaura/Downloads/RAG%20PROJECT/experiments/results/baseline_run.json) is initialized and reserved exclusively for genuine application benchmark runs (scheduled for Day 1 Task 3 / evaluation tasks).

### Reference Schema (`experiments/results/baseline_run_example.json`)

```json
[
  {
    "run_id": "run_20261124_113729_96630f",
    "timestamp": "2026-11-24T11:37:29.896667+00:00",
    "repository": {
      "url": "https://github.com/pallets/click",
      "owner": "pallets",
      "name": "click",
      "commit_sha": "7fd1a60",
      "files_indexed": 10,
      "chunks_indexed": 45,
      "loading_time_seconds": {
        "download": 0.55,
        "extraction": 0.3,
        "total_loading": 0.85
      },
      "indexing_time_seconds": {
        "repo_download": 0.55,
        "repo_extraction": 0.3,
        "repo_total_loading": 0.85,
        "chunking": 0.15,
        "chromadb_add_combined": 1.2,
        "total_index_function": 1.45,
        "total_indexing": 2.45
      },
      "embedding": {
        "model": "DefaultEmbeddingFunction (all-MiniLM-L6-v2)",
        "dimensions": 384,
        "distance_metric": "l2"
      }
    },
    "queries": [
      {
        "query_id": "query_113729_fbe4",
        "timestamp": "2026-11-24T11:37:29.896782+00:00",
        "query_text": "Where is cli defined?",
        "repository": "click",
        "retrieval": {
          "top_k_requested": 5,
          "actual_results_returned": 1,
          "total_retrieval_latency_seconds": 0.024,
          "distance_metric": "l2",
          "retrieved_chunks": [
            {
              "id": "chunk_0",
              "distance": 0.3214,
              "file_path": "click/core.py",
              "start_line": 1,
              "end_line": 10,
              "chunk_type": "FunctionDef",
              "content": "def cli(): pass"
            }
          ]
        },
        "generation": {
          "model": "gemini-3.5-flash",
          "latency_seconds": 1.1,
          "success": true,
          "error": null,
          "prompt_length_chars": 250,
          "response_length_chars": 120
        },
        "total_turn_latency_seconds": 1.124
      }
    ]
  }
]
```

---

## 4. Test Execution & Reporting Distinction

### Actual Test Suite Output
```text
======================================================================
RUNNING DAY 1 TASK 2 REVIEW CORRECTIONS SMOKE TESTS
======================================================================
Testing 0: Chunker functionality (Executed)...
  [PASS] Chunker boundary extraction and metadata verified.
Testing 1: repo_loader mocked download & commit SHA extraction (Executed)...
  [PASS] repo_loader mocked ZIP download, SHA resolution, and timing separation verified.
Testing 2: embedder indexing timing separation & retrieval instrumentation (Executed)...
  [NOTE] Retrieval consistency confirmed on test collection. Note: Full empirical ranking invariance across diverse repositories is an evaluation question to be verified in offline benchmarking, not fully proven by this smoke test alone.
  [PASS] embedder separate collection.add() timer, retrieval IDs, distances, and L2 metric verified.
Testing 3: llm.ask_gemini mocked generation & secret sanitization (Executed)...
  [PASS] llm.ask_gemini mocked success, client init failure, and secret sanitization verified.
Testing 4: telemetry module atomic persistence & corruption protection (Executed)...
  [PASS] telemetry separate timings, atomic save, and corruption protection verified.
Testing 5: Streamlit app static import & compilation (Executed)...
  [NOTE] app.py compiles cleanly and all session state variables & module imports initialize. Note: Complete interactive user workflow requires running 'streamlit run app.py' in a browser.
  [PASS] app.py compilation and import verified.
Testing 6: Conversation history excludes current question (Executed)...
  [PASS] Verified current question is excluded from conversation history and appears only in QUESTION.
======================================================================
ALL SMOKE TESTS EXECUTED AND PASSED SUCCESSFULLY!
======================================================================
```

### Distinction Between Executed Tests and Inspected Behaviors

| Test / Verification Area | Method | Scope / Limits |
| :--- | :--- | :--- |
| **Conversation History Exclusion** | **Executed** via `test_conversation_history_excludes_current_question()` | Verifies that passing `messages[:-1]` omits the current query from `CONVERSATION HISTORY` while ensuring it appears in `QUESTION`. |
| **Mocked Client Init Failure & Generation** | **Executed** via `unittest.mock` | Validates that client initialization errors occurring inside `try` are caught, sanitized, and returned in the standard error format. |
| **Mocked Repo Ingestion & SHA Extraction** | **Executed** via `unittest.mock` | Simulates in-memory ZIP archive; verifies regex commit SHA extraction and timing capture without hitting GitHub. |
| **Indexing Timing Separation** | **Executed** against ChromaDB test instance | Verifies `collection.add()` timer is isolated and `total_index_function_duration_seconds >= collection_add_duration_seconds`. |
| **Retrieval Chunk IDs, Distances, L2 Metric** | **Executed** against local ChromaDB test instance | Verifies `'id'`, `'distance'`, and `"l2"` metric are present in results. |
| **Retrieval Ranking Invariance** | **Inspected & Documented** | Retrieval code uses identical `collection.query()` without modifying scoring or sorting. Full empirical ranking invariance across diverse codebases will be benchmarked during Day 1 evaluation tasks. |
| **Atomic Persistence & File Corruption Safety** | **Executed** with temporary files | Verifies atomic replace via `tempfile` and asserts that corrupted files trigger `RuntimeError` and are preserved without modification. |
| **Streamlit UI Workflow** | **Executed (Compilation & Import Only)** | Script imports and compiles without syntax or symbol errors. Complete interactive UI workflow requires browser execution (`streamlit run app.py`). |

---

## 5. Algorithmic Invariance Confirmation

I confirm that:
- Chunking logic, AST parsing, and regex splitting remain 100% unchanged.
- Embedding model remains ChromaDB's ONNX `all-MiniLM-L6-v2` (`DefaultEmbeddingFunction`).
- Vector distance metric remains ChromaDB's default squared Euclidean distance (`"l2"`).
- Top-$K$ retrieval ranking and query logic remain unchanged.
- Gemini prompts, model (`gemini-3.5-flash`), and generation configurations remain unchanged.
