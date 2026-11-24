# DAY 1 — FINAL REPORT: V0 BASELINE BENCHMARK EVALUATION

## Executive Summary

- **Status**: Completed, Corrected & 100% Reproducible
- **Artifacts Generated & Verified**:
  - Evaluation Harness CLI: [`experiments/run_baseline_evaluation.py`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/run_baseline_evaluation.py)
  - Detailed Per-Query Results (60 records): [`experiments/results/baseline_evaluation.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/results/baseline_evaluation.json)
  - Aggregated Baseline Summary: [`experiments/results/baseline_summary.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/results/baseline_summary.json)
  - Task 4 Corrections Test Suite: [`test_task4_corrections.py`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/test_task4_corrections.py)
- **Primary Finding**:
  - The original V0 dense retrieval system achieves an overall **Hit Rate@1 of 31.5%** (`0.3148`), **Hit Rate@5 of 66.7%** (`0.6667`), **Precision@5 of 0.1815**, **MRR@5 of 0.4534**, **Span Recall@5 of 55.6%** (`0.5556`), and **NDCG@5 of 0.3339** (corrected using total indexed repository relevant chunks) across the 54 answerable evaluation queries.
  - **Latency Distribution**:
    - **Answerable Queries ($N=54$)**: Mean = **187.62 ms**, Median (P50) = **187.20 ms**, P95 = **212.50 ms** (max: 230.50 ms).
    - **All Queries ($N=60$)**: Mean = **187.80 ms**, Median (P50) = **187.60 ms**, P95 = **212.50 ms** (max: 230.50 ms).
    - **P95 Convention**: Nearest rank index $\lfloor 0.95 \times N \rfloor$ (rank 51 for $N=54$, rank 57 for $N=60$).
  - **Failure Accounting**: Exactly 0 query failures (`failed_queries_count = 0`), with status `VALID_COMPLETED`.
  - Exactly 18 queries missed completely in Top-5 ($\text{Hit@5} = 0$), identifying four distinct failure modes (test file pollution, conversational anaphora, conceptual query mismatch, and sliding-window chunk truncation) that directly motivate our Day 2–5 roadmap.

---

## 1. Experimental Setup & Pinned Repository Controls

All experiments were executed against exact pinned commit snapshots in [`experiments/datasets/cache/`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/datasets/cache) loaded via the single source of truth [`experiments/datasets/repositories.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/datasets/repositories.json):

| Repository | Primary Language | Pinned Commit SHA | Verification Evidence | Dedicated Isolated ChromaDB Collection |
| :--- | :--- | :--- | :--- | :--- |
| **`pallets/flask`** | Python | `49943b5a94413a530761350f04e896f76e8d620e` | `cache/flask/.commit_sha` matches | `v0_eval_pallets_flask` |
| **`expressjs/express`** | JavaScript | `43fca52d28a422c8be48fd489cbd4f2d6152bab6` | `cache/express/.commit_sha` matches | `v0_eval_expressjs_express` |
| **`fmtlib/fmt`** | C++ | `3be6709f575a609620f5dd82a9ab2fa91bef6a5e` | `cache/fmt/.commit_sha` matches | `v0_eval_fmtlib_fmt` |

### Pinned Snapshot Verification Mechanism
- **Verification Method**: An independent verification step in both [`validate_dataset.py`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/datasets/validate_dataset.py) and [`run_baseline_evaluation.py`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/run_baseline_evaluation.py) reads the `.commit_sha` marker stored within each cached repository root and compares it case-insensitively against the `commit_sha` field declared in `repositories.json`.
- **Integrity Guarantee**: If any `.commit_sha` marker is absent or differs from the declared hash, preflight checks halt execution immediately with a `ValueError`. All 3 cached snapshots have passed this check with 100% compliance.

### Failure Accounting Mechanism
- **Error Detection**: Retrieval calls to `embedder.search_similar()` are monitored for both unhandled exceptions and non-empty error messages returned in the retrieval metadata (`ret_meta.get("error")`).
- **Accounting Rule**: If either condition occurs:
  1. `failed_queries` is incremented.
  2. The query is flagged as `failed: True`.
  3. All ranking and retrieval metrics for that query are set strictly to `0.0`.
  4. The overall benchmark status in `baseline_summary.json` is set to `INVALID_FAILED`, and the runner exits with code 1.
- **Baseline Result**: In this benchmark run, all 60 queries completed with zero exceptions and zero metadata errors (`failed_queries = 0`, status: `VALID_COMPLETED`).

### Environment & Reproducibility Metadata
- **Run ID**: `v0_baseline_20261124_143010`
- **Python / OS**: Python 3.14.7 on Windows 11
- **Key Dependencies**: `chromadb==1.5.9`, `onnxruntime==1.29.0`, `streamlit==1.61.1`
- **Embedding Model**: ChromaDB `DefaultEmbeddingFunction` (`all-MiniLM-L6-v2`, 384 dimensions)
- **Distance Metric**: Default squared Euclidean distance ($L_2$)
- **Retrieval Top-K**: $K = 5$
- **V0 Chunker Parameters**: Max chunk lines: 100, sliding-window sub-chunk: 50 lines with 10-line overlap, Python AST parser with regex and sliding-window fallbacks.
- **Strict Controls**: Zero Gemini generation calls during retrieval benchmarking; raw question strings submitted as-is (no standalone query substitution for Turn-2 queries).

---

## 2. V0 Indexing Performance & Database Statistics

Each repository was ingested and indexed from its pinned snapshot using the unmodified V0 pipeline (`repo_loader.get_source_files()` $\to$ `chunker.chunk_code()` $\to$ `embedder.index_chunks()`):

| Repository | Source Files Indexed | Total Chunks Generated | File Extraction Time | Chunking Time | `collection.add()` Duration | Total Indexing Duration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pallets/flask`** | 83 | 1,785 | 0.0172 s | 0.1372 s | 124.7905 s | 124.9654 s |
| **`expressjs/express`** | 142 | 2,066 | 0.0436 s | 0.0269 s | 139.2077 s | 139.2353 s |
| **`fmtlib/fmt`** | 28 | 421 | 0.0118 s | 0.0690 s | 25.3969 s | 25.4160 s |
| **Total / Overall** | **253** | **4,272** | **0.0726 s** | **0.2331 s** | **289.3951 s** | **289.6167 s** |

### Observations:
- **Embedding Bottleneck**: ChromaDB's ONNX-based embedding generation combined with vector insertion (`collection.add()`) accounts for **>99.9%** of total indexing time (~289.4 seconds across 4,272 chunks on CPU).
- **Chunk Extraction Efficiency**: Python AST and sliding-window chunk extraction is fast (233 ms for all 253 files across the 3 repositories).

---

## 3. Overall V0 Retrieval Benchmark Metrics

Evaluated across all **54 answerable questions** using the formal interval-overlap relevance rule ($\text{chunk.file\_path} == \text{evidence.file\_path} \land \max(c_{\text{start}}, e_{\text{start}}) \le \min(c_{\text{end}}, e_{\text{end}})$):

| Metric | Score | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Hit Rate@1** | `0.3148` | 31.5% | Fraction of queries where the top-ranked retrieved chunk is relevant. |
| **Hit Rate@3** | `0.5741` | 57.4% | Fraction of queries with at least one relevant chunk in the Top-3. |
| **Hit Rate@5** | `0.6667` | 66.7% | Fraction of queries with at least one relevant chunk in the Top-5. |
| **Precision@5** | `0.1815` | — | Average fraction of relevant chunks in the Top-5 retrieved list (denominator = 5). |
| **MRR@5** | `0.4534` | — | Mean Reciprocal Rank of the first relevant chunk ($1/\text{rank}$). |
| **Span Recall@5** | `0.5556` | 55.6% | Mean fraction of unique ground-truth evidence spans matched in Top-5. |
| **NDCG@5 (Corrected)** | `0.3339` | — | Normalized Discounted Cumulative Gain against total relevant chunks in repo. |
| **Answerable Latency (N=54)** | `187.62 ms` | — | Mean: 187.62 ms, Median (P50): 187.20 ms, P95: 212.50 ms, Max: 230.50 ms. |
| **All Queries Latency (N=60)** | `187.80 ms` | — | Mean: 187.80 ms, Median (P50): 187.60 ms, P95: 212.50 ms, Max: 230.50 ms. |

### Note on NDCG@5 Correction:
In our initial run, IDCG was computed using the count of ground truth evidence spans ($1$ or $2$), which produced an artificially elevated NDCG@5 of `0.4514`. The mathematically correct formulation calculates $M$, the total number of chunks in the indexed repository that overlap any ground-truth evidence span for that query. The ideal DCG is then computed as $\sum_{i=1}^{\min(M, 5)} \frac{1}{\log_2(i + 1)}$. Under this rigorous formulation, the baseline V0 NDCG@5 is **`0.3339`**, reflecting that while an ideal system could return up to $\min(M, 5)$ relevant chunks in Top-5, V0 averages only $0.907$ relevant chunks in Top-5.

---

## 4. Detailed Performance Breakdowns

### A. Breakdown by Repository
| Repository | Answerable Queries | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | MRR@5 | Span Recall@5 | NDCG@5 | Mean Latency | Median P50 | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pallets/flask`** | 18 | `0.2778` | `0.5556` | `0.6667` | `0.4204` | `0.5556` | `0.3323` | 189.29 ms | 186.70 ms | 230.50 ms |
| **`expressjs/express`** | 18 | `0.2778` | `0.5000` | `0.6111` | `0.3981` | `0.4722` | `0.2080` | 188.38 ms | 189.25 ms | 212.50 ms |
| **`fmtlib/fmt`** | 18 | `0.3889` | `0.6667` | `0.7222` | `0.5417` | `0.6389` | `0.4615` | 185.19 ms | 186.85 ms | 205.70 ms |

*Insight*: `fmtlib/fmt` achieved the highest retrieval accuracy ($\text{Hit@5} = 72.2\%$, $\text{MRR@5} = 0.5417$, $\text{NDCG@5} = 0.4615$), largely because the indexed header files are modular with distinct symbol vocabularies. `expressjs/express` had the lowest retrieval scores ($\text{Hit@5} = 61.1\%$, $\text{SpanRec@5} = 47.2\%$, $\text{NDCG@5} = 0.2080$), degraded by widespread JavaScript unit test files intercepting search terms.

---

### B. Breakdown by Category
| Category | Queries | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | MRR@5 | Span Recall@5 | NDCG@5 | Median P50 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`cross_file_dependencies`** | 10 | `0.4000` | `1.0000` | `1.0000` | `0.6667` | `0.6500` | `0.4650` | 190.45 ms |
| **`function_identification`** | 12 | `0.2500` | `0.4167` | `0.5000` | `0.3361` | `0.5000` | `0.3765` | 188.30 ms |
| **`architecture_flow`** | 10 | `0.3000` | `0.7000` | `0.9000` | `0.5283` | `0.7000` | `0.3165` | 185.10 ms |
| **`debugging_configuration`** | 10 | `0.4000` | `0.4000` | `0.5000` | `0.4250` | `0.4500` | `0.2882` | 181.55 ms |
| **`implementation_understanding`** | 12 | `0.2500` | `0.4167` | `0.5000` | `0.3542` | `0.5000` | `0.2348` | 190.40 ms |

*Insight*: `cross_file_dependencies` achieved $\text{Hit@5} = 100\%$ because queries spanning two modules offer more semantic surface area to intersect with at least one evidence chunk. However, `Span Recall@5` was only $65.0\%$, showing that V0 rarely retrieves *both* participating files in Top-5. Conversely, `implementation_understanding` achieved the lowest NDCG@5 (`0.2348`) and Hit Rate@5 (`50.0%`), frequently distracted by test functions or docstring excerpts.

---

### C. Breakdown by Query Style
| Query Style | Queries | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | MRR@5 | Span Recall@5 | NDCG@5 | Median P50 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`identifier`** | 22 | `0.3182` | `0.4545` | `0.5909` | `0.4144` | `0.5455` | `0.3949` | 187.95 ms |
| **`conversational`** | 12 | `0.3333` | `0.7500` | `0.8333` | `0.5278` | `0.5417` | `0.3190` | 188.80 ms |
| **`conceptual`** | 20 | `0.3000` | `0.6000` | `0.6500` | `0.4517` | `0.5750` | `0.2758` | 185.00 ms |

*Insight*: Identifier queries achieved the highest NDCG@5 (`0.3949`), but suffered heavily whenever exact identifier tokens (e.g. `createApplication`, `from_object`, `res.cookie`) matched unit test definitions before the actual library source. Conceptual queries had the lowest NDCG@5 (`0.2758`) because dense embeddings frequently drift toward generic utility methods when specific keywords are absent.

---

### D. Breakdown by Evidence Complexity
| Evidence Complexity | Queries | Hit Rate@1 | Hit Rate@5 | Span Recall@5 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single-Evidence Queries** ($N_{\text{spans}} = 1$) | 37 | `0.2973` | `0.5405` | `0.5405` | `0.2992` |
| **Multi-Evidence Queries** ($N_{\text{spans}} \ge 2$) | 17 | `0.3529` | `0.9412` | `0.5882` | `0.4095` |

*Insight*: Multi-evidence questions show a stark disparity between Hit Rate@5 ($94.1\%$) and Span Recall@5 ($58.8\%$). V0 almost always finds *one* of the participating spans, but fails to retrieve the secondary connecting span (e.g. finding `res.send` in `lib/response.js` but completely missing `compileETag` in `lib/utils.js`).

---

## 5. Unanswerable vs. Answerable $L_2$ Distance Analysis

ChromaDB uses squared Euclidean ($L_2$) distance. A lower distance indicates higher vector similarity. We evaluated whether Top-1 $L_2$ distance naturally separates answerable from unanswerable queries:

| Query Set | Sample Count ($N$) | Min $L_2$ Distance | Median $L_2$ Distance | Mean $L_2$ Distance | Max $L_2$ Distance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Answerable Questions** | 54 | `0.3949` | `0.8626` | `0.8496` | `1.5378` |
| **Unanswerable Questions** | 6 | `0.8314` | `1.0685` | `1.0332` | `1.2868` |

```text
Answerable Top-1 L2:   [--- Min: 0.395 --- Median: 0.863 --- Mean: 0.850 ----------------- Max: 1.538 ---]
Unanswerable Top-1 L2:                     [--- Min: 0.831 --- Median: 1.069 - Mean: 1.033 - Max: 1.287 ---]
```

### Analysis for Day 6 Abstention Thresholding:
1. **Separation**: The mean Top-1 distance for unanswerable queries ($1.0332$) is noticeably higher than for answerable queries ($0.8496$). The median moves from $0.8626$ to $1.0685$.
2. **Substantial Overlap**: The unanswerable minimum ($0.8314$) is well below the answerable maximum ($1.5378$). In fact, 28 out of 54 answerable queries had a Top-1 $L_2$ distance $> 0.8314$.
3. **Conclusion**: Raw vector distance alone cannot serve as a reliable hard threshold for abstention. Without secondary signals (such as cross-encoder confidence, lexical term verification, or LLM-based hallucination detection scheduled for Day 6), a simple distance cutoff would trigger unacceptable false-rejection rates on valid developer queries.

---

## 6. Concrete V0 Retrieval Failure Modes

Out of 54 answerable queries, exactly **18 queries failed completely** in Top-5 ($\text{Hit@5} = 0$). Analysis of these failures reveals four primary structural weaknesses in the V0 system:

### 1. Test File Pollution & Identifier Shadowing (Motivates Day 4 Hybrid Search & Reranking)
- **`express_001`**: *"Where is createApplication defined in Express and how does it mix in EventEmitter and the application prototype?"*
  - *Expected*: `lib/express.js:36-57` and `lib/application.js:59-83`
  - *V0 Top-1 Retrieved*: `test/req.query.js:94-96` (`function createApp(setting)`)
  - *Failure Mechanism*: In pure dense embedding, test utility functions that instantiate `express()` match the semantic context of application creation better than the bare factory export in `lib/express.js`.
- **`flask_007`**: *"How does dispatch_request in Flask execute the endpoint matched from the active request URL?"*
  - *Expected*: `src/flask/app.py:993-1025`
  - *V0 Top-1 Retrieved*: `tests/test_views.py:21-22` (`def dispatch_request(self): return flask.request.method`)
  - *Failure Mechanism*: Test mock implementations shadow the primary framework method.
- **`flask_016`**: *"How does Config.from_object in Flask inspect and load configuration values from a Python module or class?"*
  - *Expected*: `src/flask/config.py:218-250`
  - *V0 Top-1 Retrieved*: `tests/test_config.py:26-29` (`def test_config_from_object():`) with a very close $L_2$ distance of $0.3949$.

### 2. Conversational Anaphora & Context Starvation (Motivates Day 5 Query Rewriting)
- **`flask_002`**: *"How does it configure default configuration settings during that initialization process?"*
  - *Expected*: `src/flask/sansio/app.py:480-505` (`make_config`)
  - *V0 Top-1 Retrieved*: `src/flask/sansio/blueprints.py:679-692` ($L_2 = 1.2181$)
  - *Failure Mechanism*: The query relies entirely on pronouns ("it", "that initialization process") grounded in Turn 1 (`flask_001`). With no context carried forward, the dense vector drifted to blueprint callbacks.
- **`fmt_002`**: *"How does the styled helper function apply that style to an argument value for formatted output?"*
  - *Expected*: `include/fmt/color.h:660-673` (`styled`)
  - *V0 Top-1 Retrieved*: `include/fmt/color.h:563-619` (`println`) ($L_2 = 1.0757$)

### 3. Lexical Disconnect in Conceptual Queries (Motivates Day 4 BM25 Lexical Hybrid)
- **`flask_018`**: *"How does Flask pass temporary one-off notification messages between redirected requests?"*
  - *Expected*: `src/flask/helpers.py:326-365` (`flash`)
  - *V0 Top-1 Retrieved*: `src/flask/sansio/app.py:940-959` (`redirect`)
  - *Failure Mechanism*: The user's query describes flash messaging behavior conceptually without using the specific keyword `flash`. The dense retriever matched `redirect()`, missing the session-backed flash queue entirely.
- **`fmt_003`**: *"How does the format library provide RAII management for C-style file streams?"*
  - *Expected*: `include/fmt/os.h:167-205` (`buffered_file`)
  - *V0 Top-1 Retrieved*: `include/fmt/ostream.h:109-130` ($L_2 = 0.9361$)

### 4. Sliding-Window Truncation on Header Files (Motivates Day 2 AST Chunking)
- **`fmt_004` & `fmt_005`**: In `fmtlib/fmt`, C++ headers often begin with large license notices or macro preambles. V0's sliding window chunker cut the top 60 lines (`// Formatting library for C++ ...`, lines 1–62) as the primary match, missing inner template struct definitions (`is_range`, `dynamic_format_arg_store`) located deeper in the file.

---

## 7. Reproduction Instructions

To reproduce the exact V0 baseline evaluation results from scratch:

```bash
# 1. Run the end-to-end V0 benchmark harness with verified records
python experiments/run_baseline_evaluation.py

# 2. Verify all Task 4 corrections (NDCG@5, failure accounting, latency, commit SHAs)
python test_task4_corrections.py

# 3. Verify dataset integrity
python experiments/datasets/validate_dataset.py

# 4. Verify instrumentation regression tests
python test_baseline_instrumentation.py
```

Outputs are persisted at:
- Detailed records: [`experiments/results/baseline_evaluation.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/results/baseline_evaluation.json)
- Aggregate metrics: [`experiments/results/baseline_summary.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/results/baseline_summary.json)

---

## 8. Confirmation of Strict Boundaries

1. **No Application Code Modification**: `app.py`, `chunker.py`, `embedder.py`, `llm.py`, `repo_loader.py`, and `telemetry.py` were **NOT modified**.
2. **Evaluation Datasets Untouched**: `repositories.json`, `evaluation_questions.json`, and `ground_truth.json` were strictly preserved.
3. **Task 4 Verification Passing**: All 4 correction suites in `test_task4_corrections.py` pass without regression.
4. **Validation Suite Passing**: `validate_dataset.py` passes with 100% compliance.
5. **Smoke Tests Passing**: All tests in `test_baseline_instrumentation.py` pass without regression.
6. **No Generative Calls**: Zero Gemini API calls were made during this retrieval-only evaluation.

The corrected and verified V0 baseline is locked and ready for Day 2 improvements.
