# DAY 4 — HYBRID RETRIEVAL (BM25 + DENSE + RRF) REPORT (`DAY4_HYBRID_REPORT.md`)

**Project:** CodeInsight AI Assistant (GitHub Codebase RAG)  
**Benchmark Script:** [run_hybrid_comparison.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day4_hybrid/run_hybrid_comparison.py)  
**Summary Artifact:** [hybrid_summary.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day4_hybrid/results/hybrid_summary.json)  
**Production Modules:** [retrieval/bm25_retriever.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/retrieval/bm25_retriever.py), [retrieval/hybrid_retriever.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/retrieval/hybrid_retriever.py), [embedder.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/embedder.py), [app.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/app.py)  

---

## 1. Executive Summary

Building cleanly on Day 3's winning **`code_aware_v1`** chunking strategy (where `Hit@5 = 0.7037` and `0.00%` runtime truncation were achieved with `all-MiniLM-L6-v2`), Day 4 evaluated **Code-Aware Okapi BM25** (`D4-E1`) and **Hybrid Dense + BM25 Reciprocal Rank Fusion (`rrf_k = 60`)** (`D4-E2`) against the **Dense V1 baseline** (`D4-E0`) under single-pass indexing and strict corpus parity (`3,721` V1 chunks across 253 files / 3 pinned repositories, `0` failed queries).

**Key Conclusions:**
1. **`D4-E2_hybrid_rrf` outperformed both Dense-only and BM25-only across all Top-5 retrieval metrics:**
   - **Hit@1:** `0.3704 -> 0.3889` (`+0.0185`)
   - **Hit@5:** `0.7037 -> 0.7222` (`+0.0185`, `39/54` answerable queries solved)
   - **Precision@5:** `0.2185 -> 0.2444` (`+0.0259`, `+11.9%` relative)
   - **Span Recall@5:** `0.6111 -> 0.6389` (`+0.0278`)
   - **MRR@5:** `0.5052 -> 0.5182` (`+0.0130`)
   - **NDCG@5:** `0.3217 -> 0.3737` (`+0.0520`, `+16.2%` relative)
   - **Overhead:** Hybrid P50 query latency added only **`+3.48 ms`** (`36.08 ms` Dense -> `39.56 ms` Hybrid; BM25 P50 `2.62 ms`, RRF `0.19 ms`).
2. **Candidate Generation vs. Ranking Finding (Day 5 Reranker Ceiling):**
   - `Dense@20` achieves **`0.8889`** Hit@20 (`48/54`); `BM25@20` achieves **`0.8704`** (`47/54`).
   - Their deduplicated **`Candidate Union (Dense@20 ∪ BM25@20)`** (mean `33.43`, max `40` chunks) achieves **`0.9630` Hit Rate (`52/54`)** and **`0.9074` Span Recall** — **13 of the 15 Top-5 misses ALREADY exist in the candidate pool (`12` inside RRF Top-20, `1` inside the remaining union)**, leaving only **`2/54` (`3.7%`)** true candidate-generation failures.
3. **Production Integration:** ChromaDB-hydrated BM25 and Hybrid RRF are cleanly integrated into [embedder.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/embedder.py) (`st.cache_resource` keyed by collection UUID, chunk count, and SHA-256 `corpus_fingerprint`, zero pickle persistence) and exposed in the [app.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/app.py) sidebar (`Retrieval Strategy`).

---

## 2. Corpus Representation & Code-Aware BM25 Design

### 2.1 V1 Chunk Inspection & Header Label Hygiene
Inspected actual Day 3 `code_aware_v1` chunks across Python, JavaScript, and C++ (`src/flask/app.py`, `lib/express.js`, `include/fmt/color.h`). V1 chunks already embed contextual headers directly inside `chunk["content"]`:
```text
File: src/flask/app.py | Language: python | Type: method_split
Class: Flask | Symbol: dispatch_request | Part: 2/3
Signature: def dispatch_request(self):
...source lines...
```
To prevent double-counting metadata terms and stop generic labels (`file`, `language`, `type`, `class`, `parent`, `symbol`, `part`, `signature`) from matching every corpus document:
- **Exact Corpus Representation:** BM25 indexes **`chunk["content"]` directly** without appending redundant copies of `file_path`, `symbol_name`, `class_name`, or `chunk_type`.
- **Header Label Hygiene (`strip_v1_header_labels`):** Strips ONLY the static field prefixes (`File:`, `Language:`, `Type:`, `Class:`, `Parent:`, `Symbol:`, `Part:`, `Signature:`) from the leading structured V1 header lines while preserving every field value (`src/flask/wrappers.py`, `Flask`, `dispatch_request`, `method_split`) and leaving all source code body lines untouched.

### 2.2 Code-Aware Tokenization Specification (`tokenize_code`)
Identical preprocessing is applied to corpus chunks and incoming queries:
1. **Case Normalization:** All extracted tokens are lowercased; Windows path separators `\` are normalized to `/`.
2. **Compound Preservation & Punctuation Handling:** Matches compound tokens `[A-Za-z0-9_]+(?:(?:::|\.|/|\\|-)[A-Za-z0-9_]+)*`, discarding standalone punctuation while preserving dotted chains (`res.cookie`, `app.engine`, `config.from_object`), C++ scoped names (`fmt::format`), and relative paths (`include/fmt/os.h`).
3. **Path & Identifier Splitting (Deduplicated per Compound Occurrence):** Splits `/` path segments cleanly (preserving filenames like `os.h`), splits `::`, `.`, `-`, underscores (`_`), and camelCase/acronym boundaries:
   - `createApplication` -> `['createapplication', 'create', 'application']`
   - `handle_request` -> `['handle_request', 'handle', 'request']`
   - `FMT_STRING` -> `['fmt_string', 'fmt', 'string']`
   - `HTTPResponseHandler` -> `['httpresponsehandler', 'http', 'response', 'handler']`
   - `Config.from_object` -> `['config.from_object', 'config', 'from_object', 'from', 'object']`
   - `fmt::format` -> `['fmt::format', 'fmt', 'format']`
   - `include/fmt/os.h` -> `['include/fmt/os.h', 'include', 'fmt', 'os.h', 'os', 'h']`
4. **Short Code Tokens:** 1–2 character alphanumeric identifiers (`g`, `ip`, `os`, `fg`, `bg`) are strictly preserved (never discarded).

### 2.3 BM25 Parameters & Zero-Overlap Filtering
- **Algorithm:** Okapi BM25 (`k1 = 1.5`, `b = 0.75`, nonnegative `lucene` IDF $\ln(1 + \frac{N - df + 0.5}{df + 0.5})$; unit-tested also under `raw_rsj` where $df > N/2$ yields negative scores).
- **Eligibility Rule:** Determined **strictly by actual query-token lexical overlap (`overlap_count >= 1`)**, never by `score > 0`. Zero-overlap chunks are excluded, Top-K is never padded with non-matching chunks, and zero-match queries return `[]`, triggering graceful RRF fallback to Dense-only rankings.

---

## 3. Single-Pass Experimental Controls & Corpus Parity

| Repository | Collection | Commit SHA | Files | V1 Chunks | >256 Trunc | BM25 Vocab | AvgDocLen (tok) | Corpus Fingerprint (`sha256[:12]`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `pallets/flask` | `day4_v1_flask` | `49943b5a9441...` | `83` | `1,352` | `0` | `6,484` | `92.94` | `7d996d0cf4d1` |
| `expressjs/express` | `day4_v1_express` | `43fca52d28a4...` | `142` | `1,297` | `0` | `3,967` | `78.13` | `d92e0235ef6e` |
| `fmtlib/fmt` | `day4_v1_fmt` | `3be6709f575a...` | `28` | `1,072` | `0` | `5,791` | `78.05` | `4bae980c35f2` |

- **Storage Isolation:** Persistent Day 4 DB stored cleanly in `experiments/day4_hybrid/chroma_day4_db/` (`.gitignore`d).
- **Baseline Parity Check:** `D4-E0_dense_v1` reproduced Day 3 `code_aware_v1` with **`0.0` deltas** on all 7 metrics (`VALID`, `0` failed queries). Hydration from ChromaDB matched direct V1 indexing fingerprints identically across all 3 collections.

---

## 4. Overall Retrieval Results (`N = 54` Answerable, Final Top-5)

| Method | Hit@1 | Hit@3 | Hit@5 | Precision@5 | Span Recall@5 | MRR@5 | NDCG@5 | Query P50 (All 60q) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`D4-E0`: Dense Vector (`all-MiniLM-L6-v2`)** | `0.3704` | **`0.6296`** | `0.7037` | `0.2185` | `0.6111` | `0.5052` | `0.3217` | `36.08 ms` |
| **`D4-E1`: Code-Aware BM25 (`V1 headers`)** | **`0.3889`** | `0.6111` | `0.6852` | `0.2296` | `0.6019` | `0.5043` | `0.3689` | **`2.62 ms`** |
| **`D4-E2`: Hybrid RRF (`k=60`, Top-20 Pools)** | **`0.3889`** | **`0.6296`** | **`0.7222`** | **`0.2444`** | **`0.6389`** | **`0.5182`** | **`0.3737`** | `39.56 ms` |
| **Delta (`D4-E2` Hybrid vs `D4-E0` Dense)** | **`+0.0185`** | `0.0000` | **`+0.0185`** | **`+0.0259`** | **`+0.0278`** | **`+0.0130`** | **`+0.0520`** | `+3.48 ms` |

---

## 5. Granular Evaluation Breakdowns

### 5.1 By Repository (`18` Answerable Queries Each)

| Repository | Method | Hit@1 | Hit@3 | Hit@5 | Precision@5 | Span Recall@5 | MRR@5 | NDCG@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pallets/flask`** | Dense (`D4-E0`) | `0.3889` | **`0.6667`** | **`0.7778`** | `0.2111` | `0.6944` | **`0.5463`** | `0.3207` |
| | BM25 (`D4-E1`) | `0.2778` | `0.4444` | `0.5556` | `0.1667` | `0.5278` | `0.3741` | `0.2601` |
| | **Hybrid (`D4-E2`)** | **`0.4444`** | `0.5556` | **`0.7778`** | **`0.2333`** | **`0.7500`** | `0.5407` | **`0.3768`** |
| **`expressjs/express`** | Dense (`D4-E0`) | `0.2222` | `0.5556` | `0.6667` | `0.1556` | `0.5278` | `0.3954` | `0.2794` |
| | BM25 (`D4-E1`) | **`0.4444`** | **`0.6667`** | **`0.7778`** | **`0.2222`** | **`0.6389`** | **`0.5741`** | **`0.4477`** |
| | Hybrid (`D4-E2`) | `0.2222` | **`0.6667`** | `0.7222` | `0.1667` | `0.5833` | `0.4398` | `0.3220` |
| **`fmtlib/fmt`** | Dense (`D4-E0`) | **`0.5000`** | `0.6667` | **`0.6667`** | `0.2889` | `0.6111` | **`0.5741`** | `0.3650` |
| | BM25 (`D4-E1`) | `0.4444` | **`0.7222`** | **`0.7222`** | `0.3000` | **`0.6389`** | `0.5648` | `0.3988` |
| | **Hybrid (`D4-E2`)** | **`0.5000`** | `0.6667` | **`0.6667`** | **`0.3333`** | `0.5833` | **`0.5741`** | **`0.4223`** |

**Repository Pattern:** BM25 excels dramatically on Express (`Hit@5 = 0.7778`, `MRR = 0.5741` vs Dense `0.6667 / 0.3954`) because JS middleware/API identifier names match literally without MiniLM semantic confusion across test suites, whereas Dense dominates Flask (`0.7778` vs BM25 `0.5556`). Hybrid gets the best of Flask (`Hit@1=0.4444`, `SpanRec=0.7500`) and lifts Express (`Hit@5=0.7222`), though unweighted RRF suppresses single-source BM25 #1 Express hits that Dense missed (`express_013`, `express_016`).

### 5.2 By Query Category (`Hit@5 / MRR@5 / Span Recall@5`)

| Category (`n`) | Dense (`D4-E0`) | BM25 (`D4-E1`) | Hybrid (`D4-E2`) |
| :--- | :---: | :---: | :---: |
| `architecture_flow` (`n=10`) | **`0.8000 / 0.5667 / 0.7000`** | **`0.8000`** / `0.5283` / **`0.7000`** | `0.7000 / 0.5333 / 0.6500` |
| `cross_file_dependencies` (`n=10`) | **`0.9000 / 0.7000 / 0.6000`** | `0.7000 / 0.5200 / 0.4500` | `0.8000 / 0.4667 / 0.5000` |
| `debugging_configuration` (`n=10`) | `0.4000 / 0.4000 / 0.3500` | **`0.6000 / 0.4833 / 0.5500`** | **`0.6000`** / `0.3950` / **`0.5500`** |
| `function_identification` (`n=12`) | `0.6667 / 0.4125 / 0.6250` | `0.6667 / 0.5278 / 0.6250` | **`0.7500 / 0.6111 / 0.7083`** |
| `implementation_understanding` (`n=12`) | **`0.7500`** / `0.4722` / **`0.7500`** | `0.6667 / 0.4653 / 0.6667` | **`0.7500 / 0.5583 / 0.7500`** |

### 5.3 By Query Style & Evidence Complexity (`Hit@1 / Hit@5 / MRR@5`)

| Subgroup (`n`) | Dense (`D4-E0`) | BM25 (`D4-E1`) | Hybrid (`D4-E2`) |
| :--- | :---: | :---: | :---: |
| **Style: `identifier` (`n=22`)** | `0.4545 / 0.7727 / 0.5659` | `0.5455 / 0.8636 / 0.6970` | **`0.5909 / 0.8636 / 0.7061`** |
| **Style: `conceptual` (`n=20`)** | `0.1500 / 0.6000 / 0.3500` | **`0.2000`** / `0.5000 / 0.2992` | `0.1500 / 0.6000 / 0.3058` |
| **Style: `conversational` (`n=12`)** | **`0.5833 / 0.7500 / 0.6528`** | `0.4167 / 0.6667 / 0.4930` | `0.4167 / 0.6667 / 0.5278` |
| **Evidence: Single-Span (`n=37`)** | `0.2703 / 0.6216 / 0.4041` | **`0.3784`** / `0.6486 / 0.4851` | `0.3514` / **`0.7027 / 0.4950`** |
| **Evidence: Multi-Span (`n=17`)** | **`0.5882 / 0.8824 / 0.7255`** | `0.4118 / 0.7647 / 0.5461` | `0.4706 / 0.7647 / 0.5686` |

### 5.4 Answerable (`n=54`) vs. Unanswerable (`n=6`) Score Distributions

Each metric operates on its own incompatible scale (squared L2 distance: lower is closer; BM25 & RRF: higher is more relevant) and cannot be compared numerically across retrievers or interpreted as calibrated probabilities.

| Retriever Score Scale | Cohort (`n`) | Min | Q25 | Median | Mean | Q75 | IQR | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Top-1 L2 Distance** | Answerable (`54`) | `0.3952` | `0.6540` | `0.7921` | `0.8146` | `0.9408` | `0.2868` | `1.4915` |
| | Unanswerable (`6`) | `0.7913` | `0.8760` | `1.0203` | `0.9945` | `1.0888` | `0.2128` | `1.1954` |
| **BM25 Top-1 Lexical Score** | Answerable (`54`) | `10.2289` | `19.2321` | `23.9520` | `26.4620` | `32.2183` | `12.9862` | `52.0656` |
| (`0` zero-match queries) | Unanswerable (`6`) | `10.3047` | `12.4411` | `14.8941` | `15.5567` | `18.7208` | `6.2797` | `21.6275` |
| **RRF Top-1 Fusion Score** | Answerable (`54`) | `0.01639` | `0.03129` | `0.03227` | `0.03152` | `0.03279` | `0.00150` | `0.03279` |
| | Unanswerable (`6`) | `0.02912` | `0.02936` | `0.02988` | `0.03039` | `0.03149` | `0.00213` | `0.03227` |

*Note:* Notice RRF Top-1 compresses scores tightly (`0.0291–0.0328` when both lists agree on unrelated chunks vs `0.0313–0.0328` for true hits), meaning raw RRF score cannot be used reliably for out-of-scope abstention without downstream reranker calibration.

---

## 6. Per-Query Paired Comparison (`Dense Top-5` vs. `Hybrid Top-5`, `N = 54`)

| Hit@5 Contingency Quadrant | Count | Share |
| :--- | :---: | :---: |
| **Dense Hit / Hybrid Hit** | `34` | `63.0%` |
| **Dense Miss / Hybrid Hit (Hybrid Gained)** | **`5`** | `9.3%` |
| **Dense Hit / Hybrid Miss (Hybrid Lost)** | **`4`** | `7.4%` |
| **Dense Miss / Hybrid Miss** | `11` | `20.4%` |

- **Composite MRR@5 / Span Recall Transitions:** **`14` improved**, **`10` regressed**, **`30` unchanged**.
- **Exploratory Paired Bootstrap 95% CI (`seed=42`, `B=10,000`):**
  - `Δ Hit@5`: **`+0.0185`** (`95% CI: [-0.0926, +0.1296]`)
  - `Δ MRR@5`: **`+0.0130`** (`95% CI: [-0.0725, +0.0975]`)
  - Because `N = 54` is modest and both 95% intervals span zero, these are reported as **descriptive empirical improvements** rather than claiming classical statistical significance.
- **5 Hybrid Hit@5 Gains:** `flask_005` (`route` -> Rank #5), `flask_017` (`cli.py` -> Rank #5), `flask_018` (`get_flashed_messages` -> Rank #4), `express_017` (`res.cookie` -> **Rank #2**), `fmt_014` (`compile.h` -> **Rank #2**). Major within-top-5 rank lifts include `flask_001` (#4->#1), `flask_010` (#4->#1), `flask_007` (#2->#1), `flask_012` (#2->#1), `express_005` (#2->#1).
- **4 Hybrid Hit@5 Losses:** `flask_009` (Dense #3 -> RRF #10, BM25 missed Top-20), `flask_011` (Dense #2 / BM25 #5 -> RRF #8), `flask_014` (Dense #1 / BM25 #10 -> RRF #6), `fmt_016` (Dense #2 -> RRF #8, BM25 missed Top-20). Every single regression remains inside RRF Top-10.

---

## 7. Candidate Generation vs. Ranking — Day 5 Handoff

Measuring candidate-pool coverage cleanly (full deduplicated union `Dense@20 ∪ BM25@20`, never truncated at 20, alongside `RRF Top-20` and `RRF Top-5`) isolates retrieval coverage from ranking bottlenecks:

| Pipeline Stage | Hit Rate | Span Recall | Count of Queries Solved (`n=54`) |
| :--- | :---: | :---: | :---: |
| **`Dense@20` Candidate List** | `0.8889` | `0.8056` | `48 / 54` |
| **`BM25@20` Candidate List** | `0.8704` | `0.7870` | `47 / 54` |
| **`Candidate Union (Dense@20 ∪ BM25@20)`** | **`0.9630`** | **`0.9074`** | **`52 / 54`** |
| **`RRF@20` Re-ordered Candidate List** | `0.9444` | `0.8611` | `51 / 54` |
| **`Final RRF@5` Delivered Context** | `0.7222` | `0.6389` | `39 / 54` |

- **Candidate Pool Size Distribution (`Dense@20 ∪ BM25@20`):** `min = 28`, `q25 = 31.0`, **`median = 33.5`**, **`mean = 33.43`**, `q75 = 36.0`, `max = 40` (`overlap mean = 6.57`, `median = 6.5`).

### Handoff Diagnosis (Answers to Day 5 Handoff Questions)

1. **How much relevant evidence already enters the candidate union?**
   - **`96.30%` of answerable queries (`52 / 54`)** and **`90.74%` of unique ground-truth spans** are ALREADY retrieved inside `Dense@20 ∪ BM25@20`. Dense@20 + BM25@20 are strongly complementary (`+7.4%` Hit Rate jump over Dense@20 alone).
2. **How much relevant evidence is lost due to final Top-5 ranking?**
   - **`24.07` percentage points of Hit Rate (`52 -> 39`, i.e., `13` queries)** and **`26.85` points of Span Recall (`0.9074 -> 0.6389`)** are lost purely during the Stage-2 cut from Union/Top-20 down to Final Top-5.
3. **Which queries have relevant candidates in the pool yet poor RRF ordering?**
   - **`12` queries sit inside `RRF Top-20` (Ranks #6–#19):**
     - *Strong BM25 hits demoted by single-source Dense miss:* `express_013` (`BM25 #1 -> RRF #6`), `express_016` (`BM25 #1, Dense #11 -> RRF #6`), `express_015` (`BM25 #3 -> RRF #6`), `fmt_001` (`BM25 #1 -> RRF #13`), `fmt_005` (`BM25 #2, Dense #17 -> RRF #10`).
     - *Strong Dense hits demoted by BM25 noise/miss:* `flask_014` (`Dense #1 -> RRF #6`), `flask_016`/`fmt_016` (`Dense #2 -> RRF #8`), `flask_011` (`Dense #2, BM25 #5 -> RRF #8`), `flask_009` (`Dense #3 -> RRF #10`), `flask_002` (`Dense #10 -> RRF #19`).
     - *Both middle-ranked:* `express_010` (`Dense #14, BM25 #4 -> RRF #7`), `fmt_004` (`Dense #7, BM25 #8 -> RRF #14`).
   - **`1` query sits inside `Candidate Union` beyond RRF Top-20:** `fmt_003` (`Dense #12`, `BM25 = null` due to `"file"` matching license/std headers -> `RRF #23`).
4. **Which queries have NO relevant candidates at all (`Candidate Generation Failures`)?**
   - **Only `2 / 54` queries (`3.70%`):** `express_018` (*"register custom template engines to render non-standard file extensions"* — `lib/application.js:app.engine` beaten out of Top-20 by `app.route` / `examples/`) and `fmt_013` (*"parse and format time durations with unit suffixes like seconds and milliseconds"* — `chrono.h` `get_milliseconds` filled Top-20 instead of unit suffix formatter).
5. **What is the Day 5 Reranker Ceiling?**
   - Reranking the full `Candidate Union` (~33 chunks/query) has a **strict upper bound of `Hit@5 <= 0.9630` (`52/54`)** and `Span Recall@5 <= 0.9074` (or **`0.9444` / `51/54`** if reranking `RRF Top-20`). A cross-encoder reranker **cannot** recover the 2 candidate-generation failures (`express_018`, `fmt_013`), which require query expansion/rewriting or domain vocabulary bridging rather than reranking. **Day 5 Recommendation:** Evaluate Cross-Encoder Reranking directly on the **exact Day 4 `Candidate Union` (`<=40` chunks)** (with an ablation vs `RRF Top-20`) so `fmt_003` (`RRF #23`) remains recoverable and the full `96.30%` ceiling is accessible!

---

## 8. Latency & ChromaDB-Hydrated Persistence

- **Query Latency (`N = 60`, Warm-up `0.2823s` excluded):**
  - Dense Embedding: `mean = 33.11 ms`, `P50 = 31.16 ms`, `P95 = 37.95 ms`
  - Chroma Vector Search (`top-20`): `mean = 5.29 ms`, `P50 = 5.03 ms`, `P95 = 6.31 ms`
  - BM25 Lexical Search (`top-20`): `mean = 3.09 ms`, `P50 = 2.62 ms`, `P95 = 5.99 ms`
  - RRF Fusion (`k=60`): `mean = 0.19 ms`, `P50 = 0.19 ms`, `P95 = 0.28 ms`
  - **Hybrid Total Query Latency:** `mean = 41.68 ms`, **`P50 = 39.56 ms`**, `P95 = 48.52 ms`
- **Index Build Overhead (`code_aware_v1`):** BM25 hydration + inverted-index build adds only `0.51s–0.82s` per repository (`< 0.9s` total on top of ~95–130s embedding), Peak RSS = `770.16 MB`.
- **Streamlit Hydration Persistence ([embedder.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/embedder.py)):** Hydrates BM25 directly from persisted `./chroma_db` collections (`ids`, `documents`, `metadatas`, zero pickle files), cached via `@st.cache_resource` keyed by `safe_repo_name|collection.id|collection.count()|corpus_fingerprint`. Tested across simulated app restarts, repo switches, and re-index invalidation.
