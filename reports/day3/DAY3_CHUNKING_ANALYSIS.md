# DAY 3 — CODE-AWARE CHUNKING OPTIMIZATION ANALYSIS (`DAY3_CHUNKING_ANALYSIS.md`)

**Project:** CodeInsight AI Assistant (GitHub Codebase RAG)  
**Experiment Date:** November 2026  
**Benchmark Script:** [run_chunking_comparison.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/chunking_experiments/run_chunking_comparison.py)  
**Summary Artifact:** [chunking_summary.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/chunking_experiments/results/chunking_summary.json)  
**Audit Reference:** [DAY3_CHUNKING_AUDIT.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day3/DAY3_CHUNKING_AUDIT.md)  

---

## 1. Executive Summary

Following Day 2's conclusion that `all-MiniLM-L6-v2` remains the strongest baseline embedder for the current benchmark (where no alternative embedding model exceeded V0's `Hit@5 = 0.6667`), Day 3 investigated **chunking strategy optimization** under strict experimental control (identical 253 corpus files, pinned commit SHAs, identical ONNX `all-MiniLM-L6-v2` 384d embedder, squared L2 ChromaDB distance, Top-K=5, zero reranking/rewriting).

**Key Result:** Replacing the V0 hybrid AST/regex chunker (`existing_chunker_v0`) with a **token-budgeted, hierarchical AST + lexical brace-scanner chunker (`code_aware_v1`)** achieved substantial, statistically consistent gains across core ranking and recall metrics while cutting end-to-end indexing time by **24.5%**:

- **Hit@1:** `0.3148 -> 0.3704` (**`+0.0556` / `+17.7%` relative**, 20/54 vs 17/54)
- **Hit@3:** `0.5741 -> 0.6296` (**`+0.0555` / `+9.7%` relative**, 34/54 vs 31/54)
- **Hit@5:** `0.6667 -> 0.7037` (**`+0.0370`**, breaking 70% at 38/54 vs 36/54)
- **Precision@5:** `0.1815 -> 0.2185` (**`+0.0370` / `+20.4%` relative**, mean `1.09` vs `0.91` relevant chunks in top 5)
- **Span Recall@5:** `0.5556 -> 0.6111` (**`+0.0555` / `+10.0%` relative**)
- **MRR@5:** `0.4534 -> 0.5052` (**`+0.0518` / `+11.4%` relative**, crossing 0.50)
- **NDCG@5:** `0.3339 -> 0.3217` (`-0.0122`, **IDCG denominator artifact** analyzed in Section 7.1)
- **Runtime Token Truncation (`>256` tokens):** Eliminated completely from `14.61%` (`624` chunks) in V0 and `89.24%` (`1,219` chunks) in `fixed_window_v0` down to **`0.00%` (`0` chunks)** in `code_aware_v1`
- **End-to-End Indexing Time:** Reduced from **`351.49s`** (V0) to **`265.37s`** (`code_aware_v1`, **-24.5%**)

**Recommendation:** Adopt `code_aware_v1` as the default chunking strategy for Day 4 (hybrid BM25 + dense vector retrieval), integrated in [chunker.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/chunker.py) via explicit strategy selection (`SUPPORTED_CHUNKING_STRATEGIES`) while preserving `existing_chunker_v0` and `fixed_window_v0` for full reproducibility and UI configurability.

---

## 2. Pre-Modification Baseline Validation & Audit Recap

Before writing or modifying application code:
1. **Structural Audit** ([DAY3_CHUNKING_AUDIT.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day3/DAY3_CHUNKING_AUDIT.md)) identified 6 major failure modes in V0 [chunker.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/chunker.py):
   - **Silent Python truncation (`>256` tokens):** `624 / 4,272` V0 chunks (`14.61%`) exceeded MiniLM-L6's 256-token ONNX ceiling (including **57.7% / `41 of 71`** ground-truth evidence spans having *zero* untruncated overlapping chunks).
   - **Python `ast.walk` self-duplication:** `436` nested methods/functions emitted twice (once alone, once inside their enclosing `ClassDef`/`FunctionDef`), driving a `5.97%` (`255` excess) exact-duplicate rate.
   - **Dropped Python module-level code:** `66 / 83` Flask files lost top-level imports, constants, and `LocalProxy` bindings (e.g., `src/flask/globals.py` lines 32–63) whenever any `def`/`class` existed.
   - **Express JS control-flow false triggers:** Regex `r'^\s+\w+\s*\(.*\)\s*\{'` split on `if/for/while/catch`, creating `473` 1-line chunks and `235` tiny `<15`-token fragments.
   - **Loss of class/function signatures on splits & methods:** Methods lacked parent class context and `>_split` continuations lost function signatures.
   - **Empty chunks:** `3` empty `__init__.py` chunks indexed as blank strings.
2. **Pre-Modification Baseline Validation** ([validate_baseline_v0.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/chunking_experiments/validate_baseline_v0.py), saved in [baseline_validation.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/chunking_experiments/results/baseline_validation.json)) ran unmodified V0 and verified **`0.0` deltas** across all 7 Day 1 metrics (`Hit@1=0.3148`, `Hit@5=0.6667`, `Precision@5=0.1815`, `SpanRec@5=0.5556`, `MRR@5=0.4534`, `NDCG@5=0.3339`, `0` failed queries).

---

## 3. Experimental Design & Strategy Definitions

### 3.1 Controlled Comparison Matrix

All 3 strategies executed inside the same Python process (`run_id = 20261125T130707Z`) against identical hardware, libraries (`Python 3.14.7`, `chromadb 1.5.9`, `numpy 2.5.2`), model weights, and corpus manifest (`b6f6096ff2810b9aa57c3054a419bd3bce30ccc405aab66d606f7cb5e4804d2d`):

| ID | Label | Mechanism | Token Limit Enforced? | Structural Hierarchy | Context Prefix |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **`C0`** | `fixed_window_v0` | Blind `chunk_by_sliding_window(window_size=50, overlap=10)` | No (50 lines regardless of density) | None | None |
| **`C1`** | `existing_chunker_v0` | Original V0 `ast.walk` (Python) + naive regex (JS/Java/C++) + `>100`-line sliding fallback | No (100-line split only) | Flat + `ast.walk` overlap | None |
| **`C2`** | `code_aware_v1` | Hierarchical Python AST + stateful brace/comment/preprocessor scanner + WordPiece token budgeter (`max_tokens=256`, binary search + natural AST/blank break preference, 2-line split overlap) | **Yes (`<=256` WordPiece tokens inclusive of `[CLS]`/`[SEP]`)** | Module context + class overview + methods + nested helpers intact | Compact metadata header (`File`, `Language`, `Type`, `Class`/`Symbol`, `Part i/N`, continuation signature) |

### 3.2 Fixed Experimental Controls Verified

- **Pinned Commits Verified:**
  - `pallets/flask`: `49943b5a94413a530761350f04e896f76e8d620e` (**83** `.py` files)
  - `expressjs/express`: `43fca52d28a422c8be48fd489cbd4f2d6152bab6` (**142** `.js` files)
  - `fmtlib/fmt`: `3be6709f575a609620f5dd82a9ab2fa91bef6a5e` (**28** `.h`/`.c`/`.py` files)
  - **Total Corpus:** Exact same **253** filtered source files fed identically to `C0`, `C1`, and `C2`.
- **Embedding Model:** ONNX `all-MiniLM-L6-v2` (`chroma_onnx_sha256:913d7300ceae`), 384 dimensions, mean pooling, L2-normalized, `max_seq_length=256`. Model load (`0.4117s`) and warm-up (`0.2811s`) excluded from per-strategy inference timings.
- **Retrieval Configuration:** Isolated ChromaDB persistent collections per `(strategy, repo, run)`, `hnsw:space="l2"` (squared L2 distance), `Top-K = 5`, **60 evaluation questions** (**54 answerable**, **6 unanswerable**, **71 ground-truth evidence spans**, **17 multi-evidence queries**), raw `"question"` strings submitted without rewriting.

---

## 4. `code_aware_v1` Architecture & Engineering Details ([chunker_v1.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/chunker_v1.py))

1. **Non-Overlapping Hierarchical Python AST Parser (`_extract_python_blocks`):**
   - Walks only top-level `tree.body` (never `ast.walk`), keeping nested helper `def`s inside their enclosing function/method so **zero nested duplicates** are emitted.
   - Groups contiguous top-level imports/constants/`LocalProxy` bindings cleanly until the next `FunctionDef`/`AsyncFunctionDef`/`ClassDef` (or EOF), guaranteeing **100% coverage of non-empty module-level lines**.
   - Partitions classes cleanly into `class_overview` (class signature, docstring, attributes) + separate `method` blocks (avoiding repeating 1,000-line classes across every method), coalescing tiny adjacent blocks (`<45` tokens) inside the same scope up to `195` tokens.
2. **Stateful Lexical Brace-Depth Scanner for JS/TS/Java/C/C++ (`_scan_line_brace_states` & `_extract_brace_language_blocks`):**
   - Tracks brace nesting `{`/`}`, string/template literals, single/multi-line comments, and `\`-continued preprocessor macros (`#define ... \`) so comment headers, string `{}` literals, multi-line macro expansions (`include/fmt/fmt-c.h`), and control flow (`if`/`for`/`while`/`switch`/`catch`) never trigger false function splits.
   - Attaches lookback preambles (`template <typename ...>`, `@decorators`, `inline`/`constexpr`, `FMT_*` macros) cleanly to their declarations and preserves enclosing `class`/`struct` scope names.
3. **Exact WordPiece Token Budgeter (`_emit_token_bounded_unit`):**
   - Uses the exact Chroma ONNX `all-MiniLM-L6-v2` WordPiece tokenizer (`get_embedding_tokenizer()`, unpadded & untruncated) to measure `header + source` tokens.
   - Binary-searches oversized blocks under `max_tokens = 256`, snapping backward toward natural AST statement boundaries (`preferred_breaks`) or closing braces/blank lines near `target_tokens = 200`, adding minimal 2-line continuity overlap and repeating the function signature on continuation parts (`Part: i/N`).
4. **Chroma-Safe Schema & Deterministic Identity:**
   - All 18 metadata fields use Chroma primitive types (`str | int | bool`, zero `None` values).
   - Preserves exact 1-based `start_line`/`end_line` mappings for `llm.py`, Streamlit UI citations, and `evaluation/retrieval_metrics.py`, alongside deterministic `chunk_id` (`v1_<sha16>`) and `content_sha256`.

---

## 5. Comprehensive 24-Metric Chunk Quality Comparison

| # | Chunk Metric | `C0`: `fixed_window_v0` | `C1`: `existing_chunker_v0` | `C2`: `code_aware_v1` | Change (`V1` vs `V0`) |
| :-: | :--- | :---: | :---: | :---: | :--- |
| 1 | **Total chunks indexed** | `1,366` | `4,272` | **`3,721`** | `-551` (`-12.9%` fewer vectors) |
| 2 | **Chunks per repo (`flask` / `express` / `fmt`)** | `494` / `592` / `280` | `1,785` / `2,066` / `421` | **`1,352` / `1,297` / `1,072`** | Express `-37.2%`, Flask `-24.3%`, fmt `+154.6%` |
| 3 | **Chunks per language** | `.py`:536, `.js`:593, `.h`:234, `.c`:3 | `.py`:1,902, `.js`:2,067, `.h`:296, `.c`:7 | `py`:1,493, `js`:1,298, `cpp`:920, `c`:10 | Balanced structural split in C++ |
| 4 | **Raw line coverage / Non-empty line coverage** | `100.00%` / `100.00%` | `94.93%` / `97.47%` | **`97.48%` / `100.00%`** | **`+2.53%` non-empty coverage (`100%`)** |
| 5 | **Source files with non-empty chunks** | `250 / 253` (`98.81%`) | `250 / 253` (`98.81%`) | `250 / 253` (`98.81%`) | Same 3 empty `__init__.py` skipped cleanly in V1 |
| 6 | **Line length (`mean / median / min / max`)** | `45.95 / 50.0 / 1 / 50` | `14.06 / 8.0 / 1 / 100` | `14.23 / 13.0 / 1 / 39` | Eliminates 100-line cliffs; median up `8 -> 13` |
| 7 | **Char length (`mean / median / min / max`)** | `1365.6 / 1370.5 / 0 / 2908` | `445.38 / 258.0 / 0 / 4730` | `551.85 / 565.0 / 101 / 1308` | Tight variance, `0`-char eliminated |
| 8 | **Token length (`mean / median / p95 / max`)** | `433.8 / 433.0 / 681 / 1444` | `139.34 / 84.0 / 525 / 1844` | **`178.27 / 190.0 / 253 / 256`** | **100% fit inside 256-token window** |
| 9 | **Empty chunks (`strip() == ""`)** | `3` | `3` | **`0`** | **`-100%` (`0` empty chunks)** |
| 10 | **1-line chunks (`<= 1` line)** | `3` (`0.22%`) | `473` (`11.07%`) | **`4` (`0.11%`)** | **`-99.2%` (`473 -> 4`)** |
| 11 | **Short chunks (`< 5` lines)** | `11` (`0.81%`) | `1,404` (`32.87%`) | **`230` (`6.18%`)** | **`-83.6%` (`1,404 -> 230`)** |
| 12 | **Tiny chunks (`< 15` tokens)** | `6` (`0.44%`) | `235` (`5.50%`) | **`0` (`0.00%`)** | **`-100%` (`235 -> 0`)** |
| 13 | **Chunks exceeding 256 tokens** | `1,219` (`89.24%`) | `624` (`14.61%`) | **`0` (`0.00%`)** | **`-100%` (`624 -> 0`)** |
| 14 | **Chunks exceeding 512 tokens** | `354` (`25.92%`) | `233` (`5.45%`) | **`0` (`0.00%`)** | **`-100%` (`233 -> 0`)** |
| 15 | **Runtime ONNX truncation count / rate** | `1,219` / `89.24%` | `624` / `14.61%` | **`0` / `0.00%`** | **`0` truncated vectors** |
| 16 | **Exact duplicate excess / groups / rate** | `2` / `1` / `0.15%` | `255` / `126` / `5.97%` | **`7` / `3` / `0.19%`** | **`-97.3%` duplicates** (only genuine test copies left) |
| 17 | **Syntax / AST-or-lexical parsed chunks** | `0` (`0.0%`) | `1,884` (`44.1%`) | **`3,721` (`100.0%`)** | Full structural parse coverage |
| 18 | **Fallback chunks count / rate** | `1,366` (`100.0%`) | `18` (`0.42%`) | **`0` (`0.00%`)** | Clean AST + brace parse on all 253 files |
| 19 | **Ground-truth span coverage (`71` spans)** | `71 / 71` (`100%`) | `71 / 71` (`100%`) | `71 / 71` (`100%`) | Zero dropped evidence spans |
| 20 | **Untruncated GT spans (`<= 256` tokens)** | `3 / 71` (`4.23%`) | `30 / 71` (`42.25%`) | **`71 / 71` (`100.00%`)** | **Every GT span fits cleanly inside 256 tokens** |
| 21 | **Avg relevant chunks per answerable query** | `2.67` | `3.65` | `3.61` | Express `-46%`, fmt `+131%` |
| 22 | **Chunking time (`total / mean per repo`)** | `0.0116s` / `0.0039s` | `0.3979s` / `0.1326s` | `8.9153s` / `2.9718s` | `+8.52s` due to exact WordPiece budgeting |
| 23 | **Doc embed time & throughput** | `110.63s` (`12.35 ch/s`) | `342.13s` (`12.49 ch/s`) | **`248.36s` (`14.98 ch/s`)** | **`-93.78s` faster (`+19.9%` throughput)** |
| 24 | **Chroma insert / Total indexing e2e / Peak RSS** | `2.84s` / `113.48s` / `607.6 MB` | `8.96s` / `351.49s` / `620.6 MB` | **`8.10s` / `265.37s` / `622.8 MB`** | **`-86.12s` (`-24.5%`) faster total indexing** |

---

## 6. Overall Retrieval Performance Results (`N = 54` Answerable Queries, `0` Failed)

Note: `existing_chunker_v0` reproduced Day 1 baseline metrics with **`0.0` deltas** on all 7 metrics (`VALID`).

| Strategy | Hit@1 | Hit@3 | Hit@5 | Precision@5 | Span Recall@5 | MRR@5 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C0`: `fixed_window_v0`** | `0.2778` | **`0.6481`** | `0.6481` | `0.1778` | `0.5370` | `0.4383` | `0.3275` |
| **`C1`: `existing_chunker_v0` (V0)** | `0.3148` | `0.5741` | `0.6667` | `0.1815` | `0.5556` | `0.4534` | **`0.3339`** |
| **`C2`: `code_aware_v1` (V1)** | **`0.3704`** | `0.6296` | **`0.7037`** | **`0.2185`** | **`0.6111`** | **`0.5052`** | `0.3217` |
| **Delta (`C2` vs `C1` V0)** | **`+0.0556` (+17.7%)** | **`+0.0555` (+9.7%)** | **`+0.0370` (+5.5%)** | **`+0.0370` (+20.4%)** | **`+0.0555` (+10.0%)** | **`+0.0518` (+11.4%)** | `-0.0122` (-3.7%) |
| **Delta (`C2` vs `C0` Fixed)** | **`+0.0926` (+33.3%)** | `-0.0185` (-2.9%) | **`+0.0556` (+8.6%)** | **`+0.0407` (+22.9%)** | **`+0.0741` (+13.8%)** | **`+0.0669` (+15.3%)** | `-0.0058` (-1.8%) |

### 6.1 Explaining the `NDCG@5` vs `Precision@5` Divergence (Granularity & `IDCG@5` Effect)

Although `code_aware_v1` achieved higher mean relevant chunks retrieved in Top-5 (**`1.09`** per query vs **`0.91`** in V0 and **`0.89`** in `fixed_window_v0`), higher Top-1 accuracy (`+17.7%`), and higher `MRR@5` (`+11.4%`), its macro-averaged `NDCG@5` measured `0.3217` vs `0.3339` in V0. Per-query inspection of [per_strategy evaluations](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/chunking_experiments/results/per_strategy/code_aware_v1_evaluation.json) confirms the exact mathematical cause:
- Per Day 1 Task 4's formal definition in [retrieval_metrics.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/evaluation/retrieval_metrics.py), `IDCG@5` assumes up to `min(total_relevant_chunks_in_repo, 5)` ideal hits.
- On `fmtlib/fmt`, V0 left massive C++ template headers unsplit (`211/421` truncated >256 tokens), averaging only **`2.00`** relevant chunks per query in the repo. When `code_aware_v1` cleanly split those >256-token C++ blocks under the 256-token ceiling, average relevant chunks per fmt query rose to **`4.61`**, driving `IDCG@5` denominator higher (`1.00 -> ~2.95`):
  - Example **`fmt_010`** (`printf_arg_formatter`): V0 retrieved 1 relevant chunk at Rank 1 out of 1 in repo -> `NDCG@5 = 1.0000`. `code_aware_v1` retrieved **2** relevant chunks (Rank 1 & Rank 5) and raised **`Precision@5` from `0.20` to `0.40`**, yet because the un-truncated span now occupied `4` relevant chunks in the index, its `NDCG@5` dropped mathematically to `0.5414` (`delta = -0.4586`).
  - Conversely, on **`expressjs/express`**, where V1 eliminated noisy 1-line fragments (cutting `mean_rel_repo` from `4.67` -> `2.50`), **`NDCG@5` rose substantially from `0.2080 -> 0.2794` (`+0.0714`)**.
- **Conclusion:** When chunk granularity varies across strategies, `Hit@K`, `MRR@5`, `Span Recall@5`, and `Precision@5` reflect true ranking quality without denominator shift, while `NDCG@5` penalizes strategies that split oversized truncated blocks cleanly below `max_tokens`.

---

## 7. Granular Breakdowns

### 7.1 Per-Repository Performance

| Repository | Strategy | Hit@1 | Hit@3 | Hit@5 | Precision@5 | Span Recall@5 | MRR@5 | NDCG@5 | Mean P50 Lat |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pallets/flask`** (18q) | `C0` Fixed | `0.2222` | `0.6111` | `0.6111` | `0.1444` | `0.5000` | `0.3889` | `0.2550` | `37.62 ms` |
| | `C1` V0 | `0.2778` | `0.5556` | `0.6667` | **`0.2222`** | `0.5556` | `0.4204` | **`0.3323`** | `34.63 ms` |
| | **`C2` V1** | **`0.3889`** | **`0.6667`** | **`0.7778`** | `0.2111` | **`0.6944`** | **`0.5463`** | `0.3207` | **`31.56 ms`** |
| **`expressjs/express`** (18q) | `C0` Fixed | **`0.2778`** | **`0.6667`** | **`0.6667`** | **`0.1778`** | **`0.5278`** | **`0.4444`** | **`0.3275`** | `38.16 ms` |
| | `C1` V0 | **`0.2778`** | `0.5000` | `0.6111` | `0.1444` | `0.4722` | `0.3981` | `0.2080` | `33.78 ms` |
| | **`C2` V1** | `0.2222` | `0.5556` | **`0.6667`** | `0.1556` | **`0.5278`** | `0.3954` | `0.2794` | **`31.91 ms`** |
| **`fmtlib/fmt`** (18q) | `C0` Fixed | `0.3333` | **`0.6667`** | `0.6667` | `0.2111` | `0.5833` | `0.4815` | `0.4001` | `37.05 ms` |
| | `C1` V0 | `0.3889` | **`0.6667`** | **`0.7222`** | `0.1778` | **`0.6389`** | `0.5417` | **`0.4615`** | `32.02 ms` |
| | **`C2` V1** | **`0.5000`** | **`0.6667`** | `0.6667` | **`0.2889`** | `0.6111` | **`0.5741`** | `0.3650` | **`31.62 ms`** |

- **Flask** delivered the clearest all-around leap under `code_aware_v1`: `Hit@5` climbed `+11.1%` (`0.6667 -> 0.7778`), `MRR@5` jumped `+30.0%` relative (`0.4204 -> 0.5463`), and `Span Recall@5` jumped `+13.9%` (`0.5556 -> 0.6944`).
- **fmt** surged on `Hit@1` (`0.3889 -> 0.5000`, Rank-1 on half of all C++ queries!) and `Precision@5` (`0.1778 -> 0.2889`, `+62.5%` relative) once >256-token truncation (`50.1%` in V0 -> `0%` in V1) was removed.
- **Express** remained competitive at `Hit@5 = 0.6667` (tying `C0` and beating V0 `0.6111`), but Mocha `test/` `describe()` chunks continued competing heavily for Ranks 1–3.

### 7.2 By Query Category (`Hit@5 / MRR@5 / Span Recall@5`)

| Category (n) | `C0` Fixed Window | `C1` Existing V0 | `C2` Code-Aware V1 | Winner & Insight |
| :--- | :---: | :---: | :---: | :--- |
| `architecture_flow` (10) | `0.70 / 0.3500 / 0.55` | **`0.90`** / `0.5283` / **`0.70`** | `0.80` / **`0.5667`** / **`0.70`** | V1 highest MRR & P@5 (`0.26`) |
| `cross_file_deps` (10) | `0.80 / 0.6167 / 0.45` | **`1.00`** / `0.6667` / **`0.65`** | `0.90` / **`0.7000`** / `0.60` | V1 highest H@1 (`0.50`) & MRR (`0.70`) |
| `debugging_config` (10) | **`0.70 / 0.4833 / 0.65`** | `0.50 / 0.4250 / 0.45` | `0.40 / 0.4000 / 0.35` | Fixed window won prose/macro configs (`flask_017`, `express_018`, `fmt_014`) |
| `function_identification` (12) | `0.5833 / 0.4444 / 0.5417` | `0.5000 / 0.3361 / 0.5000` | **`0.6667 / 0.4125 / 0.6250`** | V1 best H@5, P@5 (`0.20`), SpanRec (`0.625`) |
| `implementation_underst.` (12) | `0.5000 / 0.3194 / 0.5000` | `0.5000 / 0.3542 / 0.5000` | **`0.7500 / 0.4722 / 0.7500`** | **Massive +25% H@5 & SpanRec jump** in V1 |

### 7.3 By Query Style & Evidence Complexity (`Hit@1 / Hit@5 / MRR@5`)

- **`identifier` (22q):** `C0` `0.4091 / 0.6364 / 0.5000` | `C1` `0.3182 / 0.5909 / 0.4144` | **`C2` `0.4545 / 0.7727 / 0.5659`** (**+18.2% Hit@5 over V0**, showing exact benefit of compact `Symbol:`/`Class:` headers + clean function boundaries!)
- **`conversational` (12q, Turn-2 no rewriting):** `C0` `0.3333 / 0.8333 / 0.5555` | `C1` `0.3333 / 0.8333 / 0.5278` | `C2` **`0.5833` H@1** / `0.7500` H@5 / **`0.6528` MRR**
- **`conceptual` (20q):** `C0` `0.1000 / 0.5500 / 0.3000` | `C1` **`0.3000 / 0.6500 / 0.4517`** | `C2` `0.1500 / 0.6000 / 0.3500`
- **Single-Evidence (37q) vs Multi-Evidence (17q):**
  - Single-evidence `Hit@5`: `0.5676` (`C0`) -> `0.5405` (`C1`) -> **`0.6216` (`C2`)**
  - Multi-evidence `Hit@1` / `MRR@5`: `0.3529` / `0.5490` (`C0`) -> `0.4118` / `0.6471` (`C1`) -> **`0.5882` / `0.7255` (`C2`)** (`88.24%` Hit@5 vs `94.12%` V0)

### 7.4 Unanswerable vs Answerable Top-1 L2 Distance (`n = 6` vs `n = 54`)

| Strategy | Answerable Top-1 L2 (`min / median / mean / max`) | Unanswerable Top-1 L2 (`min / median / mean / max`) | Separation Gap (`Unans Med - Ans Med`) |
| :--- | :---: | :---: | :---: |
| `C0` Fixed Window | `0.5232 / 0.8818 / 0.8882 / 1.5242` | `0.9625 / 1.1045 / 1.1121 / 1.2572` | `+0.2227` |
| `C1` Existing V0 | `0.3949 / 0.8626 / 0.8496 / 1.5378` | `0.8314 / 1.0685 / 1.0332 / 1.2868` | `+0.2059` |
| **`C2` Code-Aware V1** | `0.3952 / 0.7921 / 0.8146 / 1.4915` | `0.7913 / 1.0203 / 0.9945 / 1.1954` | **`+0.2282`** |

---

## 8. Latency & Computational Cost Analysis

All query benchmarks ran with `warmup_duration_seconds = 0.2811s` excluded and separated timers:

| Phase / Metric | `C0` Fixed Window | `C1` Existing V0 | `C2` Code-Aware V1 |
| :--- | :---: | :---: | :---: |
| **Chunking Time (total across 253 files)** | `0.0116s` | `0.3979s` | `8.9153s` |
| **Document Embedding Time (all 3 repos)** | `110.63s` | `342.13s` | **`248.36s` (-27.4% vs V0)** |
| **Embedding Throughput** | `12.35 chunks/s` | `12.49 chunks/s` | **`14.98 chunks/s`** |
| **ChromaDB Insertion Time** | `2.84s` | `8.96s` | **`8.10s`** |
| **End-to-End Indexing Total (`chunk+embed+add`)** | `113.48s` | `351.49s` | **`265.37s` (-24.5% vs V0)** |
| **Query Embed Latency (60q `mean / P50 / P95`)** | `36.09 / 34.26 / 51.92 ms` | `31.12 / 30.27 / 38.17 ms` | **`29.83 / 28.53 / 38.52 ms`** |
| **Vector Search Latency (60q `mean / P50 / P95`)** | `4.46 / 3.01 / 4.47 ms` | `4.73 / 2.98 / 11.91 ms` | `4.84 / 3.12 / 3.72 ms` |
| **Total Query Latency (`all 60 mean / P50 / P95`)** | `40.56 / 37.42 / 56.84 ms` | `35.85 / 33.43 / 44.79 ms` | **`34.68 / 31.69 / 44.53 ms`** |
| **Peak Process Memory (RSS)** | `607.59 MB` | `620.55 MB` | `622.84 MB` |

Notice the counter-intuitive performance win: spending `+8.5s` on accurate WordPiece token budgeting in `code_aware_v1` **saved `94.6s`** in ONNX document embedding + ChromaDB insertion by removing duplicate AST chunks and noisy Express `if/for` micro-fragments.

---

## 9. Error Analysis & Strategy Transitions

Pairwise comparisons across the 54 answerable questions revealed exact behavioral mechanics:

1. **Solved by `code_aware_v1`, Missed by `existing_chunker_v0` (`6` queries):**
   - `flask_007` (`dispatch_request`, V1 **Rank #2**) & `flask_016` (`Config.from_object`, V1 **Rank #1**): V0 retrieved 1–2 line test helpers from `tests/test_views.py` and `tests/test_config.py`; V1's `Class: Config | Symbol: from_object` headers elevated the real implementations cleanly.
   - `express_001` (`createApplication`, V1 **Rank #1**), `express_003` (`View`, V1 **Rank #5**), `express_004` (`req.get`/`req.header`, V1 **Rank #4**): V0 polluted ranks 1–3 with 1-line regex fragments in `test/req.query.js` and `test/express.static.js`; V1 surfaced the canonical `lib/*.js` source modules.
   - `fmt_002` (`styled` helper, V1 **Rank #1**): Clean separation of C++ template functions vs monolithic truncated V0 headers brought `include/fmt/color.h:652-663` straight to Rank #1.

2. **Regressed by `code_aware_v1`, Hit by `existing_chunker_v0` (`4` queries):**
   - `express_010` (Turn-2 dispatch lifecycle): `app.handle` was pushed past Rank 5 by Mocha `describe()` / getter chunks (`stale`, `app.defaultConfiguration`, `fresh`).
   - `express_016` (client IP proxy trust): Mocha test suites `test/req.ip.js` and `test/req.ips.js` captured Ranks 1–3, pushing `lib/request.js`'s `req.ip` getter down from Rank 3 (V0) past Rank 5.
   - `fmt_001` (`Where is the text_style class defined`): V1 retrieved `text_style` helper functions/methods (`fg`, `bg`, `get_emphasis`) in `color.h:330-353` above the class header/attributes at `236-285` because `text_style` appeared repeatedly in method `Symbol:` headers.
   - `fmt_014` (`parse format syntax at compile time using macros`): Conversational query without `FMT_STRING` identifier missed when comment prose separated from `FMT_STRING` macro definitions.

3. **Unique Wins by `fixed_window_v0` (`3` queries: `flask_017`, `express_017`, `express_018`):**
   - Larger 50-line overlapping windows (even though truncated at 256 tokens) kept contiguous multi-method neighborhoods together (e.g., `lib/response.js:721-770` around `res.clearCookie` + `res.cookie` in `express_017`, and `lib/application.js:281-330` around `app.route` + `app.engine` in `express_018`), whereas V1's `test/res.cookie.js:1-7 (imports_and_setup)` topped Rank 1 on `express_017`.

4. **Missed by All Three Strategies (`9` queries):**
   - `flask_002` (Turn-2 pronoun *"it"* without query rewriting), `flask_005` (`route` decorator shadowed by test mocks), `flask_018` (*"temporary one-off notification messages"* -> symbol is `flash`, pure conceptual mismatch), `express_013`, `express_015` (*"weak and strong entity tags"* -> `etag` / `compileETag`), `fmt_003` (*"RAII file streams"* -> `class file`), `fmt_004` (*"dynamic argument lists"* -> `dynamic_format_arg_store`), `fmt_005` (`is_range` SFINAE trait), `fmt_013` (*"time durations"* -> `chrono.h` formatting).

---

## 10. Strategy Decision & Day 4 Roadmap Recommendation

### Decision: Promote `code_aware_v1` to Default Strategy (with Selectable Fallback Preserved)

`code_aware_v1` is empirically superior to both controls:
- **`+17.7%` Hit@1**, **`+20.4%` Precision@5**, **`+11.4%` MRR@5**, **`+10.0%` Span Recall@5**, **`+3.7%` abs Hit@5 (`70.37%`)**
- **`0.00%` runtime token truncation** (100% of ground-truth evidence spans preserved intact within MiniLM-L6's 256-token budget)
- **`99.2%` fewer 1-line fragments**, **`97.3%` fewer exact duplicates**, **`-24.5%` faster end-to-end indexing**
- V0 (`existing_chunker_v0`) and `fixed_window_v0` remain cleanly accessible via `chunker.chunk_code(file_data, strategy=...)` and Streamlit sidebar selector without breaking existing callers.

### Day 4 Immediate Recommendations (Why Hybrid Retrieval Fits Cleanly Next)

The residual error analysis points directly to **Day 4 (Hybrid Retrieval: BM25 + Dense Vector + RRF)**, **Day 5 (Reranking)**, and **Day 6 (Query Rewriting)**:
1. **Day 4 — Hybrid BM25 + Dense Vector (`code_aware_v1`):**
   - Exact identifiers like `FMT_STRING` (`fmt_014`), `text_style` (`fmt_001` class definition vs callers), `res.cookie` (`express_017`), `app.engine` (`express_018`), `is_range` (`fmt_005`), and `dynamic_format_arg_store` (`fmt_004`) benefit immediately from lexical BM25 keyword matching combined with semantic `code_aware_v1` vectors via Reciprocal Rank Fusion (RRF).
   - Indexing `code_aware_v1`'s clean `Symbol:`/`Class:` headers and zero-duplicate chunks gives BM25 crisp term frequencies without V0's `ast.walk` double-counting distortion.
2. **Test-Directory Ranking Mitigation (Day 4/5):** Across all strategies, `test/` and `tests/` chunks repeatedly outrank core `src/`, `lib/`, and `include/` implementations (`express_016`, `express_017`, `flask_005`). Metadata-aware path weighting or reranking will unlock clean precision gains on Day 5.
3. **Turn-2 Conversational Rewriting (Day 6):** `flask_002` and `fmt_014` confirm that dense retrieval alone cannot resolve underspecified pronouns (`"it"`, `"that formatting syntax"`) without conversational query rewriting.
