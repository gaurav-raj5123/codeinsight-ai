# 📊 Project Evaluation & Engineering Reports

This directory houses the structured, day-by-day engineering audits, benchmark reports, and evaluation summaries for the **CodeInsight AI Assistant** project.

---

## 📅 Daily Engineering Progress

### [Day 1: Baseline Architecture, Instrumentation & V0 Benchmark](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day1/)
- **[DAY1_PROJECT_AUDIT.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day1/DAY1_PROJECT_AUDIT.md)**: Comprehensive architectural audit of the pre-existing system, component boundaries, runtime lifecycles, and backward-compatibility constraints.
- **[DAY1_INSTRUMENTATION_REPORT.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day1/DAY1_INSTRUMENTATION_REPORT.md)**: Verification of minimal baseline instrumentation (loading duration, extraction time, `collection.add()` vs total function time, query latency, secret sanitization).
- **[DAY1_TASK3_REPORT.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day1/DAY1_TASK3_REPORT.md)**: Selection and curation of the 60-question gold-standard codebase benchmark across 3 repositories (`pallets/flask`, `expressjs/express`, `fmtlib/fmt`) with pinned commit SHAs and zero chunk ID dependence.
- **[DAY1_FINAL_REPORT.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day1/DAY1_FINAL_REPORT.md)**: Verified V0 dense retrieval baseline results (`Hit@1=0.3148`, `Hit@5=0.6667`, `MRR@5=0.4534`, `NDCG@5=0.3339`, `SpanRec@5=0.5556`) and failure-mode catalog.

### [Day 2: Embedding Optimization & Controlled Retrieval Experiment](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day2/)
- **[DAY2_TASK1_REPORT.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day2/DAY2_TASK1_REPORT.md)**: Controlled ONNX embedding benchmark comparing V0 `all-MiniLM-L6-v2` against `BAAI/bge-small-en-v1.5` (no prefix & retrieval instruct), `intfloat/e5-small-v2`, and `all-MiniLM-L12-v2` across identical V0 chunks, squared $L_2$, and separated indexing/query timings:
  - **Overall Winner on Hit@5 & MRR@5**: V0 `all-MiniLM-L6-v2` (`0.6667` / `0.4534`, `86.2 MB`, `625.6 MB` peak RSS) retained as production baseline.
  - **Runner-Up Candidate**: `BAAI/bge-small-en-v1.5` + query instruction led `NDCG@5` (`0.3661`) and `identifier` queries (`Hit@5=0.7727`), staged for re-evaluation after Day 3 code-aware chunking removes noisy JS test fragments and C++ license preambles.
- **[day2_embedding_analysis.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/results/day2_embedding_analysis.md)**: Concise tradeoff digest and empirical summary.

### [Day 3: Code-Aware Chunking Optimization](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day3/)
- **[DAY3_CHUNKING_AUDIT.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day3/DAY3_CHUNKING_AUDIT.md)**: Pre-modification audit identifying Python `ast.walk` self-duplication (`436` nested chunks), dropped Python module globals (`66/83` Flask files), Express JS regex control-flow fragmentation (`473` 1-line chunks), and `624` (`14.61%`) >256-token truncations.
- **[DAY3_CHUNKING_ANALYSIS.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day3/DAY3_CHUNKING_ANALYSIS.md)**: Controlled Day 3 benchmark (`fixed_window_v0` vs `existing_chunker_v0` vs `code_aware_v1`) establishing `code_aware_v1` as the clear winner (`Hit@1`: `0.3148 -> 0.3704`, `Hit@5`: `0.6667 -> 0.7037`, `Precision@5`: `0.1815 -> 0.2185`, `MRR@5`: `0.4534 -> 0.5052`, `SpanRec@5`: `0.5556 -> 0.6111`, `0.00%` runtime truncation, `-24.5%` faster indexing).

### [Day 4: Hybrid Retrieval (BM25 + Dense + RRF)](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day4/)
- **[DAY4_HYBRID_REPORT.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day4/DAY4_HYBRID_REPORT.md)**: Single-pass controlled comparison of Dense V1 (`Hit@5=0.7037`, `NDCG@5=0.3217`), Code-Aware Okapi BM25 with V1 header label hygiene & zero-overlap filtering (`Hit@5=0.6852`, `NDCG@5=0.3689`), and Hybrid RRF `k=60` (`Hit@1=0.3889`, `Hit@5=0.7222`, `Precision@5=0.2444`, `SpanRec@5=0.6389`, `MRR@5=0.5182`, `NDCG@5=0.3737`, `P50=39.56 ms`). Establishes `Candidate Union` (`Dense@20 ∪ BM25@20`) ceiling of **`0.9630` Hit Rate (`52/54`)** and **`0.9074` Span Recall** for Day 5 reranking.

### [Day 5: Cross-Encoder Reranking & Conversational Query Rewriting](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day5/)
- **[DAY5_RERANK_REWRITE_REPORT.md](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day5/DAY5_RERANK_REWRITE_REPORT.md)**: ONNX-only CPU Cross-Encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`, rev `233902d25c44`, `86.80 MB`, `0.0%` pair truncation) over `D4-E2` Hybrid RRF (`0.0` Day 4 parity deltas):
  - **`D5-E3a` (RRF Top-20 -> Rerank, Production Default):** Lifts `Hit@1` `0.3889 -> 0.4630`, `Hit@5` `0.7222 -> 0.7963` (`43/54`), `Precision@5` `0.2444 -> 0.2704`, `SpanRec@5` `0.6389 -> 0.6852`, `MRR@5` `0.5182 -> 0.5861`, `NDCG@5` `0.3737 -> 0.4366` (`P50 = 2,598.23 ms`).
  - **`D5-E3b` (Full Candidate Union `<=40` -> Rerank):** Reaches `Hit@1 = 0.4815`, `MRR@5 = 0.5948`, `NDCG@5 = 0.4391` (`Hit@5 = 0.7778`, `P50 = 4,498.75 ms`).
  - **Leakage-Free Conversational Rewriting (`Q1` vs `Q0` & `Q2-Oracle`):** `D5-E3a` already lifts unrewritten Turn-2 `Hit@5` to `0.9167` (`11/12`); live Gemini `Q1` suffers `4/12` API timeouts/503s (`4.2s–32.7s`, safely falling back) and query over-specification (`Hit@5 = 0.7500`), so rewriting is integrated cleanly as an **opt-in toggle (default `False`)**. Includes exploratory Day 6 raw-logit/sigmoid distribution handoff (`Answerable` vs `Unanswerable`).

---

### Upcoming Milestones
- **Day 6**: Score Calibration, Unanswerable Abstention Boundaries & Retrieval Grounding Analysis (building on Day 5 raw-logit/sigmoid score distributions)
- **Day 7**: End-to-End Generation Evaluation, Final System Polish & Release Verification
