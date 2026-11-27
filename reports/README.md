# 📊 Project Evaluation & Engineering Reports

This directory houses the structured, day-by-day engineering audits, benchmark reports, reproducible charts, and executive release documentation for the **CodeInsight AI Assistant (`v1.0.0`)** project.

---

## 🏆 Final Capstone & Visual Portfolio (Day 7 Complete)

- **[FINAL_PROJECT_SUMMARY.md](FINAL_PROJECT_SUMMARY.md)**: Master 7-day technical synthesis covering dataset controls, **V0 $\rightarrow$ V3** retrieval progression (`Hit@5: 0.6667` $\rightarrow$ **`0.7963`**, `MRR@5: 0.4534` $\rightarrow$ **`0.5861`**), **G0 $\rightarrow$ G4** trustworthy generation (`100%` valid citations across `154/154` tokens, `100%` Cohort C abstention, `0` fabricated file/line citations), rejected alternatives, and honest limitations.
- **[INTERVIEW_AND_DEMO_GUIDE.md](INTERVIEW_AND_DEMO_GUIDE.md)**: 60-second pitch, architectural deep dive, high-signal *"Why Not X?"* tradeoff Q&A backed strictly by saved numbers, and 5-minute live UI demo script.
- **[`assets/`](assets/)** (Reproduced via [`experiments/generate_final_charts.py`](../experiments/generate_final_charts.py)):
  - [`v0_to_v3_retrieval_progression.png`](assets/v0_to_v3_retrieval_progression.png) — `Hit@5`, `Span Recall@5`, `MRR@5`, `Hit@1`, and `NDCG@5` trajectory across Days 1, 3, 4, and 5.
  - [`stage1_ceiling_vs_reranking.png`](assets/stage1_ceiling_vs_reranking.png) — Stage-1 candidate generation ceilings (`96.3%` Union / `94.4%` RRF@20) vs Stage-2 delivered Top-5 (`79.6%`).
  - [`latency_vs_quality_tradeoff.png`](assets/latency_vs_quality_tradeoff.png) — Median P50 retrieval latency (`ms`, log scale) vs `Hit@5` and `MRR@5` across CPU retrieval modes.
  - [`day6_grounding_reliability.png`](assets/day6_grounding_reliability.png) — `G0` baseline vs `G4` grounded generation: abstention, blinded-judge faithfulness/leakage, verified citation coverage, and unretrieved mention counts.

---

## 📅 Daily Engineering Reports (Days 1–6 Complete)

### [Day 1: Baseline Architecture, Instrumentation & V0 Benchmark](day1/)
- **[DAY1_PROJECT_AUDIT.md](day1/DAY1_PROJECT_AUDIT.md)**: Architectural audit of the pre-existing V0 system, component boundaries, runtime lifecycles, and compatibility anchors.
- **[DAY1_INSTRUMENTATION_REPORT.md](day1/DAY1_INSTRUMENTATION_REPORT.md)**: Verification of minimal baseline instrumentation (loading duration, extraction time, `collection.add()` vs total indexing time, query latency, secret sanitization).
- **[DAY1_TASK3_REPORT.md](day1/DAY1_TASK3_REPORT.md)**: Curation of the 60-question gold-standard codebase benchmark across 3 pinned repositories (`pallets/flask`, `expressjs/express`, `fmtlib/fmt`) with `71` evidence spans.
- **[DAY1_FINAL_REPORT.md](day1/DAY1_FINAL_REPORT.md)**: Verified V0 dense retrieval baseline (`Hit@1=0.3148`, `Hit@5=0.6667`, `MRR@5=0.4534`, `NDCG@5=0.3339`, `SpanRec@5=0.5556`) and failure-mode catalog.

### [Day 2: Embedding Optimization & Controlled Retrieval Experiment](day2/)
- **[DAY2_TASK1_REPORT.md](day2/DAY2_TASK1_REPORT.md)**: Controlled ONNX embedding comparison across `all-MiniLM-L6-v2` (`E0`), `BAAI/bge-small-en-v1.5` (`E1a`/`E1b`), `intfloat/e5-small-v2` (`E2`), and `all-MiniLM-L12-v2` (`E3`): V0 `all-MiniLM-L6-v2` retained top `Hit@5` (`0.6667`) and `MRR@5` (`0.4534`) at `86.2 MB` / `625.6 MB` peak RSS while proving `128`-token truncation (`E3`) harms code retrieval (`0.4815`).
- **[day2_embedding_analysis.md](../experiments/results/day2_embedding_analysis.md)**: Concise tradeoff digest and empirical summary.

### [Day 3: Code-Aware Chunking Optimization](day3/)
- **[DAY3_CHUNKING_AUDIT.md](day3/DAY3_CHUNKING_AUDIT.md)**: Pre-modification audit uncovering Python `ast.walk` self-duplication (`436` nested blocks / `255` duplicates), dropped Python module globals (`66/83` Flask files), Express JS control-flow regex splitting (`473` 1-line fragments), and `624` (`14.61%`) >256-token truncations.
- **[DAY3_CHUNKING_ANALYSIS.md](day3/DAY3_CHUNKING_ANALYSIS.md)**: Controlled comparison establishing `code_aware_v1` as winner (`Hit@1: 0.3148 -> 0.3704`, `Hit@5: 0.6667 -> 0.7037`, `Precision@5: 0.1815 -> 0.2185`, `MRR@5: 0.4534 -> 0.5052`, `SpanRec@5: 0.5556 -> 0.6111`, `0.00%` runtime truncation, `-24.5%` indexing time).

### [Day 4: Hybrid Retrieval (BM25 + Dense + RRF)](day4/)
- **[DAY4_HYBRID_REPORT.md](day4/DAY4_HYBRID_REPORT.md)**: Single-pass comparison of Dense V1 (`Hit@5=0.7037`), Code-Aware Okapi BM25 with V1 header label hygiene & zero-overlap filtering (`Hit@5=0.6852`, `P50=2.62 ms`), and Hybrid RRF `k=60` (`Hit@1=0.3889`, `Hit@5=0.7222`, `Precision@5=0.2444`, `SpanRec@5=0.6389`, `MRR@5=0.5182`, `NDCG@5=0.3737`, `P50=39.56 ms`), revealing **`0.9630` (`52/54`) Candidate Union ceiling**.

### [Day 5: Cross-Encoder Reranking & Conversational Query Rewriting](day5/)
- **[DAY5_RERANK_REWRITE_REPORT.md](day5/DAY5_RERANK_REWRITE_REPORT.md)**: CPU ONNX Cross-Encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `86.80 MB`, `0.0%` truncation) with exact `0.0` Day 4 parity deltas: `D5-E3a` (`RRF Top-20 -> Rerank`) lifts `Hit@1` **`0.4630`**, `Hit@5` **`0.7963` (`43/54`)**, `MRR@5` **`0.5861`**, `NDCG@5` **`0.4366`** (`P50=2,598.23 ms`); rejects default query rewriting after empirical regression (`Turn-2 Hit@5: 0.9167 -> 0.7500`).

### [Day 6: Trustworthy Grounded Generation & Citation Validation](day6/)
- **[DAY6_GENERATION_AUDIT.md](day6/DAY6_GENERATION_AUDIT.md)**: Audit identifying `llm.build_prompt()` directory-tree/history leakage hazards, V1 header duplication, missing timeouts/retries, `20 RPD` free-tier cap on `gemini-3.5-flash`, and verified `300/300` frozen cohort counts (`43 / 11 / 6`).
- **[DAY6_GROUNDED_GENERATION_REPORT.md](day6/DAY6_GROUNDED_GENERATION_REPORT.md)**: Controlled `G0–G4` & Layer 2 Blinded Judge study (`204/204` live `gemini-3.5-flash-lite` calls): **`100.0%` token citation validity (`154/154`)**, `0` unretrieved file/line mentions (`2 -> 0` / `9 -> 0`), `100.0%` Cohort C abstention (`6/6`), `100.0%` approximate blinded-judge faithfulness (`24/24`), and `0.0%` parametric leakage (`0/24`).
