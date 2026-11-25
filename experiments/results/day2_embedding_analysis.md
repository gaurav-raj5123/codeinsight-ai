# Day 2: Embedding Optimization & Controlled Retrieval Analysis

- **Run ID**: `d2_emb_20261125_082152`
- **Status**: `VALID_COMPLETED` (0 failed queries across all 5 runs / 300 query executions)
- **Full Benchmark Artifacts**: [`experiments/embedding_experiments/results/embedding_summary.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/embedding_experiments/results/embedding_summary.json) | [`embedding_comparison.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/embedding_experiments/results/embedding_comparison.json)
- **Comprehensive Report**: [`reports/day2/DAY2_TASK1_REPORT.md`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/reports/day2/DAY2_TASK1_REPORT.md)

---

## 1. Controlled Comparison Overview (N=54 Answerable Queries, Top-K=5, Squared L2)

All embeddings are verified L2 unit-normalized 384-dimensional vectors evaluated under identical V0 chunks (4,272 chunks), identical ground-truth spans (71 spans), and isolated ChromaDB collections (`hnsw:space = "l2"`).

| Experiment ID | Model | Pooling / MaxLen | Prefixes (Doc / Query) | Hit@1 | Hit@3 | Hit@5 | Precision@5 | SpanRec@5 | MRR@5 | NDCG@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E0_v0_minilm_l6_v2`** | `all-MiniLM-L6-v2` (V0) | mean / 256 | none / none | **`0.3148`** | **`0.5741`** | **`0.6667`** | 0.1815 | 0.5556 | **`0.4534`** | 0.3339 |
| **`E1a_bge_small_en_v15_no_prefix`** | `BAAI/bge-small-en-v1.5` | cls / 512 | none / none | 0.2593 | **`0.5741`** | 0.6111 | 0.1889 | 0.5185 | 0.3963 | 0.3370 |
| **`E1b_bge_small_en_v15_instruct`** | `BAAI/bge-small-en-v15` | cls / 512 | none / BGE instruct | **`0.3148`** | **`0.5741`** | 0.6296 | **`0.1963`** | 0.5463 | 0.4503 | **`0.3661`** |
| **`E2_e5_small_v2`** | `intfloat/e5-small-v2` | mean / 512 | `passage: ` / `query: ` | 0.2963 | 0.5185 | 0.6481 | 0.1889 | **`0.5741`** | 0.4164 | 0.3529 |
| **`E3_minilm_l12_v2`** | `all-MiniLM-L12-v2` | mean / 128 | none / none | 0.2778 | 0.4074 | 0.4815 | 0.1185 | 0.3796 | 0.3509 | 0.2679 |

---

## 2. Key Analytical Findings

1. **Exact V0 Baseline Reproduction**: `E0_v0_minilm_l6_v2` reproduced Day 1's validated V0 metrics identically (`Hit@1=0.3148`, `Hit@5=0.6667`, `MRR@5=0.4534`, `NDCG@5=0.3339`, `SpanRec@5=0.5556`) while cleanly separating ONNX embedding time (`300.04s` docs, `29.46ms` query P50) from ChromaDB vector operations (`4.74s` insert, `2.62ms` query P50).
2. **V0 Unit Normalization Control**: Inspection of ChromaDB's `ONNXMiniLM_L6_V2._forward()` and runtime norm assertions proved that V0 vectors already have Euclidean norm `1.000000`. Thus all 5 experiments share an identical unit-hypersphere geometry under squared $L_2$ ($\|u-v\|^2 = 2 - 2\cos(u,v)$).
3. **Why V0 Held Up on Overall Hit@5 & MRR@5**:
   - **`sentence-transformers/all-MiniLM-L6-v2` pretraining**: Trained on `CodeSearchNet` paired code-text data alongside general QA, making it resilient on `conceptual` (`Hit@5=0.6500` vs `E1b` `0.5000`), `conversational` (`Hit@5=0.8333` vs `0.5833`), and `expressjs/express` (`0.6111` vs `E1a` `0.3333` / `E1b` `0.3889`).
   - **`E1b` (`BGE-small-en-v1.5 + instruct`)** outperformed V0 markedly on **`identifier` queries** (`Hit@5`: `0.7727` vs `0.5909`, `MRR@5`: `0.5848` vs `0.4144`) and **corpus-level `NDCG@5`** (`0.3661` vs `0.3339`), while `512`-token context helped on `fmtlib/fmt` (`0.7778` vs `0.7222`), where V0 truncated `50.12%` of chunks. However, `E1b` pulled test/example JS files heavily in `expressjs/express` (`Hit@5` dropped from `0.6111` to `0.3889`).
   - **`E2` (`e5-small-v2`)** achieved top `Span Recall@5` (`0.5741`) and strong `fmtlib/fmt` retrieval (`0.8333`), including conceptual Express matches (`express_017`, `express_018`) and `flask_002`, but lagged in `MRR@5` (`0.4164`) and compressed $L_2$ distances into a tight `[0.178, 0.353]` band.
   - **`E3` (`all-MiniLM-L12-v2`)** suffered severe degradation (`Hit@5=0.4815`, `NDCG@5=0.2679`) because its official `max_seq_length=128` config truncated **31.98% (1,366 / 4,272)** of chunks.
4. **Universal Dense Failures (`8 / 54`)**: All 5 dense models missed `flask_005`, `flask_017`, `flask_018`, `express_003`, `express_004`, `express_015`, `fmt_003`, and `fmt_005` because regex/sliding-window chunking emits noisy license preambles & 1-line fragments (`fmt_005`, `express_003`) while unit tests shadow source implementations (`flask_005`, `express_004`) and conceptual queries lack exact lexical tokens (`flask_018`).

---

## 3. Decision & Recommendation

- **Production Baseline Decision**: **Retain `V0` (`all-MiniLM-L6-v2` via Chroma ONNX Runtime)** per the balanced selection criterion — no candidate meaningfully outperformed V0 on overall `Hit@5` or `MRR@5`, while candidates incur +47% disk footprint (`126.93 MB` vs `86.2 MB`) and up to ~3x higher peak RSS (`1.85–1.88 GB` vs `625.6 MB`) during 512-token indexing.
- **Best Candidate Runner-Up**: Keep **`BAAI/bge-small-en-v1.5` (with retrieval query instruction)** staged for re-evaluation in subsequent days *after* Day 3 AST/Tree-sitter chunking eliminates 1-line test noise and license preambles.
- **Next Priority (Day 3)**: Fix **chunking quality** (Day 3 code-aware chunking) and subsequently introduce **hybrid BM25 + cross-encoder reranking** and **query rewriting**, where real structural gains are bounded.
