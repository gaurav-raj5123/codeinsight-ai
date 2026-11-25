# DAY 2 — TASK 1 REPORT: EMBEDDING OPTIMIZATION & CONTROLLED RETRIEVAL EXPERIMENT

**Project:** CodeInsight AI Assistant (RAG)  
**Experiment Suite ID:** `d2_emb_20261125_082152`  
**Execution Timestamp (UTC):** `2026-11-25T08:21:52.131273+00:00`  
**Status:** Completed (`VALID_COMPLETED`, 0 failed queries across 300 total evaluations)  
**Primary Artifacts:**  
- Reusable Metrics Module: [`evaluation/retrieval_metrics.py`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/evaluation/retrieval_metrics.py)  
- Model Registry Config: [`experiments/embedding_experiments/configs/embedding_models.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/embedding_experiments/configs/embedding_models.json)  
- Isolated Experiment Runner: [`experiments/embedding_experiments/run_embedding_comparison.py`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/embedding_experiments/run_embedding_comparison.py)  
- Aggregated & Detailed Results: [`embedding_summary.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/embedding_experiments/results/embedding_summary.json), [`embedding_comparison.json`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/embedding_experiments/results/embedding_comparison.json)  
- Concise Analysis Summary: [`experiments/results/day2_embedding_analysis.md`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/results/day2_embedding_analysis.md)  

---

## 1. Objective

Evaluate whether alternative ONNX-compatible 384-dimensional dense embedding models improve code retrieval accuracy over the V0 `all-MiniLM-L6-v2` baseline while maintaining acceptable indexing time, query latency, disk size, and RAM consumption on CPU. Embedding configuration is the **sole independent variable**; repository snapshots, chunks, metadata, Top-$K=5$, squared $L_2$ distance, and interval-overlap evaluation metrics are strictly held constant. No application code or Day 1 baseline files were modified.

---

## 2. Environment & V0 Pipeline Audit

### Host Runtime & Package Inventory
| Component | Observed Value |
| :--- | :--- |
| **OS / Platform** | Windows 11 (`Windows-11-10.0.26300-SP0`) |
| **Processor** | `AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD` (6 physical / 12 logical cores) |
| **System RAM** | 7.34 GB |
| **Python Version** | `3.14.7` |
| **ONNX Runtime** | `onnxruntime==1.29.0` (Providers: `['AzureExecutionProvider', 'CPUExecutionProvider']`) |
| **Vector Store** | `chromadb==1.5.9` |
| **Tokenizers & Hub** | `tokenizers==0.23.2`, `huggingface_hub==1.30.0`, `numpy==2.5.2`, `psutil==7.2.2` |

### Exact 10-Point Audit of Existing V0 Embedding Pipeline
1. **Model**: `all-MiniLM-L6-v2` (6 Transformer layers, hidden size 384, BERT WordPiece architecture).
2. **Backend**: ChromaDB `DefaultEmbeddingFunction` delegating to `ONNXMiniLM_L6_V2` running on `onnxruntime` (`CPUExecutionProvider`, `GraphOptimizationLevel.ORT_ENABLE_ALL`).
3. **Preprocessing & Tokenization**: HuggingFace `tokenizers.Tokenizer` loaded from cached `tokenizer.json` with `enable_truncation(max_length=256)` and fixed padding `enable_padding(pad_id=0, pad_token="[PAD]", length=256)`. Inputs passed to ONNX are `input_ids`, `attention_mask`, and zero-filled `token_type_ids` (`int64`). Pooling is `attention_mask`-weighted **mean pooling** over `last_hidden_state`.
4. **Vector Normalization**: Verified via direct inspection of `ONNXMiniLM_L6_V2._forward()` and runtime norm assertions (`[0.99999994, 1.0]`): **V0 vectors are already L2 unit-normalized** (`v / max(||v||_2, 1e-12)`).
5. **Embedding Dimensions**: `384` (`float32`).
6. **ChromaDB Distance Metric**: Default squared Euclidean distance (`l2`). Note: For unit vectors, $\|u-v\|_2^2 = 2 - 2\cos(u,v)$, yielding exact rank-equivalence with cosine similarity.
7. **Chunk Embedding Procedure (V0 App)**: `embedder.index_chunks()` deletes any existing collection and inserts chunks in batches of 100 (`collection.add(documents=..., metadatas=..., ids=...)`), allowing ChromaDB to invoke `DefaultEmbeddingFunction` internally.
8. **Query Embedding Procedure (V0 App)**: `embedder.search_similar()` passes raw `query_texts=[query]` to `collection.query(..., n_results=5)`.
9. **Existing Model / Cache Paths**:
   - Chroma ONNX model cache: `C:\Users\gaura\.cache\chroma\onnx_models\all-MiniLM-L6-v2\onnx\model.onnx` (`86.20 MB`, SHA256-verified archive from Chroma S3).
   - HuggingFace Hub cache: `C:\Users\gaura\.cache\huggingface\hub\`.
   - V0 / Day 1 ChromaDB store: `./chroma_db` (contains isolated `v0_eval_*` collections).
10. **Current V0 Timing Measurements & Limitations**:
    - Day 1 `embedder.index_chunks()` timed `collection.add()` (`289.40s` across 4,272 chunks), which conflated ONNX tokenization/forward passes with SQLite/HNSW insertion.
    - Day 1 `embedder.search_similar()` timed `collection.query()` (`187.62 ms` mean), conflating client/tokenizer overhead, fixed-256 query padding, ONNX inference, and vector search.
    - Day 2 explicitly decouples document embedding from `collection.add(embeddings=...)` and query embedding from `collection.query(query_embeddings=...)`.

---

## 3. Model & Backend Compatibility Preflight

All four candidate architectures were verified directly on Windows Python 3.14.7 using official ONNX exports and `onnxruntime 1.29.0` (`CPUExecutionProvider`), eliminating PyTorch runtime dependencies from the benchmark and production candidate path. Exact numerical parity against reference `SentenceTransformer` / Chroma outputs was confirmed via preflight smoke tests:

| Candidate | Model Repository & ONNX File | Size (MB) | Parity vs Reference (`max_abs_diff` / `min_cos_sim`) | Preflight Status |
| :--- | :--- | :---: | :---: | :--- |
| **E0** | Chroma `all-MiniLM-L6-v2` (`onnx/model.onnx`) | 86.20 | `0.00e+00` / `1.00000000` (vs Chroma `DefaultEF`) | Available (Direct ONNX) |
| **E1** | `BAAI/bge-small-en-v1.5` (`onnx/model.onnx`) | 126.93 | `1.38e-07` / `1.00000000` (vs `SentenceTransformer`) | Available (Direct ONNX) |
| **E2** | `intfloat/e5-small-v2` (`model.onnx`) | 126.93 | `2.78e-07` / `1.00000000` (vs `SentenceTransformer`) | Available (Direct ONNX) |
| **E3** | `sentence-transformers/all-MiniLM-L12-v2` (`onnx/model.onnx`) | 126.96 | `1.31e-07` / `1.00000001` (vs `SentenceTransformer`) | Available (Direct ONNX) |

*Note on `fastembed`*: `fastembed` is not installed in the active environment; direct `onnxruntime` + HuggingFace `tokenizers` execution ran all models identically under one unified ONNX Runtime backend with zero package bloat.

---

## 4. Experimental Methodology & Controls

1. **Corpus & Chunking Fixed Once**: Pinned snapshots (`pallets/flask` `49943b5`, `expressjs/express` `43fca52`, `fmtlib/fmt` `3be6709`) verified via `.commit_sha` were loaded and chunked once (`repo_loader.get_source_files()` $\to$ `chunker.chunk_code()`), producing **4,272 chunks** (Flask: 1,785; Express: 2,066; fmt: 421) in `0.3839s`. Identical chunk texts and metadata were fed to every model.
2. **Vector Store Isolation**: Each model indexed into fresh collections inside a dedicated persistent directory (`experiments/embedding_experiments/chroma_exp_db/`, ignored in `.gitignore`), deleting any colliding name before creation and enforcing `metadata={"hnsw:space": "l2"}`. Zero stale collections or cached Day 1 records were reused.
3. **Metric Extraction Validation**: [`evaluation/retrieval_metrics.py`](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/evaluation/retrieval_metrics.py) uses Day 1's corrected corpus-level IDCG@5 (capped at 5), fixed Precision@5 denominator, unique span coverage, `statistics.median` for P50, and `math.ceil(0.95 * n) - 1` for nearest-rank P95. Unit tests confirmed exact Day 1 parity (`MRR@5 = 0.4534`, `NDCG@5 = 0.3339`).
4. **Warm-Up & Steady-State Timing Separation**: After timed model load (`model_load_duration_seconds`), 3 synthetic chunks + 2 queries execute in a warm-up pass (`warmup_duration_seconds`, discarded). Document embedding (`batch_size=32`), Chroma insertion (`batch_size=100`), single-query embedding, and Chroma vector search (`n_results=5`) are all timed independently. Length-sorted dynamic batching (verified `0.00e+00` numerical diff vs unsorted dynamic batching) was used for candidate `dynamic_batch` models to avoid padding waste while returning vectors in exact original chunk order.

---

## 5. Models Evaluated

| Experiment ID | Group | Model Identifier | Architecture Depth | Official Source / Version |
| :--- | :---: | :--- | :---: | :--- |
| **`E0_v0_minilm_l6_v2`** | `E0` | `all-MiniLM-L6-v2` (V0 baseline) | 6 layers, 384-d | `chroma_onnx_sha256:913d7300ceae` |
| **`E1a_bge_small_en_v15_no_prefix`** | `E1` | `BAAI/bge-small-en-v1.5` (no prefix) | 12 layers, 384-d | `hf_commit:5c38ec7c405e` |
| **`E1b_bge_small_en_v15_instruct`** | `E1` | `BAAI/bge-small-en-v1.5` (query instruct) | 12 layers, 384-d | `hf_commit:5c38ec7c405e` |
| **`E2_e5_small_v2`** | `E2` | `intfloat/e5-small-v2` | 12 layers, 384-d | `hf_commit:ffb93f3bd404` |
| **`E3_minilm_l12_v2`** | `E3` | `sentence-transformers/all-MiniLM-L12-v2` | 12 layers, 384-d | `hf_commit:a50ef00143b4` |

---

## 6. Normalization and Preprocessing Configuration

Because our audit confirmed **V0 vectors are already unit-normalized** (`||v||_2 = 1.0`), Task 4's rule (*"avoid redundant experiments"* when V0 is already normalized) applies: one exact V0 control (`E0`) places all models in the exact same normalized squared-$L_2$ metric space.

To test BGE v1.5's documentation faithfully (*"The best method to decide whether to add instructions for queries is choosing the setting that achieves better performance on your task"* — BGE FAQ #3), `E1` was evaluated both without a query prefix (`E1a`) and with BGE's official retrieval prefix (`E1b`), sharing BGE's identical freshly generated zero-prefix document vectors within the same run.

| Exp ID | Model Identifier | Pooling | Max Seq Len | Padding Strategy | Doc Prefix | Query Prefix | Measured Chunks Truncated | Measured Doc/Query Norm Range |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E0`** | `all-MiniLM-L6-v2` (V0) | `mean` | 256 | `fixed_256` | `""` | `""` | **624 (14.61%)** | `[1.000000, 1.000000]` |
| **`E1a`** | `BAAI/bge-small-en-v1.5` | `cls` | 512 | `dynamic_batch` | `""` | `""` | **233 (5.45%)** | `[1.000000, 1.000000]` |
| **`E1b`** | `BAAI/bge-small-en-v1.5` | `cls` | 512 | `dynamic_batch` | `""` | `"Represent this sentence for searching relevant passages: "` | *(shared w/ E1a)* | `[1.000000, 1.000000]` |
| **`E2`** | `intfloat/e5-small-v2` | `mean` | 512 | `dynamic_batch` | `"passage: "` | `"query: "` | **236 (5.52%)** | `[1.000000, 1.000000]` |
| **`E3`** | `all-MiniLM-L12-v2` | `mean` | 128 | `dynamic_batch` | `""` | `""` | **1,366 (31.98%)** | `[1.000000, 1.000000]` |

*Note on `E3` (`all-MiniLM-L12-v2`)*: Its official `sentence_bert_config.json` and `tokenizer.json` both mandate `max_seq_length = 128`, truncating nearly one-third of chunks (`50.12%` of `fmt` chunks exceed 256 tokens, and even more exceed 128).

---

## 7. Overall Metrics (54 Answerable Queries)

| Model Config | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | Precision@5 | Span Recall@5 | MRR@5 | NDCG@5 | Failures |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E0` (`all-MiniLM-L6-v2`, V0)** | **`0.3148`** | **`0.5741`** | **`0.6667`** | 0.1815 | 0.5556 | **`0.4534`** | 0.3339 | 0 / 60 |
| **`E1a` (`bge-small-en-v1.5`, No Prefix)** | 0.2593 | **`0.5741`** | 0.6111 | 0.1889 | 0.5185 | 0.3963 | 0.3370 | 0 / 60 |
| **`E1b` (`bge-small-en-v1.5`, Instruct)** | **`0.3148`** | **`0.5741`** | 0.6296 | **`0.1963`** | 0.5463 | 0.4503 | **`0.3661`** | 0 / 60 |
| **`E2` (`e5-small-v2`)** | 0.2963 | 0.5185 | 0.6481 | 0.1889 | **`0.5741`** | 0.4164 | 0.3529 | 0 / 60 |
| **`E3` (`all-MiniLM-L12-v2`)** | 0.2778 | 0.4074 | 0.4815 | 0.1185 | 0.3796 | 0.3509 | 0.2679 | 0 / 60 |

*Key Observations*:
- `E0` exactly reproduced Day 1's baseline (`MRR@5 = 0.4534`, `NDCG@5 = 0.3339`) and retained the highest overall `Hit@5` (`0.6667`) and `MRR@5` (`0.4534`).
- Adding BGE's retrieval query instruction (`E1b` vs `E1a`) delivered consistent improvements across every metric (`Hit@1`: `+0.0555`, `MRR@5`: `+0.0540`, `NDCG@5`: `+0.0291`), achieving the highest overall `NDCG@5` (`0.3661`) and `Precision@5` (`0.1963`).
- `E2` achieved the highest overall `Span Recall@5` (`0.5741`), narrowly ahead of V0 (`0.5556`), while trailing in `Hit@1` and `MRR@5`.

---

## 8. Repository-Wise Metrics (18 Answerable Queries per Repository)

| Repository | Config | Hit@1 | Hit@3 | Hit@5 | Precision@5 | SpanRec@5 | MRR@5 | NDCG@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pallets/flask`** (Python, AST) | **`E0` (V0)** | **`0.2778`** | 0.5556 | 0.6667 | 0.2222 | 0.5556 | 0.4204 | 0.3323 |
| | `E1a` (BGE NoPrefix) | 0.1111 | 0.6111 | 0.6667 | 0.2556 | 0.5556 | 0.3167 | 0.3202 |
| | **`E1b` (BGE Instruct)** | **`0.2778`** | **`0.6667`** | **`0.7222`** | **`0.2667`** | **`0.6111`** | **`0.4833`** | **`0.3986`** |
| | `E2` (E5-small-v2) | 0.1667 | 0.4444 | 0.6111 | 0.1556 | 0.5278 | 0.3287 | 0.2748 |
| | `E3` (MiniLM-L12) | 0.1667 | 0.2778 | 0.3889 | 0.1000 | 0.3333 | 0.2407 | 0.2025 |
| **`expressjs/express`** (JS, Regex) | **`E0` (V0)** | 0.2778 | **`0.5000`** | **`0.6111`** | 0.1444 | **`0.4722`** | **`0.3981`** | 0.2080 |
| | `E1a` (BGE NoPrefix) | 0.2222 | 0.2778 | 0.3333 | 0.0889 | 0.2500 | 0.2611 | 0.1496 |
| | `E1b` (BGE Instruct) | 0.2222 | 0.3333 | 0.3889 | 0.1000 | 0.3056 | 0.2824 | 0.1677 |
| | **`E2` (E5-small-v2)** | **`0.3333`** | 0.4444 | 0.5000 | **`0.1556`** | 0.4167 | 0.3907 | **`0.2319`** |
| | `E3` (MiniLM-L12) | 0.2222 | 0.2778 | 0.3889 | 0.0778 | 0.2222 | 0.2750 | 0.1201 |
| **`fmtlib/fmt`** (C++, Regex/Split) | `E0` (V0) | 0.3889 | 0.6667 | 0.7222 | 0.1778 | 0.6389 | 0.5417 | 0.4615 |
| | **`E1a` (BGE NoPrefix)** | **`0.4444`** | **`0.8333`** | **`0.8333`** | 0.2222 | 0.7500 | **`0.6111`** | 0.5412 |
| | `E1b` (BGE Instruct) | **`0.4444`** | 0.7222 | 0.7778 | 0.2222 | 0.7222 | 0.5852 | 0.5320 |
| | **`E2` (E5-small-v2)** | 0.3889 | 0.6667 | **`0.8333`** | **`0.2556`** | **`0.7778`** | 0.5296 | **`0.5520`** |
| | `E3` (MiniLM-L12) | **`0.4444`** | 0.6667 | 0.6667 | 0.1778 | 0.5833 | 0.5370 | 0.4812 |

*Insight*: A sharp repository split emerges. `E1b` leads on clean Python AST chunks (`flask` Hit@5 = `0.7222`, NDCG = `0.3986`), and `E1a`/`E2` dominate C++ headers (`fmt` Hit@5 = `0.8333`, because 512 tokens avoids V0's `50.12%` truncation on `fmt`), whereas **V0 preserves strong performance on noisy Express JS regex chunks (`0.6111`)** where BGE (`0.3333`–`0.3889`) disproportionately promotes `test/*.js` and `examples/*` matches.

---

## 9. Category-Wise Metrics

| Category (N) | Config | Hit@1 | Hit@5 | SpanRec@5 | MRR@5 | NDCG@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`architecture_flow`** ($N=10$) | `E0` / `E1a` / **`E1b`** / `E2` / `E3` | 0.30 / 0.30 / 0.30 / 0.20 / **0.40** | **0.90** / 0.70 / 0.80 / 0.80 / 0.60 | **0.70** / 0.55 / 0.65 / **0.70** / 0.50 | 0.5283 / 0.4667 / **0.5333** / 0.4167 / 0.4533 | 0.3165 / 0.3779 / **0.4192** / 0.3578 / 0.2940 |
| **`cross_file_dependencies`** ($N=10$) | **`E0`** / `E1a` / `E1b` / `E2` / `E3` | **0.40** / 0.20 / 0.30 / **0.40** / 0.20 | **1.00** / 0.80 / 0.80 / 0.70 / 0.70 | **0.65** / 0.50 / 0.55 / 0.50 / 0.35 | **0.6667** / 0.3900 / 0.5083 / 0.5083 / 0.4250 | **0.4650** / 0.3067 / 0.3490 / 0.4041 / 0.2619 |
| **`debugging_configuration`** ($N=10$) | `E0` / `E1a` / `E1b` / **`E2`** / `E3` | **0.40** / 0.30 / 0.30 / 0.20 / 0.20 | 0.50 / 0.40 / 0.40 / **0.70** / 0.40 | 0.45 / 0.40 / 0.40 / **0.65** / 0.35 | **0.4250** / 0.3333 / 0.3200 / 0.3650 / 0.2583 | 0.2882 / 0.2088 / 0.2256 / **0.3221** / 0.1851 |
| **`function_identification`** ($N=12$) | `E0` / `E1a` / **`E1b`** / **`E2`** / `E3` | 0.25 / 0.25 / 0.25 / **0.4167** / **0.4167** | 0.50 / 0.5833 / **0.6667** / 0.5833 / 0.50 | 0.50 / 0.5417 / **0.6250** / 0.5417 / 0.4583 | 0.3361 / 0.4167 / 0.4333 / **0.4722** / 0.4444 | 0.3765 / 0.3957 / **0.4280** / 0.4007 / 0.4141 |
| **`implementation_understanding`** ($N=12$) | `E0` / **`E1a`** / **`E1b`** / `E2` / `E3` | 0.25 / 0.25 / **0.4167** / 0.25 / 0.1667 | 0.50 / **0.5833** / 0.50 / 0.50 / 0.25 | 0.50 / **0.5833** / 0.50 / 0.50 / 0.25 | 0.3542 / 0.3750 / **0.4583** / 0.3264 / 0.1875 | 0.2348 / 0.3762 / **0.3911** / 0.2839 / 0.1740 |

---

## 10. Query-Style Metrics

| Query Style (N) | Config | Hit@1 | Hit@3 | Hit@5 | SpanRec@5 | MRR@5 | NDCG@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`identifier`** ($N=22$) | `E0` (V0) | 0.3182 | 0.5000 | 0.5909 | 0.5455 | 0.4144 | 0.3949 |
| | `E1a` (BGE NoPrefix) | 0.3636 | 0.6818 | 0.6818 | 0.5909 | 0.5000 | 0.4117 |
| | **`E1b` (BGE Instruct)** | **`0.4545`** | **`0.7273`** | **`0.7727`** | **`0.7045`** | **`0.5848`** | **`0.4865`** |
| | `E2` (E5-small-v2) | **`0.4545`** | **`0.7273`** | 0.7273 | 0.6591 | 0.5606 | 0.4591 |
| | `E3` (MiniLM-L12) | 0.4091 | 0.5455 | 0.6364 | 0.5455 | 0.4826 | 0.4006 |
| **`conceptual`** ($N=20$) | **`E0` (V0)** | **`0.3000`** | **`0.6000`** | **`0.6500`** | **`0.5750`** | **`0.4517`** | 0.2758 |
| | `E1a` (BGE NoPrefix) | 0.2000 | 0.5000 | 0.5500 | 0.5000 | 0.3267 | **`0.2771`** |
| | `E1b` (BGE Instruct) | 0.2500 | 0.4500 | 0.5000 | 0.4500 | 0.3600 | 0.2594 |
| | `E2` (E5-small-v2) | 0.2000 | 0.4000 | 0.6000 | 0.5250 | 0.3333 | 0.2678 |
| | `E3` (MiniLM-L12) | 0.1500 | 0.1500 | 0.2000 | 0.1500 | 0.1625 | 0.1049 |
| **`conversational`** ($N=12$) | **`E0` (V0)** | **`0.3333`** | **`0.6667`** | **`0.8333`** | **`0.5417`** | **`0.5278`** | 0.3190 |
| | `E1a` (BGE NoPrefix) | 0.1667 | 0.5000 | 0.5833 | 0.4167 | 0.3222 | 0.2997 |
| | `E1b` (BGE Instruct) | 0.1667 | 0.5000 | 0.5833 | 0.4167 | 0.3542 | **`0.3230`** |
| | `E2` (E5-small-v2) | 0.1667 | 0.3333 | 0.5833 | 0.5000 | 0.2903 | 0.3000 |
| | `E3` (MiniLM-L12) | 0.2500 | 0.5833 | 0.6667 | 0.4583 | 0.4236 | 0.2964 |

---

## 11. Single Versus Multi-Evidence Results

| Complexity | Config | Hit@1 | Hit@3 | Hit@5 | SpanRec@5 | MRR@5 | NDCG@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Evidence** ($N=37$) | `E0` (V0) | 0.2703 | 0.4054 | 0.5405 | 0.5405 | 0.3644 | 0.2992 |
| | `E1a` (BGE NoPrefix) | 0.2703 | **`0.5405`** | 0.5405 | 0.5405 | 0.3829 | 0.3373 |
| | **`E1b` (BGE Instruct)** | **`0.3243`** | 0.5135 | 0.5676 | 0.5676 | **`0.4252`** | **`0.3660`** |
| | **`E2` (E5-small-v2)** | 0.2703 | 0.4865 | **`0.6216`** | **`0.6216`** | 0.3928 | 0.3432 |
| | `E3` (MiniLM-L12) | 0.2162 | 0.2703 | 0.3514 | 0.3514 | 0.2532 | 0.2404 |
| **Multi-Evidence** ($N=17$) | **`E0` (V0)** | **`0.4118`** | **`0.9412`** | **`0.9412`** | **`0.5882`** | **`0.6471`** | **`0.4095`** |
| | `E1a` (BGE NoPrefix) | 0.2353 | 0.6471 | 0.7647 | 0.4706 | 0.4255 | 0.3364 |
| | `E1b` (BGE Instruct) | 0.2941 | 0.7059 | 0.7647 | 0.5000 | 0.5049 | 0.3662 |
| | `E2` (E5-small-v2) | 0.3529 | 0.5882 | 0.7059 | 0.4706 | 0.4676 | 0.3740 |
| | `E3` (MiniLM-L12) | **`0.4118`** | 0.7059 | 0.7647 | 0.4412 | 0.5637 | 0.3278 |

---

## 12. Unanswerable-Query Analysis

All distances represent ChromaDB squared Euclidean ($L_2$) distances on unit-normalized 384-d vectors across answerable ($N=54$) vs unanswerable ($N=6$) queries:

| Config | Answerable Top-1 $L_2$ (Min / Med / Mean / Max) | Unanswerable Top-1 $L_2$ (Min / Med / Mean / Max) | Unanswerable Mean Top-5 $L_2$ (Min / Med / Mean / Max) |
| :--- | :---: | :---: | :---: |
| **`E0` (V0)** | `0.3949 / 0.8626 / 0.8496 / 1.5378` | `0.8314 / 1.0685 / 1.0332 / 1.2868` | `0.9107 / 1.0985 / 1.1018 / 1.3427` |
| **`E1a` (BGE NoPrefix)** | `0.2414 / 0.3969 / 0.3926 / 0.5747` | `0.4291 / 0.5182 / 0.5019 / 0.5429` | `0.4418 / 0.5524 / 0.5288 / 0.5831` |
| **`E1b` (BGE Instruct)** | `0.2921 / 0.4345 / 0.4317 / 0.6334` | `0.4873 / 0.5612 / 0.5620 / 0.6227` | `0.5021 / 0.5844 / 0.5850 / 0.6682` |
| **`E2` (E5-small-v2)** | `0.1781 / 0.2490 / 0.2492 / 0.3209` | `0.2360 / 0.2835 / 0.2921 / 0.3535` | `0.2532 / 0.2893 / 0.3019 / 0.3716` |
| **`E3` (MiniLM-L12)** | `0.4070 / 0.8239 / 0.8170 / 1.3467` | `0.8768 / 1.1085 / 1.1049 / 1.3341` | `1.0060 / 1.1377 / 1.1809 / 1.4895` |

*Analysis*: Across **every** model, the minimum unanswerable Top-1 $L_2$ distance overlaps substantially inside the answerable distribution (e.g., `E0`: `0.8314` vs `1.5378`; `E1b`: `0.4873` vs `0.6334`; `E2`: `0.2360` vs `0.3209`). BGE and E5 additionally compress squared $L_2$ scores into narrow bands due to low contrastive training temperatures (noted in BGE FAQ #2). Raw $L_2$ distance alone **cannot** safely gate abstention without unacceptable false rejections.

---

## 13. Performance Comparison

### A. Indexing, Footprint & Cold-Start Telemetry (4,272 Chunks)
| Config | Size (MB) | Cold Load (s) | Warm-Up (s, excl.) | Doc Embed (s) | Avg/Chunk (ms) | Chroma Insert (s) | Total Index (s) | Load RSS Delta (MB) | Max Sampled RSS (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E0` (V0)** | **86.20** | **0.3383** | 0.2466 | 300.0373 | 70.233 | **4.7447** | 304.7820 | **+108.45** | **625.64** |
| **`E1a` (BGE)** | 126.93 | 0.5529 | 0.0438 | 269.0415 | 62.978 | 6.1248 | 275.1663 | +149.69 | 1,852.93 |
| **`E1b` (BGE Instruct)** | 126.93 | 0.5824 | 0.0437 | *(0.00 / E1a)* | *—* | 8.1397 | *(8.1397)* | +153.40 | *(323.99)* |
| **`E2` (E5-small-v2)** | 126.93 | 0.4642 | 0.0495 | 288.4379 | 67.518 | 6.3410 | 294.7789 | +142.53 | 1,875.68 |
| **`E3` (MiniLM-L12)** | 126.96 | 0.6475 | 0.0647 | **117.6595** | **27.542** | 6.8824 | **124.5419** | +144.42 | 539.08 |

*Note*: `E0` runs slower on `doc_embed` than 12-layer `E1a` solely because V0 fixes padding at 256 tokens (`fixed_256`), whereas `E1a`/`E2`/`E3` use length-sorted dynamic batch padding (`dynamic_batch`, where median chunks are 84 tokens). `E1b` shares `E1a`'s identical zero-prefix passage vectors within the same run; `323.99 MB` peak RSS for `E1b` only reflects query execution + Chroma insert without 512-token doc ONNX forward passes.

### B. Steady-State Query Latency Separation (ms)
| Config | Answerable Total (N=54) Mean / P50 / P95 | All Queries Total (N=60) Mean / P50 / P95 | Query Embed Only (N=54) Mean / P50 / P95 | Vector Search Only (N=54) Mean / P50 / P95 |
| :--- | :---: | :---: | :---: | :---: |
| **`E0` (V0, fixed 256)** | `33.31 / 32.27 / 41.57` | `33.50 / 32.34 / 42.16` | `29.59 / 29.46 / 36.09` | `3.72 / 2.62 / 9.34` |
| **`E1a` (BGE NoPrefix)** | **`13.26 / 11.52 / 25.48`** | **`13.12 / 11.55 / 13.32`** | **`9.15 / 9.05 / 10.58`** | `4.11 / 2.38 / 16.18` |
| **`E1b` (BGE Instruct)** | `18.19 / 15.79 / 42.43` | `17.97 / 15.82 / 24.75` | `12.85 / 12.75 / 15.38` | `5.34 / 3.02 / 28.95` |
| **`E2` (E5-small-v2)** | `14.83 / 12.93 / 35.76` | `14.63 / 12.93 / 15.03` | `10.41 / 10.35 / 11.81` | `4.42 / 2.53 / 25.06` |
| **`E3` (MiniLM-L12)** | `15.24 / 13.07 / 42.79` | `15.02 / 13.07 / 14.89` | `10.62 / 10.48 / 12.30` | `4.62 / 2.57 / 31.71` |

*Insight*: ChromaDB HNSW vector search across ~2K chunks takes only **2.4–3.0 ms** median once embeddings are computed. Total steady-state query latency is overwhelmingly dominated by ONNX query encoding, where dynamic padding reduces encoding time from `29.5 ms` (fixed 256 in V0) down to `9.1–12.8 ms` even on 12-layer models.

---

## 14. Error Analysis

### Transition Counts vs V0 Baseline (54 Answerable Queries)
| Candidate Config | Solved by Model (Missed by V0) | Regressed vs V0 (Solved by V0) | Net Hit@5 Delta | Universal Misses Across All Models |
| :--- | :---: | :---: | :---: | :---: |
| **`E1a` (`bge-small` NoPrefix)** | **6** (`flask_007, flask_016, express_001, fmt_002, fmt_004, fmt_013`) | **9** (`flask_001, flask_015, express_002, express_008, express_010, express_011, express_014, express_016, fmt_014`) | `-3` (`33` vs `36`) | **8** |
| **`E1b` (`bge-small` Instruct)** | **7** (`flask_007, flask_016, express_001, express_013, fmt_002, fmt_004, fmt_013`) | **9** (`flask_015, express_002, express_008, express_010, express_011, express_014, express_016, fmt_014, fmt_016`) | `-2` (`34` vs `36`) | **8** |
| **`E2` (`e5-small-v2`)** | **8** (`flask_002, flask_007, express_001, express_013, express_017, express_018, fmt_002, fmt_013`) | **9** (`flask_001, flask_006, flask_009, express_002, express_006, express_007, express_008, express_009, express_010`) | `-1` (`35` vs `36`) | **8** |
| **`E3` (`all-MiniLM-L12`)** | **4** (`flask_007, express_001, express_013, fmt_002`) | **14** (`flask_004, flask_008, flask_009, flask_011, flask_012, flask_015, express_002, express_005, express_007, express_008, express_012, express_016, fmt_014, fmt_016`) | `-10` (`26` vs `36`) | **8** |

### Representative Failure & Recovery Case Studies (Evidence-Supported)

1. **Solved by Alternatives, Missed by V0 (Identifier Precision & 512-Token Context)**:
   - **`express_001`** (`createApplication`): Ground truth `lib/express.js:36-57`. V0 top-1 was `test/req.query.js:94-96` ($L_2=0.7332$), missing `lib/express.js` completely. **All 4 candidate configs (`E1a`, `E1b`, `E2`, `E3`) ranked `lib/express.js:36-82` at Rank #1** (e.g., `E1b` $L_2=0.3975$, `E2` $L_2=0.2058$). *(Classification: Identifier mismatch / test pollution in V0)*.
   - **`flask_016`** (`Config.from_object`): Ground truth `src/flask/config.py:218-250`. V0 matched `tests/test_config.py:26-29` ($L_2=0.3949$). `E1a` hit `src/flask/config.py:210-259` at Rank #3 and **`E1b` promoted it to Rank #1** ($L_2=0.3413$). *(Classification: Identifier mismatch / test pollution in V0)*.
   - **`fmt_004`** (`dynamic_format_arg_store`): Ground truth `include/fmt/args.h:74-110`. V0 truncated past 256 tokens and matched `args.h:1-23`. **`E1a` & `E1b` retrieved `include/fmt/args.h:43-92` at Rank #2** (`E1a` $L_2=0.3127$, `E1b` $L_2=0.3549$). *(Classification: Chunking / truncation limitation)*.
   - **`express_017` & `express_018`** (`res.cookie`, `app.engine`): Both missed by V0 and BGE; **`E2` solved both at Rank #1** (`lib/response.js:687-752`, $L_2=0.2467$; `lib/application.js:221-294`, $L_2=0.2370$).

2. **Solved by V0, Regressed by Candidates (Test Shadowing on Express & Conversational Drift)**:
   - **`express_008`** (`res.sendFile`): Ground truth `lib/response.js:375-425`. `E0` retrieved `lib/response.js:408-412` at Rank #2 ($L_2=1.0287$). **`E1a`, `E1b`, `E2`, and `E3` all ranked `examples/static-files/index.js:1-44` at Rank #1** followed by `test/res.sendFile.js:471-486`, pushing `lib/response.js` outside Top-5. *(Classification: Semantic mismatch toward tutorial/test prose)*.
   - **`express_011`** (`app.render` + `View.prototype.render`): Ground truth `lib/application.js:522-575` + `lib/view.js:133-165`. `E0` and `E2` both hit Rank #1 (`lib/application.js:544-548`), whereas `E1a` & `E1b` ranked `test/res.render.js:189-206` and `test/app.render.js` top-3. *(Classification: Cross-file evidence & test file pollution)*.
   - **`express_014`** (`compileTrust` / `req.ip`, Turn-2 conversational): `E0` hit `lib/request.js:343-392` at Rank #1 ($L_2=0.9533$); `E1a` & `E1b` ranked `test/req.ips.js:10-24` and `test/req.ip.js` top-3. *(Classification: Query ambiguity without Turn-1 rewrite)*.

3. **Universal Dense Failures — Missed by All 5 Configs (`8 / 54`)**:
   - **`flask_005`** (`route` decorator, `src/flask/sansio/scaffold.py:344-370`): All 5 models pulled `tests/test_blueprints.py` & `tests/test_basic.py` into Top-3. *(Test pollution)*.
   - **`flask_017`** (`with_appcontext`, `src/flask/cli.py:380-410`): Conceptual phrasing matched `AppGroup`/`load_dotenv` in `cli.py` instead of `with_appcontext`. *(Semantic mismatch)*.
   - **`flask_018`** (`flash`, `src/flask/helpers.py:326-365`): Pure conceptual query lacking keyword `flash` pulled `redirect` (`E0`/`E3`) or `tests/test_signals.py` (`E1a`/`E1b`/`E2`). *(Semantic mismatch)*.
   - **`express_003`** (`View`, `lib/view.js:52-80`): All 5 models ranked 1-line `test/app.render.js:277-277` at Rank #1. *(Chunking fragmentation + test pollution)*.
   - **`express_004`** (`req.header`, `lib/request.js:63-85`): All 5 models matched `test/res.get.js`, `test/req.get.js`, or `test/express.static.js` above `lib/request.js`. *(Test pollution)*.
   - **`express_015`** (`compileETag`, `lib/utils.js:130-152`): Models matched `lib/utils.js:1-88`, `test/utils.js`, or `test/req.stale.js`, missing line 130. *(Wrong abstraction / header split)*.
   - **`fmt_003`** (`buffered_file`, `include/fmt/os.h:167-205`): All 5 models ranked `include/fmt/ostream.h` Rank #1 due to conceptual phrase "file streams". *(Semantic mismatch)*.
   - **`fmt_005`** (`struct is_range`, `include/fmt/ranges.h:370-410`): All 5 models ranked `include/fmt/ranges.h:1-62` (license header & macros!) at Rank #1. *(Chunking limitation — regex header preamble)*.

---

## 15. Quality-versus-Cost Tradeoffs

| Criterion | Winner | Assessment |
| :--- | :--- | :--- |
| **Overall Hit@5 & MRR@5** | **`E0` (V0 MiniLM-L6)** | `0.6667` / `0.4534` — undefeated overall despite 6-layer size |
| **Corpus NDCG@5 & Identifier Precision** | **`E1b` (BGE-small Instruct)** | `0.3661` NDCG@5 (`+0.0322` over V0), `+18.2%` Hit@5 on identifier queries |
| **Span Recall@5 & Single-Evidence Hit@5** | **`E2` (E5-small-v2)** | `0.5741` SpanRec@5 (`+0.0185` over V0), best on single-evidence (`0.6216`) |
| **ExpressJS / Multi-Evidence / Conversational** | **`E0` (V0 MiniLM-L6)** | Leads Express (`0.6111`), Multi-evidence (`0.9412`), Conversational (`0.8333`) |
| **Memory & Disk Footprint** | **`E0` (V0 MiniLM-L6)** | `86.20 MB` disk (-32%), `625.64 MB` max RSS vs `1,853–1,876 MB` (~3x lower) |
| **Steady-State Query Latency** | **`E1a` / `E1b` (dynamic)** | `11.52 / 15.79 ms` P50 (note: V0's `32.27 ms` stems solely from fixed-256 query padding) |

---

## 16. Recommended Model & Configuration

**Decision: Retain V0 (`all-MiniLM-L6-v2` via Chroma ONNX Runtime) as the active production dense embedding baseline.**  
Per Task 11 (*"If no candidate meaningfully improves on the baseline, retain V0"*), replacing V0 right now is **not** engineering-justified:
- Neither `BGE-small-en-v1.5` nor `E5-small-v2` outperformed V0 on overall `Hit@5` (`0.6667`) or `MRR@5` (`0.4534`).
- `BGE` (`E1b`) incurred a severe regression on `expressjs/express` (`-22.2%` Hit@5) and `conversational` queries (`-25.0%` Hit@5), while requiring +47% disk storage (`126.93 MB`) and ~3x higher peak RSS (`1.85 GB` vs `626 MB`) during 512-token indexing on our 7.34 GB host.
- `MiniLM-L6-v2` benefited directly from `CodeSearchNet` pretraining data paired with natural language, explaining its resilience on `conceptual` and `expressjs/express` queries.
- **Keep `E1b` (`BAAI/bge-small-en-v1.5` + query instruction) as the top alternative candidate** for a follow-up check *after* Day 3 chunking cleanup fixes noisy regex fragments and header preambles.

---

## 17. Limitations

1. **Chunking Confounding Effect on Express & C++**: Regex chunking in `expressjs/express` creates 1-line fragments (`test/app.render.js:277`), and C++ headers emit 60-line license preambles (`fmt_005`). Stronger semantic models (`BGE`, `E5`) over-index on comment/test prose in noisy chunks, partially masking their representation potential until chunking is fixed.
2. **In-Run Shared Document Embeddings for `E1b`**: Within run `d2_emb_20261125_082152`, `E1b` shared `E1a`'s freshly generated zero-prefix document vectors (mathematically identical for BGE) while generating fresh query embeddings and fresh Chroma collections (`8.14s` Chroma insert); thus `E1b`'s standalone `doc_embedding_duration_seconds` mirrors `E1a` (`269.04s`).
3. **Host Process RSS Measurement**: `psutil` RSS reflects Windows working set across sequential model stages within one process after explicit `gc.collect()`, rather than cold-boot isolated subprocess RSS.
4. **No Lexical or Reranking Stages Tested**: Per Day 2 isolation rules, dense embeddings were tested strictly standalone without BM25 or cross-encoder reranking.

---

## 18. Recommendation for Day 3

1. **Prioritize Code-Aware Chunking (Tree-sitter / Improved AST) on Day 3**: Fixing broken regex boundaries (`fmt_005`, `express_003`, `express_015`), dropping duplicate nested Python `ast.walk` chunks, and capturing enclosing class/function signatures in chunk context will directly attack both universal failures and V0's `50.12%` `fmt` truncation bottleneck.
2. **Re-benchmark `BGE-small-en-v1.5 (instruct)` on Clean Day 3 Chunks**: Once 1-line test noise and license preambles are eliminated, `BGE`'s strong `NDCG@5` (`0.3661`) and `+18.2%` identifier accuracy should translate cleanly without Express regressions.
3. **Plan Hybrid Lexical (BM25) + Cross-Encoder Reranking & Query Rewriting for Subsequent Days**: Universal misses like `flask_005`, `flask_017`, `flask_018`, `express_004`, and `fmt_003` demonstrate that pure dense embeddings—regardless of backbone—cannot resolve exact symbol shadowing (`tests/` vs `lib/`) or zero-lexical-overlap conceptual queries alone.
