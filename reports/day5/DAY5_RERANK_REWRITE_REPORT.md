# DAY 5 — ONNX CROSS-ENCODER RERANKING & CONVERSATIONAL QUERY REWRITING REPORT (`DAY5_RERANK_REWRITE_REPORT.md`)

**Project:** CodeInsight AI Assistant (GitHub Codebase RAG)  
**Benchmark Script:** [run_day5_benchmark.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day5_rerank_rewrite/run_day5_benchmark.py)  
**Summary Artifacts:** [day5_summary.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day5_rerank_rewrite/results/day5_summary.json), [conversational_subset_analysis.json](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/experiments/day5_rerank_rewrite/results/conversational_subset_analysis.json)  
**Production Modules:** [retrieval/reranker.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/retrieval/reranker.py), [retrieval/query_rewriter.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/retrieval/query_rewriter.py), [embedder.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/embedder.py), [app.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/app.py)  

---

## 1. Executive Summary

Day 5 evaluated **Stage-2 CPU ONNX Cross-Encoder Reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and **Leakage-Free Gated Conversational Query Rewriting** (`llm.ask_gemini`) atop the Day 4 **`code_aware_v1` + Hybrid RRF (`k=60`)** baseline, reusing the exact pinned 3-repository corpus (`3,721` V1 chunks across 253 files) and achieving **exact `0.0` parity deltas** against Day 4 (`VALID`, `0` failed retrieval queries across all experiments).

**Key Conclusions:**
1. **Cross-Encoder Reranking Delivered Major Precision & Ranking Gains (`D5-E3a` Selected as Production Default):**
   - Reranking RRF Top-20 (`D5-E3a`) lifted **`Hit@1` from `0.3889` -> `0.4630` (`+0.0741`)**, **`Hit@5` from `0.7222` -> `0.7963` (`+0.0741`, `43/54` queries solved)**, **`Precision@5` from `0.2444` -> `0.2704`**, **`Span Recall@5` from `0.6389` -> `0.6852`**, **`MRR@5` from `0.5182` -> `0.5861` (`+0.0679`)**, and **`NDCG@5` from `0.3737` -> `0.4366` (`+0.0629`)**.
   - Reranking the full `Candidate Union` (`Dense@20 ∪ BM25@20`, mean `33.77`, max `40` chunks; `D5-E3b`) achieved higher Top-1 rank quality (**`Hit@1 = 0.4815`**, **`MRR@5 = 0.5948`**, **`NDCG@5 = 0.4391`**) at **`1.73x` higher median latency** (`4,498.75 ms` vs. `2,598.23 ms`) and lower `Hit@5` (`0.7778` vs. `0.7963`), so **`D5-E3a` (`rrf_top_20`)** is promoted as the default `hybrid_rerank` pool while `candidate_union` remains selectable via API.
2. **Zero Runtime Pair Truncation:** Pairing `code_aware_v1` (`<=256` WordPiece tokens/chunk) with `ms-marco-MiniLM-L-6-v2` (`max_length = 512`) resulted in **`0 / 1,200` (`0.0%`) truncated pairs** in `D5-E3a` and **`0 / 2,026` (`0.0%`) truncated pairs** in `D5-E3b`.
3. **Honest Finding on Live Conversational Query Rewriting (`Q1` Opt-In Default `False`):**
   - Cross-encoder reranking (`D5-E3a` `Q0`, unrewritten) already raised Turn-2 conversational `Hit@5` to **`0.9167` (`11/12` solved)**.
   - Enabling leakage-free Gemini rewriting (`D5-Q1`) resolved `flask_002` (`null -> #1`) and lifted `express_010` (`#3 -> #1`), but **regressed Turn-2 `Hit@5` from `0.9167` -> `0.7500` (`MRR@5: 0.6139 -> 0.5694`)** due to `4/12` live API stalls (`503 UNAVAILABLE` / `30.0s` timeout, safely falling back) and rewrite over-specification hurting `flask_009`, `flask_014`, and `fmt_014`. Consequently, conversational rewriting is integrated as an **opt-in sidebar toggle (`default = False`)**.
4. **Day 6 Score Handoff Warning:** Raw cross-encoder logits and `sigmoid(logit)` transforms exhibit substantial overlap between `Answerable (N=54)` (`[-10.0492, +7.7011]`) and `Unanswerable (N=6)` (`[-9.8669, +2.4591]`, including `fmt_020` scoring `+2.4591` / `0.9212`), proving `sigmoid(logit)` is **not a calibrated probability** and cannot alone serve as an abstention threshold without Day 6 verification/calibration.

---

## 2. ONNX Preflight, Pair Encoding & Truncation Verification

### 2.1 Environment & Verified Cross-Encoder Artifact
Implemented strictly with `onnxruntime`, `tokenizers`, and `huggingface_hub` on CPU (`CPUExecutionProvider`) — **zero PyTorch or `sentence-transformers` dependencies** (matching Day 2 Windows/Python 3.14 compatibility):

| Property | Verified Artifact Value |
| :--- | :--- |
| **Runtime Environment** | Windows (`win32`), Python `3.14.7`, `chromadb 1.5.9`, `numpy 2.5.2`, Peak RSS `742.2 MB` |
| **Model Name & Revision** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (rev `233902d25c440f23af6f7d6e94d2946bac0bee0a`) |
| **ONNX Artifact & Size** | `onnx/model.onnx` — **`86.80 MB`** (`CPUExecutionProvider`) |
| **Max Length & Pair Overhead** | `512` tokens (`3` special tokens: `[CLS] query [SEP] candidate [SEP]`) |
| **Input Signatures** | `input_ids`, `attention_mask`, `token_type_ids` — `tensor(int64)` `["batch_size", "sequence_length"]` |
| **Output Signature** | `logits` — `tensor(float)` `["batch_size", 1]` (`single_raw_relevance_logit_descending`) |
| **Load & Warmup Timings** | Locate: `0.0026s`, Session Load: `0.4748s`, Warmup: `0.0241s` (excluded from query latency) |

### 2.2 Pair Encoding, Segment IDs & Oversized-Query Safety
- Native pair encoding in [ONNXCrossEncoderReranker.encode_pair](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/retrieval/reranker.py#L152-L227) passes `[CLS] query [SEP] candidate [SEP]` with verified `token_type_ids`: **`0` across `[CLS] query [SEP]`** and **`1` across `candidate [SEP]`**.
- Candidate-first truncation (`strategy="only_second"`) trims long code blocks before touching queries. Preflight testing on `tokenizers` uncovered that `only_second` raises a `Truncation error` if a raw query alone exceeds `509` tokens; `encode_pair` deterministically caps query tokens at `512 - 3 - 64 = 445` whenever needed so inference never crashes and at least `64` candidate tokens remain intact, while emitting separate `query_truncated` and `candidate_truncated` telemetry.
- **Observed Production Truncation:** Across the entire Day 5 benchmark, `D5-E3a` recorded **`0 / 1,200` (`0.0%`)** truncated pairs and `D5-E3b` recorded **`0 / 2,026` (`0.0%`)** truncated pairs.

---

## 3. Experimental Controls & Day 4 Baseline Parity

All Day 5 experiments reused the persisted Day 4 V1 collections (`experiments/day4_hybrid/chroma_day4_db/`, leaving `./chroma_db` and Day 1–4 artifacts untouched):

| Repository | Pinned Commit SHA | Collection | V1 Chunks | Verified Corpus SHA-256 (`[:12]`) | Matches Day 4 |
| :--- | :--- | :--- | :---: | :--- | :---: |
| `pallets/flask` | `49943b5a9441...` | `day4_v1_flask` | `1,352` | `7d996d0cf4d1` | `True` |
| `expressjs/express` | `43fca52d28a4...` | `day4_v1_express` | `1,297` | `d92e0235ef6e` | `True` |
| `fmtlib/fmt` | `3be6709f575a...` | `day4_v1_fmt` | `1,072` | `4bae980c35f2` | `True` |

**Day 4 Parity Audit (`D5-Base_hybrid_rrf` vs. `D4-E2_hybrid_rrf`):** All 7 retrieval metrics reproduced Day 4 (`Hit@1=0.3889`, `Hit@3=0.6296`, `Hit@5=0.7222`, `Precision@5=0.2444`, `SpanRec@5=0.6389`, `MRR@5=0.5182`, `NDCG@5=0.3737`) with exact **`0.0` deltas** (`validity_status = "VALID"`, `0` failed retrieval queries across all experiments).

---

## 4. Stage-2 Reranking Evaluation (`D5-Base` vs. `D5-E3a` vs. `D5-E3b`)

### 4.1 Overall Top-5 Retrieval Quality (`N = 54` Answerable Queries)

| Experiment | Candidate Pool (per query) | Hit@1 | Hit@3 | Hit@5 | Precision@5 | Span Recall@5 | MRR@5 | NDCG@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`D5-Base`: Hybrid RRF (`k=60`)** | No reranking (`Top-5` cut) | `0.3889` | `0.6296` | `0.7222` | `0.2444` | `0.6389` | `0.5182` | `0.3737` |
| **`D5-E3a`: Rerank `RRF Top-20`** | Fixed `20` (`1,200` pairs) | `0.4630` | **`0.7037`** | **`0.7963`** | **`0.2704`** | **`0.6852`** | `0.5861` | `0.4366` |
| **`D5-E3b`: Rerank `Candidate Union`** | `28–40`, mean `33.77` (`2,026` pairs) | **`0.4815`** | **`0.7037`** | `0.7778` | `0.2667` | `0.6759` | **`0.5948`** | **`0.4391`** |
| **Delta (`D5-E3a` vs. `D5-Base`)** | — | **`+0.0741`** | **`+0.0741`** | **`+0.0741`** | **`+0.0260`** | **`+0.0463`** | **`+0.0679`** | **`+0.0629`** |

### 4.2 Stage-Wise Latency, Cost & Truncation (`N = 60` All Queries, Warmup Excluded)

| Pipeline Metric | `D5-Base` (Hybrid RRF) | `D5-E3a` (RRF Top-20 -> Rerank) | `D5-E3b` (Union `<=40` -> Rerank) |
| :--- | :---: | :---: | :---: |
| **Candidate Pool Size (Min / Median / Mean / Max)** | `—` | `20 / 20.0 / 20.00 / 20` | `28 / 34.0 / 33.77 / 40` |
| **Total Pairs Scored / Pairs Truncated** | `0 / 0` | `1,200 / 0 (0.0%)` | `2,026 / 0 (0.0%)` |
| **Stage-1 Dense Embed + Search P50 (`ms`)** | `101.72 + 16.05` | `101.72 + 16.05` | `101.72 + 16.05` |
| **Stage-1 BM25 + RRF Fusion P50 (`ms`)** | `6.36 + 0.64` | `6.36 + 0.64` | `6.36 + 0.64` |
| **Stage-2 Tokenization P50 (`ms`)** | `—` | `33.12` (mean `34.67`) | `48.73` (mean `54.34`) |
| **Stage-2 ONNX CPU Inference P50 (`ms`)** | `—` | `2,412.49` (mean `2,321.37`) | `4,245.90` (mean `4,007.23`) |
| **Stage-2 Reranker Total P50 / P95 (`ms`)** | `—` | `2,456.04 / 3,174.22` | `4,297.15 / 5,553.92` |
| **End-to-End Total Retrieval P50 / P95 (`N=60`, `ms`)** | `130.77 / 244.24` | `2,598.23 / 3,409.95` | `4,498.75 / 5,654.54` |
| **End-to-End Answerable Latency P50 / P95 (`N=54`, `ms`)** | `128.31 / 244.24` | `2,522.89 / 3,446.91` | `4,405.90 / 5,597.43` |

*(Note: Stage-1 `D5-Base` P50 under Day 5 full-model memory footprint was `130.77 ms` vs. `39.56 ms` in isolated Day 4 measurements; identical Stage-1 candidate pools were shared fairly across `D5-Base`, `D5-E3a`, and `D5-E3b`.)*

### 4.3 Granular Breakdowns (`Hit@1 / Hit@5 / MRR@5 / NDCG@5`)

| Cohort (`n`) | `D5-Base` (Hybrid RRF) | `D5-E3a` (`RRF Top-20`) | `D5-E3b` (`Candidate Union`) |
| :--- | :---: | :---: | :---: |
| **Repo: `pallets/flask` (`18`)** | `0.4444 / 0.7778 / 0.5407 / 0.3768` | `0.4444` / **`0.8333 / 0.5898`** / `0.4374` | `0.4444 / 0.7778 / 0.5787` / **`0.4413`** |
| **Repo: `expressjs/express` (`18`)** | `0.2222 / 0.7222 / 0.4398 / 0.3220` | `0.4444` / **`0.7778`** / `0.5667 / 0.4425` | **`0.5000 / 0.7778 / 0.6037 / 0.4533`** |
| **Repo: `fmtlib/fmt` (`18`)** | `0.5000 / 0.6667 / 0.5741 / 0.4223` | **`0.5000 / 0.7778 / 0.6018 / 0.4300`** | **`0.5000 / 0.7778 / 0.6018`** / `0.4228` |
| **Style: `identifier` (`22`)** | `0.5909 / 0.8636 / 0.7061 / 0.5203` | **`0.6818 / 0.9545 / 0.7856`** / `0.5970` | **`0.6818 / 0.9545 / 0.7856 / 0.6002`** |
| **Style: `conversational` (`12`)** | `0.4167 / 0.6667 / 0.5278 / 0.3463` | `0.4167` / **`0.9167`** / `0.6139 / 0.4360` | **`0.5000`** / `0.8333` / **`0.6528 / 0.4413`** |
| **Style: `conceptual` (`20`)** | `0.1500` / **`0.6000`** / `0.3058 / 0.2288` | **`0.2500`** / `0.5500` / **`0.3500 / 0.2607`** | **`0.2500`** / `0.5500` / **`0.3500 / 0.2607`** |
| **Cat: `architecture_flow` (`10`)** | `0.4000 / 0.7000 / 0.5333 / 0.3512` | `0.5000` / **`0.9000`** / `0.6283 / 0.4307` | **`0.6000`** / `0.8000` / **`0.6750 / 0.4371`** |
| **Cat: `cross_file_dependencies` (`10`)** | `0.3000 / 0.8000 / 0.4667 / 0.3436` | **`0.4000 / 0.9000 / 0.5917 / 0.4156`** | **`0.4000 / 0.9000 / 0.5917`** / `0.4025` |
| **Cat: `debugging_configuration` (`10`)** | `0.3000 / 0.6000 / 0.3950` / **`0.2807`** | **`0.3000 / 0.6000 / 0.4167`** / `0.2603` | **`0.3000 / 0.6000 / 0.4167`** / `0.2603` |
| **Cat: `function_identification` (`12`)** | `0.5000 / 0.7500 / 0.6111 / 0.4474` | **`0.5000 / 0.8333 / 0.6208`** / `0.5621` | **`0.5000 / 0.8333 / 0.6208 / 0.5789`** |
| **Cat: `implementation_understanding` (`12`)** | `0.4167 / 0.7500 / 0.5583 / 0.4213` | **`0.5833 / 0.7500 / 0.6528 / 0.4807`** | **`0.5833 / 0.7500 / 0.6528 / 0.4807`** |
| **Evidence: Single-Span (`37`)** | `0.3514 / 0.7027 / 0.4950 / 0.3680` | **`0.4865 / 0.7297 / 0.5730`** / `0.4376` | **`0.4865 / 0.7297 / 0.5730 / 0.4431`** |
| **Evidence: Multi-Span (`17`)** | `0.4706 / 0.7647 / 0.5686 / 0.3860` | `0.4118` / **`0.9412`** / `0.6147` / **`0.4345`** | **`0.4706`** / `0.8824` / **`0.6422`** / `0.4305` |

### 4.4 Paired Query Transitions & Selection (`D5-E3a` vs. `D5-Base` & `D5-E3b`)
- **`D5-E3a` vs. `D5-Base` (`N=54`):** Quadrants = `35 Hit/Hit`, **`8 Gained Hit@5`** (`flask_009` `#5`, `flask_011` `#4`, `flask_014` `#3`, `express_010` `#3`, `express_015` `#3`, `express_016` `#3`, `fmt_001` `#4`, `fmt_005` `#3`), **`4 Lost Hit@5`** (`flask_017` `#5->null`, `flask_018` `#4->null`, `express_008` `#2->null`, `express_011` `#3->null`), `7 Miss/Miss`. Exploratory paired bootstrap 95% CI (`seed=42`, `B=10,000`): `ΔHit@5 = +0.0741 [-0.0370, +0.2037]`, `ΔMRR@5 = +0.0679 [-0.0130, +0.1472]`.
- **`D5-E3b` vs. `D5-E3a` (`N=54`):** Identical Top-5 outcomes on `52/54` queries; only `2` differences occurred: `express_010` improved `#3 -> #1` (because `lib/application.js:app.handle` lay outside `RRF Top-20` in the `21–40` union tail), while `flask_009` regressed `#5 -> null` (a noisy union-tail chunk outranked the relevant context block). Exploratory bootstrap 95% CI (`E3b - E3a`): `ΔHit@5 = -0.0185 [-0.0556, 0.0000]`, `ΔMRR@5 = +0.0086 [-0.0111, +0.0370]`.
- **Pipeline Promotion:** **`D5-E3a_rerank_rrf20`** is selected as the default production `hybrid_rerank` pool because it achieves higher `Hit@5` (`0.7963`), `Span Recall@5` (`0.6852`), and Conversational `Hit@5` (`0.9167`) while cutting CPU inference time nearly in half (`2.46s` vs. `4.30s` reranker P50). Notice that `fmt_003` (`Rank #23` in union) and test/license preambles (`flask_017`, `express_008`, `express_011`) still tripped the web-passage MS MARCO model on abstract conceptual code queries.

---

## 5. Leakage-Free Conversational Query Rewriting Ablation (`Q0` vs. `Q1` vs. `Q2-Oracle`)

Implemented cleanly in [retrieval/query_rewriter.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/retrieval/query_rewriter.py) with strict isolation from dataset oracle/ground-truth files:
- **Permitted Inputs Only:** `query`, `previous_context`, and `repository_name`.
- **Deterministic Gating:** Verified on all 60 evaluation queries — **`48/48` standalone queries passed through unchanged (`rewritten=False`, `0.0 ms`, `0` Gemini calls)**, and **`12/12` Turn-2 conversational queries triggered rewriting**.
- **Shared Client:** Uses `llm.ask_gemini` with a `30.0s` non-blocking timeout guard and transient `429/503` backoff retry, falling back cleanly to the original query on failure.

> [!CAUTION]
> **`D5-Q2-Oracle` (`expected_standalone_query`) is an Offline Oracle Reference — Not a Deployable Configuration.** It is reported strictly separately as an idealized benchmark reference, never mixed into production defaults.

### 5.1 Fixed-Pipeline (`D5-E3a`) Rewriting Comparison

| Evaluation Scope (`N`) | Variant | Hit@1 | Hit@3 | Hit@5 | Precision@5 | Span Recall@5 | MRR@5 | NDCG@5 | Retrieval P50 (`ms`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall (`N=54`)** | **`D5-Q0` (Original Query)** | `0.4630` | **`0.7037`** | **`0.7963`** | **`0.2704`** | **`0.6852`** | **`0.5861`** | **`0.4366`** | **`2,522.89`** |
| **Overall (`N=54`)** | **`D5-Q1` (Leakage-Free Rewriter)** | `0.4630` | `0.6852` | `0.7593` | `0.2593` | `0.6574` | `0.5762` | `0.4262` | `2,562.93` *(excl. API wait)* |
| **Overall (`N=54`)** | *`D5-Q2-Oracle` (Not Deployable)* | *`0.5000`* | *`0.6852`* | *`0.7778`* | *`0.2852`* | *`0.6852`* | *`0.6025`* | *`0.4510`* | *`2,493.96`* |
| **Conversational (`N=12`)** | **`D5-Q0` (Original Query)** | `0.4167` | **`0.8333`** | **`0.9167`** | `0.3000` | **`0.6667`** | `0.6139` | `0.4360` | **`2,520.89`** |
| **Conversational (`N=12`)** | **`D5-Q1` (Leakage-Free Rewriter)** | `0.4167` | `0.7500` | `0.7500` | `0.2500` | `0.5417` | `0.5694` | `0.3891` | `2,596.19` *(+15.67s API P50)* |
| **Conversational (`N=12`)** | *`D5-Q2-Oracle` (Not Deployable)* | *`0.5833`* | *`0.7500`* | *`0.8333`* | *`0.3667`* | *`0.6667`* | *`0.6875`* | *`0.5006`* | *`1,941.09`* |

### 5.2 Per-Query Turn-2 Outcome & Live Gemini Reliability Audit (`N = 12`)

| Query ID | Rewritten? / Error Status | Rewrite Latency (`ms`) | Q0 Rank | Q1 Rank | Observed Mechanism |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `flask_002` | Rewritten (`Flask class constructor`) | `17,607.96` | `null` | **`1`** (+Gain) | Resolved missing `"Flask"` entity -> `src/flask/config.py:from_object` `#1` |
| `flask_006` | Rewritten (`add_url_rule`) | `6,975.26` | `2` | `2` | Preserved Rank #2 cleanly |
| `flask_009` | Rewritten (`wsgi_app`, `AppContext`, `RequestContext`) | `4,247.92` | **`5`** | `null` (-Loss) | Specific class names boosted `request_context`/`AppContext` (`logit +3.16`), pushing Rank #5 span out of Top-5 |
| `flask_014` | Rewritten (`SecureCookieSessionInterface`) | `5,224.49` | **`3`** | `null` (-Loss) | Boosted `save_session` sub-chunks (`logits +4.18, +3.86`), displacing `get_cookie_secure` (`#3`) |
| `express_002` | Rewritten (`createApplication`, `app.init`) | `17,520.78` | **`1`** | `3` | Extra function names lowered exact `defaultConfiguration` `#1 -> #3` |
| `express_006` | **Fallback (`503 UNAVAILABLE`)** | `13,813.28` | `1` | `1` | Live Gemini transient spike; safely preserved `#1` via fallback |
| `express_010` | Rewritten (`Express application handler`) | `4,561.49` | `3` | **`1`** | Lifted `lib/application.js:app.handle` `#3 -> #1` |
| `express_014` | **Fallback (`30.0s timeout`)** | `30,018.85` | `2` | `2` | Timed out cleanly; preserved `#2` via fallback |
| `fmt_002` | **Fallback (`30.0s timeout`)** | `30,022.58` | `2` | `2` | Timed out cleanly; preserved `#2` via fallback |
| `fmt_006` | Rewritten (`is_range`) | `24,007.81` | `1` | `1` | Preserved `#1` cleanly |
| `fmt_009` | **Fallback (`30.0s timeout`)** | `32,747.14` | `1` | `1` | Timed out after retry; preserved `#1` cleanly |
| `fmt_014` | Rewritten (`chrono duration`) | `4,879.38` | **`1`** | `null` (-Loss) | Mis-bound Turn-1 `chrono` context to Turn-2 `compile.h` `FMT_STRING` query, retrieving `chrono.h` Top-3 instead of `compile.h` |

**Architectural Decision:** Because unrewritten `D5-E3a` already achieves **`0.9167` Turn-2 `Hit@5` (`11/12`)**, while live Gemini rewriting incurs **`4.2s–32.7s` latency** (`4/12` `503`/timeout failures) and net Turn-2 regression (`ΔHit@5 = -0.1667` conversational, `-0.0370` overall), query rewriting is wired safely as an **opt-in toggle (`enable_query_rewrite = False` by default)** in [app.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/app.py).

---

## 6. Exploratory Day 6 Score-Handoff Analysis (Non-Calibrated)

The cross-encoder scores documented below (`D5-E3a`) are **uncalibrated MS MARCO logits** and their **strictly exploratory monotonic transform `sigmoid(logit)`** — they are **not** calibrated code-answerability probabilities and cannot be used alone as an abstention threshold.

| Score Scale (`D5-E3a`) | Cohort (`N`) | Min | Q25 | Median | Mean | Q75 | IQR | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top-1 Raw Logit** | Answerable (`54`) | `-10.0492` | `-0.2904` | `2.2159` | `1.7473` | `4.5334` | `4.8239` | `7.7011` |
| | Unanswerable (`6`) | `-9.8669` | `-7.1078` | `-2.9457` | `-3.8535` | `-1.7314` | `5.3763` | **`+2.4591`** |
| **Top-1 Sigmoid** | Answerable (`54`) | `0.000043` | `0.4286` | `0.9013` | `0.6979` | `0.9894` | `0.5607` | `0.9995` |
| | Unanswerable (`6`) | `0.000052` | `0.0060` | `0.0635` | `0.2030` | `0.1530` | `0.1470` | **`0.9212`** |

- **Distribution Overlap (`D5-E3a`):** Wide bidirectional overlap spanning logits **`[-9.8669, +2.4591]`** — **`28 / 54` answerable queries (`51.9%`)** scored below the unanswerable maximum (`+2.4591`), and **`6 / 6` unanswerable queries** exceeded the answerable minimum (`-10.0492`). `D5-E3b` showed an identical overlap upper bound (`+2.4591`, `28` answerable below unanswerable max, `5` unanswerable above answerable min `-9.2009`).
- **Diagnostic Examples Explaining Non-Calibration:**
  - **False-High Unanswerable (`fmt_020`):** *"How does fmt parse and serialize JSON documents into C++ struct fields?"* produced top-1 logit **`+2.4591` (`sigmoid = 0.9212`)** against `include/fmt/compile.h:24` (`license_header` / header comments).
  - **False-Low Answerable Hits:** `express_010` (`Hit@5=1.0`, logit `-10.0492` / `sigmoid = 0.000043` in `E3a`), `express_009` (`Hit@5=1.0`, logit `-7.6015`), and `fmt_009` (`Hit@5=1.0`, logit `-4.1170`) all solved correctly with deeply negative logits because dense terse code chunks lacked natural-language MS MARCO query phrasing. Conversely, answerable misses like `fmt_016` scored `+4.3161` (`sigmoid = 0.9868`) on wrong candidate chunks.

---

## 7. Production Integration & Verification

- **[embedder.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/embedder.py):**
  - Preserves `DEFAULT_RETRIEVAL_MODE = "dense"` for V0 callers while extending `SUPPORTED_RETRIEVAL_MODES = ("hybrid_rerank", "hybrid_rrf", "dense", "bm25")`.
  - Adds `@st.cache_resource` loader `load_cross_encoder_reranker()`, `candidate_pool_mode="rrf_top_20"` (also supports `"candidate_union"`), `enable_query_rewrite: bool = False`, and `chat_history: Any = None`.
  - Automatically falls back cleanly to Hybrid RRF on reranker errors and preserves the original query on rewrite errors.
- **[app.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/app.py):**
  - Sets sidebar default `retrieval_mode = "hybrid_rerank"` (`Hybrid RRF + ONNX Cross-Encoder`), adds `Enable Conversational Query Rewriting (Experimental)` checkbox defaulting to `False`, passes `chat_history=st.session_state.messages[:-1]`, and surfaces optional rewrite/fallback transparency notices without modifying [llm.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/llm.py).
- **Automated Test Suite ([test_day5_rerank_and_rewrite.py](file:///C:/Users/gaura/Downloads/RAG%20PROJECT/tests/test_day5_rerank_and_rewrite.py)):** Covers pair tokenization + `0`/`1` `token_type_ids`, `only_second` & `>512`-token query truncation telemetry, logit tie-breaking & RRF fallback, zero-leakage AST check + deterministic gating across all 60 questions, retry/fallback behavior, live ChromaDB end-to-end `hybrid_rerank` + rewrite integration, and Day 1–4 preservation.
