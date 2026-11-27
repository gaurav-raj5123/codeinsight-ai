# CodeInsight AI Assistant — Interview, Technical Walkthrough & Live Demo Guide

This guide equips you to present **CodeInsight AI Assistant** end-to-end in engineering interviews, ML system design rounds, and live technical demos. Every figure matches the locked Day 1–6 artifacts under `experiments/`.

---

## 1. 60-Second Elevator Pitch

> *"I built **CodeInsight AI Assistant**, a framework-free Codebase RAG system over Python, JavaScript, and C++ repositories. Instead of trusting LangChain defaults, I instrumented the pipeline first (`V0`) and ran a controlled 7-day optimization roadmap across a pinned 253-file / 60-question benchmark (`pallets/flask`, `expressjs/express`, `fmtlib/fmt`). By fixing AST duplication and WordPiece token truncation (`code_aware_v1`), fusing ChromaDB ONNX embeddings with code-aware Okapi BM25 via RRF (`Hit@5` `0.6667` $\rightarrow$ `0.7222`, Candidate Union ceiling `96.3%`), adding a zero-PyTorch ONNX CPU cross-encoder reranker (`Hit@5` $\rightarrow$ **`0.7963`**, `MRR@5` `0.4534` $\rightarrow$ **`0.5861`**), and enforcing structured Pydantic grounding with deterministic `[EVIDENCE_i]` citation verification (`100%` valid citations across `154/154` tokens, `0` fabricated file/line references), the system turns noisy naive code search into verifiable, abstention-capable code intelligence."*

---

## 2. End-to-End Architecture Summary (`V4`)

1. **Ingestion (`repo_loader.py`):** GitHub REST ZIP download (zero local `git` dependency), deterministic extension filtering, and path normalization (`\` $\rightarrow$ `/`).
2. **Token-Budgeted Structural Chunking (`chunker_v1.py`, Day 3):** Hierarchical top-level Python `ast` + stateful JS/C++ brace/comment/preprocessor scanner + binary-search WordPiece budgeting (`<=256` tokens using the live Chroma ONNX tokenizer), emitting compact metadata headers (`File`, `Language`, `Type`, `Class`/`Symbol`, `Part i/N`, `Signature`).
3. **Stage-1 Hybrid Candidate Generation (`embedder.py`, `retrieval/bm25_retriever.py`, `retrieval/hybrid_retriever.py`, Day 4):**
   - **Dense@20:** ChromaDB `all-MiniLM-L6-v2` (384d ONNX, squared $L_2$).
   - **BM25@20:** Code-aware tokenization (`res.cookie`, `fmt::format`, `camelCase`, `snake_case`, short symbols like `g`/`ip`/`os`), static V1 header label hygiene (`strip_v1_header_labels`), strict `overlap_count >= 1` filtering (no zero-match noise), and pickle-free ChromaDB hydration cached via `st.cache_resource`.
   - **Fusion:** Reciprocal Rank Fusion (`rrf_k = 60`) producing Top-20 (`0.9444` candidate hit rate) and `Candidate Union` (`<=40`, **`0.9630`** hit rate).
4. **Stage-2 Cross-Encoder Reranking (`retrieval/reranker.py`, Day 5):** CPU ONNX Runtime inference (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `86.80 MB`, `max_length = 512`, `0.0%` truncation) scoring `[CLS] query [SEP] candidate [SEP]` pairs with candidate-first `only_second` truncation.
5. **Grounded Structured Generation + Offline Citation Validation (`generation/`, Day 6):** Single-call Gemini structured output (`GroundedGenerationSchema` assessing `strong` / `partial` / `insufficient` sufficiency + `abstained` flag + `[EVIDENCE_i]` IDs), directory-tree & `(UNVERIFIED)` history isolation, bounded retries/timeouts, and pure-Python citation resolution/blocking (`0.33 ms` P50).

---

## 3. High-Impact Engineering Tradeoffs & "Why Not X?" Interview Q&A

### Q1: Why run everything on ONNX Runtime instead of PyTorch + `sentence-transformers`?
- **Answer:** On Windows + Python `3.14.7`, native PyTorch stacks frequently hit heavy wheel/DLL conflicts and bloat deployment footprints (`>1 GB`). Using Chroma's ONNX embedder (`86.2 MB`) paired directly with `onnxruntime` + `tokenizers` + `huggingface_hub` for `ms-marco-MiniLM-L-6-v2` (`86.8 MB`) delivers identical vector math with fast cold start (`0.34s`–`0.48s` load), `626–742 MB` total RSS, and zero PyTorch dependency.

### Q2: Why didn't you switch embedding models on Day 2 when `BAAI/bge-small-en-v1.5` or `all-MiniLM-L12-v2` existed?
- **Answer:** Controlled ONNX benchmarking on all `4,272` V0 chunks showed **`all-MiniLM-L12-v2` (`max_seq=128`) truncated `31.98%` of chunks and dropped `Hit@5` from `0.6667` to `0.4815`**, while `BGE-small` and `E5-small` nearly **tripled peak RSS (`~1.85–1.88 GB` vs `626 MB`)** and still trailed V0 `all-MiniLM-L6-v2` on overall `Hit@5` (`0.6111` / `0.6296` / `0.6481` vs `0.6667`) and `MRR@5`. Rather than chasing model size, Day 2 proved the true bottleneck was V0 chunk truncation and structural noise — validated immediately on Day 3 (`+5.56%` `Hit@1`, `-24.5%` indexing time).

### Q3: Why did Day 3 `NDCG@5` dip slightly (`0.3339 -> 0.3217`) even though every other metric jumped cleanly?
- **Answer:** Corpus-level `IDCG@5` depends on total relevant chunks matching ground-truth line ranges in the indexed repository (capped at 5). V0 dropped Python module-level globals (`66/83` Flask files) and kept `fmt` headers monolithic (`1,219` tokens). When `code_aware_v1` properly preserved globals and split large `fmt` C++ files under 256 tokens, multiple relevant sibling chunks entered the index per query, raising the ideal `IDCG@5` normalization denominator. Meanwhile `Hit@1` jumped `0.3148 -> 0.3704`, `MRR@5` `0.4534 -> 0.5052`, and `Span Recall@5` `0.5556 -> 0.6111`.

### Q4: How did Day 4 decide Day 5's architecture?
- **Answer:** Inspecting Stage-1 Top-20 pools (`Candidate Pool Analysis`) showed `Dense@20` (`0.8889`) $\cup$ `BM25@20` (`0.8704`) already achieved **`0.9630` Hit Rate (`52/54`)** and **`0.9074` Span Recall**, while `RRF@20` held **`0.9444` (`51/54`)**. Meaning **13 of the 15 Top-5 misses were retrieval-ranking problems, not missing-document problems**. That directly motivated Stage-2 Cross-Encoder reranking on Day 5 (`Hit@5` jumped `0.7222 -> 0.7963`, `MRR@5` `0.5182 -> 0.5861`).

### Q5: Why did `D5-E3a` (`RRF Top-20`) beat `D5-E3b` (`Candidate Union <= 40`) as default?
- **Answer:** `D5-E3b` scored `~69%` more cross-encoder pairs (`2,026` vs `1,200`), driving CPU P50 latency up **`1.73x` (`4,498.75 ms` vs `2,598.23 ms`)**. While `D5-E3b` gained `+1.85%` `Hit@1` (`0.4815` vs `0.4630`) and `+0.87%` `MRR@5` (`0.5948`), it actually slipped `-1.85%` on `Hit@5` (`0.7778` vs `0.7963`) due to cross-encoder false positives (`MS MARCO` prose prior pulling noisy JS/test snippets into Top-5). `D5-E3a` hits higher `Hit@5` (`79.6%`) with far lower latency.

### Q6: Why is Conversational Query Rewriting OFF (`False`) by default when every RAG blog recommends it?
- **Answer:** Our leakage-free Day 5 evaluation (`12` Turn-2 questions) proved **empirical intuition wrong**: `D5-E3a` without rewriting already achieved **`0.9167` (`11/12`)** Turn-2 `Hit@5` because V1 structural headers + BM25 + cross-encoder attention handled conversational pointers cleanly. Enabling live Gemini rewriting suffered `4/12` API timeouts/503s (`4.2s–32.7s`, safely falling back) and **over-specified** symbol names on `flask_009`, `flask_014`, and `fmt_014`, dropping Turn-2 `Hit@5` from **`0.9167` to `0.7500`**. Shipping it as an explicit opt-in toggle demonstrates data-driven restraint.

### Q7: Can you use vector distance (`L2`) or cross-encoder `sigmoid(logit)` to abstain automatically?
- **Answer:** **No** — verified across both Day 1 and Day 5. In Day 1, unanswerable Top-1 $L_2$ (`[0.83, 1.29]`) fell squarely inside answerable Top-1 $L_2$ (`[0.39, 1.54]`). In Day 5, answerable `express_010` scored logit `-10.05` (`sigmoid = 0.000043`) yet had exact working code (correctly answered on Day 6!), whereas unanswerable `fmt_020` (JSON parsing question on a formatting library) scored **`+2.4591` (`sigmoid = 0.9212`)** due to lexical overlap on `compile.h`/`chrono.h`. Semantic sufficiency must be adjudicated jointly by grounded LLM evidence inspection (`G3`), never raw retrieval thresholds alone.

### Q8: How does Day 6 prevent directory-tree hallucinations and fake citations (`G0 -> G4`)?
- **Answer:** Three concrete guards validated across `N = 60`:
  1. **Directory Tree & History Isolation:** `prompts.py` removes `G0`'s *"use repository structure"* rule and marks prior assistant turns `(UNVERIFIED)` — stopping directory-tree file synthesis (`2 -> 0` unretrieved file mentions, e.g., `express_011` referencing unretrieved `lib/response.js`).
  2. **Pydantic Structured Sufficiency (`G3`):** Forces `[EVIDENCE_i]` handles, `confidence_tier` (`strong` / `partial` / `insufficient`), and `missing_information` list before answering (`6/6` Cohort C abstention vs `5/6` for `G0`, raising blinded-judge faithfulness from `95.8%` to `100.0%` and zeroing parametric leakage `4.2% -> 0.0%`, catching `G0`'s fabricated `lib/utils.js:51-60` in `express_015`).
  3. **Deterministic Offline Citation Validator (`G4`, `0.33 ms`):** Resolves `[EVIDENCE_i]` tokens directly against retrieved chunk metadata (`154/154` valid tokens, `0` extra LLM calls) and flags any invalid ID once as `[UNVERIFIED_CITATION: ...]` instead of emitting broken links.

---

## 4. 5-Minute Live Demo Walkthrough Script

1. **Launch App:** `streamlit run app.py` (`http://localhost:8501`).
2. **Show Engineering Controls in Sidebar:** Point out live selectors:
   - Chunking: `code_aware_v1` (default) vs `existing_chunker_v0` / `fixed_window_v0`.
   - Retrieval: `hybrid_rerank` (default) vs `hybrid_rrf`, `dense_only`, `bm25_only`.
   - Query Rewriting: opt-in toggle (`False` default — explain Day 5 `0.9167 -> 0.7500` finding).
   - Grounded Mode (`G3 + G4`): `True` default with `gemini-3.5-flash-lite` (avoiding `20 RPD` Free Tier exhaustion).
3. **Index a Target Repo (`Index Repository`):** Paste a GitHub URL (e.g., `https://github.com/pallets/flask`) and show sidebar indexing metrics (`source_files_count`, `chunks_count`, `chunk_tokens <= 256`).
4. **Demo 1 — Verified Multi-File Grounding (Cohort A):** Ask `"How does Flask initialize configuration defaults and dispatch requests?"`
   - Show green/amber sufficiency banner, inline clickable `path:start–end` citations, **`📚 Cited Source Navigation`**, and **`🔍 Retrieved Code Chunks`** showing Stage-1 RRF ranks vs Stage-2 Cross-Encoder logits + stripped clean headers.
5. **Demo 2 — Honest Unanswerable Abstention (Cohort C):** Ask `"Does this repository implement built-in GraphQL schema federation or JWT verification middleware?"`
   - Point out the **`🛑 Abstention Notice`** banner (`confidence_tier: insufficient`), explicit **`Missing Evidence`** gaps listed cleanly, `0` fabricated citations, and safe non-hallucination despite the directory tree being present.
6. **Inspect Latency & Diagnostics Expander:** Open **`📊 Baseline Diagnostics`** at the bottom of the turn to show decoupled timings (`Embedding / BM25 / Rerank / Generation / Validation ms`) + sanitized telemetry saved atomically in `telemetry.py`.
