# 🧪 Test Suites

This directory contains the automated regression and verification test suites for the codebase assistant.

- **`test_baseline_instrumentation.py`**:
  - Smoke tests for chunker, repo loader, embedder timing separation, mocked LLM calls, telemetry atomic persistence, and Streamlit import.
- **`test_task4_corrections.py`**:
  - Verification of Day 1 NDCG@5 mathematical validity, failure accounting on errors, dual latency reporting (answerable vs all queries), and pinned commit SHA markers.
- **`test_retrieval_metrics.py`**:
  - Unit tests for `evaluation/retrieval_metrics.py` covering fixed Precision@5 denominator, deduplicated span recall, corpus-level NDCG@5, P50/nearest-rank P95 latencies, failure zeroing, and exact Day 1 metric parity (`MRR@5=0.4534`, `NDCG@5=0.3339`).
- **`test_embedding_runner.py`**:
  - Verification of `embedding_models.json`, ONNX Runtime embedder initialization & normalization across all candidate models, Day 2 result artifact integrity, and simulated failure accounting.
- **`test_chunker_v1.py`**:
  - Verification of `code_aware_v1` (`chunker_v1.py`) and `chunker.py` strategy dispatch parity: V0 backward compatibility, Python hierarchical AST without duplicate nested functions, module-level imports/constants preservation, decorator/docstring attachment, accurate 1-based line mapping, WordPiece `<=256`-token budgeting on oversized/single-line blocks, JS/C++ brace & macro scanning, fallback handling, deterministic IDs, and live ChromaDB insertion.
- **`test_chunking_comparison.py`**:
  - Verification of Day 3 benchmark runner fairness controls (253 files), isolated collection naming, metadata validation, 24 chunk quality metrics computation, failed-query zeroing, error-analysis transitions, and Day 1/2 artifact preservation.
- **`test_bm25_and_hybrid.py`**:
  - Verification of Day 4 V1 header-label hygiene (`strip_v1_header_labels`), code-aware tokenization (dotted/scoped/path/camelCase/snake_case/1–2 char symbols), BM25 zero-overlap filtering without padding (including negative RSJ edge cases), RRF fusion & empty-BM25 fallback, un-truncated `Candidate Union` evaluation, ChromaDB persistent BM25 hydration across restarts/re-indexing/repo switching without pickles, and Day 1–3 artifact preservation.

### Running Tests

```bash
# Run all pytest suites
python -m pytest tests/ -v
```
