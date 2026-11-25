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

### Running Tests

```bash
# Run all pytest suites
python -m pytest tests/ -v
```
