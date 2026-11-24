# 🧪 Test Suites

This directory contains the automated regression and verification test suites for the codebase assistant.

- **`test_baseline_instrumentation.py`**:
  - Smoke tests for chunker, repo loader, embedder timing separation, mocked LLM calls, telemetry atomic persistence, and Streamlit import.
- **`test_task4_corrections.py`**:
  - Verification of NDCG@5 mathematical validity, failure accounting on errors, dual latency reporting (answerable vs all queries), and pinned commit SHA markers.

### Running Tests

```bash
# Run baseline instrumentation smoke tests
python tests/test_baseline_instrumentation.py

# Run benchmark verification tests
python tests/test_task4_corrections.py
```
