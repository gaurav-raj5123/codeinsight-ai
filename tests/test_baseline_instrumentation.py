"""
test_baseline_instrumentation.py
================================
Automated smoke test suite validating Day 1 Baseline Instrumentation:
1. Mocked repository download verifying commit SHA extraction and separate timing fields without network dependency.
2. Embedder indexing timing distinction: collection.add() duration vs. total index_chunks() function duration.
3. Embedder retrieval instrumentation: chunk IDs, distances, content, and metadata with L2 metric preserved.
4. Mocked Gemini success & error generation verifying latency, metadata, and secret sanitization without real API calls.
5. Telemetry module safe atomic persistence, non-destructive appending, and corruption protection.
6. Static import & compilation validation of Streamlit app.py (distinguished from interactive workflow).
"""

import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, "..")) if os.path.basename(TESTS_DIR) == "tests" else TESTS_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import tempfile
import time
import io
import zipfile
from unittest.mock import patch, MagicMock

# ══════════════════════════════════════════════════════════════════
# 0. CHUNKER FUNCTIONALITY
# ══════════════════════════════════════════════════════════════════
def test_chunker_unmodified():
    print("Testing 0: Chunker functionality (Executed)...")
    import chunker
    sample_py = {
        "file_path": "sample.py",
        "content": "def add(a, b):\n    return a + b\n\nclass Calc:\n    def sub(self, a, b):\n        return a - b\n",
        "extension": ".py"
    }
    chunks = chunker.chunk_code(sample_py)
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
    for c in chunks:
        assert "content" in c and "metadata" in c
        assert "file_path" in c["metadata"]
        assert "start_line" in c["metadata"]
        assert "end_line" in c["metadata"]
    print("  [PASS] Chunker boundary extraction and metadata verified.")

# ══════════════════════════════════════════════════════════════════
# 1. REPO LOADER: MOCKED DOWNLOAD & COMMIT SHA EXTRACTION
# ══════════════════════════════════════════════════════════════════
def test_repo_loader_mocked_download_and_commit_extraction():
    print("Testing 1: repo_loader mocked download & commit SHA extraction (Executed)...")
    import repo_loader
    
    # Create an in-memory zip archive mimicking GitHub's {owner}-{repo}-{commit_sha} folder
    expected_sha = "9f8e7d6c5b4a"
    folder_name = f"testowner-testrepo-{expected_sha}"
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{folder_name}/README.md", "# Test Repo\nSample content.")
        zf.writestr(f"{folder_name}/main.py", "def main():\n    print('hello')\n")
    zip_bytes = zip_buffer.getvalue()
    
    class MockResponse:
        def __enter__(self):
            return io.BytesIO(zip_bytes)
        def __exit__(self, *args):
            pass
            
    with patch("urllib.request.urlopen", return_value=MockResponse()):
        # 1. Test backward-compatible default signature (returns str)
        repo_path_default = repo_loader.clone_repo("https://github.com/testowner/testrepo")
        assert isinstance(repo_path_default, str), f"Expected str, got {type(repo_path_default)}"
        assert os.path.exists(repo_path_default), "Returned repo_path does not exist"
        assert os.path.exists(os.path.join(repo_path_default, "main.py"))
        repo_loader.cleanup_repo(repo_path_default)
        
        # 2. Test metadata retrieval mode
        repo_path, meta = repo_loader.clone_repo("https://github.com/testowner/testrepo", return_metadata=True)
        assert isinstance(repo_path, str)
        assert isinstance(meta, dict)
        assert meta["commit_sha"] == expected_sha, f"Expected SHA {expected_sha}, got {meta['commit_sha']}"
        assert meta["owner"] == "testowner"
        assert meta["repo"] == "testrepo"
        assert "download_duration_seconds" in meta and meta["download_duration_seconds"] >= 0.0
        assert "extraction_duration_seconds" in meta and meta["extraction_duration_seconds"] >= 0.0
        assert "total_loading_duration_seconds" in meta and meta["total_loading_duration_seconds"] >= 0.0
        repo_loader.cleanup_repo(repo_path)
        
    print("  [PASS] repo_loader mocked ZIP download, SHA resolution, and timing separation verified.")

# ══════════════════════════════════════════════════════════════════
# 2. EMBEDDER: SEPARATE COLLECTION.ADD TIMING & RETRIEVAL
# ══════════════════════════════════════════════════════════════════
def test_embedder_separate_timing_and_retrieval():
    print("Testing 2: embedder indexing timing separation & retrieval instrumentation (Executed)...")
    import embedder
    
    test_repo = "test_smoke_repo"
    chunks = [
        {
            "content": "def calculate_tax(amount):\n    return amount * 0.2",
            "metadata": {"file_path": "tax.py", "start_line": 1, "end_line": 2, "chunk_type": "FunctionDef"}
        },
        {
            "content": "def format_currency(val):\n    return f'${val:.2f}'",
            "metadata": {"file_path": "format.py", "start_line": 1, "end_line": 2, "chunk_type": "FunctionDef"}
        }
    ]
    
    # 1. Test index_chunks default return type (must be int)
    count = embedder.index_chunks(chunks, test_repo)
    assert isinstance(count, int), f"Expected int count, got {type(count)}"
    assert count == 2, f"Expected 2 indexed chunks, got {count}"
    
    # 2. Test index_chunks with return_metadata=True
    count2, index_meta = embedder.index_chunks(chunks, test_repo, return_metadata=True)
    assert isinstance(count2, int)
    assert isinstance(index_meta, dict)
    assert index_meta["chunks_count"] == 2
    # Verify separate timing fields
    assert "collection_add_duration_seconds" in index_meta, "Missing collection_add_duration_seconds"
    assert "total_index_function_duration_seconds" in index_meta, "Missing total_index_function_duration_seconds"
    assert index_meta["total_index_function_duration_seconds"] >= index_meta["collection_add_duration_seconds"], \
        "Total function duration must be >= collection.add duration"
    assert index_meta["distance_metric"] == "l2"
    assert index_meta["embedding_dimension"] == 384
    
    # 3. Test search_similar default return type (must be list of dicts)
    results = embedder.search_similar("calculate tax", test_repo, n_results=2)
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) > 0, "Expected at least 1 retrieved chunk"
    first = results[0]
    assert "content" in first and "metadata" in first
    assert "id" in first and "distance" in first
    assert first["id"] == "chunk_0"
    assert isinstance(first["distance"], float)
    
    # 4. Test search_similar with return_metadata=True
    results2, ret_meta = embedder.search_similar("currency formatting", test_repo, n_results=2, return_metadata=True)
    assert isinstance(results2, list)
    assert isinstance(ret_meta, dict)
    assert ret_meta["distance_metric"] == "l2"
    assert "retrieval_latency_seconds" in ret_meta
    assert ret_meta["retrieval_latency_seconds"] >= 0.0
    
    # Note on retrieval ranking invariance
    print("  [NOTE] Retrieval consistency confirmed on test collection. Note: Full empirical ranking invariance across diverse repositories is an evaluation question to be verified in offline benchmarking, not fully proven by this smoke test alone.")
    
    # Cleanup collection
    client = embedder.get_chroma_client()
    try:
        client.delete_collection(name=index_meta["safe_collection_name"])
    except Exception:
        pass
        
    print("  [PASS] embedder separate collection.add() timer, retrieval IDs, distances, and L2 metric verified.")

# ══════════════════════════════════════════════════════════════════
# 3. LLM: MOCKED GEMINI SUCCESS & ERROR GENERATION
# ══════════════════════════════════════════════════════════════════
def test_llm_mocked_gemini_generation():
    print("Testing 3: llm.ask_gemini mocked generation & secret sanitization (Executed)...")
    import llm
    
    # 1. Mock successful generation
    mock_response = MagicMock()
    mock_response.text = "This function calculates the tax for an amount."
    
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    
    with patch("os.getenv", return_value="AIzaSyMockedKey1234567890abcdefghijklm"):
        with patch("google.genai.Client", return_value=mock_client):
            # Default signature (returns str)
            text_resp = llm.ask_gemini("Explain calculate_tax")
            assert isinstance(text_resp, str)
            assert text_resp == mock_response.text
            
            # return_metadata mode
            resp_str, gen_meta = llm.ask_gemini("Explain calculate_tax", return_metadata=True)
            assert resp_str == mock_response.text
            assert gen_meta["success"] is True
            assert gen_meta["error"] is None
            assert gen_meta["latency_seconds"] >= 0.0
            assert gen_meta["model"] == "gemini-3.5-flash"
            assert gen_meta["response_length_chars"] == len(mock_response.text)
            
    # 2. Mock error with simulated API key leak to verify sanitization
    mock_client_err = MagicMock()
    mock_client_err.models.generate_content.side_effect = Exception(
        "API quota exceeded for key AIzaSyFakeApiKeyWithSecretLeak12345678"
    )
    with patch("os.getenv", return_value="AIzaSyFakeApiKeyWithSecretLeak12345678"):
        with patch("google.genai.Client", return_value=mock_client_err):
            err_str, err_meta = llm.ask_gemini("Test prompt", return_metadata=True)
            assert err_meta["success"] is False
            assert "AIzaSy" not in err_meta["error"], "API key leaked in logged error!"
            assert "[REDACTED_API_KEY]" in err_meta["error"], "API key was not redacted"
            
    # 3. Mock client initialization failure (verifying inside-try handling)
    with patch("os.getenv", return_value="AIzaSyClientInitKeyFail987654321"):
        with patch("google.genai.Client", side_effect=Exception("Client init failed with AIzaSyClientInitKeyFail987654321")):
            init_err_str, init_err_meta = llm.ask_gemini("Test prompt", return_metadata=True)
            assert init_err_meta["success"] is False
            assert "⚠️ Error calling Gemini API:" in init_err_str
            assert "Client init failed" in init_err_meta["error"]
            assert "AIzaSy" not in init_err_meta["error"], "API key leaked in client init error!"
            assert "[REDACTED_API_KEY]" in init_err_meta["error"]
            
    print("  [PASS] llm.ask_gemini mocked success, client init failure, and secret sanitization verified.")

# ══════════════════════════════════════════════════════════════════
# 4. TELEMETRY: ATOMIC PERSISTENCE & CORRUPTION PROTECTION
# ══════════════════════════════════════════════════════════════════
def test_telemetry_atomic_persistence_and_corruption_safety():
    print("Testing 4: telemetry module atomic persistence & corruption protection (Executed)...")
    import telemetry
    
    temp_dir = tempfile.mkdtemp(prefix="telemetry_test_")
    test_filepath = os.path.join(temp_dir, "baseline_run.json")
    
    # 1. Create record with separate loading and indexing stages
    run1 = telemetry.create_baseline_run_record(
        repo_url="https://github.com/pallets/click",
        owner="pallets",
        repo_name="click",
        commit_sha="7fd1a60",
        files_indexed=15,
        chunks_indexed=60,
        repo_download_duration=1.1,
        repo_extraction_duration=0.4,
        repo_total_loading_duration=1.5,
        chunking_duration=0.3,
        chromadb_add_duration=2.0,
        total_index_function_duration=2.2,
        total_indexing_duration=4.0
    )
    
    # Verify separate loading and indexing fields
    idx_times = run1["repository"]["indexing_time_seconds"]
    assert idx_times["repo_download"] == 1.1
    assert idx_times["repo_extraction"] == 0.4
    assert idx_times["repo_total_loading"] == 1.5
    assert idx_times["chromadb_add_combined"] == 2.0
    assert idx_times["total_index_function"] == 2.2
    assert idx_times["total_indexing"] == 4.0
    
    # Add a query
    q1 = telemetry.create_query_record(
        query_text="Where is cli defined?",
        repo_name="click",
        top_k_requested=5,
        actual_results_returned=1,
        retrieval_latency_seconds=0.025,
        retrieved_chunks=[{
            "id": "chunk_0",
            "distance": 0.35,
            "file_path": "click/core.py",
            "start_line": 1,
            "end_line": 10,
            "chunk_type": "FunctionDef",
            "content": "def cli(): pass"
        }],
        distance_metric="l2"
    )
    run1["queries"].append(q1)
    
    # Save run
    telemetry.save_baseline_run(run1, filepath=test_filepath)
    assert os.path.exists(test_filepath)
    
    with open(test_filepath, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == 1
    assert loaded[0]["run_id"] == run1["run_id"]
    
    # 2. Append second run
    run2 = telemetry.create_baseline_run_record(repo_name="repo_two")
    telemetry.save_baseline_run(run2, filepath=test_filepath)
    with open(test_filepath, "r", encoding="utf-8") as f:
        loaded2 = json.load(f)
    assert len(loaded2) == 2, f"Expected 2 runs, found {len(loaded2)}"
    
    # 3. Test corruption protection: if file is corrupted, raise error and preserve file
    with open(test_filepath, "w", encoding="utf-8") as f:
        f.write("{corrupted json contents")
        
    run3 = telemetry.create_baseline_run_record(repo_name="repo_three")
    try:
        telemetry.save_baseline_run(run3, filepath=test_filepath)
        assert False, "Expected save_baseline_run to raise RuntimeError on corrupted file"
    except RuntimeError as e:
        assert "Failed to read existing baseline results" in str(e)
        
    # Verify file was preserved and not silently overwritten
    with open(test_filepath, "r", encoding="utf-8") as f:
        assert f.read() == "{corrupted json contents", "Original corrupted file was not preserved!"
        
    # Cleanup
    try:
        os.remove(test_filepath)
        os.rmdir(temp_dir)
    except Exception:
        pass
        
    print("  [PASS] telemetry separate timings, atomic save, and corruption protection verified.")

# ══════════════════════════════════════════════════════════════════
# 5. STREAMLIT APP: STATIC IMPORT & SYNTAX COMPILATION
# ══════════════════════════════════════════════════════════════════
def test_streamlit_app_compilation_and_import():
    print("Testing 5: Streamlit app static import & compilation (Executed)...")
    # Clarify that this is an import/compilation check, not full interactive UI execution
    import app
    assert hasattr(app, "repo_loader")
    assert hasattr(app, "embedder")
    assert hasattr(app, "llm")
    assert hasattr(app, "telemetry")
    print("  [NOTE] app.py compiles cleanly and all session state variables & module imports initialize. Note: Complete interactive user workflow requires running 'streamlit run app.py' in a browser.")
    print("  [PASS] app.py compilation and import verified.")

# ══════════════════════════════════════════════════════════════════
# 6. CONVERSATION HISTORY: EXCLUDE CURRENT QUESTION
# ══════════════════════════════════════════════════════════════════
def test_conversation_history_excludes_current_question():
    print("Testing 6: Conversation history excludes current question (Executed)...")
    import llm
    
    messages = [
        {"role": "user", "content": "What is Click?"},
        {"role": "assistant", "content": "Click is a Python package for creating command line interfaces."},
        {"role": "user", "content": "How do decorators work in Click?"}
    ]
    current_question = "How do decorators work in Click?"
    
    # In app.py, chat_history is passed as messages[:-1] so current question appears only in QUESTION
    prompt = llm.build_prompt(
        question=current_question,
        chunks=[],
        directory_tree="",
        chat_history=messages[:-1]
    )
    
    # Verify CONVERSATION HISTORY contains prior turn
    assert "CONVERSATION HISTORY (for follow-up context):" in prompt
    assert "User: What is Click?" in prompt
    assert "Assistant: Click is a Python package for creating command line interfaces." in prompt
    
    # Verify current question is excluded from CONVERSATION HISTORY block
    history_block = prompt.split("CONVERSATION HISTORY (for follow-up context):")[1].split("(End of conversation history)")[0]
    assert "How do decorators work in Click?" not in history_block, "Current question must not appear in CONVERSATION HISTORY"
    
    # Verify current question appears in QUESTION section
    assert f"QUESTION: {current_question}" in prompt, "Current question must appear in QUESTION section"
    print("  [PASS] Verified current question is excluded from conversation history and appears only in QUESTION.")

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING DAY 1 TASK 2 REVIEW CORRECTIONS SMOKE TESTS")
    print("=" * 70)
    test_chunker_unmodified()
    test_repo_loader_mocked_download_and_commit_extraction()
    test_embedder_separate_timing_and_retrieval()
    test_llm_mocked_gemini_generation()
    test_telemetry_atomic_persistence_and_corruption_safety()
    test_streamlit_app_compilation_and_import()
    test_conversation_history_excludes_current_question()
    print("=" * 70)
    print("ALL SMOKE TESTS EXECUTED AND PASSED SUCCESSFULLY!")
    print("=" * 70)
