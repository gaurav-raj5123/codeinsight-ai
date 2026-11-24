"""
telemetry.py — Baseline Instrumentation & Telemetry Recorder
===========================================================
Handles structured logging of baseline metrics for:
- Repository loading and extraction
- Source file extraction and chunking
- ChromaDB indexing (combined embedding and insertion)
- Similarity retrieval (chunk IDs, distances, content, metadata)
- Gemini LLM generation (latencies, success, error tracking)
- End-to-end run persistence under experiments/results/baseline_run.json
"""

import os
import json
import uuid
import tempfile
from datetime import datetime, timezone

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiments", "results")
BASELINE_LOG_PATH = os.path.join(RESULTS_DIR, "baseline_run.json")


def ensure_results_dir() -> str:
    """Ensures the experiments/results directory exists and returns its path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def get_current_iso_timestamp() -> str:
    """Returns current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def create_baseline_run_record(
    repo_url: str = "",
    owner: str = "",
    repo_name: str = "",
    commit_sha: str = "unknown",
    files_indexed: int = 0,
    chunks_indexed: int = 0,
    repo_download_duration: float = 0.0,
    repo_extraction_duration: float = 0.0,
    repo_total_loading_duration: float = 0.0,
    chunking_duration: float = 0.0,
    chromadb_add_duration: float = 0.0,
    total_index_function_duration: float = 0.0,
    total_indexing_duration: float = 0.0,
    embedding_model: str = "DefaultEmbeddingFunction (all-MiniLM-L6-v2)",
    embedding_dimensions: int = 384,
    distance_metric: str = "l2",
    # Backward compatibility aliases
    download_extraction_duration: float | None = None,
    chromadb_indexing_duration: float | None = None
) -> dict:
    """Creates a new baseline run structure with separated loading and indexing stages."""
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    # Handle backward-compatible aliases if provided
    dl_dur = repo_download_duration
    ext_dur = repo_extraction_duration
    tot_load = repo_total_loading_duration if repo_total_loading_duration > 0.0 else (dl_dur + ext_dur)
    if download_extraction_duration is not None and tot_load == 0.0:
        tot_load = download_extraction_duration

    add_dur = chromadb_add_duration if chromadb_add_duration > 0.0 else (chromadb_indexing_duration or 0.0)
    func_dur = total_index_function_duration if total_index_function_duration > 0.0 else add_dur

    return {
        "run_id": run_id,
        "timestamp": get_current_iso_timestamp(),
        "repository": {
            "url": repo_url,
            "owner": owner,
            "name": repo_name,
            "commit_sha": commit_sha,
            "files_indexed": files_indexed,
            "chunks_indexed": chunks_indexed,
            "loading_time_seconds": {
                "download": round(dl_dur, 4),
                "extraction": round(ext_dur, 4),
                "total_loading": round(tot_load, 4)
            },
            "indexing_time_seconds": {
                "repo_download": round(dl_dur, 4),
                "repo_extraction": round(ext_dur, 4),
                "repo_total_loading": round(tot_load, 4),
                "chunking": round(chunking_duration, 4),
                "chromadb_add_combined": round(add_dur, 4),
                "total_index_function": round(func_dur, 4),
                "total_indexing": round(total_indexing_duration, 4)
            },
            "embedding": {
                "model": embedding_model,
                "dimensions": embedding_dimensions,
                "distance_metric": distance_metric
            }
        },
        "queries": []
    }


def create_query_record(
    query_text: str,
    repo_name: str,
    top_k_requested: int,
    actual_results_returned: int,
    retrieval_latency_seconds: float,
    retrieved_chunks: list[dict],
    distance_metric: str = "l2",
    generation_model: str = "gemini-3.5-flash",
    generation_latency_seconds: float = 0.0,
    generation_success: bool = True,
    generation_error: str | None = None,
    prompt_length_chars: int = 0,
    response_length_chars: int = 0,
    total_turn_latency_seconds: float = 0.0
) -> dict:
    """Creates a structured record for a single query turn."""
    # Format retrieved chunks for logging
    formatted_chunks = []
    for c in retrieved_chunks:
        meta = c.get("metadata", {}) or {}
        chunk_entry = {
            "id": c.get("id", "unknown"),
            "distance": round(c.get("distance", 0.0), 6) if isinstance(c.get("distance"), (int, float)) else None,
            "file_path": meta.get("file_path", c.get("file_path", "unknown")),
            "start_line": meta.get("start_line", c.get("start_line")),
            "end_line": meta.get("end_line", c.get("end_line")),
            "chunk_type": meta.get("chunk_type", c.get("chunk_type", "unknown")),
            "content": c.get("content", "")
        }
        formatted_chunks.append(chunk_entry)

    # Sanitize generation error (remove any sensitive tokens or full API keys if present)
    safe_error = None
    if generation_error:
        safe_error = str(generation_error)
        # Mask out any apparent API key patterns
        import re
        safe_error = re.sub(r'AIza[0-9A-Za-z-_]+', '[REDACTED_API_KEY]', safe_error)

    query_id = f"query_{datetime.now(timezone.utc).strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}"
    return {
        "query_id": query_id,
        "timestamp": get_current_iso_timestamp(),
        "query_text": query_text,
        "repository": repo_name,
        "retrieval": {
            "top_k_requested": top_k_requested,
            "actual_results_returned": actual_results_returned,
            "total_retrieval_latency_seconds": round(retrieval_latency_seconds, 4),
            "distance_metric": distance_metric,
            "retrieved_chunks": formatted_chunks
        },
        "generation": {
            "model": generation_model,
            "latency_seconds": round(generation_latency_seconds, 4),
            "success": generation_success,
            "error": safe_error,
            "prompt_length_chars": prompt_length_chars,
            "response_length_chars": response_length_chars
        },
        "total_turn_latency_seconds": round(total_turn_latency_seconds, 4)
    }


def save_baseline_run(run_record: dict, filepath: str = BASELINE_LOG_PATH) -> str:
    """
    Saves a baseline run record to the JSON results log.
    If the file exists, it loads existing runs and appends the new run (or updates it if matching run_id).
    Protects existing results: raises RuntimeError on corrupted JSON to prevent silent resets/overwrites.
    Uses safe atomic persistence to prevent file corruption during writes.
    """
    ensure_results_dir()

    existing_runs = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    data = json.loads(content)
                    if isinstance(data, list):
                        existing_runs = data
                    elif isinstance(data, dict):
                        existing_runs = [data]
                    else:
                        raise ValueError(f"Unexpected JSON data type: {type(data)}")
        except Exception as e:
            # Strictly do not overwrite or reset corrupted results file
            raise RuntimeError(
                f"Failed to read existing baseline results from {filepath}: {e}. "
                f"Original file has been preserved to prevent data loss."
            ) from e

    # Check if run already exists in file; update if so, else append
    updated = False
    for i, r in enumerate(existing_runs):
        if r.get("run_id") == run_record.get("run_id"):
            existing_runs[i] = run_record
            updated = True
            break

    if not updated:
        existing_runs.append(run_record)

    # Safe atomic persistence: write to temporary file in same directory, flush, sync, then replace
    dir_name = os.path.dirname(os.path.abspath(filepath))
    fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix="baseline_tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(existing_runs, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, filepath)
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise

    return filepath
