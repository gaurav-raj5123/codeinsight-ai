"""
# ══════════════════════════════════════════════════════════════════════════════
# FILE 3: EMBEDDING & VECTOR STORE LAYER
# ══════════════════════════════════════════════════════════════════════════════
# This file handles Step 3 of our RAG pipeline. It takes the text chunks produced
# by chunker.py, converts them into mathematical vectors (embeddings), and stores
# them in a Vector Database (ChromaDB). Later, it takes user questions, embeds
# them, and finds the most relevant chunks using similarity search.
#
# 🔗 CONNECTS TO:
# - chunker.py (Input: receives raw text chunks)
# - llm.py (Output: sends retrieved relevant chunks to the LLM)
# ══════════════════════════════════════════════════════════════════════════════
"""
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import time
from typing import Any

from retrieval.bm25_retriever import (
    BM25Retriever,
    build_bm25_from_collection,
    compute_corpus_fingerprint,
)
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_rewriter import rewrite_conversational_query
from retrieval.reranker import (
    DEFAULT_CROSS_ENCODER_MODEL,
    HybridRerankRetriever,
    ONNXCrossEncoderReranker,
)

SUPPORTED_RETRIEVAL_MODES = ("hybrid_rerank", "hybrid_rrf", "dense", "bm25")
DEFAULT_RETRIEVAL_MODE = "dense"

# Global telemetry caches for last indexing and retrieval operations
_LAST_INDEX_METADATA: dict | None = None
_LAST_RETRIEVAL_METADATA: dict | None = None
_SEEN_BM25_CACHE_KEYS: dict[str, set[str]] = {}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING & CACHING
# ══════════════════════════════════════════════════════════════════════════════

# 💡 WHY: Streamlit re-runs the entire script on every user interaction. 
# Loading an embedding model can take time.
# @st.cache_resource keeps the model in memory across all these re-runs.
@st.cache_resource
def load_embedding_model():
    """Loads ChromaDB's built-in ONNX-based all-MiniLM-L6-v2 embedding model.
    Runs locally on CPU using onnxruntime (fast, lightweight, and avoids
    heavy PyTorch DLL dependencies that Windows Security policies block)."""
    # 📚 CONCEPT: Vector Embeddings
    # Embeddings represent text as arrays of numbers (vectors) in high-dimensional space.
    # Texts with similar meanings will have vectors that are mathematically close.
    # We use all-MiniLM-L6-v2 (384-dimensional vectors) via ONNX Runtime.
    # We use all-MiniLM-L6-v2 (384-dimensional vectors) via ONNX Runtime.
    return embedding_functions.DefaultEmbeddingFunction()


@st.cache_resource(show_spinner=False)
def load_cross_encoder_reranker() -> ONNXCrossEncoderReranker:
    """Load and cache the CPU ONNX Runtime cross-encoder reranker."""
    return ONNXCrossEncoderReranker()

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def get_chroma_client():
    # 📚 CONCEPT: Vector Database (ChromaDB)
    # Similar to Pinecone (but open-source and local), ChromaDB is designed to 
    # store, index, and query high-dimensional vectors efficiently.
    
    # 💡 WHY: PersistentClient vs In-Memory
    # We use PersistentClient so our vector database is saved to the hard drive 
    # ("./chroma_db" folder). If the app restarts, we don't have to re-process 
    # the entire GitHub repository!
    return chromadb.PersistentClient(path="./chroma_db")

# ══════════════════════════════════════════════════════════════════════════════
# INDEXING / STORING DATA
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_repo_collection_name(repo_name: str) -> str:
    safe_repo_name = "".join(c if c.isalnum() or c in ('_', '-') else "_" for c in repo_name)
    safe_repo_name = safe_repo_name[:63]
    if len(safe_repo_name) < 3:
        safe_repo_name = safe_repo_name.ljust(3, '_')
    return safe_repo_name


@st.cache_resource(show_spinner=False)
def _cached_hydrate_bm25_from_chroma(cache_key: str, _collection: Any) -> tuple[BM25Retriever, str]:
    """Hydrate and cache a BM25Retriever directly from a persisted ChromaDB collection."""
    return build_bm25_from_collection(_collection, k1=1.5, b=0.75, idf_mode="lucene")


def clear_bm25_cache() -> None:
    """Clear hydrated BM25 resource cache (useful after manual DB changes or tests)."""
    _SEEN_BM25_CACHE_KEYS.clear()
    _cached_hydrate_bm25_from_chroma.clear()


def get_or_build_bm25_index(
    repo_name: str,
    client: Any = None,
    collection: Any = None,
    return_metadata: bool = False,
) -> BM25Retriever | tuple[BM25Retriever, dict[str, Any]]:
    """Hydrate BM25 directly from ChromaDB using IDs/documents/metadata without pickle.

    Uses a composite cache key derived from normalized repo name, Chroma collection
    UUID (`collection.id`), document count, and stored `corpus_fingerprint` metadata so
    re-indexing or repo switching automatically invalidates stale lexical indices.
    """
    safe_repo_name = _normalize_repo_collection_name(repo_name)
    if collection is None:
        ch_client = client if client is not None else get_chroma_client()
        embedding_fn = load_embedding_model()
        collection = ch_client.get_collection(
            name=safe_repo_name,
            embedding_function=embedding_fn,
        )

    col_meta = getattr(collection, "metadata", None) or {}
    col_uuid = str(getattr(collection, "id", ""))
    col_count = int(collection.count())
    fp_hint = str(col_meta.get("corpus_fingerprint", ""))
    cache_key = f"{safe_repo_name}|{col_uuid}|{col_count}|{fp_hint}"

    seen_for_repo = _SEEN_BM25_CACHE_KEYS.setdefault(safe_repo_name, set())
    was_cached = cache_key in seen_for_repo
    # If repo was re-indexed under a new key, drop prior tracked keys for clarity
    if not was_cached and seen_for_repo:
        seen_for_repo.clear()

    t_hyd0 = time.perf_counter()
    bm25, actual_fp = _cached_hydrate_bm25_from_chroma(cache_key, collection)
    hyd_s = time.perf_counter() - t_hyd0
    seen_for_repo.add(cache_key)

    meta = {
        "safe_collection_name": safe_repo_name,
        "collection_uuid": col_uuid,
        "corpus_fingerprint": actual_fp,
        "cache_key": cache_key,
        "cache_hit": was_cached,
        "hydration_seconds": round(hyd_s, 4),
        "chunks_count": bm25.num_docs,
    }
    if return_metadata:
        return bm25, meta
    return bm25


def index_chunks(chunks: list[dict], repo_name: str, return_metadata: bool = False):
    """Stores all chunks in ChromaDB. Returns count of chunks indexed.
    Uses repo_name as collection name.
    Deletes old collection if re-indexing same repo.
    Preserves backward compatibility: returns int by default.
    If return_metadata=True, returns (total_indexed, index_metadata)."""
    global _LAST_INDEX_METADATA
    t_index_start = time.perf_counter()
    client = get_chroma_client()
    
    embedding_fn = load_embedding_model()
    
    # [1] Make sure repo name is safe for ChromaDB (no weird characters)
    safe_repo_name = _normalize_repo_collection_name(repo_name)
    _SEEN_BM25_CACHE_KEYS.pop(safe_repo_name, None)

    # Pre-resolve deterministic IDs and compute corpus fingerprint for BM25 sync
    resolved_ids = [
        str(c['metadata'].get('chunk_id')) if c['metadata'].get('chunk_id') else f"chunk_{idx}"
        for idx, c in enumerate(chunks)
    ]
    resolved_docs = [c['content'] for c in chunks]
    corpus_fingerprint = compute_corpus_fingerprint(resolved_ids, resolved_docs)

    # [2] Delete existing collection if we are re-indexing
    try:
        client.delete_collection(name=safe_repo_name)
    except Exception:
        pass # Collection didn't exist

    # [3] Create a new fresh collection
    # A "collection" in Chroma is like a table in a relational database
    collection = client.get_or_create_collection(
        name=safe_repo_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "l2", "corpus_fingerprint": corpus_fingerprint},
    )

    # [4] Prepare batch insertion
    # 💡 WHY: Inserting items one-by-one is very slow due to network/DB overhead.
    # Batching (processing multiple chunks at once) is much more efficient. 100 is a safe chunk size.
    batch_size = 100
    total_indexed = 0
    total_collection_add_duration = 0.0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # Split our chunk dictionaries into parallel lists expected by ChromaDB
        documents = resolved_docs[i:i + batch_size]
        metadatas = [c['metadata'] for c in batch]
        ids = resolved_ids[i:i + batch_size]

        # [5] Add vectors to the database
        # Under the hood, ChromaDB calls our EmbeddingFunction to convert
        # the 'documents' into embeddings before saving them.
        t_add_start = time.perf_counter()
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        total_collection_add_duration += (time.perf_counter() - t_add_start)
        total_indexed += len(batch)

    total_index_function_duration = time.perf_counter() - t_index_start
    index_metadata = {
        "repo_name": repo_name,
        "safe_collection_name": safe_repo_name,
        "chunks_count": total_indexed,
        "corpus_fingerprint": corpus_fingerprint,
        "collection_add_duration_seconds": round(total_collection_add_duration, 4),
        "total_index_function_duration_seconds": round(total_index_function_duration, 4),
        "indexing_duration_seconds": round(total_collection_add_duration, 4), # Preserved for backward compatibility
        "embedding_model": "DefaultEmbeddingFunction (all-MiniLM-L6-v2)",
        "embedding_dimension": 384,
        "distance_metric": "l2"
    }
    _LAST_INDEX_METADATA = index_metadata

    if return_metadata:
        return total_indexed, index_metadata
    return total_indexed

def get_last_index_metadata() -> dict | None:
    """Returns metadata from the most recent index_chunks call."""
    return _LAST_INDEX_METADATA

# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL / SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def search_similar(
    query: str,
    repo_name: str,
    n_results: int = 5,
    return_metadata: bool = False,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    candidate_k: int = 20,
    rrf_k: int = 60,
    candidate_pool_mode: str = "rrf_top_20",
    enable_query_rewrite: bool = False,
    chat_history: Any = None,
):
    """Finds top N most relevant code chunks for a query.
    Supports 'dense' (default V0 backward compatibility), 'bm25', 'hybrid_rrf',
    and 'hybrid_rerank' (Hybrid RRF + CPU ONNX Cross-Encoder reranking).
    Optionally rewrites context-dependent follow-up queries via a leakage-free
    gate before retrieval, falling back safely on errors.
    If return_metadata=True, returns (retrieved_chunks, retrieval_metadata)."""
    global _LAST_RETRIEVAL_METADATA
    if retrieval_mode not in SUPPORTED_RETRIEVAL_MODES:
        raise ValueError(
            f"Unsupported retrieval_mode {retrieval_mode!r}. Supported: {SUPPORTED_RETRIEVAL_MODES}"
        )
    if candidate_pool_mode not in ("rrf_top_20", "candidate_union"):
        raise ValueError(
            f"Unsupported candidate_pool_mode {candidate_pool_mode!r}; expected 'rrf_top_20' or 'candidate_union'"
        )

    client = get_chroma_client()
    embedding_fn = load_embedding_model()
    safe_repo_name = _normalize_repo_collection_name(repo_name)

    try:
        collection = client.get_collection(
            name=safe_repo_name,
            embedding_function=embedding_fn
        )
    except Exception as e:
        empty_meta = {
            "query": query,
            "effective_query": query,
            "repo_name": repo_name,
            "safe_collection_name": safe_repo_name,
            "retrieval_mode": retrieval_mode,
            "n_results_requested": n_results,
            "n_results_returned": 0,
            "retrieval_latency_seconds": 0.0,
            "distance_metric": "l2" if retrieval_mode != "bm25" else "bm25",
            "enable_query_rewrite": bool(enable_query_rewrite),
            "query_rewritten": False,
            "rewrite_failed": False,
            "rewrite_error": None,
            "rewrite_latency_seconds": 0.0,
            "error": str(e)
        }
        _LAST_RETRIEVAL_METADATA = empty_meta
        if return_metadata:
            return [], empty_meta
        return []

    effective_query = query
    rewrite_meta: dict[str, Any] = {
        "original_query": query,
        "retrieval_query": query,
        "rewritten": False,
        "gate_triggered": False,
        "gate_reason": "disabled",
        "rewrite_failed": False,
        "error": None,
        "rewrite_latency_ms": 0.0,
        "gemini_calls": 0,
    }
    if enable_query_rewrite:
        try:
            rewrite_meta = rewrite_conversational_query(
                query=query,
                previous_context=chat_history,
                repository_name=repo_name,
            )
            effective_query = rewrite_meta.get("retrieval_query") or query
        except Exception as exc:
            rewrite_meta = {
                "original_query": query,
                "retrieval_query": query,
                "rewritten": False,
                "gate_triggered": True,
                "gate_reason": "exception_fallback",
                "rewrite_failed": True,
                "error": str(exc),
                "rewrite_latency_ms": 0.0,
                "gemini_calls": 0,
            }
            effective_query = query

    def _query_chroma_dense(q_txt: str, n_req: int) -> list[dict]:
        total_in_col = int(collection.count())
        if total_in_col <= 0:
            return []
        res = collection.query(
            query_texts=[q_txt],
            n_results=min(int(n_req), total_in_col),
        )
        entries = []
        if res.get('documents') and len(res['documents'][0]) > 0:
            for i in range(len(res['documents'][0])):
                entries.append({
                    'content': res['documents'][0][i],
                    'metadata': res['metadatas'][0][i],
                    'id': res['ids'][0][i] if ('ids' in res and res['ids']) else f"chunk_{i}",
                    'distance': res['distances'][0][i] if ('distances' in res and res['distances']) else None,
                })
        return entries

    t_query_start = time.perf_counter()
    extra_meta: dict[str, Any] = {}

    if retrieval_mode == "dense":
        retrieved_chunks = _query_chroma_dense(effective_query, n_results)
        metric_name = "l2"
    elif retrieval_mode == "bm25":
        bm25, bm25_hyd_meta = get_or_build_bm25_index(
            repo_name, client=client, collection=collection, return_metadata=True
        )
        bm25_hits, bm25_q_meta = bm25.search(effective_query, top_k=n_results, return_metadata=True)
        retrieved_chunks = [
            {
                'content': hit['content'],
                'metadata': hit['metadata'],
                'id': hit['id'],
                'distance': None,
                'bm25_score': hit['bm25_score'],
                'bm25_rank': hit['bm25_rank'],
                'retrieval_sources': ['bm25'],
            }
            for hit in bm25_hits
        ]
        metric_name = "bm25"
        extra_meta = {
            "bm25_corpus_fingerprint": bm25_hyd_meta["corpus_fingerprint"],
            "bm25_cache_hit": bm25_hyd_meta["cache_hit"],
            "bm25_eligible_overlap_docs": bm25_q_meta["eligible_overlap_docs"],
        }
    elif retrieval_mode == "hybrid_rrf":
        bm25, bm25_hyd_meta = get_or_build_bm25_index(
            repo_name, client=client, collection=collection, return_metadata=True
        )
        cand_pool_k = max(int(candidate_k), int(n_results))
        hybrid = HybridRetriever(
            dense_search_fn=_query_chroma_dense,
            bm25_retriever=bm25,
            candidate_k=cand_pool_k,
            rrf_k=rrf_k,
        )
        retrieved_chunks, diag = hybrid.retrieve(
            effective_query, top_k=n_results, candidate_k=cand_pool_k, return_diagnostics=True
        )
        metric_name = f"hybrid_rrf_k{rrf_k}"
        extra_meta = {
            "bm25_corpus_fingerprint": bm25_hyd_meta["corpus_fingerprint"],
            "bm25_cache_hit": bm25_hyd_meta["cache_hit"],
            "dense_candidates_count": diag["rrf_stats"]["dense_candidates_count"],
            "bm25_candidates_count": diag["rrf_stats"]["bm25_candidates_count"],
            "union_candidates_count": diag["rrf_stats"]["union_candidates_count"],
            "bm25_fallback_to_dense_only": diag["rrf_stats"]["bm25_fallback_to_dense_only"],
            "rrf_k": rrf_k,
        }
    else:  # hybrid_rerank
        bm25, bm25_hyd_meta = get_or_build_bm25_index(
            repo_name, client=client, collection=collection, return_metadata=True
        )
        cand_pool_k = max(int(candidate_k), int(n_results))
        hybrid = HybridRetriever(
            dense_search_fn=_query_chroma_dense,
            bm25_retriever=bm25,
            candidate_k=cand_pool_k,
            rrf_k=rrf_k,
        )
        try:
            reranker = load_cross_encoder_reranker()
            rerank_retriever = HybridRerankRetriever(
                hybrid_retriever=hybrid,
                reranker=reranker,
                candidate_pool_mode=candidate_pool_mode,
            )
            retrieved_chunks, r_diag = rerank_retriever.retrieve(
                effective_query,
                top_k=n_results,
                candidate_k=cand_pool_k,
                fallback_to_rrf_on_error=True,
                return_diagnostics=True,
            )
            h_diag = r_diag["stage1_hybrid"]
            r_meta = r_diag["reranker_meta"]
            fallback_used = bool(r_diag["fallback_to_rrf_used"])
            rerank_failed = bool(r_diag["rerank_failed"])
            rerank_error = r_diag["rerank_error"]
            cand_pool_count = r_diag["candidate_pool_count"]
            metric_name = (
                f"hybrid_rrf_k{rrf_k}_fallback"
                if fallback_used
                else f"hybrid_rerank_{candidate_pool_mode}_k{rrf_k}"
            )
        except Exception as exc:
            retrieved_chunks, h_diag = hybrid.retrieve(
                effective_query, top_k=n_results, candidate_k=cand_pool_k, return_diagnostics=True
            )
            r_meta = {
                "pairs_count": 0,
                "pairs_truncated_count": 0,
                "inference_ms": 0.0,
            }
            fallback_used = True
            rerank_failed = True
            rerank_error = str(exc)
            cand_pool_count = len(retrieved_chunks)
            metric_name = f"hybrid_rrf_k{rrf_k}_fallback"

        extra_meta = {
            "bm25_corpus_fingerprint": bm25_hyd_meta["corpus_fingerprint"],
            "bm25_cache_hit": bm25_hyd_meta["cache_hit"],
            "dense_candidates_count": h_diag["rrf_stats"]["dense_candidates_count"],
            "bm25_candidates_count": h_diag["rrf_stats"]["bm25_candidates_count"],
            "union_candidates_count": h_diag["rrf_stats"]["union_candidates_count"],
            "bm25_fallback_to_dense_only": h_diag["rrf_stats"]["bm25_fallback_to_dense_only"],
            "rrf_k": rrf_k,
            "candidate_pool_mode": candidate_pool_mode,
            "candidate_pool_count": cand_pool_count,
            "reranker_model": DEFAULT_CROSS_ENCODER_MODEL,
            "rerank_failed": rerank_failed,
            "rerank_error": rerank_error,
            "fallback_to_rrf_used": fallback_used,
            "rerank_pairs_truncated_count": r_meta.get("pairs_truncated_count", 0),
            "reranker_inference_ms": r_meta.get("inference_ms", 0.0),
            "top1_reranker_raw_logit": (
                retrieved_chunks[0].get("reranker_raw_logit") if retrieved_chunks else None
            ),
            "top1_reranker_sigmoid_score": (
                retrieved_chunks[0].get("reranker_sigmoid_score") if retrieved_chunks else None
            ),
        }

    retrieval_latency = time.perf_counter() - t_query_start

    retrieval_metadata = {
        "query": query,
        "effective_query": effective_query,
        "repo_name": repo_name,
        "safe_collection_name": safe_repo_name,
        "retrieval_mode": retrieval_mode,
        "n_results_requested": n_results,
        "n_results_returned": len(retrieved_chunks),
        "retrieval_latency_seconds": round(retrieval_latency, 4),
        "distance_metric": metric_name,
        "enable_query_rewrite": bool(enable_query_rewrite),
        "query_rewritten": bool(rewrite_meta.get("rewritten", False)),
        "rewrite_gate_triggered": bool(rewrite_meta.get("gate_triggered", False)),
        "rewrite_gate_reason": rewrite_meta.get("gate_reason"),
        "rewrite_failed": bool(rewrite_meta.get("rewrite_failed", False)),
        "rewrite_error": rewrite_meta.get("error"),
        "rewrite_latency_seconds": round(float(rewrite_meta.get("rewrite_latency_ms", 0.0)) / 1000.0, 4),
        **extra_meta,
    }
    _LAST_RETRIEVAL_METADATA = retrieval_metadata

    if return_metadata:
        return retrieved_chunks, retrieval_metadata
    return retrieved_chunks

def get_last_retrieval_metadata() -> dict | None:
    """Returns metadata from the most recent search_similar call."""
    return _LAST_RETRIEVAL_METADATA

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def is_repo_indexed(repo_name: str) -> bool:
    """Checks if a repo has already been indexed in ChromaDB."""
    client = get_chroma_client()
    safe_repo_name = _normalize_repo_collection_name(repo_name)

    try:
        # If we can successfully get the collection, it exists!
        client.get_collection(name=safe_repo_name)
        return True
    except ValueError:
        # ValueError is raised if the collection does not exist
        return False
