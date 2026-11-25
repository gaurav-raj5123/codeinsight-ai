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

# Global telemetry caches for last indexing and retrieval operations
_LAST_INDEX_METADATA: dict | None = None
_LAST_RETRIEVAL_METADATA: dict | None = None

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
    return embedding_functions.DefaultEmbeddingFunction()

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
    # 💡 WHY: ChromaDB throws errors if collection names contain invalid characters.
    safe_repo_name = "".join(c if c.isalnum() or c in ('_', '-') else "_" for c in repo_name)
    # Chroma requires collection names to be between 3 and 63 characters
    safe_repo_name = safe_repo_name[:63]
    if len(safe_repo_name) < 3:
        safe_repo_name = safe_repo_name.ljust(3, '_')
        
    # [2] Delete existing collection if we are re-indexing
    try:
        client.delete_collection(name=safe_repo_name)
    except Exception:
        pass # Collection didn't exist
        
    # [3] Create a new fresh collection
    # A "collection" in Chroma is like a table in a relational database
    collection = client.get_or_create_collection(
        name=safe_repo_name,
        embedding_function=embedding_fn
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
        documents = [c['content'] for c in batch]
        metadatas = [c['metadata'] for c in batch]
        # Prefer deterministic chunk_id when provided (e.g., code_aware_v1), otherwise fall back cleanly to V0 IDs
        ids = [
            str(c['metadata'].get('chunk_id')) if c['metadata'].get('chunk_id') else f"chunk_{total_indexed + j}"
            for j, c in enumerate(batch)
        ]
        
        # [5] Add vectors to the database
        # Under the hood, ChromaDB calls our CustomSentenceTransformerEF to convert
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

def search_similar(query: str, repo_name: str, n_results: int = 5, return_metadata: bool = False):
    """Finds top N most similar code chunks to the query.
    Preserves backward compatibility: returns list[dict] by default.
    Adds 'id' and 'distance' to each returned chunk dictionary.
    If return_metadata=True, returns (retrieved_chunks, retrieval_metadata)."""
    global _LAST_RETRIEVAL_METADATA
    client = get_chroma_client()
    
    embedding_fn = load_embedding_model()
    
    # Sanitize the repo name exactly like we did during indexing
    safe_repo_name = "".join(c if c.isalnum() or c in ('_', '-') else "_" for c in repo_name)
    safe_repo_name = safe_repo_name[:63]
    if len(safe_repo_name) < 3:
        safe_repo_name = safe_repo_name.ljust(3, '_')
        
    try:
        # Connect to the specific repository's data
        collection = client.get_collection(
            name=safe_repo_name,
            embedding_function=embedding_fn
        )
    except Exception as e:
        empty_meta = {
            "query": query,
            "repo_name": repo_name,
            "safe_collection_name": safe_repo_name,
            "n_results_requested": n_results,
            "n_results_returned": 0,
            "retrieval_latency_seconds": 0.0,
            "distance_metric": "l2",
            "error": str(e)
        }
        _LAST_RETRIEVAL_METADATA = empty_meta
        if return_metadata:
            return [], empty_meta
        return [] # Collection not found, meaning repo wasn't indexed
        
    # 📚 CONCEPT: Similarity Search (ChromaDB default L2 squared Euclidean distance)
    # 1. First, the query string is converted into a vector using the same model.
    # 2. Then, ChromaDB calculates the distance between the query vector and all chunk vectors.
    # 3. It returns the top N chunks that are closest mathematically.
    t_query_start = time.perf_counter()
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    retrieval_latency = time.perf_counter() - t_query_start
    
    # Reconstruct the results into our standard dictionary format with IDs and distances preserved
    retrieved_chunks = []
    if results['documents'] and len(results['documents'][0]) > 0:
        for i in range(len(results['documents'][0])):
            chunk_entry = {
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'id': results['ids'][0][i] if ('ids' in results and results['ids']) else f"chunk_{i}",
                'distance': results['distances'][0][i] if ('distances' in results and results['distances']) else None
            }
            retrieved_chunks.append(chunk_entry)
            
    retrieval_metadata = {
        "query": query,
        "repo_name": repo_name,
        "safe_collection_name": safe_repo_name,
        "n_results_requested": n_results,
        "n_results_returned": len(retrieved_chunks),
        "retrieval_latency_seconds": round(retrieval_latency, 4),
        "distance_metric": "l2"
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
    # Sanitize name to match the collection
    safe_repo_name = "".join(c if c.isalnum() or c in ('_', '-') else "_" for c in repo_name)
    safe_repo_name = safe_repo_name[:63]
    if len(safe_repo_name) < 3:
        safe_repo_name = safe_repo_name.ljust(3, '_')
        
    try:
        # If we can successfully get the collection, it exists!
        client.get_collection(name=safe_repo_name)
        return True
    except ValueError:
        # ValueError is raised if the collection does not exist
        return False
