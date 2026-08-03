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
from sentence_transformers import SentenceTransformer
import chromadb

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING & CACHING
# ══════════════════════════════════════════════════════════════════════════════

# 💡 WHY: Streamlit re-runs the entire script on every user interaction. 
# Loading a machine learning model is slow (can take several seconds).
# @st.cache_resource keeps the model in memory across all these re-runs.
@st.cache_resource
def load_embedding_model():
    """Loads SentenceTransformer ONCE and caches it across all Streamlit reruns."""
    # 📚 CONCEPT: Vector Embeddings
    # Embeddings are a way to represent text as arrays of numbers (vectors) in a 
    # high-dimensional space. Texts with similar meanings will have vectors that 
    # are physically close to each other in this mathematical space.
    
    # 📚 CONCEPT: SentenceTransformer ('all-MiniLM-L6-v2')
    # We use 'all-MiniLM-L6-v2', a small, fast transformer model that converts 
    # sentences into 384-dimensional embeddings. It's highly optimized for 
    # semantic search and creates vectors where cosine similarity works well.
    return SentenceTransformer('all-MiniLM-L6-v2')

# ══════════════════════════════════════════════════════════════════════════════
# ADAPTER PATTERN FOR CHROMA
# ══════════════════════════════════════════════════════════════════════════════

# 📚 CONCEPT: Custom Embedding Function (Adapter Pattern)
# ChromaDB needs to know HOW to turn text into embeddings. By default, it might
# download its own model. We want it to use OUR pre-loaded, cached model to save
# time and memory. This class acts as a bridge (adapter) between Chroma and our model.
class CustomSentenceTransformerEF(chromadb.EmbeddingFunction):
    def __init__(self, model):
        self.model = model

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        # [1] Convert the list of texts (input) into vectors
        embeddings = self.model.encode(input)
        # [2] Convert numpy arrays back to standard Python lists for ChromaDB
        return embeddings.tolist()

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

def index_chunks(chunks: list[dict], repo_name: str) -> int:
    """Stores all chunks in ChromaDB. Returns count of chunks indexed.
    Uses repo_name as collection name.
    Deletes old collection if re-indexing same repo."""
    client = get_chroma_client()
    
    model = load_embedding_model()
    custom_ef = CustomSentenceTransformerEF(model)#create Chroma-compatible embedding function
    
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
        embedding_function=custom_ef#You're telling Chroma:
        # Whenever you need embeddings for text in this collection, use my CustomSentenceTransformerEF."

    )
    
    # [4] Prepare batch insertion
    # 💡 WHY: Inserting items one-by-one is very slow due to network/DB overhead.
    # Batching (processing multiple chunks at once) is much more efficient. 100 is a safe chunk size.
    batch_size = 100
    total_indexed = 0
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Split our chunk dictionaries into parallel lists expected by ChromaDB
        documents = [c['content'] for c in batch]
        metadatas = [c['metadata'] for c in batch]
        # Generate unique IDs for each chunk (e.g., chunk_0, chunk_1)
        ids = [f"chunk_{total_indexed + j}" for j in range(len(batch))]
        
        # [5] Add vectors to the database
        # Under the hood, ChromaDB calls our CustomSentenceTransformerEF to convert
        # the 'documents' into embeddings before saving them.
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        total_indexed += len(batch)
        
    return total_indexed

# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL / SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def search_similar(query: str, repo_name: str, n_results: int = 5) -> list[dict]:
    """Finds top N most similar code chunks to the query.
    Returns [{content, metadata: {file_path, start_line, end_line, chunk_type}}]"""
    client = get_chroma_client()
    
    model = load_embedding_model()
    custom_ef = CustomSentenceTransformerEF(model)
    
    # Sanitize the repo name exactly like we did during indexing
    safe_repo_name = "".join(c if c.isalnum() or c in ('_', '-') else "_" for c in repo_name)
    safe_repo_name = safe_repo_name[:63]
    if len(safe_repo_name) < 3:
        safe_repo_name = safe_repo_name.ljust(3, '_')
        
    try:
        # Connect to the specific repository's data
        collection = client.get_collection(
            name=safe_repo_name,
            embedding_function=custom_ef
        )
    except Exception:
        return [] # Collection not found, meaning repo wasn't indexed
        
    # 📚 CONCEPT: Similarity Search (Cosine Similarity)
    # 1. First, the query string is converted into a vector using the same model.
    # 2. Then, ChromaDB calculates the "distance" (typically cosine similarity) 
    #    between the query vector and all chunk vectors in the database.
    # 3. It returns the top N chunks that are closest mathematically.
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    # Reconstruct the results into our standard dictionary format
    retrieved_chunks = []
    if results['documents'] and len(results['documents'][0]) > 0:
        for i in range(len(results['documents'][0])):
            retrieved_chunks.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i]
            })
            
    return retrieved_chunks

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
