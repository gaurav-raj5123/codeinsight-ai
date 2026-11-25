"""
# 📚 CONCEPT: The Orchestration & UI Layer (Step 5 in our RAG Pipeline)
# This file serves as the "brain" and "face" of our RAG application. 
# It uses Streamlit, a rapid prototyping framework for Python, to create a web interface.
#
# 🔗 CONNECTS TO: 
# - repo_loader.py (for fetching code)
# - chunker.py (for breaking code into pieces)
# - embedder.py (for vector storage and retrieval)
# - llm.py (for generating the final answer)
#
# 💡 WHY Streamlit? It allows us to build interactive web apps using pure Python,
# without needing to write HTML, CSS, or JavaScript frontend code.
"""

import streamlit as st
import os
import time
from dotenv import load_dotenv

# ══════════════════
# MODULE IMPORTS
# ══════════════════
# Import our custom modules that make up the RAG pipeline
import repo_loader
import chunker
import embedder
import llm
import telemetry

# Load environment variables (like API keys) from the .env file
load_dotenv()

# Configure the main Streamlit page settings
st.set_page_config(page_title="Codebase AI Assistant", page_icon="🧠", layout="wide")

st.title("🧠 Codebase AI Assistant")

# ══════════════════
# SESSION STATE MANAGEMENT
# ══════════════════
# 📚 CONCEPT: Streamlit's Execution Model
# Streamlit re-runs the ENTIRE script from top to bottom every time a user 
# interacts with a widget (like clicking a button or typing text).
# 
# 💡 WHY session_state? Because the script reruns constantly, regular variables 
# get reset. `st.session_state` is a dictionary that persists data across these reruns, 
# allowing us to remember the chat history and which repo is loaded.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "repo_name" not in st.session_state:
    st.session_state.repo_name = None
if "directory_tree" not in st.session_state:
    st.session_state.directory_tree = ""
if "baseline_run" not in st.session_state:
    st.session_state.baseline_run = None
if "indexing_diagnostics" not in st.session_state:
    st.session_state.indexing_diagnostics = None
if "chunking_strategy" not in st.session_state:
    st.session_state.chunking_strategy = "code_aware_v1"
if "retrieval_mode" not in st.session_state:
    st.session_state.retrieval_mode = "hybrid_rerank"
if "enable_query_rewrite" not in st.session_state:
    st.session_state.enable_query_rewrite = False

# ══════════════════
# PHASE 1: INDEXING (Sidebar)
# ══════════════════
# 📚 CONCEPT: The Indexing Phase 
# Before we can answer questions, we need to process the knowledge base (the code).
# This section orchestrates Steps 1-3 of our pipeline.
with st.sidebar:
    st.header("1️⃣ Index Repository")
    github_url = st.text_input("GitHub URL", placeholder="https://github.com/pallets/click")
    strategy_labels = {
        "code_aware_v1": "code_aware_v1 (Recommended — <=256 tok AST/Brace)",
        "existing_chunker_v0": "existing_chunker_v0 (Original V0 Baseline)",
        "fixed_window_v0": "fixed_window_v0 (Fixed 50-line Window)",
    }
    selected_strategy = st.selectbox(
        "Chunking Strategy",
        options=list(chunker.SUPPORTED_CHUNKING_STRATEGIES),
        index=list(chunker.SUPPORTED_CHUNKING_STRATEGIES).index(st.session_state.chunking_strategy)
        if st.session_state.chunking_strategy in chunker.SUPPORTED_CHUNKING_STRATEGIES
        else 2,
        format_func=lambda s: strategy_labels.get(s, s),
    )
    st.session_state.chunking_strategy = selected_strategy

    retrieval_labels = {
        "hybrid_rerank": "hybrid_rerank (Recommended — Hybrid RRF + ONNX Cross-Encoder)",
        "hybrid_rrf": "hybrid_rrf (Fast — Dense + BM25 RRF k=60)",
        "dense": "dense (Vector L2 Only)",
        "bm25": "bm25 (Lexical Code-Aware BM25 Only)",
    }
    selected_retrieval_mode = st.selectbox(
        "Retrieval Strategy",
        options=list(embedder.SUPPORTED_RETRIEVAL_MODES),
        index=list(embedder.SUPPORTED_RETRIEVAL_MODES).index(st.session_state.retrieval_mode)
        if st.session_state.retrieval_mode in embedder.SUPPORTED_RETRIEVAL_MODES
        else 0,
        format_func=lambda m: retrieval_labels.get(m, m),
    )
    st.session_state.retrieval_mode = selected_retrieval_mode
    st.session_state.enable_query_rewrite = st.checkbox(
        "Enable Conversational Query Rewriting (Experimental)",
        value=bool(st.session_state.enable_query_rewrite),
        help=(
            "Rewrites pronoun-dependent follow-up queries via Gemini before retrieval. "
            "Standalone queries bypass rewriting cleanly (0 API calls); falls back to original query on errors."
        ),
    )
    
    # When the user clicks the index button, start the pipeline
    if st.button("Index Repository", type="primary"):
        if not github_url:
            st.error("Please enter a GitHub URL")
        else:
            # Clean up the URL to extract a clean repository name
            repo_name = github_url.rstrip('/').split('/')[-1]
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]
                
            # Save the repo name in session state so we remember it across reruns
            st.session_state.repo_name = repo_name
                
            try:
                repo_path = None
                t_index_all_start = time.perf_counter()
                
                # [1] Data Ingestion (repo_loader)
                # st.spinner gives visual feedback to the user while long tasks run
                with st.spinner("📥 Cloning repository... (Shallow clone)"):
                    repo_path, repo_meta = repo_loader.clone_repo(github_url, return_metadata=True)
                    
                with st.spinner("🧹 Extracting source files..."):
                    files = repo_loader.get_source_files(repo_path)
                    st.session_state.directory_tree = repo_loader.generate_directory_tree(repo_path)
                    
                if not files:
                    st.error("No supported source files found in repository (.py, .js, .ts, .java, .cpp, .c, .h).")
                    if repo_path:
                        # 💡 WHY cleanup? Always clean up temporary files to avoid filling up the disk
                        repo_loader.cleanup_repo(repo_path)
                else:
                    # [2] Data Processing / Chunking (chunker)
                    with st.spinner(f"✂️ Chunking {len(files)} files ({selected_strategy})..."):
                        t_chunk_start = time.perf_counter()
                        v1_tokenizer = None
                        if selected_strategy == "code_aware_v1":
                            import chunker_v1

                            v1_tokenizer, _ = chunker_v1.get_embedding_tokenizer()
                        all_chunks = []
                        for file_data in files:
                            chunks = chunker.chunk_code(
                                file_data,
                                strategy=selected_strategy,
                                repository=repo_name,
                                tokenizer=v1_tokenizer,
                            )
                            all_chunks.extend(chunks)
                        chunking_duration = time.perf_counter() - t_chunk_start
                            
                    # [3] Embedding & Storage (embedder)
                    with st.spinner(f"💾 Indexing {len(all_chunks)} chunks into ChromaDB..."):
                        count, index_meta = embedder.index_chunks(all_chunks, repo_name, return_metadata=True)
                        
                    if repo_path:
                        repo_loader.cleanup_repo(repo_path)
                        
                    total_indexing_duration = time.perf_counter() - t_index_all_start
                    
                    # Record baseline telemetry
                    run_record = telemetry.create_baseline_run_record(
                        repo_url=github_url,
                        owner=repo_meta.get("owner", ""),
                        repo_name=repo_name,
                        commit_sha=repo_meta.get("commit_sha", "unknown"),
                        files_indexed=len(files),
                        chunks_indexed=count,
                        repo_download_duration=repo_meta.get("download_duration_seconds", 0.0),
                        repo_extraction_duration=repo_meta.get("extraction_duration_seconds", 0.0),
                        repo_total_loading_duration=repo_meta.get("total_loading_duration_seconds", 0.0),
                        chunking_duration=chunking_duration,
                        chromadb_add_duration=index_meta.get("collection_add_duration_seconds", 0.0),
                        total_index_function_duration=index_meta.get("total_index_function_duration_seconds", 0.0),
                        total_indexing_duration=total_indexing_duration,
                        embedding_model=index_meta.get("embedding_model", "DefaultEmbeddingFunction (all-MiniLM-L6-v2)"),
                        embedding_dimensions=index_meta.get("embedding_dimension", 384),
                        distance_metric=index_meta.get("distance_metric", "l2")
                    )
                    telemetry.save_baseline_run(run_record)
                    st.session_state.baseline_run = run_record
                    st.session_state.indexing_diagnostics = {
                        "commit_sha": repo_meta.get("commit_sha", "unknown"),
                        "chunking_strategy": selected_strategy,
                        "files": len(files),
                        "chunks": count,
                        "download_sec": repo_meta.get("download_duration_seconds", 0.0),
                        "extraction_sec": repo_meta.get("extraction_duration_seconds", 0.0),
                        "total_loading_sec": repo_meta.get("total_loading_duration_seconds", 0.0),
                        "chunking_sec": round(chunking_duration, 4),
                        "chromadb_add_sec": index_meta.get("collection_add_duration_seconds", 0.0),
                        "index_function_sec": index_meta.get("total_index_function_duration_seconds", 0.0),
                        "total_indexing_sec": round(total_indexing_duration, 4)
                    }
                    st.success(f"✅ Successfully indexed {len(files)} files into {count} chunks!")
            except Exception as e:
                # 💡 WHY catch exceptions? So the app doesn't crash completely. 
                # We show a nice error message to the user instead.
                st.error(f"Error during indexing: {str(e)}")
                if 'repo_path' in locals() and repo_path:
                    repo_loader.cleanup_repo(repo_path)

    # Display current state to the user
    if st.session_state.repo_name:
        st.success(f"Currently indexed: **{st.session_state.repo_name}**")
        
    if st.session_state.indexing_diagnostics:
        with st.expander("📊 Indexing Diagnostics", expanded=False):
            diag = st.session_state.indexing_diagnostics
            st.write(f"**Commit SHA:** `{diag['commit_sha']}`")
            st.write(f"**Chunking Strategy:** `{diag.get('chunking_strategy', chunker.DEFAULT_CHUNKING_STRATEGY)}`")
            st.write(f"**Files Indexed:** {diag['files']}")
            st.write(f"**Chunks Indexed:** {diag['chunks']}")
            st.write(f"**Download Time:** {diag['download_sec']:.2f}s")
            st.write(f"**Extraction Time:** {diag['extraction_sec']:.2f}s")
            st.write(f"**Total Repo Load Time:** {diag['total_loading_sec']:.2f}s")
            st.write(f"**Chunking Time:** {diag['chunking_sec']:.2f}s")
            st.write(f"**ChromaDB Add (Embed+Insert):** {diag['chromadb_add_sec']:.2f}s")
            st.write(f"**Index Function Total:** {diag['index_function_sec']:.2f}s")
            st.write(f"**Total Indexing Time:** {diag['total_indexing_sec']:.2f}s")
        
    if st.session_state.directory_tree:
        with st.expander("📂 Repository Structure", expanded=False):
            st.code(st.session_state.directory_tree, language="text")

# ══════════════════
# PHASE 2: QUERYING & GENERATION (Main Interface)
# ══════════════════
# 📚 CONCEPT: The Retrieval-Augmented Generation Phase
# Now that the data is indexed, the user can chat with it. This section 
# handles user input, fetches relevant context, and gets the LLM response.
st.header("2️⃣ Chat with your Code")

# Check for API key early to warn the user
if not os.getenv("GEMINI_API_KEY"):
    st.warning("⚠️ GEMINI_API_KEY is not set. Please add it to your .env file or export it.")

# Re-draw the chat history on every rerun so previous messages stay visible
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If this message was from the assistant, show the chunks it used as context
        if "chunks" in message and message["chunks"]:
            with st.expander("📄 Retrieved Code Chunks"):
                for chunk in message["chunks"]:
                    meta = chunk['metadata']
                    
                    # Determine formatting language based on file extension
                    lang = "python"
                    if meta.get('file_path', '').endswith(('.js', '.ts')): lang = "javascript"
                    elif meta.get('file_path', '').endswith('.java'): lang = "java"
                    elif meta.get('file_path', '').endswith(('.cpp', '.c', '.h')): lang = "cpp"
                        
                    id_str = f" | ID: {chunk.get('id')}" if chunk.get('id') else ""
                    dist_str = f" | Distance: {chunk.get('distance'):.4f}" if chunk.get('distance') is not None else ""
                    st.code(f"// {meta.get('file_path')} | Lines {meta.get('start_line')}-{meta.get('end_line')}{id_str}{dist_str}\n" + chunk['content'], language=lang)
                    
        if "diagnostics" in message and message["diagnostics"]:
            with st.expander("📊 Baseline Diagnostics", expanded=False):
                diag = message["diagnostics"]
                r = diag.get("retrieval", {})
                g = diag.get("generation", {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retrieval Latency", f"{r.get('total_retrieval_latency_seconds', 0.0):.3f}s")
                with col2:
                    st.metric("Generation Latency", f"{g.get('latency_seconds', 0.0):.3f}s")
                with col3:
                    st.metric("Total Turn Latency", f"{diag.get('total_turn_latency_seconds', 0.0):.3f}s")
                st.write(f"**Chunks:** {r.get('actual_results_returned', 0)} / {r.get('top_k_requested', 5)} | **Metric:** {r.get('distance_metric', 'l2')} | **Model:** {g.get('model', 'gemini-3.5-flash')}")

# Wait for user to enter a new prompt
# `st.chat_input` pauses execution here until the user submits text
if prompt := st.chat_input("Ask a question about the code..."):
    # Guard clauses: prevent querying if we aren't ready
    if not st.session_state.repo_name:
        st.warning("Please index a repository first!")
    elif not os.getenv("GEMINI_API_KEY"):
        st.error("Please set GEMINI_API_KEY to ask questions.")
    else:
        # Display the user's new message instantly
        st.chat_message("user").markdown(prompt)
        # Save it to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response area
        with st.chat_message("assistant"):
            retrieved_chunks = []
            t_turn_start = time.perf_counter()
            
            # [4] Retrieval (embedder)
            # Find the chunks of code most relevant to the user's question
            with st.spinner("Searching codebase..."):
                retrieved_chunks, retrieval_meta = embedder.search_similar(
                    prompt,
                    st.session_state.repo_name,
                    return_metadata=True,
                    retrieval_mode=st.session_state.retrieval_mode,
                    enable_query_rewrite=st.session_state.enable_query_rewrite,
                    chat_history=st.session_state.messages[:-1],
                )
            
            # [5] Generation (llm)
            # Combine the user's question, the retrieved code chunks, repo context,
            # AND conversation history to build a comprehensive prompt.
            # 📚 CONCEPT: Conversation Memory
            # We pass the previous Q&A pairs so Gemini can understand follow-up 
            # questions like "explain more" or "what about the auth module?"
            with st.spinner("Generating answer..."):
                llm_prompt = llm.build_prompt(
                    prompt, 
                    retrieved_chunks, 
                    st.session_state.directory_tree,
                    chat_history=st.session_state.messages[:-1]  # Exclude current prompt so it appears only in QUESTION
                )
                response, gen_meta = llm.ask_gemini(llm_prompt, return_metadata=True)
                
            total_turn_duration = time.perf_counter() - t_turn_start
            
            # Render the final AI answer to the screen
            st.markdown(response)
            
            # Show the chunks that were used so the user can verify the context (transparency)
            if retrieved_chunks:
                with st.expander("📄 Retrieved Code Chunks"):
                    if retrieval_meta.get("query_rewritten"):
                        st.caption(f"🔎 Rewritten retrieval query: `{retrieval_meta.get('effective_query')}`")
                    elif retrieval_meta.get("rewrite_failed"):
                        st.caption("⚠️ Conversational query rewrite failed; retrieved with original query.")
                    if retrieval_meta.get("fallback_to_rrf_used"):
                        st.caption("⚠️ Cross-encoder reranker unavailable; fell back to Hybrid RRF scores.")
                    for chunk in retrieved_chunks:
                        meta = chunk['metadata']
                        
                        lang = "python"
                        if meta.get('file_path', '').endswith(('.js', '.ts')): lang = "javascript"
                        elif meta.get('file_path', '').endswith('.java'): lang = "java"
                        elif meta.get('file_path', '').endswith(('.cpp', '.c', '.h')): lang = "cpp"
                            
                        id_str = f" | ID: {chunk.get('id')}" if chunk.get('id') else ""
                        dist_str = f" | Distance: {chunk.get('distance'):.4f}" if chunk.get('distance') is not None else ""
                        logit_str = (
                            f" | Rerank Logit: {chunk.get('reranker_raw_logit'):.3f}"
                            if chunk.get("reranker_raw_logit") is not None
                            else ""
                        )
                        st.code(f"// {meta.get('file_path')} | Lines {meta.get('start_line')}-{meta.get('end_line')}{id_str}{dist_str}{logit_str}\n" + chunk['content'], language=lang)
            
            # Show diagnostic metrics in expander
            with st.expander("📊 Baseline Diagnostics", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retrieval Latency", f"{retrieval_meta.get('retrieval_latency_seconds', 0.0):.3f}s")
                with col2:
                    st.metric("Generation Latency", f"{gen_meta.get('latency_seconds', 0.0):.3f}s")
                with col3:
                    st.metric("Total Turn Latency", f"{total_turn_duration:.3f}s")
                st.write(
                    f"**Chunks:** {len(retrieved_chunks)} / {retrieval_meta.get('n_results_requested', 5)} | "
                    f"**Mode:** {retrieval_meta.get('retrieval_mode', 'dense')} | "
                    f"**Metric:** {retrieval_meta.get('distance_metric', 'l2')} | "
                    f"**Model:** {gen_meta.get('model', 'gemini-3.5-flash')}"
                )
                if retrieval_meta.get("enable_query_rewrite"):
                    st.caption(
                        f"**Query Rewrite:** {'Rewritten' if retrieval_meta.get('query_rewritten') else 'Bypassed/Fallback'} "
                        f"(`{retrieval_meta.get('rewrite_gate_reason')}`, {retrieval_meta.get('rewrite_latency_seconds', 0.0):.3f}s)"
                    )
            
            # Record query event into baseline run if active
            query_event = telemetry.create_query_record(
                query_text=prompt,
                repo_name=st.session_state.repo_name,
                top_k_requested=retrieval_meta.get("n_results_requested", 5),
                actual_results_returned=len(retrieved_chunks),
                retrieval_latency_seconds=retrieval_meta.get("retrieval_latency_seconds", 0.0),
                retrieved_chunks=retrieved_chunks,
                distance_metric=retrieval_meta.get("distance_metric", "l2"),
                generation_model=gen_meta.get("model", "gemini-3.5-flash"),
                generation_latency_seconds=gen_meta.get("latency_seconds", 0.0),
                generation_success=gen_meta.get("success", True),
                generation_error=gen_meta.get("error"),
                prompt_length_chars=gen_meta.get("prompt_length_chars", 0),
                response_length_chars=gen_meta.get("response_length_chars", 0),
                total_turn_latency_seconds=total_turn_duration
            )
            
            if st.session_state.baseline_run:
                st.session_state.baseline_run["queries"].append(query_event)
                telemetry.save_baseline_run(st.session_state.baseline_run)
            else:
                run_record = telemetry.create_baseline_run_record(
                    repo_name=st.session_state.repo_name
                )
                run_record["queries"].append(query_event)
                telemetry.save_baseline_run(run_record)
                st.session_state.baseline_run = run_record
            
        # Add the assistant's complete response to the session history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "chunks": retrieved_chunks,
            "diagnostics": query_event
        })
