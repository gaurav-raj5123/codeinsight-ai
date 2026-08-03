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
from dotenv import load_dotenv

# ══════════════════
# MODULE IMPORTS
# ══════════════════
# Import our custom modules that make up the RAG pipeline
import repo_loader
import chunker
import embedder
import llm

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

# ══════════════════
# PHASE 1: INDEXING (Sidebar)
# ══════════════════
# 📚 CONCEPT: The Indexing Phase 
# Before we can answer questions, we need to process the knowledge base (the code).
# This section orchestrates Steps 1-3 of our pipeline.
with st.sidebar:
    st.header("1️⃣ Index Repository")
    github_url = st.text_input("GitHub URL", placeholder="https://github.com/pallets/click")
    
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
                # [1] Data Ingestion (repo_loader)
                # st.spinner gives visual feedback to the user while long tasks run
                with st.spinner("📥 Cloning repository... (Shallow clone)"):
                    repo_path = repo_loader.clone_repo(github_url)
                    
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
                    with st.spinner(f"✂️ Chunking {len(files)} files..."):
                        all_chunks = []
                        for file_data in files:
                            chunks = chunker.chunk_code(file_data)
                            all_chunks.extend(chunks)
                            
                    # [3] Embedding & Storage (embedder)
                    with st.spinner(f"💾 Indexing {len(all_chunks)} chunks into ChromaDB..."):
                        count = embedder.index_chunks(all_chunks, repo_name)
                        
                    if repo_path:
                        repo_loader.cleanup_repo(repo_path)
                        
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
                        
                    st.code(f"// {meta.get('file_path')} | Lines {meta.get('start_line')}-{meta.get('end_line')}\n" + chunk['content'], language=lang)

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
            # [4] Retrieval (embedder)
            # Find the chunks of code most relevant to the user's question
            with st.spinner("Searching codebase..."):
                retrieved_chunks = embedder.search_similar(prompt, st.session_state.repo_name)
            
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
                    chat_history=st.session_state.messages  # ← THIS IS THE KEY CHANGE
                )
                response = llm.ask_gemini(llm_prompt)
                
            # Render the final AI answer to the screen
            st.markdown(response)
            
            # Show the chunks that were used so the user can verify the context (transparency)
            if retrieved_chunks:
                with st.expander("📄 Retrieved Code Chunks"):
                    for chunk in retrieved_chunks:
                        meta = chunk['metadata']
                        
                        lang = "python"
                        if meta.get('file_path', '').endswith(('.js', '.ts')): lang = "javascript"
                        elif meta.get('file_path', '').endswith('.java'): lang = "java"
                        elif meta.get('file_path', '').endswith(('.cpp', '.c', '.h')): lang = "cpp"
                        
                        st.code(f"// {meta.get('file_path')} | Lines {meta.get('start_line')}-{meta.get('end_line')}\n" + chunk['content'], language=lang)
            
        # Add the assistant's complete response to the session history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "chunks": retrieved_chunks
        })
