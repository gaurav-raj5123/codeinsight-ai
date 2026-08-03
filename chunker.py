"""
# ════════════════════════════════════════════════════════════════════════════
# 📄 FILE: chunker.py (Step 2 in the RAG Pipeline)
# ════════════════════════════════════════════════════════════════════════════
# 
# 🔗 CONNECTS TO: 
# - Input from: repo_loader.py (raw file contents)
# - Output to: embedder.py (smaller, meaningful chunks of code)
#
# 📚 CONCEPT: What is Chunking and Why do we need it?
# In a RAG (Retrieval-Augmented Generation) pipeline, we can't feed an entire 
# repository (or even a massive file) directly into an LLM because:
# 1. Context Window Limits: LLMs have a maximum number of tokens they can process at once.
# 2. Relevance: If we embed an entire 10,000-line file as one vector, a search for 
#    a specific function will yield noisy results. Smaller chunks mean our vector 
#    search will find the *exact* function or class relevant to the user's query.
#
# This file implements a "Semantic Chunker". It tries to break code intelligently 
# at logical boundaries (like functions or classes) rather than just cutting 
# arbitrarily every N lines.
# ════════════════════════════════════════════════════════════════════════════
"""

import ast
import re

# ════════════════════════════════════════════════════════════════════════════
# 🛠️ MAIN ROUTER FUNCTION
# ════════════════════════════════════════════════════════════════════════════
def chunk_code(file_data: dict) -> list[dict]:
    """Main entry point. Routes to language-specific chunker.
    file_data format: {"file_path": str, "content": str, "extension": str}
    Returns [{content, metadata: {file_path, start_line, end_line, chunk_type}}]
    """
    content = file_data['content']
    file_path = file_data['file_path']
    ext = file_data['extension']
    
    chunks = []
    
    # 💡 WHY: Strategy Pattern
    # Different programming languages have different syntaxes. We use a simple 
    # routing mechanism (if/elif) to apply the right chunking strategy based on 
    # the file extension.
    if ext == '.py':
        chunks = chunk_python(content, file_path)
    elif ext in {'.js', '.ts'}:
        chunks = chunk_javascript(content, file_path)
    elif ext == '.java':
        chunks = chunk_java(content, file_path)
    elif ext in {'.cpp', '.c', '.h'}:
        chunks = chunk_cpp(content, file_path)
        
    # [1] FALLBACK MECHANISM
    # If smart chunking failed (e.g., syntax errors) or returned nothing (e.g., 
    # an unsupported language or a file with no functions), fallback to sliding window.
    if not chunks:
        chunks = chunk_by_sliding_window(content, file_path)
        
    # [2] SIZE ENFORCEMENT
    # 📚 CONCEPT: Maximum Chunk Size
    # Even if we extract a class semantically, it might be 2000 lines long! 
    # We must enforce a maximum size limit so it fits nicely into the LLM context.
    final_chunks = []
    for chunk in chunks:
        # If a chunk is larger than 100 lines, it's too big, split it further
        chunk_lines = chunk['content'].split('\n')
        if len(chunk_lines) > 100:
            # 💡 WHY: We use a sliding window here to blindly cut the massive chunk
            # into manageable 50-line sub-chunks with a 10-line overlap.
            sub_chunks = chunk_by_sliding_window(chunk['content'], file_path, window_size=50, overlap=10)
            
            # Update line numbers to match original file if possible, or just keep as is
            for sc in sub_chunks:
                sc['metadata']['chunk_type'] = f"{chunk['metadata'].get('chunk_type', 'unknown')}_split"
                # Approximate start line
                sc['metadata']['start_line'] = chunk['metadata'].get('start_line', 1) + sc['metadata']['start_line'] - 1
                sc['metadata']['end_line'] = chunk['metadata'].get('start_line', 1) + sc['metadata']['end_line'] - 1
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)
            
    return final_chunks

# ════════════════════════════════════════════════════════════════════════════
# 🐍 PYTHON CHUNKER (AST-Based)
# ════════════════════════════════════════════════════════════════════════════
def chunk_python(content: str, file_path: str) -> list[dict]:
    """Uses Python's ast module for EXACT function/class boundaries."""
    # 📚 CONCEPT: AST (Abstract Syntax Tree)
    # Instead of guessing where functions start using Regex, we use Python's built-in 
    # AST parser. It reads the code exactly like the Python interpreter does, 
    # giving us 100% accurate boundaries for classes and functions.
    try:
        tree = ast.parse(content)
    except SyntaxError:
        # Fallback to regex if there's a syntax error (e.g., incomplete code)
        return chunk_by_regex(content, file_path, [r'^class \w+', r'^def \w+', r'^\s+def \w+'])
        
    lines = content.split('\n')
    chunks = []
    
    # Walk through the parsed syntax tree
    for node in ast.walk(tree):
        # We only care about major structural blocks: Functions and Classes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', start_line)
            
            # 💡 WHY: Decorators (like @staticmethod or @app.route) sit *above* the 
            # def keyword. If we don't include them, the chunk loses important context.
            if getattr(node, 'decorator_list', None) and node.decorator_list:
                start_line = node.decorator_list[0].lineno
                
            chunk_content = '\n'.join(lines[start_line - 1 : end_line])
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    'file_path': file_path,
                    'start_line': start_line,
                    'end_line': end_line,
                    'chunk_type': type(node).__name__
                }
            })
    return chunks

# ════════════════════════════════════════════════════════════════════════════
# 🌐 REGEX-BASED CHUNKERS (JS, Java, C++)
# ════════════════════════════════════════════════════════════════════════════
# 📚 CONCEPT: Regex Parsing vs AST
# For non-Python languages, we don't have built-in AST parsers available in standard 
# library. Instead, we use Regular Expressions (Regex) to *guess* where functions 
# and classes start. It's faster and requires zero dependencies, but it's less 
# accurate (e.g., it might get confused by a function declared inside a string).

def chunk_javascript(content: str, file_path: str) -> list[dict]:
    JS_PATTERNS = [
        r'^function \w+',
        r'^(?:export )?(?:const|let|var) \w+ = .*(=>|function)',
        r'^(?:export )?class \w+',
        r'^\s+\w+\s*\(.*\)\s*\{'
    ]
    return chunk_by_regex(content, file_path, JS_PATTERNS)

def chunk_java(content: str, file_path: str) -> list[dict]:
    JAVA_PATTERNS = [
        r'^\s*(?:public|private|protected|static|final)\s+[\w<>, ]+\s+\w+\s*\(.*\)\s*(?:throws\s+[\w, ]+)?\s*\{',
        r'^\s*(?:public|private|protected|static|final|abstract)\s+class \w+'
    ]
    return chunk_by_regex(content, file_path, JAVA_PATTERNS)

def chunk_cpp(content: str, file_path: str) -> list[dict]:
    CPP_PATTERNS = [
        r'^\w+.*\w+\s*\(.*\)\s*\{',
        r'^class \w+',
        r'^struct \w+'
    ]
    return chunk_by_regex(content, file_path, CPP_PATTERNS)

def chunk_by_regex(content: str, file_path: str, patterns: list[str]) -> list[dict]:
    """Helper to split code based on regex boundaries."""
    lines = content.split('\n')
    chunks = []
    current_chunk_lines = []
    start_line = 1
    
    # Combine all language patterns into one massive OR statement (pattern1|pattern2|...)
    combined_pattern = '|'.join(patterns)
    
    for i, line in enumerate(lines):
        # 💡 WHY: If the line matches a class/function signature AND we already have 
        # lines in our buffer, it means we hit a boundary. Time to save the old chunk!
        if re.search(combined_pattern, line) and current_chunk_lines:
            # We found a new boundary, save the old chunk
            chunk_content = '\n'.join(current_chunk_lines)
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    'file_path': file_path,
                    'start_line': start_line,
                    'end_line': i,
                    'chunk_type': 'regex_split'
                }
            })
            # Start a new buffer with the current line (the new function signature)
            current_chunk_lines = [line]
            start_line = i + 1
        else:
            current_chunk_lines.append(line)
            
    # Add the final trailing chunk left in the buffer
    if current_chunk_lines:
        chunk_content = '\n'.join(current_chunk_lines)
        if chunk_content.strip(): # Don't add empty chunks
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    'file_path': file_path,
                    'start_line': start_line,
                    'end_line': len(lines),
                    'chunk_type': 'regex_split'
                }
            })
            
    return chunks

# ════════════════════════════════════════════════════════════════════════════
# 📏 FALLBACK: SLIDING WINDOW
# ════════════════════════════════════════════════════════════════════════════
def chunk_by_sliding_window(content: str, file_path: str, window_size: int = 50, overlap: int = 10) -> list[dict]:
    """Fallback: sliding window of N lines with M-line overlap."""
    # 📚 CONCEPT: Fixed-Size Chunking (Sliding Window)
    # When semantic parsing fails, we just cut the code into blocks of N lines (window_size).
    # 
    # 📚 CONCEPT: Overlap (Why is it important?)
    # Imagine a function is split right down the middle into Chunk A and Chunk B. 
    # Neither chunk alone has full context! By having a 10-line overlap, Chunk B 
    # will contain the last 10 lines of Chunk A, acting as "glue" and preserving 
    # context across the artificial boundary.
    lines = content.split('\n')
    chunks = []
    
    if len(lines) <= window_size:
        return [{
            'content': content,
            'metadata': {
                'file_path': file_path,
                'start_line': 1,
                'end_line': len(lines),
                'chunk_type': 'sliding_window'
            }
        }]
        
    start = 0
    while start < len(lines):
        end = min(start + window_size, len(lines))
        chunk_lines = lines[start:end]
        chunk_content = '\n'.join(chunk_lines)
        
        chunks.append({
            'content': chunk_content,
            'metadata': {
                'file_path': file_path,
                'start_line': start + 1,
                'end_line': end,
                'chunk_type': 'sliding_window'
            }
        })
        
        if end == len(lines):
            break
            
        # 💡 WHY: Advance the start pointer but step back by the `overlap` amount
        # This creates the overlapping window effect.
        start += (window_size - overlap)
        
    return chunks
