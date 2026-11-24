"""
======================================================================
FILE 1: DATA INGESTION LAYER (repo_loader.py)
======================================================================
Welcome to the first step of our RAG (Retrieval-Augmented Generation) pipeline! 🚀

This file handles "Data Ingestion". Before our AI can answer questions 
about a codebase, we need to download it and extract its readable content.

What this file does:
1. Downloads a GitHub repository into a temporary folder.
2. Scans the folder to extract source code (ignoring junk/binary files).
3. Generates a "directory tree" string so the AI understands project structure.
4. Cleans up temporary files to save space.

🔗 CONNECTS TO: 
After this file extracts the raw text from files, the data is passed to `chunker.py` 
(Step 2) to be broken into smaller, digestible pieces for the AI.
"""

# ══════════════════════════════════════════════════════════════════
# IMPORTS & DEPENDENCIES
# ══════════════════════════════════════════════════════════════════
import os            # For file path and directory operations
import shutil        # For high-level file operations (like deleting folders)
import tempfile      # 📚 CONCEPT: OS-managed temporary workspaces that keep our system clean
import urllib.request # For making HTTP requests to GitHub
import zipfile       # For extracting downloaded repository archives
import re            # For parsing GitHub URLs using Regular Expressions
import time          # For baseline telemetry timing (Day 1 instrumentation)

# Global telemetry cache for last cloned repository metadata
_LAST_REPO_METADATA: dict | None = None

# ══════════════════════════════════════════════════════════════════
# 1. REPOSITORY CLONER
# ══════════════════════════════════════════════════════════════════
def clone_repo(github_url: str, return_metadata: bool = False):
    """Downloads repo as a ZIP and extracts to temp folder, returns path.
    Preserves backward compatibility: returns str by default.
    If return_metadata=True, returns (repo_path, metadata_dict)."""
    global _LAST_REPO_METADATA
    t_total_start = time.perf_counter()
    
    # [1] Parse the URL to get the owner and repo name
    match = re.search(r'github\.com/([^/]+)/([^/]+)', github_url)
    if not match:
        raise ValueError("Invalid GitHub URL format. Use https://github.com/user/repo")
        
    owner, repo = match.groups()
    repo = repo.replace('.git', '')
    
    # 💡 WHY: Instead of running 'git clone', we use GitHub's API to download a ZIP.
    # This is faster and doesn't require Git to be installed on the user's computer!
    zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
    
    # Create a unique temporary directory
    temp_dir = tempfile.mkdtemp(prefix="rag_repo_")
    zip_path = os.path.join(temp_dir, "repo.zip")
    
    try:
        # [2] Download the ZIP file with timing
        t_download_start = time.perf_counter()
        req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        download_duration = time.perf_counter() - t_download_start
            
        # [3] Extract the ZIP with timing
        t_extract_start = time.perf_counter()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        extract_duration = time.perf_counter() - t_extract_start
            
        os.remove(zip_path) # Clean up the zip file itself
        
        # [4] Find the actual repository folder and resolve commit SHA
        # GitHub ZIPs extract into a subfolder with a commit hash (e.g., owner-repo-1a2b3c4). 
        # We return that subfolder path so the file tree looks normal to our pipeline.
        extracted_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
        commit_sha = "unknown"
        if extracted_dirs:
            dir_name = extracted_dirs[0]
            sha_match = re.search(r'-([a-f0-9]{7,40})$', dir_name)
            if sha_match:
                commit_sha = sha_match.group(1)
            final_path = os.path.join(temp_dir, dir_name)
        else:
            final_path = temp_dir
            
        total_duration = time.perf_counter() - t_total_start
        
        metadata = {
            "url": github_url,
            "owner": owner,
            "repo": repo,
            "commit_sha": commit_sha,
            "download_duration_seconds": round(download_duration, 4),
            "extraction_duration_seconds": round(extract_duration, 4),
            "total_loading_duration_seconds": round(total_duration, 4)
        }
        _LAST_REPO_METADATA = metadata
        
        if return_metadata:
            return final_path, metadata
        return final_path
    except Exception as e:
        # If anything fails, clean up the temp folder so we don't leak disk space
        cleanup_repo(temp_dir)
        raise Exception(f"Failed to download repository. Make sure the URL is public. Error: {e}")

def get_last_repo_metadata() -> dict | None:
    """Returns the metadata dictionary captured during the most recent clone_repo call."""
    return _LAST_REPO_METADATA

# ══════════════════════════════════════════════════════════════════
# 2. SOURCE FILE EXTRACTOR
# ══════════════════════════════════════════════════════════════════
def get_source_files(repo_path: str) -> list[dict]:
    """Returns [{file_path, content, extension}] for valid files only.
    Skips: .git, node_modules, __pycache__, venv, files > 100KB, etc."""
    
    # 💡 WHY: LLMs have a limited "context window". Feeding them massive build files,
    # dependencies, or compiled binaries is useless and expensive. (Faltu data skip karna zaroori hai).
    supported_extensions = {".py", ".js", ".java", ".cpp", ".c", ".h", ".ts"}
    ignore_dirs = {".git", "node_modules", "__pycache__", "venv", "env", ".venv", "build", "dist", "target", ".idea", ".vscode"}
    max_file_size = 100 * 1024 # 100 KB limit per file
    
    files_data = []
    
    # 📚 CONCEPT: os.walk is a powerful generator that recursively visits every folder and file in a tree.
    for root, dirs, files in os.walk(repo_path):
        # [1] Optimize traversal
        # Modify dirs in-place to skip ignored directories. This stops os.walk from even entering them!
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        
        for file in files:
            if file.startswith('.'):
                continue
                
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                file_path = os.path.join(root, file)
                
                # [2] Size Check
                if os.path.getsize(file_path) > max_file_size:
                    continue
                    
                try:
                    # [3] Read Content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    rel_path = os.path.relpath(file_path, repo_path).replace("\\", "/")
                    
                    # 🔗 CONNECTS TO: This dictionary format is exactly what `chunker.py` expects next!
                    files_data.append({
                        "file_path": rel_path,
                        "content": content,
                        "extension": ext
                    })
                except UnicodeDecodeError:
                    # Skip binary files or files with weird encodings that accidentally passed the extension check
                    continue
                    
    return files_data

# ══════════════════════════════════════════════════════════════════
# 3. DIRECTORY TREE GENERATOR
# ══════════════════════════════════════════════════════════════════
def generate_directory_tree(repo_path: str) -> str:
    """Returns a string representing the directory tree."""
    # 💡 WHY: The AI needs "architectural context". Seeing a map of folders helps the AI 
    # understand how different parts of the system connect together.
    
    tree_lines = []
    ignore_dirs = {".git", "node_modules", "__pycache__", "venv", "env", ".venv", "build", "dist", "target", ".idea", ".vscode"}
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        
        # Calculate folder depth/level to know how many spaces to indent
        rel_path = os.path.relpath(root, repo_path)
        if rel_path == '.':
            level = 0
            dir_name = os.path.basename(os.path.normpath(repo_path))
            # Clean up the weird github subfolder name (owner-repo-hash) for a cleaner UI
            dir_name = re.sub(r'-[a-f0-9]{7,}$', '', dir_name)
            if not dir_name:
                dir_name = "repo"
        else:
            level = rel_path.count(os.sep) + 1
            dir_name = os.path.basename(root)
            
        indent = '  ' * level
        tree_lines.append(f"{indent}📂 {dir_name}/")
        
        sub_indent = '  ' * (level + 1)
        for f in sorted(files):
            if not f.startswith('.'):
                tree_lines.append(f"{sub_indent}📄 {f}")
                
    return '\n'.join(tree_lines)

# ══════════════════════════════════════════════════════════════════
# 4. RESOURCE CLEANUP
# ══════════════════════════════════════════════════════════════════
def cleanup_repo(repo_path: str) -> None:
    """Deletes the temp folder containing the repo."""
    if repo_path and os.path.exists(repo_path):
        # Since we return a subfolder in clone_repo(), we must go up one level 
        # to delete the actual temporary parent directory.
        parent_dir = os.path.dirname(repo_path)
        if os.path.basename(parent_dir).startswith("rag_repo_"):
            target_dir = parent_dir
        elif os.path.basename(repo_path).startswith("rag_repo_"):
            target_dir = repo_path
        else:
            return # Safety check: Don't accidentally delete something unexpected!

        try:
            # 💡 WHY: Windows OS often marks files inside '.git' folders as Read-Only.
            # A normal shutil.rmtree() crashes on read-only files. 
            # This 'onerror' callback automatically fixes file permissions and retries.
            def onerror(func, path, exc_info):
                import stat
                if not os.access(path, os.W_OK):
                    os.chmod(path, stat.S_IWUSR)
                    func(path)
                else:
                    raise
            shutil.rmtree(target_dir, onerror=onerror)
        except Exception as e:
            print(f"Warning: Failed to cleanup repo {target_dir}: {e}")
