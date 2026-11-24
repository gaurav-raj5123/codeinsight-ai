import os
import sys
import shutil
import urllib.request
import zipfile
import json

# Add project root to sys.path so repo_loader can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from repo_loader import get_source_files

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

REPOS = {
    "flask": {
        "url": "https://github.com/pallets/flask/archive/49943b5a94413a530761350f04e896f76e8d620e.zip",
        "sha": "49943b5a94413a530761350f04e896f76e8d620e",
        "dir": "flask"
    },
    "express": {
        "url": "https://github.com/expressjs/express/archive/43fca52d28a422c8be48fd489cbd4f2d6152bab6.zip",
        "sha": "43fca52d28a422c8be48fd489cbd4f2d6152bab6",
        "dir": "express"
    },
    "fmt": {
        "url": "https://github.com/fmtlib/fmt/archive/3be6709f575a609620f5dd82a9ab2fa91bef6a5e.zip",
        "sha": "3be6709f575a609620f5dd82a9ab2fa91bef6a5e",
        "dir": "fmt"
    }
}

def download_and_extract(name, info):
    target_dir = os.path.join(CACHE_DIR, info["dir"])
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"[{name}] Already cached at {target_dir}")
        return target_dir

    print(f"[{name}] Downloading from {info['url']}...")
    zip_path = os.path.join(CACHE_DIR, f"{name}.zip")
    req = urllib.request.Request(info["url"], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f:
        shutil.copyfileobj(resp, f)

    print(f"[{name}] Extracting...")
    extract_temp = os.path.join(CACHE_DIR, f"temp_{name}")
    os.makedirs(extract_temp, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_temp)

    # Move extracted inner folder to target_dir
    inner = [os.path.join(extract_temp, d) for d in os.listdir(extract_temp) if os.path.isdir(os.path.join(extract_temp, d))][0]
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(inner, target_dir)
    shutil.rmtree(extract_temp)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    print(f"[{name}] Extracted to {target_dir}")
    return target_dir

def inspect_repos():
    summary = {}
    for name, info in REPOS.items():
        repo_dir = download_and_extract(name, info)
        source_files = get_source_files(repo_dir)
        summary[name] = {
            "source_files_count": len(source_files),
            "extensions": {},
            "files": []
        }
        for sf in source_files:
            ext = sf["extension"]
            summary[name]["extensions"][ext] = summary[name]["extensions"].get(ext, 0) + 1
            line_count = len(sf["content"].splitlines())
            summary[name]["files"].append({
                "path": sf["file_path"],
                "lines": line_count
            })
        print(f"[{name}] Processed {len(source_files)} valid source files. Extensions: {summary[name]['extensions']}")
    
    with open(os.path.join(CACHE_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Inspection complete. Manifest written to cache/manifest.json")

if __name__ == "__main__":
    inspect_repos()
