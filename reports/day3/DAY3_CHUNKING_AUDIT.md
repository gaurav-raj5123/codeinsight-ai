# DAY 3 — INITIAL CHUNKING AUDIT (`DAY3_CHUNKING_AUDIT.md`)

**Project:** CodeInsight AI Assistant (RAG)  
**Audit Date:** November 2026  
**Scope:** Pre-modification technical audit of `chunker.py`, `repo_loader.py`, `embedder.py`, callers (`app.py`, `llm.py`), actual MiniLM-L6 tokenizer behavior, and the pinned 3-repository corpus (`pallets/flask`, `expressjs/express`, `fmtlib/fmt`).  
**Baseline State Confirmed:** Git working tree clean at commit `7353477` prior to Day 3 changes.

---

## 1. Existing Architecture & Caller Contracts

```text
repo_loader.get_source_files() -> [{"file_path", "content", "extension"}]
   -> chunker.chunk_code(file_data) -> [{"content": str, "metadata": {...}}]
   -> embedder.index_chunks(chunks, repo_name) -> ChromaDB (ONNX all-MiniLM-L6-v2, L2, max_tokens=256)
   -> embedder.search_similar(query, repo_name) -> Top-5 [{"content", "metadata", "id", "distance"}]
   -> llm.build_prompt() / app.py UI citations (consumes metadata['file_path'], metadata['start_line'], metadata['end_line'])
```

- **Downstream Consumer Requirements (`app.py`, `llm.py`, `embedder.py`, `evaluation/`)**:
  - `embedder.py` passes `chunk["content"]` directly as the ChromaDB document text to be embedded and `chunk["metadata"]` directly into `collection.add(metadatas=...)` (ChromaDB requires flat primitive values: `str | int | float | bool`, no `None` values).
  - `llm.py` (`build_prompt`) and `app.py` (Streamlit chunk expanders) depend strictly on `metadata["file_path"]`, `metadata["start_line"]`, and `metadata["end_line"]` for syntax highlighting and source line citations.
  - `evaluation/retrieval_metrics.py` matches retrieved chunks to ground-truth evidence spans using `(file_path == span.file_path)` and interval overlap `max(start_line, span.start) <= min(end_line, span.end)`. Accurate 1-based source line mappings are therefore non-negotiable.

---

## 2. Actual Chunking Behavior by Language (`chunker.py` V0)

| Item Investigated | Observed V0 Behavior |
| :--- | :--- |
| **1. Supported languages** | `.py` -> `chunk_python`; `.js`, `.ts` -> `chunk_javascript`; `.java` -> `chunk_java`; `.cpp`, `.c`, `.h` -> `chunk_cpp`; all others -> `chunk_by_sliding_window`. |
| **2. Python AST implementation** | Uses `ast.parse(content)` and traverses **all** descendants via `ast.walk(tree)`, extracting only `ast.FunctionDef`, `ast.AsyncFunctionDef`, and `ast.ClassDef`. Falls back to regex `[r'^class \w+', r'^def \w+', r'^\s+def \w+']` on `SyntaxError`. |
| **3. Regex-based parsing (`.js`, `.ts`, `.java`, `.c`, `.cpp`, `.h`)** | Single-pass line scanner (`chunk_by_regex`) emitting a new chunk whenever a line matches any regex via `re.search` (when buffer is non-empty). Never tracks brace depth (`{`/`}`), multi-line signatures, comments, or string literals. |
| **4. Fallback chunking** | `chunk_by_sliding_window(window_size=50, overlap=10)` activates ONLY if `chunks` is empty OR if a chunk has `>100` lines (which re-splits that chunk into 50-line overlapping windows labeled `<type>_split`). |
| **5. Chunk boundaries** | Python uses `node.decorator_list[0].lineno` through `node.end_lineno`. Regex languages cut purely at line-start pattern hits regardless of balanced blocks. |
| **6. Nested functions & methods** | `ast.walk(tree)` emits every method and nested inner function separately **while also keeping it inside the enclosing `ClassDef` or outer `FunctionDef` chunk**. Regex JS/C++ splits nested callbacks indiscriminately. |
| **7. Duplicate source inclusion** | **436** Python chunks (`389` in Flask, `47` in fmt utility `.py`) are completely contained within parent `ClassDef`/`FunctionDef` chunks in the same file. |
| **8. Class context** | Standalone Python `FunctionDef` method chunks **lose enclosing `class Foo:` context completely**. `ClassDef_split` sub-chunks #2+ lose the class header and docstring. |
| **9. Function signatures** | Long functions/methods split past line 100 lose their `def` signature on sub-chunks #2+. JS arrow functions / prototype assignments (`app.use = ...`, `res.send = ...`) without `function` keywords fail `^function \w+` unless indented. |
| **10. Imports** | Module-level `import` / `from ... import` lines in Python are **silently discarded** whenever the file has at least one `def`/`class`. |
| **11. Module-level constants & globals** | Python module-level globals/Singletons/proxies outside classes/functions are **silently discarded** in files that contain any `def`/`class`. Example: `src/flask/globals.py` loses lines 32–63 (`request`, `current_app`, `session`, `g` `LocalProxy` definitions). |
| **12. Docstrings** | Included inside top-level `ClassDef`/`FunctionDef` chunks, but often push small 6-line function signatures >256 tokens (e.g., `class Flask` in `src/flask/app.py:108-157` = **570 tokens**, truncating constructor details). |
| **13. Metadata** | Only 4 fields populated: `file_path`, `start_line`, `end_line`, `chunk_type`. Missing: `repository`, `language`, `symbol_name`, `class_name`, `parent_symbol`, `token_count`, `chunking_strategy`, `parser_type`, `chunk_id`. |
| **14. Source-line mapping** | Python AST line mappings are accurate (`0` out-of-bounds ranges across 4,272 chunks). In `chunk_code`, `chunk['metadata']['start_line'] + sc['metadata']['start_line'] - 1` maps split lines correctly. |
| **15–22. Coverage & quality stats** | Detailed below across all 253 files / 4,272 chunks using the exact Chroma ONNX `all-MiniLM-L6-v2` tokenizer. |
| **23. Language-specific limitations** | **JS**: `r'^\s+\w+\s*\(.*\)\s*\{'` matches `if (...) {`, `for (...) {`, `while (...) {`, `describe(..., function () {`, causing **164** control-flow splits (`7.9%` of Express). Prototype methods (`proto.handle = ...`) are not recognized directly.<br>**C/C++**: `r'^\w+.*\w+\s*\(.*\)\s*\{'` misses indented methods, templates (`template <...>`), and macros (`FMT_API`), leaving 60-line license headers as standalone chunks (`fmt/ranges.h:1-62`) and emitting massive dense chunks (`>256` tokens).<br>**Python**: Drops module-level code in **66/83** Flask files and duplicates **389** nested chunks. |
| **24. Existing test coverage** | `tests/test_baseline_instrumentation.py` checks only a trivial 6-line Python snippet (`test_chunker_unmodified`). No tests exist for JS, C++, Java, truncation, token budgets, nested duplication, empty files, oversized functions, or module-level globals. |

---

## 3. Source-File Coverage & Corpus Composition (`repo_loader.get_source_files`)

All counts measured against the verified Day 1/Day 2 pinned snapshots (`repositories.json`):

| Repository | Supported Files Extracted | Skipped (`>100 KB`) | Non-Supported Extensions Excluded | Total Chunks (V0) | Core Source vs Test/Example/Docs Chunks |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **`pallets/flask`** | **83** (all `.py`) | `0` | `.rst` (79), `.html` (20), `.txt` (10), `.toml` (5), etc. | **1,785** | Core `src/`: **592** (33.2%) / `tests/, docs/`: **1,193** (66.8%) |
| **`expressjs/express`** | **142** (all `.js`) | `0` | `.ejs` (21), `.txt` (11), `.html` (8), `.tmpl` (7), etc. | **2,066** | Core `lib/`: **133** (6.4%) / `test/, examples/`: **1,933** (93.6%) |
| **`fmtlib/fmt`** | **28** (`.h`/`.c`/`.py`) | **5** (`core.h`, `format-inl.h`, `format.h`, `gmock.h`, `gtest.h`) | `.cc` (48), `.txt` (12), `.md` (9), `.in` (4), etc. | **421** | Core `include/`: **249** (59.1%) / `test/, support/`: **172** (40.9%) |
| **Total Corpus** | **253** | **5** | — | **4,272** | — |

### Line-Level Coverage by Repository (`V0`)
- **`pallets/flask`**: `14,119` non-empty lines -> `13,257` covered (**`93.89%`**); **66 / 83 files (`79.5%`) have uncovered non-empty lines** (imports, constants, `LocalProxy` bindings, `cli.py`/`app.py` module code).
- **`expressjs/express`**: `18,004` non-empty lines -> `18,004` covered (**`100.00%`**) because `chunk_by_regex` sweeps all lines from line 1 to EOF.
- **`fmtlib/fmt`**: `9,473` non-empty lines -> `9,282` covered (**`97.98%`**); **5 / 28 files** have uncovered non-empty lines (the 5 Python utility scripts inside `fmtlib/fmt`, e.g. `support/release.py` drops 119 of 192 non-empty lines!).

---

## 4. Quantitative Chunk & Tokenizer Audit (`all-MiniLM-L6-v2` Tokenizer)

Measured with the actual ChromaDB ONNX tokenizer (`~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx/tokenizer.json`, untruncated & unpadded, inclusive of `[CLS]` and `[SEP]`):

### Overall Statistics (`N = 4,272` V0 Chunks)
- **Token Count**: `min = 2`, `median = 84.0`, `mean = 139.34`, `p95 = 525`, `max = 1,844`
- **Line Count**: `min = 1`, `median = 8.0`, `mean = 14.06`, `p95 = 50`, `max = 100`
- **Chunk Type Counts**: `regex_split`: 2,234; `FunctionDef`: 1,528; `ClassDef`: 165; `ClassDef_split`: 147; `regex_split_split`: 136; `AsyncFunctionDef`: 18; `FunctionDef_split`: 26; `sliding_window`: 18
- **Empty Chunks** (`strip() == ""`): **3** (`0.07%`) — empty `__init__.py` files
- **1-Line Chunks**: **473 (`11.07%`)**
- **`< 5` Line Chunks**: **1,404 (`32.87%`)**
- **Tiny Chunks (`< 15` tokens)**: **235 (`5.50%`)**
- **Truncated Chunks (`> 256` tokens, MiniLM-L6 ceiling)**: **624 (`14.61%`)**
- **Severe Oversized Chunks (`> 512` tokens)**: **233 (`5.45%`)**
- **Invalid Line Ranges**: `0`
- **Exact Content Duplicates**: **255 excess chunks across 126 duplicate hash groups (`5.97%`)**
- **Python Nested AST Overlap Duplicates**: **436 chunks (`22.9%` of Python chunks)**

### Language-Wise Token Distribution
| Extension | Chunks | Min | Median | Mean | P95 | Max | Truncated (`>256` tok) | Tiny (`<15` tok) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`.h` (C/C++)** | 296 | 15 | 406.0 | 416.6 | 906 | 1,524 | **185 (`62.5%`)** | 0 (`0.0%`) |
| **`.py` (Python)** | 1,902 | 2 | 76.0 | 148.7 | 520 | 1,844 | **369 (`19.4%`)** | 51 (`2.7%`) |
| **`.c` (C)** | 7 | 50 | 136.0 | 170.4 | 440 | 440 | **2 (`28.6%`)** | 0 (`0.0%`) |
| **`.js` (JavaScript)** | 2,067 | 5 | 79.0 | 90.9 | 229 | 771 | **68 (`3.3%`)** | **184 (`8.9%`)** |

### Ground-Truth Truncation Risk
Checking all **71 verified ground-truth evidence spans** across the 54 answerable benchmark questions revealed that **41 out of 71 spans (`57.7%`)** are ONLY covered by V0 chunks exceeding `256` tokens — meaning `all-MiniLM-L6-v2` silently drops the tail of the chunk containing or surrounding critical evidence (e.g., `flask_001` `570` tokens, `fmt_006` `920` tokens, `fmt_010` `880` tokens, `fmt_012` `591–1,007` tokens).

---

## 5. Concrete Examples of Problematic V0 Chunks

1. **Empty Chunks (`2` tokens `[CLS] [SEP]`)**:
   - `pallets/flask`: `tests/test_apps/blueprintapp/apps/__init__.py:1-1` (`sliding_window`, `''`)
   - `pallets/flask`: `tests/test_apps/cliapp/__init__.py:1-1` (`sliding_window`, `''`)
   - `pallets/flask`: `tests/test_apps/cliapp/inner1/inner2/__init__.py:1-1` (`sliding_window`, `''`)
2. **JS Control-Flow Regex False Splits (Single-Line Fragments)**:
   - `expressjs/express`: `test/res.format.js:182-182` (`regex_split`, `8` tokens): `'function test(app) {'`
   - `expressjs/express`: `examples/route-map/index.js:16-16` (`regex_split`, `10` tokens): `'  for (var key in a) {'`
   - `expressjs/express`: `lib/response.js:166-166` (`regex_split`, `12` tokens): `'  if (chunk !== undefined) {'`
   - `expressjs/express`: `test/express.json.js:197` (`regex_split`): `"    describe('when false', function () {"` (duplicated **11 times** across the repo!).
3. **Dropped Module-Level Python Globals**:
   - `pallets/flask`: `src/flask/globals.py` (`77` lines total, `58` non-empty). Only `18` non-empty lines (`_FakeStack` type-check class + `__getattr__`) are indexed; lines `32–63` (`request`, `current_app`, `g`, `session` `LocalProxy` definitions) are **completely absent** from ChromaDB.
4. **Severe C/C++ Token Overflow Despite `<=100` Line Limit**:
   - `fmtlib/fmt`: `include/fmt/chrono.h:892-983` (`92` lines = **1,524 tokens**; `1,268` tokens truncated!)
   - `fmtlib/fmt`: `include/fmt/compile.h:457-556` (`100` lines = **1,498 tokens**; `1,242` tokens truncated!)
   - `fmtlib/fmt`: `include/fmt/fmt-c.h:152-201` (`50` lines `regex_split_split` = **1,484 tokens** — even V0's 50-line split is 5.8x too large for MiniLM-L6!).
5. **C++ Preamble Splitting Without Context**:
   - `fmtlib/fmt`: `include/fmt/ranges.h:1-62` (MIT license block & `#include` preamble) retrieved Rank #1 across all Day 2 models on `fmt_005`, while `struct is_range` (`370-410`) stays buried inside oversized template blocks.

---

## 6. Proposed `code_aware_v1` Improvements, Risks & Tradeoffs

### Proposed V1 Design
1. **Strict Tokenizer-Verified 256-Token Budget (Zero Silent Truncation)**:
   - Measure token length directly with the active `all-MiniLM-L6-v2` `Tokenizer` **including** the contextual metadata prefix (`File:`, `Language:`, `Symbol:`, `Class:`, `Type:`).
   - Target `120–220` tokens per chunk and guarantee final embedded strings `<= 256` tokens (`<= 254` content+prefix tokens + `[CLS]/[SEP]`).
2. **Eliminate Empty & Micro-Fragment Chunks**:
   - Drop whitespace-only chunks (`not content.strip()`).
   - Fix JS/TS/Java/C/C++ structural heuristics: reject control-flow keywords (`if`, `for`, `while`, `switch`, `catch`, `else`) as function boundaries, detect JS `proto.foo = function`, `exports.foo =`, `defineGetter(...)`, `function foo(`, C++ `template`/`namespace`/`class`/`struct`/signature blocks cleanly, and merge consecutive tiny declarations/snippets (`< 40` tokens) within logical file windows so single-line stubs like `if (x) {` never become standalone vectors.
3. **Clean Non-Redundant Python AST Extraction + Module-Level Gap Coverage**:
   - Walk top-level AST nodes recursively *without* emitting giant duplicate full-class text alongside every method:
     - Emit a compact `ClassDef` **header/overview chunk** (signature, bases, class docstring, class attributes, method outline) plus individual `MethodDef` chunks carrying explicit `Class:` + `Parent:` context.
     - Capture uncovered module-level spans (imports, constants, `LocalProxy` bindings, top-level statements) into token-budgeted `module_context` chunks so **100% of non-empty Python code lines are indexed**.
     - Split oversized functions/methods at logical AST block/statement boundaries first (fallback to token-bounded line splits) while **re-attaching `File`, `Class`, and `Function` signature context headers to every continuation chunk** (`is_split = True`).
4. **Richer Chroma-Compatible Metadata & Deterministic Chunk IDs**:
   - Include `repository`, `file_path`, `language`, `chunk_type`, `symbol_name`, `class_name`, `parent_symbol`, `start_line`, `end_line`, `chunking_strategy`, `parser_type`, `token_count`, `content_sha256`, and deterministic `chunk_id`.

### Risks & Tradeoffs
- **Header Prefix Overhead**: Adding compact structural headers consumes ~15–25 tokens per chunk out of MiniLM-L6's 256-token budget, slightly increasing total chunk count on dense C++ files. Headers must be kept concise (`3–4` short key-value lines).
- **Eliminating Python `ast.walk` Redundancy**: V0's redundant emission of both `ClassDef_split` and `FunctionDef` gave multi-evidence queries multiple chances to match in Top-5; splitting large classes cleanly into `ClassSummary` + `Methods` avoids duplicate bloat (`436` nested dupes) and raises precision, which must be measured empirically against V0 (`C1`) and `fixed_window_v0` (`C0`).
- **Regex/Heuristic Bounds for JS/Java/C++**: Without full Tree-sitter grammars installed, non-Python structural parsing uses brace/keyword-aware boundary scanning plus token-aware splitting (`parser_type` recorded honestly in metadata so no fragile parsing is misrepresented as a full AST).
