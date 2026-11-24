# Codebase AI Assistant — Evaluation Benchmark Dataset (Day 1 Task 3)

## 1. Dataset Overview and Purpose

This evaluation benchmark provides a manually verified, reproducible test suite for evaluating codebase retrieval-augmented generation (RAG) performance in the **Codebase AI Assistant** project.

It establishes the empirical ground truth needed to benchmark the existing **V0 baseline** (in Day 1 Task 4) and to objectively evaluate subsequent algorithmic improvements in Days 2 through 7:
- **Day 2**: Code-aware chunking (AST-based vs. fixed sliding window)
- **Day 3**: Embedding model evaluation and upgrades (BGE-code, CodeBERT, dense vector search)
- **Day 4**: Hybrid retrieval (combining BM25 lexical search with dense vector embeddings) and Cross-Encoder reranking
- **Day 5**: Multi-turn conversational query rewriting and context compression
- **Day 6**: Answer grounding, citation accuracy, and hallucination evaluation
- **Day 7**: Production optimization, latency tuning, and end-to-end evaluation

---

## 2. Selected Repositories and Pinned Commit Hashes

Three open-source repositories were selected to represent different software ecosystems, paradigms, and complexities:

| Repository | GitHub URL | Primary Language | Pinned Commit SHA | Indexed Files | Key Architectural Characteristics Tested |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`pallets/flask`** | [https://github.com/pallets/flask](https://github.com/pallets/flask) | Python | `49943b5a94413a530761350f04e896f76e8d620e` | 83 | WSGI request context stack, Blueprint registration, decorators (`@app.route`), config inheritance, Werkzeug URL mapping. |
| **`expressjs/express`** | [https://github.com/expressjs/express](https://github.com/expressjs/express) | JavaScript | `43fca52d28a422c8be48fd489cbd4f2d6152bab6` | 142 | Prototypical inheritance, middleware pipelines (`app.use`), event dispatching, lazy router initialization, content negotiation. |
| **`fmtlib/fmt`** | [https://github.com/fmtlib/fmt](https://github.com/fmtlib/fmt) | C++ | `3be6709f575a609620f5dd82a9ab2fa91bef6a5e` | 28 | C++ template metaprogramming, compile-time format string validation (`FMT_COMPILE`), range traits (`is_range`), OS error wrappers. |

### Single Source of Truth
Repository metadata, URLs, and pinned commit SHAs are dynamically loaded from [`repositories.json`](repositories.json). All evaluations and caching scripts use these exact SHAs to ensure 100% reproducibility across environments.

---

## 3. Question Categories & Query Styles

The benchmark consists of **exactly 60 questions** (20 per repository), distributed across six categories and four realistic query styles:

### Category Breakdown
| Category | Description | Count (Per Repo) | Total Count |
| :--- | :--- | :--- | :--- |
| **`function_identification`** | Locating specific functions, classes, methods, or constructor definitions and signatures. | 4 | 12 |
| **`implementation_understanding`** | Understanding specific algorithms, data transformations, serialization, or internal mechanisms. | 4 | 12 |
| **`cross_file_dependencies`** | Explaining architectural linkages and contract flows across multiple modules/files. | 3–4 | 10 |
| **`architecture_flow`** | Tracing end-to-end request dispatch, pipeline lifecycles, and exception processing flows. | 3–4 | 10 |
| **`debugging_configuration`** | Configuration option parsing, cookie security flags, CLI wrappers, or compile-time flags. | 3–4 | 10 |
| **`unanswerable`** | Plausible codebase questions that cannot be answered because the feature is external or absent. | 2 | 6 |
| **Total** | | **20** | **60** |

### Query Style Breakdown
To eliminate retrieval evaluation bias and simulate realistic developer inquiries:
- **`conceptual` (20 questions)**: Formulated in terms of intent, behavior, or high-level architecture without naming internal symbols (e.g., *"How does Express stream static files from disk to the client while handling range headers and transmission errors?"*).
- **`identifier` (22 questions)**: Targeted questions naming explicit functions or classes without revealing file paths (e.g., *"Where is the text_style class defined in fmt and what style attributes does it store?"*).
- **`conversational` (12 questions)**: Natural context-dependent follow-up queries that depend on Turn-1 context (e.g., *"How does that created application set up its default configuration settings like etag, env, and query parser?"*).
- **`unanswerable` (6 questions)**: Inquiries targeting absent or external capabilities (e.g., built-in PostgreSQL pools, JWT middleware, async logging sinks).
- **Zero Leaked File Paths**: 0 out of 60 questions (0.0%) leak internal file paths in the question text.

---

## 4. Multi-Turn Conversational Question Design (Day 5 Alignment)

To evaluate multi-turn conversational query rewriting and context tracking in **Day 5**, the dataset includes **12 two-turn conversation pairs** (4 per repository, 12 follow-up questions total):

- **Turn 1 (Root Question)**: Establishes the initial technical topic.
  - `turn_number`: `1`
  - `previous_context`: `null`
  - `expected_standalone_query`: `null`
- **Turn 2 (Follow-up Question)**: Genuinely context-dependent using conversational anaphora ("it", "that method", "those settings", "that stream") and CANNOT be cleanly retrieved without Turn-1 context.
  - `turn_number`: `2`
  - `previous_context`: A concise summary of the topic established in Turn 1.
  - `expected_standalone_query`: A fully self-contained, disambiguated standalone search query that a query-rewriting model should generate.
  - Strictly different from the `question` string (enforced by validation).

### Examples of Context-Dependent Rewrites:
1. **Flask (`flask_conv_01`)**:
   - *Turn 1 (`flask_001`)*: "Where is the Flask application class defined and what parameters does its \_\_init\_\_ constructor accept?"
   - *Turn 2 (`flask_002`)*: "How does it configure default configuration settings during that initialization process?"
   - *Expected Standalone Query*: "How does the Flask application initialize and configure its default configuration settings during initialization?"
2. **Express (`express_conv_02`)**:
   - *Turn 1 (`express_005`)*: "How does Express register middleware and mount nested sub-applications under a specific route prefix?"
   - *Turn 2 (`express_006`)*: "How does that instance lazily initialize its underlying router with case-sensitive and strict routing settings?"
   - *Expected Standalone Query*: "How does the Express application lazily initialize its router with case-sensitive and strict routing settings?"
3. **fmt (`fmt_conv_01`)**:
   - *Turn 1 (`fmt_001`)*: "Where is the text_style class defined in fmt and what style attributes does it store?"
   - *Turn 2 (`fmt_002`)*: "How does the styled helper function apply that style to an argument value for formatted output?"
   - *Expected Standalone Query*: "How does the styled helper function apply a text_style to an argument value for formatted output in fmt?"

---

## 5. Ground Truth Annotation Schema (Zero Chunk IDs & Multi-File Spans)

Ground truth annotations **NEVER reference chunk IDs** or vector store document IDs. Vector database chunk IDs change dynamically whenever chunk sizes, chunk overlap, or chunking strategies (e.g., character-based vs. AST-based) are modified.

### Multi-File Evidence Spans
Across the 54 answerable questions, **17 questions (31.5%)** contain multi-file evidence ($\ge 2$ evidence spans), covering all 10 `cross_file_dependencies` questions plus multi-module architecture/flow questions. Total evidence spans: **71 verified spans**.

### Example Multi-File Ground Truth (`express_009`):
```json
{
  "question_id": "express_009",
  "repository": "expressjs/express",
  "answerable": true,
  "evidence": [
    {
      "file_path": "lib/response.js",
      "symbol": "res.send",
      "start_line": 126,
      "end_line": 190,
      "relevance_reason": "Shows response body conversion, Content-Type setting, and calling app.get('etag fn') to generate ETag headers."
    },
    {
      "file_path": "lib/utils.js",
      "symbol": "exports.compileETag",
      "start_line": 130,
      "end_line": 152,
      "relevance_reason": "Compiles the etag setting to an etag generator function (weak or strong)."
    }
  ]
}
```

### Unanswerable Question Schema (`answerable: false`):
```json
{
  "question_id": "flask_019",
  "repository": "pallets/flask",
  "answerable": false,
  "evidence": [],
  "justification": "Flask does not include a built-in database connection pool or ORM. Database connectivity is handled by third-party extensions such as Flask-SQLAlchemy or external drivers (e.g. psycopg2)."
}
```

---

## 6. Practical Relevance Proxy Rule

When evaluating retrieval engines (e.g. calculating **Hit Rate@K**, **MRR@K**, or **Precision@K**), a retrieved chunk is defined as **Relevant** to a question if:

$$\text{chunk.file\_path} == \text{evidence.file\_path} \quad \land \quad \max(\text{chunk.start\_line}, \text{evidence.start\_line}) \le \min(\text{chunk.end\_line}, \text{evidence.end\_line})$$

### Explanation of the Rule:
1. **File Matching**: The retrieved chunk must come from the exact source file cited in the ground truth evidence.
2. **Interval Overlap**: A 1D line range $[s_1, e_1]$ overlaps with $[s_2, e_2]$ if and only if $\max(s_1, s_2) \le \min(e_1, e_2)$. If this condition holds, the chunk contains at least one line of the annotated evidence range.
3. **Multi-evidence Support**: If a question has multiple evidence items (e.g. cross-file questions), retrieving a chunk overlapping *any* valid evidence item constitutes a hit for that item.
4. **Unanswerable Questions**: For `answerable: false`, any retrieval is considered non-relevant, and generation evaluation evaluates whether the LLM correctly recognizes the feature as absent rather than hallucinating code.

---

## 7. Known Limitations and Scope Boundaries

1. **Java Language Omission**:
   - `google/guava` was originally evaluated as a Java candidate. However, `guava` contains **3,255 source files** and an extracted footprint exceeding 30 MB. Indexing Guava on a standard single-node CPU environment with ChromaDB takes over 30 minutes, exceeding reasonable CI/evaluation runtimes and risking memory exhaustion.
   - Therefore, Python (`pallets/flask`), JavaScript (`expressjs/express`), and C++ (`fmtlib/fmt`) were chosen to provide rich language and paradigm coverage. Java coverage is omitted from this initial benchmark suite to keep benchmark runs fast and responsive.
2. **Repository Loader Extension Constraints (`fmtlib/fmt`)**:
   - The current `repo_loader.py` recognizes `{".py", ".js", ".java", ".cpp", ".c", ".h", ".ts"}` and enforces a strict `100 KB` file size limit (`max_file_size = 100 * 1024`).
   - In `fmtlib/fmt`, core header files `format.h` (214 KB), `format-inl.h` (106 KB), and `core.h` (103 KB) exceed the 100 KB ceiling, and `.cc` implementation files are skipped by the loader's extension whitelist.
   - Consequently, the 28 files indexed by the current loader for `fmtlib/fmt` are modular headers (`color.h`, `ranges.h`, `chrono.h`, `os.h`, `compile.h`, `args.h`, `printf.h`, `xchar.h`, `enum.h`, etc.). 100% of ground truth files pass `repo_loader.get_source_files()` verification.
3. **Subjective Line Boundaries**:
   - Functions and classes may include extensive docstrings or comments. Ground truth line ranges span the function declaration, parameter definitions, and core implementation body. Minor deviations in line spans across different tool formatters are absorbed by the interval overlap relevance proxy rule.

---

## 8. Hardened Validation Suite

The dataset is validated by running:
```bash
python experiments/datasets/validate_dataset.py
```
Enforced checks:
1. **Single Source of Truth**: Repository URLs and SHAs loaded dynamically from `repositories.json`.
2. **Loader Indexability Check**: Imports `repo_loader.get_source_files()` to ensure 100% of annotated files pass the extension whitelist and 100 KB ceiling.
3. **Multi-Turn Strictness**: Fails if any Turn-2 question is identical to its expected standalone query.
4. **Multi-File Reporting**: Tracks total evidence spans (71) and multi-evidence questions (17).
5. **Path Leak Detection**: Tracks questions leaking file paths (0% detected).
6. **Ground Truth Integrity**: Zero chunk IDs, verified line ranges, and verified symbol presence in snapshots.
7. Exits with status `0` on success and non-zero on failure.
