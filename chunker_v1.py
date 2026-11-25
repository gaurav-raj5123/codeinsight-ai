"""
Code-Aware Token-Budgeted Chunker (V1 — code_aware_v1).

Designed for dense code retrieval under a strict 256-token embedding budget
(ChromaDB ONNX all-MiniLM-L6-v2, inclusive of [CLS]/[SEP] special tokens).

Key improvements over V0:
- Python: hierarchical AST visitor (no ast.walk duplicate method+class inflation),
  class overview + method chunks with class context, nested helper awareness, and
  100% coverage of module-level imports/constants/globals (e.g., LocalProxy bindings).
- JS / TS / Java / C / C++: stateful comment/string/brace-depth lexical scanner
  (filtering out control-flow false triggers like if/for/while/switch/catch) with
  template/decorator lookback and documented fallback metadata.
- Token-aware splitting: measures exact WordPiece tokens (header + source) via the
  active embedding tokenizer so zero chunks exceed 256 tokens.
- Zero empty chunks, elimination of 1-line fragments via safe small-block merging,
  deterministic chunk IDs, content SHA256 hashes, and Chroma-compatible metadata.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import warnings
from typing import Any

MAX_MODEL_TOKENS = 256
TARGET_CHUNK_TOKENS = 200
TINY_MERGE_THRESHOLD_TOKENS = 45
SPLIT_LINE_OVERLAP = 2

EXT_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".h": "cpp",
}

CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "catch",
        "try",
        "finally",
        "with",
        "return",
        "throw",
        "sizeof",
        "typeof",
        "delete",
        "new",
        "await",
    }
)

_TOKENIZER_CACHE: Any = None
_TOKENIZER_MODE: str = "uninitialized"


def get_embedding_tokenizer() -> tuple[Any, str]:
    """Load the unpadded, untruncated MiniLM-L6 WordPiece tokenizer singleton."""
    global _TOKENIZER_CACHE, _TOKENIZER_MODE
    if _TOKENIZER_CACHE is not None:
        return _TOKENIZER_CACHE, _TOKENIZER_MODE

    from tokenizers import Tokenizer

    chroma_tok_path = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "chroma",
        "onnx_models",
        "all-MiniLM-L6-v2",
        "onnx",
        "tokenizer.json",
    )
    if os.path.exists(chroma_tok_path):
        tok = Tokenizer.from_file(chroma_tok_path)
        tok.no_truncation()
        tok.no_padding()
        _TOKENIZER_CACHE = tok
        _TOKENIZER_MODE = "chroma_onnx_all_minilm_l6_v2"
        return _TOKENIZER_CACHE, _TOKENIZER_MODE

    try:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

        loader = ONNXMiniLM_L6_V2()
        loader._download_model_if_not_exists()
        tok = Tokenizer.from_file(chroma_tok_path)
        tok.no_truncation()
        tok.no_padding()
        _TOKENIZER_CACHE = tok
        _TOKENIZER_MODE = "chroma_onnx_all_minilm_l6_v2"
        return _TOKENIZER_CACHE, _TOKENIZER_MODE
    except Exception:
        pass

    try:
        from huggingface_hub import hf_hub_download

        hf_tok_path = hf_hub_download("sentence-transformers/all-MiniLM-L6-v2", "tokenizer.json")
        tok = Tokenizer.from_file(hf_tok_path)
        tok.no_truncation()
        tok.no_padding()
        _TOKENIZER_CACHE = tok
        _TOKENIZER_MODE = "hf_all_minilm_l6_v2"
        return _TOKENIZER_CACHE, _TOKENIZER_MODE
    except Exception:
        _TOKENIZER_CACHE = None
        _TOKENIZER_MODE = "char_fallback"
        return None, _TOKENIZER_MODE


def count_tokens(text: str, tokenizer: Any = None) -> int:
    """Count WordPiece tokens (inclusive of [CLS]/[SEP]) matching ONNX MiniLM-L6."""
    if not text:
        return 2
    tok = tokenizer
    if tok is None:
        tok, _ = get_embedding_tokenizer()
    if tok is not None:
        return len(tok.encode(text).ids)
    # Conservative fallback (~2.8 chars per code token + 2 special tokens)
    return max(2, (len(text) + 2) // 3 + 2)


def detect_language(file_path: str, extension: str = "") -> str:
    ext = (extension or os.path.splitext(file_path)[1]).lower()
    return EXT_TO_LANGUAGE.get(ext, "text")


def format_context_header(
    file_path: str,
    language: str,
    chunk_type: str,
    symbol_name: str = "",
    class_name: str = "",
    parent_symbol: str = "",
    signature: str = "",
    part_index: int = 1,
    total_parts: int = 1,
) -> str:
    """Format a compact contextual header that stays within a minimal token footprint."""
    lines = [f"File: {file_path} | Language: {language} | Type: {chunk_type}"]
    details: list[str] = []
    if class_name and class_name != symbol_name:
        details.append(f"Class: {class_name}")
    elif parent_symbol and parent_symbol != symbol_name:
        details.append(f"Parent: {parent_symbol}")
    if symbol_name:
        details.append(f"Symbol: {symbol_name}")
    if total_parts > 1:
        details.append(f"Part: {part_index}/{total_parts}")
    if details:
        lines.append(" | ".join(details))
    if signature and part_index > 1:
        clean_sig = re.sub(r"\s+", " ", signature).strip()
        if len(clean_sig) > 120:
            clean_sig = clean_sig[:117] + "..."
        lines.append(f"Signature: {clean_sig}")
    return "\n".join(lines)


def make_chunk_id(
    repository: str,
    file_path: str,
    symbol_name: str,
    start_line: int,
    end_line: int,
    content_sha256: str,
    part_index: int = 1,
) -> str:
    """Create a deterministic chunk identifier grounded in location and content hash."""
    raw = f"{repository}|{file_path}|{symbol_name}|{start_line}-{end_line}|p{part_index}|{content_sha256[:12]}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"v1_{digest}"


def _split_long_single_line(
    line_text: str,
    line_num: int,
    header_tokens_est: int,
    max_tokens: int,
    tokenizer: Any,
) -> list[tuple[int, int, str]]:
    """Hard safety fallback when a single physical line exceeds the token budget."""
    budget = max(32, max_tokens - header_tokens_est - 6)
    words = re.findall(r"\S+\s*", line_text)
    if not words:
        return [(line_num, line_num, line_text)]

    pieces: list[tuple[int, int, str]] = []
    cur: list[str] = []
    for w in words:
        candidate = "".join(cur) + w
        if cur and count_tokens(candidate, tokenizer) > budget:
            pieces.append((line_num, line_num, "".join(cur).rstrip()))
            cur = [w]
        elif not cur and count_tokens(w, tokenizer) > budget:
            # Extremely long contiguous token/string with no whitespace
            step = max(40, budget * 2)
            for idx in range(0, len(w), step):
                sub = w[idx : idx + step].rstrip()
                if sub:
                    pieces.append((line_num, line_num, sub))
            cur = []
        else:
            cur.append(w)
    if cur and "".join(cur).strip():
        pieces.append((line_num, line_num, "".join(cur).rstrip()))
    return pieces or [(line_num, line_num, line_text[:200])]


def _emit_token_bounded_unit(
    lines: list[str],
    start_line: int,
    end_line: int,
    file_path: str,
    language: str,
    chunk_type: str,
    parser_type: str,
    symbol_name: str = "",
    class_name: str = "",
    parent_symbol: str = "",
    signature: str = "",
    repository: str = "",
    strategy: str = "code_aware_v1",
    preferred_breaks: set[int] | None = None,
    max_tokens: int = MAX_MODEL_TOKENS,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Emit 1 or more strictly token-bounded chunks from a contiguous source line range."""
    raw_slice = lines[start_line - 1 : end_line]
    raw_text = "\n".join(raw_slice)
    if not raw_text.strip():
        return []

    base_header = format_context_header(
        file_path=file_path,
        language=language,
        chunk_type=chunk_type,
        symbol_name=symbol_name,
        class_name=class_name,
        parent_symbol=parent_symbol,
        signature=signature,
        part_index=1,
        total_parts=1,
    )
    single_full = f"{base_header}\n{raw_text}"
    single_tok = count_tokens(single_full, tokenizer)

    if single_tok <= max_tokens:
        sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        return [
            {
                "content": single_full,
                "metadata": {
                    "repository": repository,
                    "file_path": file_path,
                    "language": language,
                    "chunk_type": chunk_type,
                    "symbol_name": symbol_name,
                    "class_name": class_name,
                    "parent_symbol": parent_symbol,
                    "start_line": start_line,
                    "end_line": end_line,
                    "chunking_strategy": strategy,
                    "parser_type": parser_type,
                    "fallback_used": parser_type in ("python_regex_fallback", "token_window_fallback"),
                    "token_count": single_tok,
                    "content_sha256": sha,
                    "chunk_id": make_chunk_id(
                        repository, file_path, symbol_name, start_line, end_line, sha, 1
                    ),
                    "is_split": False,
                    "part_index": 1,
                    "total_parts": 1,
                },
            }
        ]

    # Split oversized unit cleanly along preferred boundaries, respecting token budget
    split_header_sample = format_context_header(
        file_path=file_path,
        language=language,
        chunk_type=f"{chunk_type}_split",
        symbol_name=symbol_name,
        class_name=class_name,
        parent_symbol=parent_symbol,
        signature=signature,
        part_index=2,
        total_parts=99,
    )
    header_budget = count_tokens(split_header_sample + "\n", tokenizer) - 2

    parts_raw: list[tuple[int, int, str]] = []
    cur_idx = 0
    total_unit_lines = len(raw_slice)
    pref_set = preferred_breaks or set()

    while cur_idx < total_unit_lines:
        cur_start_ln = start_line + cur_idx
        # First check single line vs budget
        first_line = raw_slice[cur_idx]
        probe_header = format_context_header(
            file_path=file_path,
            language=language,
            chunk_type=f"{chunk_type}_split",
            symbol_name=symbol_name,
            class_name=class_name,
            parent_symbol=parent_symbol,
            signature=signature,
            part_index=len(parts_raw) + 1,
            total_parts=10,
        )
        if count_tokens(f"{probe_header}\n{first_line}", tokenizer) > max_tokens:
            sub_pieces = _split_long_single_line(
                first_line, cur_start_ln, header_budget, max_tokens, tokenizer
            )
            parts_raw.extend(sub_pieces)
            cur_idx += 1
            continue

        # Fast binary search + structural-break scan within the hard token ceiling
        lo, hi = cur_idx, total_unit_lines - 1
        best_end_idx = cur_idx
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate_text = "\n".join(raw_slice[cur_idx : mid + 1])
            toks = count_tokens(f"{probe_header}\n{candidate_text}", tokenizer)
            if toks <= max_tokens:
                best_end_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1

        # Prefer natural structural break near target size unless the slice reaches unit end
        if best_end_idx < total_unit_lines - 1:
            last_good_break_idx: int | None = None
            last_good_soft_idx: int | None = None
            min_accept_tok = min(target_tokens, max_tokens - 60)
            for b_idx in range(best_end_idx, cur_idx - 1, -1):
                abs_ln = start_line + b_idx
                line_stripped = raw_slice[b_idx].strip()
                is_pref = abs_ln in pref_set
                is_soft = line_stripped in ("", "}", "};", "),") or line_stripped.endswith("}")
                if not is_pref and not is_soft:
                    continue
                cand_tok = count_tokens(
                    f"{probe_header}\n" + "\n".join(raw_slice[cur_idx : b_idx + 1]),
                    tokenizer,
                )
                if cand_tok < min_accept_tok:
                    break
                if is_pref and last_good_break_idx is None:
                    last_good_break_idx = b_idx
                    break
                if is_soft and last_good_soft_idx is None:
                    last_good_soft_idx = b_idx

            if last_good_break_idx is not None:
                chosen_end_idx = last_good_break_idx
            elif last_good_soft_idx is not None:
                chosen_end_idx = last_good_soft_idx
            else:
                chosen_end_idx = best_end_idx
        else:
            chosen_end_idx = best_end_idx

        slice_text = "\n".join(raw_slice[cur_idx : chosen_end_idx + 1])
        if slice_text.strip():
            parts_raw.append((start_line + cur_idx, start_line + chosen_end_idx, slice_text))

        if chosen_end_idx >= total_unit_lines - 1:
            break

        # Minimal intentional overlap only if continuing mid-block, and never stall progress
        overlap_lines = min(SPLIT_LINE_OVERLAP, chosen_end_idx - cur_idx)
        next_idx = chosen_end_idx + 1 - overlap_lines
        if next_idx <= cur_idx:
            next_idx = chosen_end_idx + 1
        # Verify overlap doesn't drop next step into >max_tokens on next lines cleanly
        cur_idx = next_idx

    total_parts = len(parts_raw)
    emitted: list[dict[str, Any]] = []
    split_type = f"{chunk_type}_split" if total_parts > 1 else chunk_type
    for p_idx, (s_ln, e_ln, txt) in enumerate(parts_raw, start=1):
        if not txt.strip():
            continue
        hdr = format_context_header(
            file_path=file_path,
            language=language,
            chunk_type=split_type,
            symbol_name=symbol_name,
            class_name=class_name,
            parent_symbol=parent_symbol,
            signature=signature,
            part_index=p_idx,
            total_parts=total_parts,
        )
        full_content = f"{hdr}\n{txt}"
        tok_cnt = count_tokens(full_content, tokenizer)
        # Re-check exact total_parts formatting in edge cases at exact ceiling 256
        if tok_cnt > max_tokens:
            txt_lines = txt.split("\n")
            while len(txt_lines) > 1 and count_tokens(f"{hdr}\n" + "\n".join(txt_lines), tokenizer) > max_tokens:
                txt_lines.pop()
                e_ln -= 1
            txt = "\n".join(txt_lines)
            full_content = f"{hdr}\n{txt}"
            tok_cnt = count_tokens(full_content, tokenizer)

        sha = hashlib.sha256(txt.encode("utf-8")).hexdigest()
        emitted.append(
            {
                "content": full_content,
                "metadata": {
                    "repository": repository,
                    "file_path": file_path,
                    "language": language,
                    "chunk_type": split_type,
                    "symbol_name": symbol_name,
                    "class_name": class_name,
                    "parent_symbol": parent_symbol,
                    "start_line": s_ln,
                    "end_line": e_ln,
                    "chunking_strategy": strategy,
                    "parser_type": parser_type,
                    "fallback_used": parser_type in ("python_regex_fallback", "token_window_fallback"),
                    "token_count": tok_cnt,
                    "content_sha256": sha,
                    "chunk_id": make_chunk_id(
                        repository, file_path, symbol_name, s_ln, e_ln, sha, p_idx
                    ),
                    "is_split": total_parts > 1,
                    "part_index": p_idx,
                    "total_parts": total_parts,
                },
            }
        )
    return emitted


def _merge_small_raw_blocks(
    raw_blocks: list[dict[str, Any]],
    lines: list[str],
    tokenizer: Any,
    merge_threshold_tokens: int = TINY_MERGE_THRESHOLD_TOKENS,
    max_merge_tokens: int = 195,
) -> list[dict[str, Any]]:
    """Coalesce adjacent tiny structural/gap blocks in the same scope to avoid single-line junk."""
    if not raw_blocks:
        return []

    merged: list[dict[str, Any]] = []
    for blk in raw_blocks:
        s_l = blk["start_line"]
        e_l = blk["end_line"]
        txt = "\n".join(lines[s_l - 1 : e_l])
        if not txt.strip():
            continue
        blk["tok_est"] = count_tokens(txt, tokenizer)

        if not merged:
            merged.append(blk)
            continue

        prev = merged[-1]
        same_parent = prev.get("class_name", "") == blk.get("class_name", "")
        can_combine_line_gap = s_l <= prev["end_line"] + 4
        either_tiny = (
            prev["tok_est"] < merge_threshold_tokens or blk["tok_est"] < merge_threshold_tokens
        )
        both_gap_or_header = (
            prev["chunk_type"] in ("module_context", "module_imports", "preamble", "block")
            and blk["chunk_type"] in ("module_context", "module_imports", "preamble", "block")
        )

        if same_parent and can_combine_line_gap and (either_tiny or both_gap_or_header):
            cand_text = "\n".join(lines[prev["start_line"] - 1 : e_l])
            cand_tok = count_tokens(cand_text, tokenizer)
            # Reserve ~35 tokens for contextual header
            if cand_tok <= max_merge_tokens:
                prev["end_line"] = e_l
                prev["tok_est"] = cand_tok
                if not prev["symbol_name"]:
                    prev["symbol_name"] = blk["symbol_name"]
                elif blk["symbol_name"] and blk["symbol_name"] not in prev["symbol_name"]:
                    combined_syms = f"{prev['symbol_name']}, {blk['symbol_name']}"
                    prev["symbol_name"] = combined_syms[:80]
                if prev["chunk_type"] in ("preamble", "module_imports", "module_context") and blk["chunk_type"] not in (
                    "preamble",
                    "module_imports",
                    "module_context",
                    "block",
                ):
                    prev["chunk_type"] = blk["chunk_type"]
                    prev["signature"] = blk.get("signature", "")
                prev["preferred_breaks"] = (prev.get("preferred_breaks") or set()) | (
                    blk.get("preferred_breaks") or set()
                ) | {prev["end_line"]}
                continue

        merged.append(blk)

    return merged


# ----------------------------------------------------------------------
# PYTHON HIERARCHICAL AST SCANNER
# ----------------------------------------------------------------------
def _ast_node_span(node: ast.AST) -> tuple[int, int]:
    start = node.lineno
    decs = getattr(node, "decorator_list", None)
    if decs:
        start = min(start, decs[0].lineno)
    end = getattr(node, "end_lineno", start) or start
    return start, end


def _python_signature(lines: list[str], node: ast.AST) -> str:
    start = node.lineno
    end = min(len(lines), start + 5)
    collected = []
    for idx in range(start - 1, end):
        collected.append(lines[idx].strip())
        if lines[idx].rstrip().endswith(":"):
            break
    return re.sub(r"\s+", " ", " ".join(collected))


def _collect_gap_symbols(tree_nodes: list[ast.stmt], start_ln: int, end_ln: int) -> tuple[str, str]:
    """Identify imports / top-level variables inside a module-level line range."""
    syms: list[str] = []
    only_imports = True
    for stmt in tree_nodes:
        s_l, e_l = _ast_node_span(stmt)
        if e_l < start_ln or s_l > end_ln:
            continue
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                syms.append(alias.asname or alias.name.split(".")[0])
        else:
            only_imports = False
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        syms.append(tgt.id)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                syms.append(stmt.target.id)
    chunk_t = "module_imports" if only_imports and syms else "module_context"
    uniq = list(dict.fromkeys(syms))
    return chunk_t, ", ".join(uniq[:6])


def _python_preferred_breaks(node: ast.AST) -> set[int]:
    breaks: set[int] = set()
    for stmt in getattr(node, "body", []):
        end_ln = getattr(stmt, "end_lineno", None)
        if end_ln:
            breaks.add(end_ln)
    return breaks


def _extract_python_blocks(content: str, file_path: str, tokenizer: Any) -> tuple[list[dict[str, Any]], str]:
    """Extract non-overlapping hierarchical AST blocks plus uncovered module-level spans."""
    lines = content.split("\n")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return [], "python_regex_fallback"

    blocks: list[dict[str, Any]] = []
    cursor = 1
    gap_breaks: set[int] = set()

    for node in tree.body:
        n_start, n_end = _ast_node_span(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n_start > cursor:
                gap_txt = "\n".join(lines[cursor - 1 : n_start - 1])
                if gap_txt.strip():
                    g_type, g_sym = _collect_gap_symbols(tree.body, cursor, n_start - 1)
                    blocks.append(
                        {
                            "start_line": cursor,
                            "end_line": n_start - 1,
                            "chunk_type": g_type,
                            "symbol_name": g_sym,
                            "class_name": "",
                            "parent_symbol": "",
                            "signature": "",
                            "preferred_breaks": set(gap_breaks),
                        }
                    )
                gap_breaks.clear()
        else:
            # Top-level statements (imports, constants, assignments) remain grouped until the next def/class or EOF
            gap_breaks.add(n_end)
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ftype = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            sig = _python_signature(lines, node)
            nested_funcs = [
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            # Keep single-unit if fits in token budget; otherwise expose statement/nested breaks
            p_breaks = _python_preferred_breaks(node)
            for nf in nested_funcs:
                ns, ne = _ast_node_span(nf)
                p_breaks.add(max(n_start, ns - 1))
                p_breaks.add(ne)
            blocks.append(
                {
                    "start_line": n_start,
                    "end_line": n_end,
                    "chunk_type": ftype,
                    "symbol_name": node.name,
                    "class_name": "",
                    "parent_symbol": "",
                    "signature": sig,
                    "preferred_breaks": p_breaks,
                }
            )
            cursor = max(cursor, n_end + 1)

        elif isinstance(node, ast.ClassDef):
            cls_sig = _python_signature(lines, node)
            # Partition class body cleanly: class header/docstring/attributes + direct methods (avoid duplicating full class!)
            method_nodes = [
                stmt
                for stmt in node.body
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if not method_nodes:
                blocks.append(
                    {
                        "start_line": n_start,
                        "end_line": n_end,
                        "chunk_type": "class_overview",
                        "symbol_name": node.name,
                        "class_name": node.name,
                        "parent_symbol": "",
                        "signature": cls_sig,
                        "preferred_breaks": _python_preferred_breaks(node),
                    }
                )
            else:
                cls_cursor = n_start
                first_m_start, _ = _ast_node_span(method_nodes[0])
                if first_m_start > cls_cursor:
                    overview_txt = "\n".join(lines[cls_cursor - 1 : first_m_start - 1])
                    if overview_txt.strip():
                        blocks.append(
                            {
                                "start_line": cls_cursor,
                                "end_line": first_m_start - 1,
                                "chunk_type": "class_overview",
                                "symbol_name": node.name,
                                "class_name": node.name,
                                "parent_symbol": "",
                                "signature": cls_sig,
                                "preferred_breaks": set(),
                            }
                        )
                    cls_cursor = first_m_start

                for m_node in method_nodes:
                    m_start, m_end = _ast_node_span(m_node)
                    if m_start > cls_cursor:
                        mid_txt = "\n".join(lines[cls_cursor - 1 : m_start - 1])
                        if mid_txt.strip():
                            blocks.append(
                                {
                                    "start_line": cls_cursor,
                                    "end_line": m_start - 1,
                                    "chunk_type": "class_attributes",
                                    "symbol_name": node.name,
                                    "class_name": node.name,
                                    "parent_symbol": node.name,
                                    "signature": cls_sig,
                                    "preferred_breaks": set(),
                                }
                            )
                    m_sig = _python_signature(lines, m_node)
                    blocks.append(
                        {
                            "start_line": m_start,
                            "end_line": m_end,
                            "chunk_type": "method",
                            "symbol_name": m_node.name,
                            "class_name": node.name,
                            "parent_symbol": node.name,
                            "signature": m_sig,
                            "preferred_breaks": _python_preferred_breaks(m_node),
                        }
                    )
                    cls_cursor = max(cls_cursor, m_end + 1)

                if cls_cursor <= n_end:
                    tail_txt = "\n".join(lines[cls_cursor - 1 : n_end])
                    if tail_txt.strip():
                        blocks.append(
                            {
                                "start_line": cls_cursor,
                                "end_line": n_end,
                                "chunk_type": "class_attributes",
                                "symbol_name": node.name,
                                "class_name": node.name,
                                "parent_symbol": node.name,
                                "signature": cls_sig,
                                "preferred_breaks": set(),
                            }
                        )
            cursor = max(cursor, n_end + 1)

    if cursor <= len(lines):
        tail = "\n".join(lines[cursor - 1 :])
        if tail.strip():
            g_type, g_sym = _collect_gap_symbols(tree.body, cursor, len(lines))
            blocks.append(
                {
                    "start_line": cursor,
                    "end_line": len(lines),
                    "chunk_type": g_type,
                    "symbol_name": g_sym,
                    "class_name": "",
                    "parent_symbol": "",
                    "signature": "",
                    "preferred_breaks": set(gap_breaks),
                }
            )

    merged = _merge_small_raw_blocks(blocks, lines, tokenizer, merge_threshold_tokens=TINY_MERGE_THRESHOLD_TOKENS)
    return merged, "python_ast"


# ----------------------------------------------------------------------
# LEXICAL BRACE-DEPTH SCANNER (JS / TS / JAVA / C / C++)
# ----------------------------------------------------------------------
def _scan_line_brace_states(lines: list[str]) -> list[tuple[int, int, bool, bool]]:
    """Return [(depth_before_code, depth_after, is_pure_comment_or_blank, is_preproc)] per line."""
    states: list[tuple[int, int, bool, bool]] = []
    depth = 0
    in_block_comment = False
    in_preproc = False

    for line in lines:
        depth_before = depth
        in_str: str | None = None
        escape = False
        has_code = False
        stripped = line.strip()
        line_is_preproc = in_preproc or stripped.startswith("#")
        in_preproc = line_is_preproc and stripped.endswith("\\")

        i = 0
        n = len(line)

        while i < n:
            ch = line[i]
            nxt = line[i + 1] if i + 1 < n else ""

            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue

            if in_str is not None:
                has_code = True
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_str:
                    in_str = None
                i += 1
                continue

            if ch in (" ", "\t", "\r"):
                i += 1
                continue

            if ch == "/" and nxt == "/":
                break
            if ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue

            has_code = True
            if ch in ('"', "'", "`"):
                in_str = ch
            elif not line_is_preproc:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth = max(0, depth - 1)
            i += 1

        states.append((depth_before, depth, not has_code, line_is_preproc))
    return states


def _detect_js_declaration(line: str, depth_before: int) -> tuple[str, str] | None:
    s = line.strip()
    if not s or s.startswith(("//", "/*", "*")):
        return None
    first_word = re.match(r"^([A-Za-z_$]\w*)\b", s)
    if first_word and first_word.group(1) in CONTROL_KEYWORDS:
        return None

    m = re.match(r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", s)
    if m:
        return ("function", m.group(1))

    m = re.match(r"^(?:export\s+(?:default\s+)?)?class\s+([A-Za-z_$][\w$]*)\b", s)
    if m:
        return ("class", m.group(1))

    # Express-style prototype / module exports: proto.foo = ..., exports.foo = ..., app.foo = ...
    m = re.match(
        r"^(?:module\.)?exports(?:\.([A-Za-z_$][\w$]*))?\s*=\s*(?:async\s+)?(?:function|\(|[A-Za-z_$])",
        s,
    )
    if m:
        return ("export", m.group(1) or "module.exports")

    m = re.match(
        r"^([A-Za-z_$][\w$]*)\.([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>)",
        s,
    )
    if m and m.group(1) not in CONTROL_KEYWORDS:
        return ("method", f"{m.group(1)}.{m.group(2)}")

    m = re.match(
        r"^defineGetter\s*\(\s*\w+\s*,\s*['\"]([^'\"]+)['\"]",
        s,
    )
    if m:
        return ("getter", m.group(1))

    m = re.match(
        r"^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>)",
        s,
    )
    if m:
        return ("function", m.group(1))

    # Class method inside depth 1 (avoid control flow like if/for/catch and test describe/it fragments)
    if depth_before == 1:
        m = re.match(r"^(?:async\s+)?([A-Za-z_$][\w$]*)\s*\([^;]*\)\s*\{", s)
        if m and m.group(1) not in CONTROL_KEYWORDS:
            return ("method", m.group(1))

    return None


def _detect_java_declaration(line: str, depth_before: int) -> tuple[str, str] | None:
    s = line.strip()
    if not s or s.startswith(("//", "/*", "*", "@")):
        return None
    m = re.match(
        r"^(?:(?:public|private|protected|static|final|abstract|sealed|non-sealed)\s+)*(class|interface|enum|record)\s+([A-Za-z_]\w*)",
        s,
    )
    if m:
        return (m.group(1), m.group(2))

    if depth_before <= 1:
        m = re.match(
            r"^(?:(?:public|private|protected|static|final|synchronized|abstract|native|default)\s+)*[\w<>\[\], ?]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:throws\s+[\w.,\s]+)?\{?",
            s,
        )
        if m and m.group(1) not in CONTROL_KEYWORDS:
            return ("method", m.group(1))
    return None


def _detect_cpp_declaration(line: str, depth_before: int) -> tuple[str, str] | None:
    s = line.strip()
    if not s or s.startswith(("//", "/*", "*", "#")):
        return None
    if re.match(r"^(if|else|for|while|switch|catch|return|typedef|using)\b", s):
        return None

    m = re.match(r"^(?:class|struct)\s+(?:[A-Z0-9_]+\s+)*([A-Za-z_]\w*)\b", s)
    if m and not s.endswith(";"):
        kw = "struct" if s.startswith("struct") else "class"
        return (kw, m.group(1))

    m = re.match(r"^enum(?:\s+class)?\s+([A-Za-z_]\w*)\b", s)
    if m and not s.endswith(";"):
        return ("enum", m.group(1))

    # C/C++ function/constructor signature at top-level or inside namespace (depth <= 2)
    if depth_before <= 2 and "(" in s and not s.endswith(";") and "=" not in s.split("(")[0]:
        # Extract function identifier directly before '('
        m = re.search(r"([~A-Za-z_][\w:]*)\s*\(", s)
        if m:
            fn_name = m.group(1).split("::")[-1]
            if fn_name not in CONTROL_KEYWORDS and not fn_name.isupper():
                return ("function", fn_name)
    return None


def _lookback_preamble_start(lines: list[str], decl_idx: int, min_bound_idx: int, language: str) -> int:
    """Attach preceding template<...>, decorator/attribute, or immediate doc comments to a declaration."""
    cur = decl_idx
    while cur > min_bound_idx:
        prev = lines[cur - 1].strip()
        if language == "cpp":
            if (
                prev.startswith("template")
                or prev.startswith("FMT_")
                or prev.startswith("[[")
                or prev.startswith("inline")
                or prev.startswith("constexpr")
            ):
                cur -= 1
                continue
        elif language == "java":
            if prev.startswith("@"):
                cur -= 1
                continue
        break
    return cur


def _extract_brace_language_blocks(
    content: str, file_path: str, language: str, tokenizer: Any
) -> tuple[list[dict[str, Any]], str]:
    lines = content.split("\n")
    states = _scan_line_brace_states(lines)
    n_lines = len(lines)

    detector = (
        _detect_js_declaration
        if language in ("javascript", "typescript")
        else _detect_java_declaration if language == "java" else _detect_cpp_declaration
    )

    boundaries: list[dict[str, Any]] = []
    # Identify leading license/file-header pure-comment block so it stays cleanly isolated from code declarations
    first_code_idx = 0
    while first_code_idx < n_lines and states[first_code_idx][2]:
        first_code_idx += 1

    min_decl_search = first_code_idx
    if first_code_idx > 0:
        header_txt = "\n".join(lines[:first_code_idx])
        if header_txt.strip():
            boundaries.append(
                {
                    "start_line": 1,
                    "decl_line": 1,
                    "chunk_type": "file_header",
                    "symbol_name": "license_header",
                }
            )

    last_start_0idx = min_decl_search
    class_stack: list[tuple[str, int]] = []
    for idx in range(min_decl_search, n_lines):
        depth_before, _, is_comment_only, is_preproc = states[idx]
        while class_stack and depth_before <= class_stack[-1][1]:
            class_stack.pop()
        if is_comment_only or is_preproc:
            continue
        # For C/C++, avoid splitting inside deep function bodies when not directly in a class/struct
        if language in ("cpp", "c") and class_stack and depth_before > class_stack[-1][1] + 1:
            continue
        detected = detector(lines[idx], depth_before)
        if detected is None:
            continue
        c_type, sym = detected
        adj_start = _lookback_preamble_start(lines, idx, last_start_0idx, language)
        # Avoid creating boundary inside previous signature lookback
        if boundaries and adj_start + 1 <= boundaries[-1]["start_line"]:
            continue

        enclosing_class = class_stack[-1][0] if class_stack else ""
        if c_type in ("class", "struct", "interface"):
            block_class = sym
            block_parent = ""
            class_stack.append((sym, depth_before))
        elif enclosing_class and c_type == "function":
            c_type = "method"
            block_class = enclosing_class
            block_parent = enclosing_class
        else:
            block_class = enclosing_class
            block_parent = enclosing_class

        boundaries.append(
            {
                "start_line": adj_start + 1,
                "decl_line": idx + 1,
                "chunk_type": c_type,
                "symbol_name": sym,
                "class_name": block_class,
                "parent_symbol": block_parent,
            }
        )
        last_start_0idx = idx + 1

    raw_blocks: list[dict[str, Any]] = []
    if not boundaries:
        raw_blocks.append(
            {
                "start_line": 1,
                "end_line": n_lines,
                "chunk_type": "module_context",
                "symbol_name": os.path.basename(file_path),
                "class_name": "",
                "parent_symbol": "",
                "signature": "",
                "preferred_breaks": set(),
            }
        )
    else:
        if boundaries[0]["start_line"] > 1:
            pre_txt = "\n".join(lines[: boundaries[0]["start_line"] - 1])
            if pre_txt.strip():
                raw_blocks.append(
                    {
                        "start_line": 1,
                        "end_line": boundaries[0]["start_line"] - 1,
                        "chunk_type": "module_context",
                        "symbol_name": "imports_and_setup",
                        "class_name": "",
                        "parent_symbol": "",
                        "signature": "",
                        "preferred_breaks": set(),
                    }
                )

        for i, b in enumerate(boundaries):
            s_ln = b["start_line"]
            e_ln = boundaries[i + 1]["start_line"] - 1 if i + 1 < len(boundaries) else n_lines
            if e_ln < s_ln:
                continue
            txt = "\n".join(lines[s_ln - 1 : e_ln])
            if not txt.strip():
                continue
            sig_line = lines[b["decl_line"] - 1].strip()
            p_breaks = {
                ln
                for ln in range(s_ln, e_ln + 1)
                if not lines[ln - 1].strip() or lines[ln - 1].strip() in ("}", "};")
            }
            raw_blocks.append(
                {
                    "start_line": s_ln,
                    "end_line": e_ln,
                    "chunk_type": b["chunk_type"],
                    "symbol_name": b["symbol_name"],
                    "class_name": b.get("class_name", ""),
                    "parent_symbol": b.get("parent_symbol", ""),
                    "signature": sig_line,
                    "preferred_breaks": p_breaks,
                }
            )

    # Separate license header cleanly from declarations, only merge tiny adjacent blocks excluding license headers
    separated: list[dict[str, Any]] = []
    if raw_blocks and raw_blocks[0]["chunk_type"] == "file_header":
        separated.append(raw_blocks[0])
        rest = _merge_small_raw_blocks(
            raw_blocks[1:], lines, tokenizer, merge_threshold_tokens=TINY_MERGE_THRESHOLD_TOKENS
        )
        separated.extend(rest)
    else:
        separated = _merge_small_raw_blocks(
            raw_blocks, lines, tokenizer, merge_threshold_tokens=TINY_MERGE_THRESHOLD_TOKENS
        )
    return separated, "lexical_brace_scanner"


def chunk_code_v1(
    file_data: dict[str, Any],
    repository: str = "",
    tokenizer: Any = None,
    max_tokens: int = MAX_MODEL_TOKENS,
) -> list[dict[str, Any]]:
    """
    Token-aware, code-aware V1 chunker.

    Input format matches V0: {"file_path": str, "content": str, "extension": str}.
    Guarantees every returned chunk is non-empty, has Chroma-safe metadata,
    accurate 1-based line numbers, and token_count <= max_tokens (default 256).
    """
    content = file_data.get("content", "")
    file_path = file_data["file_path"]
    ext = file_data.get("extension", os.path.splitext(file_path)[1]).lower()

    if not content or not content.strip():
        return []

    tok, _ = (tokenizer, "provided") if tokenizer is not None else get_embedding_tokenizer()
    language = detect_language(file_path, ext)
    lines = content.split("\n")

    raw_blocks: list[dict[str, Any]] = []
    parser_type = "token_window_fallback"

    if language == "python":
        raw_blocks, parser_type = _extract_python_blocks(content, file_path, tok)
        if parser_type == "python_regex_fallback":
            # Python SyntaxError fallback: token-aware window over lines
            raw_blocks = [
                {
                    "start_line": 1,
                    "end_line": len(lines),
                    "chunk_type": "fallback_module",
                    "symbol_name": os.path.basename(file_path),
                    "class_name": "",
                    "parent_symbol": "",
                    "signature": "",
                    "preferred_breaks": {
                        ln for ln in range(1, len(lines) + 1) if not lines[ln - 1].strip()
                    },
                }
            ]
    elif language in ("javascript", "typescript", "java", "cpp", "c"):
        raw_blocks, parser_type = _extract_brace_language_blocks(content, file_path, language, tok)
    else:
        raw_blocks = [
            {
                "start_line": 1,
                "end_line": len(lines),
                "chunk_type": "fallback_module",
                "symbol_name": os.path.basename(file_path),
                "class_name": "",
                "parent_symbol": "",
                "signature": "",
                "preferred_breaks": {
                    ln for ln in range(1, len(lines) + 1) if not lines[ln - 1].strip()
                },
            }
        ]
        parser_type = "token_window_fallback"

    final_chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for blk in raw_blocks:
        emitted = _emit_token_bounded_unit(
            lines=lines,
            start_line=blk["start_line"],
            end_line=blk["end_line"],
            file_path=file_path,
            language=language,
            chunk_type=blk["chunk_type"],
            parser_type=parser_type,
            symbol_name=blk.get("symbol_name", ""),
            class_name=blk.get("class_name", ""),
            parent_symbol=blk.get("parent_symbol", ""),
            signature=blk.get("signature", ""),
            repository=repository,
            strategy="code_aware_v1",
            preferred_breaks=blk.get("preferred_breaks"),
            max_tokens=max_tokens,
            target_tokens=TARGET_CHUNK_TOKENS,
            tokenizer=tok,
        )
        for ch in emitted:
            cid = ch["metadata"]["chunk_id"]
            if cid in seen_ids:
                suffix = 2
                while f"{cid}_{suffix}" in seen_ids:
                    suffix += 1
                cid = f"{cid}_{suffix}"
                ch["metadata"]["chunk_id"] = cid
            seen_ids.add(cid)
            final_chunks.append(ch)

    return final_chunks
