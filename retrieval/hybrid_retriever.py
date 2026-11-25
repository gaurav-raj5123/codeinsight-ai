"""Hybrid Dense + BM25 Reciprocal Rank Fusion Retriever (Day 4)."""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence


def reciprocal_rank_fusion(
    dense_results: Sequence[dict[str, Any]],
    bm25_results: Sequence[dict[str, Any]],
    rrf_k: int = 60,
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fuse dense and BM25 candidate lists via Reciprocal Rank Fusion (RRF).

    Score(d) = 1/(rrf_k + rank_dense(d)) + 1/(rrf_k + rank_bm25(d)).
    Handles empty BM25 (or empty dense) inputs gracefully and deduplicates on chunk ID.
    """
    candidates: dict[str, dict[str, Any]] = {}
    rrf_k = int(rrf_k)

    for rank_i, chunk in enumerate(dense_results, start=1):
        cid = str(chunk.get("id") or f"dense_{rank_i}")
        entry = candidates.get(cid)
        if entry is None:
            entry = {
                "id": cid,
                "content": chunk.get("content", ""),
                "metadata": dict(chunk.get("metadata") or {}),
                "distance": chunk.get("distance"),
                "dense_rank": rank_i,
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 0.0,
                "retrieval_sources": [],
            }
            candidates[cid] = entry
        if "dense" not in entry["retrieval_sources"]:
            entry["dense_rank"] = min(entry["dense_rank"], rank_i) if entry["dense_rank"] else rank_i
            entry["distance"] = chunk.get("distance") if chunk.get("distance") is not None else entry["distance"]
            entry["retrieval_sources"].append("dense")
            entry["rrf_score"] += 1.0 / (rrf_k + rank_i)

    for rank_i, chunk in enumerate(bm25_results, start=1):
        cid = str(chunk.get("id") or f"bm25_{rank_i}")
        entry = candidates.get(cid)
        if entry is None:
            entry = {
                "id": cid,
                "content": chunk.get("content", ""),
                "metadata": dict(chunk.get("metadata") or {}),
                "distance": chunk.get("distance"),
                "dense_rank": None,
                "bm25_rank": rank_i,
                "bm25_score": chunk.get("bm25_score"),
                "rrf_score": 0.0,
                "retrieval_sources": [],
            }
            candidates[cid] = entry
        if "bm25" not in entry["retrieval_sources"]:
            entry["bm25_rank"] = rank_i
            entry["bm25_score"] = chunk.get("bm25_score")
            entry["retrieval_sources"].append("bm25")
            entry["rrf_score"] += 1.0 / (rrf_k + rank_i)

    ordered = list(candidates.values())
    ordered.sort(
        key=lambda c: (
            -round(float(c["rrf_score"]), 12),
            min(
                c["dense_rank"] if c["dense_rank"] is not None else 10**9,
                c["bm25_rank"] if c["bm25_rank"] is not None else 10**9,
            ),
            c["dense_rank"] if c["dense_rank"] is not None else 10**9,
            c["bm25_rank"] if c["bm25_rank"] is not None else 10**9,
            str(c["id"]),
        )
    )

    for rank_i, item in enumerate(ordered, start=1):
        item["rrf_rank"] = rank_i
        item["rrf_score"] = round(float(item["rrf_score"]), 8)

    dense_ids = {str(c.get("id")) for c in dense_results if c.get("id") is not None}
    bm25_ids = {str(c.get("id")) for c in bm25_results if c.get("id") is not None}
    stats = {
        "rrf_k": rrf_k,
        "dense_candidates_count": len(dense_results),
        "bm25_candidates_count": len(bm25_results),
        "union_candidates_count": len(ordered),
        "overlap_candidates_count": len(dense_ids & bm25_ids),
        "bm25_fallback_to_dense_only": len(bm25_results) == 0,
    }

    sliced = ordered[:top_k] if top_k is not None else ordered
    return sliced, stats


class HybridRetriever:
    """Orchestrates Dense Top-M, BM25 Top-M, full candidate union, and RRF Top-K."""

    def __init__(
        self,
        dense_search_fn: Callable[[str, int], list[dict[str, Any]]],
        bm25_retriever: Any,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.dense_search_fn = dense_search_fn
        self.bm25_retriever = bm25_retriever
        self.candidate_k = int(candidate_k)
        self.rrf_k = int(rrf_k)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
        return_diagnostics: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        cand_k = int(candidate_k or self.candidate_k)

        t_dense0 = time.perf_counter()
        dense_cands = list(self.dense_search_fn(query, cand_k))
        dense_ms = (time.perf_counter() - t_dense0) * 1000.0

        t_bm250 = time.perf_counter()
        bm25_cands, bm25_meta = self.bm25_retriever.search(
            query, top_k=cand_k, return_metadata=True
        )
        bm25_ms = (time.perf_counter() - t_bm250) * 1000.0

        t_rrf0 = time.perf_counter()
        fused_all, rrf_stats = reciprocal_rank_fusion(
            dense_cands, bm25_cands, rrf_k=self.rrf_k, top_k=None
        )
        rrf_ms = (time.perf_counter() - t_rrf0) * 1000.0

        final_top_k = fused_all[:top_k]
        if not return_diagnostics:
            return final_top_k

        # Build deduplicated candidate union (Dense@K U BM25@K, size <= 2*cand_k)
        union_map: dict[str, dict[str, Any]] = {}
        for ch in dense_cands:
            cid = str(ch["id"])
            union_map.setdefault(cid, ch)
        for ch in bm25_cands:
            cid = str(ch["id"])
            union_map.setdefault(cid, ch)

        diagnostics = {
            "candidate_k": cand_k,
            "top_k": top_k,
            "dense_candidates": dense_cands,
            "bm25_candidates": bm25_cands,
            "bm25_meta": bm25_meta,
            "candidate_union": list(union_map.values()),
            "rrf_all_ranked": fused_all,
            "rrf_top_20": fused_all[:20],
            "rrf_stats": rrf_stats,
            "timings_ms": {
                "dense_ms": round(dense_ms, 2),
                "bm25_ms": round(bm25_ms, 2),
                "rrf_ms": round(rrf_ms, 3),
                "total_hybrid_ms": round(dense_ms + bm25_ms + rrf_ms, 2),
            },
        }
        return final_top_k, diagnostics
