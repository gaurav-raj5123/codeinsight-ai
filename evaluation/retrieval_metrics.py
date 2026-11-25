"""Reusable retrieval and latency metrics extracted from the validated Day 1 baseline.

All ranking definitions follow the formal Day 1 Task 4 specifications:
- Formal interval overlap for chunk-to-evidence relevance matching.
- Hit Rate@K: binary indicator of >= 1 relevant chunk within top K.
- Precision@5: relevant count in top K divided by fixed denominator K (default 5).
- Span Recall@5: unique matched evidence spans divided by total ground-truth spans.
- MRR@5: reciprocal of the 1-based rank of the first relevant chunk (0.0 if none).
- NDCG@5: DCG@5 normalized by ideal DCG@5 computed from total relevant chunks in
  the indexed corpus (capped at K), NOT ground-truth span count.
- P50 latency: standard statistics.median().
- P95 latency: nearest-rank index math.ceil(0.95 * n) - 1.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence


def check_overlap(c_start: int, c_end: int, e_start: int, e_end: int) -> bool:
    """Return True if two inclusive 1-based line intervals overlap."""
    return max(c_start, e_start) <= min(c_end, e_end)


def _extract_chunk_location(chunk: dict[str, Any]) -> tuple[str | None, int | None, int | None]:
    """Extract (file_path, start_line, end_line) from flat or metadata-nested chunk dicts."""
    meta = chunk.get("metadata")
    source = meta if isinstance(meta, dict) else chunk
    return source.get("file_path"), source.get("start_line"), source.get("end_line")


def build_chunks_by_file_index(chunks: Sequence[dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    """Pre-index chunk line intervals by relative file_path for fast corpus overlap counting."""
    by_file: dict[str, list[tuple[int, int]]] = {}
    for ch in chunks:
        c_file, c_start, c_end = _extract_chunk_location(ch)
        if c_file is not None and c_start is not None and c_end is not None:
            by_file.setdefault(c_file, []).append((c_start, c_end))
    return by_file


def count_relevant_chunks_in_repo(
    chunks: Sequence[dict[str, Any]] | None = None,
    evidence_spans: Sequence[dict[str, Any]] = (),
    chunks_by_file: dict[str, list[tuple[int, int]]] | None = None,
) -> int:
    """Count unique chunks in an indexed repository overlapping >= 1 ground-truth span."""
    if not evidence_spans:
        return 0

    if chunks_by_file is not None:
        spans_by_file: dict[str, list[tuple[int, int]]] = {}
        for span in evidence_spans:
            spans_by_file.setdefault(span["file_path"], []).append(
                (span["start_line"], span["end_line"])
            )
        count = 0
        for file_path, file_spans in spans_by_file.items():
            for c_start, c_end in chunks_by_file.get(file_path, ()):
                if any(check_overlap(c_start, c_end, e_s, e_e) for e_s, e_e in file_spans):
                    count += 1
        return count

    if not chunks:
        return 0

    count = 0
    for ch in chunks:
        c_file, c_start, c_end = _extract_chunk_location(ch)
        if c_file is None or c_start is None or c_end is None:
            continue
        for span in evidence_spans:
            if c_file == span["file_path"] and check_overlap(
                c_start, c_end, span["start_line"], span["end_line"]
            ):
                count += 1
                break
    return count


def evaluate_retrieved_chunks(
    retrieved_chunks: Sequence[dict[str, Any]],
    evidence_spans: Sequence[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Match retrieved chunks against evidence spans, tracking unique span coverage."""
    matched_span_indices: set[int] = set()
    per_chunk_relevance: list[bool] = []

    for ch in list(retrieved_chunks)[:top_k]:
        c_file, c_start, c_end = _extract_chunk_location(ch)
        chunk_is_relevant = False
        if c_file is not None and c_start is not None and c_end is not None:
            for s_idx, span in enumerate(evidence_spans):
                if c_file == span["file_path"] and check_overlap(
                    c_start, c_end, span["start_line"], span["end_line"]
                ):
                    chunk_is_relevant = True
                    matched_span_indices.add(s_idx)
        per_chunk_relevance.append(chunk_is_relevant)

    while len(per_chunk_relevance) < top_k:
        per_chunk_relevance.append(False)

    first_relevant_rank: int | None = None
    for idx, is_rel in enumerate(per_chunk_relevance, start=1):
        if is_rel:
            first_relevant_rank = idx
            break

    return {
        "per_chunk_relevance": per_chunk_relevance,
        "binary_relevance_vector": [1 if r else 0 for r in per_chunk_relevance],
        "matched_span_indices": matched_span_indices,
        "first_relevant_rank": first_relevant_rank,
    }


def hit_rate_at_k(per_chunk_relevance: Sequence[bool | int], k: int) -> float:
    """Return 1.0 if at least one relevant item appears in top k, else 0.0."""
    if k <= 0:
        return 0.0
    return 1.0 if any(bool(r) for r in per_chunk_relevance[:k]) else 0.0


def precision_at_k(per_chunk_relevance: Sequence[bool | int], k: int = 5) -> float:
    """Compute Precision@K with fixed denominator K regardless of returned list size."""
    if k <= 0:
        return 0.0
    relevant_count = sum(1 for r in per_chunk_relevance[:k] if r)
    return relevant_count / float(k)


def span_recall(matched_spans_count: int, total_evidence_spans: int) -> float:
    """Compute fraction of unique ground-truth evidence spans covered."""
    if total_evidence_spans <= 0:
        return 0.0
    return min(1.0, matched_spans_count / float(total_evidence_spans))


def reciprocal_rank_at_k(per_chunk_relevance: Sequence[bool | int], k: int = 5) -> float:
    """Compute reciprocal rank of the first relevant item within top k."""
    if k <= 0:
        return 0.0
    for idx, is_rel in enumerate(per_chunk_relevance[:k], start=1):
        if is_rel:
            return 1.0 / float(idx)
    return 0.0


def calculate_dcg(relevance_vector: Sequence[int | float]) -> float:
    """Compute DCG@K = sum(rel_i / log2(i + 1)) for 1-based rank i."""
    dcg = 0.0
    for idx, rel in enumerate(relevance_vector, start=1):
        if rel > 0:
            dcg += float(rel) / math.log2(idx + 1)
    return dcg


def calculate_ndcg(
    relevance_vector: Sequence[int | float],
    total_relevant_chunks_in_repo: int,
    k: int | None = None,
) -> float:
    """Compute binary NDCG@K using ideal DCG derived from corpus relevant chunk count."""
    cutoff = len(relevance_vector) if k is None else k
    if cutoff <= 0 or total_relevant_chunks_in_repo <= 0:
        return 0.0

    dcg = calculate_dcg(relevance_vector[:cutoff])
    if dcg == 0.0:
        return 0.0

    ideal_rel_count = min(total_relevant_chunks_in_repo, cutoff)
    ideal_vector = [1] * ideal_rel_count + [0] * (cutoff - ideal_rel_count)
    idcg = calculate_dcg(ideal_vector)
    if idcg == 0.0:
        return 0.0
    return min(1.0, dcg / idcg)


def compute_query_metrics(
    retrieved_chunks: Sequence[dict[str, Any]],
    evidence_spans: Sequence[dict[str, Any]],
    total_relevant_chunks_in_repo: int,
    top_k: int = 5,
    failed: bool = False,
) -> dict[str, Any]:
    """Compute full per-query retrieval evaluation metrics and matching diagnostics."""
    total_spans = len(evidence_spans)
    if failed:
        return {
            "per_chunk_relevance": [False] * top_k,
            "first_relevant_rank": None,
            "matched_evidence_spans_count": 0,
            "total_evidence_spans_count": total_spans,
            "total_relevant_chunks_in_repo": total_relevant_chunks_in_repo,
            "dcg_at_5": 0.0,
            "metrics": {
                "hit_rate_at_1": 0.0,
                "hit_rate_at_3": 0.0,
                "hit_rate_at_5": 0.0,
                "precision_at_5": 0.0,
                "mrr_at_5": 0.0,
                "span_recall_at_5": 0.0,
                "ndcg_at_5": 0.0,
            },
        }

    match = evaluate_retrieved_chunks(retrieved_chunks, evidence_spans, top_k=top_k)
    rel_flags = match["per_chunk_relevance"]
    rel_vec = match["binary_relevance_vector"]
    matched_count = len(match["matched_span_indices"])

    dcg_val = calculate_dcg(rel_vec)
    ndcg_val = calculate_ndcg(rel_vec, total_relevant_chunks_in_repo, k=top_k)

    return {
        "per_chunk_relevance": rel_flags,
        "first_relevant_rank": match["first_relevant_rank"],
        "matched_evidence_spans_count": matched_count,
        "total_evidence_spans_count": total_spans,
        "total_relevant_chunks_in_repo": total_relevant_chunks_in_repo,
        "dcg_at_5": round(dcg_val, 4),
        "metrics": {
            "hit_rate_at_1": hit_rate_at_k(rel_flags, 1),
            "hit_rate_at_3": hit_rate_at_k(rel_flags, 3),
            "hit_rate_at_5": hit_rate_at_k(rel_flags, 5),
            "precision_at_5": round(precision_at_k(rel_flags, top_k), 4),
            "mrr_at_5": round(reciprocal_rank_at_k(rel_flags, top_k), 4),
            "span_recall_at_5": round(span_recall(matched_count, total_spans), 4),
            "ndcg_at_5": round(ndcg_val, 4),
        },
    }


def nearest_rank_p95(values: Sequence[float]) -> float | None:
    """Compute nearest-rank 95th percentile using index math.ceil(0.95 * n) - 1."""
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = max(0, min(math.ceil(0.95 * n) - 1, n - 1))
    return float(sorted_vals[idx])


def compute_latency_stats(latencies_ms: Sequence[float]) -> dict[str, Any]:
    """Compute count, mean, P50 (statistics.median), P95 (ceil(0.95*n)-1), min, and max."""
    if not latencies_ms:
        return {}
    n = len(latencies_ms)
    p50 = statistics.median(latencies_ms)
    p95 = nearest_rank_p95(latencies_ms)
    p95_idx = max(0, min(math.ceil(0.95 * n) - 1, n - 1))
    return {
        "count": n,
        "mean": round(sum(latencies_ms) / n, 2),
        "median_p50": round(float(p50), 2),
        "p95": round(float(p95), 2) if p95 is not None else None,
        "min": round(float(min(latencies_ms)), 2),
        "max": round(float(max(latencies_ms)), 2),
        "p95_rank_index": p95_idx,
        "p95_convention": "nearest rank index math.ceil(0.95 * n) - 1",
    }


def aggregate_retrieval_metrics(
    records: Sequence[dict[str, Any]],
    latency_key: str = "retrieval_latency_ms",
) -> dict[str, Any]:
    """Aggregate query-level retrieval metrics and latency statistics across records."""
    if not records:
        return {}
    n = len(records)
    m_list = [r["metrics"] for r in records]
    lat_list = [r[latency_key] for r in records if r.get(latency_key) is not None]
    lat_stats = compute_latency_stats(lat_list)

    return {
        "query_count": n,
        "hit_rate_at_1": round(sum(m["hit_rate_at_1"] for m in m_list) / n, 4),
        "hit_rate_at_3": round(sum(m["hit_rate_at_3"] for m in m_list) / n, 4),
        "hit_rate_at_5": round(sum(m["hit_rate_at_5"] for m in m_list) / n, 4),
        "precision_at_5": round(sum(m["precision_at_5"] for m in m_list) / n, 4),
        "mrr_at_5": round(sum(m["mrr_at_5"] for m in m_list) / n, 4),
        "span_recall_at_5": round(sum(m["span_recall_at_5"] for m in m_list) / n, 4),
        "ndcg_at_5": round(sum(m["ndcg_at_5"] for m in m_list) / n, 4),
        "latency_ms": {
            "mean": lat_stats.get("mean"),
            "median_p50": lat_stats.get("median_p50"),
            "p95": lat_stats.get("p95"),
            "max": lat_stats.get("max"),
        } if lat_stats else {},
    }


def calc_distance_distribution(distances: Sequence[float]) -> dict[str, Any]:
    """Summarize distance statistics (min, median, mean, max) safely for a sample list."""
    if not distances:
        return {}
    n = len(distances)
    return {
        "count": n,
        "min": round(float(min(distances)), 6),
        "median": round(float(statistics.median(distances)), 6),
        "mean": round(sum(distances) / n, 6),
        "max": round(float(max(distances)), 6),
    }
