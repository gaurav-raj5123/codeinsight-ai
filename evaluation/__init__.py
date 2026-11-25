"""Reusable evaluation metrics and latency utilities for RAG retrieval benchmarks."""

from .retrieval_metrics import (
    aggregate_retrieval_metrics,
    build_chunks_by_file_index,
    calc_distance_distribution,
    calculate_dcg,
    calculate_ndcg,
    check_overlap,
    compute_latency_stats,
    compute_query_metrics,
    count_relevant_chunks_in_repo,
    evaluate_retrieved_chunks,
    hit_rate_at_k,
    nearest_rank_p95,
    precision_at_k,
    reciprocal_rank_at_k,
    span_recall,
)

__all__ = [
    "aggregate_retrieval_metrics",
    "build_chunks_by_file_index",
    "calc_distance_distribution",
    "calculate_dcg",
    "calculate_ndcg",
    "check_overlap",
    "compute_latency_stats",
    "compute_query_metrics",
    "count_relevant_chunks_in_repo",
    "evaluate_retrieved_chunks",
    "hit_rate_at_k",
    "nearest_rank_p95",
    "precision_at_k",
    "reciprocal_rank_at_k",
    "span_recall",
]
