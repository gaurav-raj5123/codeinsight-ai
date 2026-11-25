"""Retrieval modules for CodeInsight AI Assistant (Day 4+ modular architecture).

Organized to allow future additions such as retrieval/reranker.py and
retrieval/query_rewriter.py without altering Day 4 interfaces.
"""

from retrieval.bm25_retriever import (
    BM25Retriever,
    V1_HEADER_LABELS,
    build_bm25_from_collection,
    compute_corpus_fingerprint,
    strip_v1_header_labels,
    tokenize_code,
)
from retrieval.hybrid_retriever import (
    HybridRetriever,
    reciprocal_rank_fusion,
)

__all__ = [
    "BM25Retriever",
    "HybridRetriever",
    "V1_HEADER_LABELS",
    "build_bm25_from_collection",
    "compute_corpus_fingerprint",
    "reciprocal_rank_fusion",
    "strip_v1_header_labels",
    "tokenize_code",
]
