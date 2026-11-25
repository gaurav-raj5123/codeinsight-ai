"""Retrieval modules for CodeInsight AI Assistant (Day 4+ modular architecture).

Organized to allow modular composition across BM25, Hybrid RRF fusion,
ONNX Cross-Encoder reranking, and conversational query rewriting.
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
from retrieval.query_rewriter import (
    format_previous_context,
    needs_conversational_rewrite,
    rewrite_conversational_query,
)
from retrieval.reranker import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_REVISION,
    HybridRerankRetriever,
    ONNXCrossEncoderReranker,
    stable_sigmoid,
)

__all__ = [
    "BM25Retriever",
    "DEFAULT_CROSS_ENCODER_MODEL",
    "DEFAULT_CROSS_ENCODER_REVISION",
    "HybridRerankRetriever",
    "HybridRetriever",
    "ONNXCrossEncoderReranker",
    "V1_HEADER_LABELS",
    "build_bm25_from_collection",
    "compute_corpus_fingerprint",
    "format_previous_context",
    "needs_conversational_rewrite",
    "reciprocal_rank_fusion",
    "rewrite_conversational_query",
    "stable_sigmoid",
    "strip_v1_header_labels",
    "tokenize_code",
]
