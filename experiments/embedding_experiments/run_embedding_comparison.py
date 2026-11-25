#!/usr/bin/env python3
"""
======================================================================
DAY 2: CONTROLLED EMBEDDING OPTIMIZATION & RETRIEVAL COMPARISON RUNNER
======================================================================
Evaluates candidate ONNX embedding models against the V0 baseline control
under strictly identical corpus chunks, metadata, ground truth, interval-overlap
relevance rules, Top-K=5, and ChromaDB squared L2 distance.

Key Scientific Controls:
- Generates source files and V0 chunks ONCE and reuses identical chunks across models
- Verifies dataset counts and .commit_sha markers before running
- Explicitly computes embeddings outside ChromaDB to isolate:
  * Cold-start model load time
  * Warm-up latency (excluded from steady-state measurements)
  * Document embedding latency vs ChromaDB insertion latency
  * Query embedding latency vs ChromaDB vector search latency
- Never reuses stale Chroma collections or cached results from previous runs
- Never calls Gemini or alters chunking/retrieval heuristics
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any

import chromadb
import numpy as np
import onnxruntime as ort
import psutil
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chunker import chunk_code
from evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    build_chunks_by_file_index,
    calc_distance_distribution,
    compute_latency_stats,
    compute_query_metrics,
    count_relevant_chunks_in_repo,
)
from repo_loader import get_source_files

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(EXP_DIR, "configs", "embedding_models.json")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
PER_MODEL_DIR = os.path.join(RESULTS_DIR, "per_model")
CHROMA_EXP_DIR = os.path.join(EXP_DIR, "chroma_exp_db")

DATASETS_DIR = os.path.join(PROJECT_ROOT, "experiments", "datasets")
CACHE_DIR = os.path.join(DATASETS_DIR, "cache")
REPOS_FILE = os.path.join(DATASETS_DIR, "repositories.json")
QUESTIONS_FILE = os.path.join(DATASETS_DIR, "evaluation_questions.json")
GROUND_TRUTH_FILE = os.path.join(DATASETS_DIR, "ground_truth.json")


def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0), 2)


def run_preflight_checks(
    repos: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate dataset counts and verify .commit_sha markers for all cached snapshots."""
    total_q = len(questions)
    answerable_q = sum(1 for g in ground_truth if g.get("answerable"))
    unanswerable_q = sum(1 for g in ground_truth if not g.get("answerable"))
    total_spans = sum(len(g.get("evidence", [])) for g in ground_truth)
    multi_ev = sum(1 for g in ground_truth if len(g.get("evidence", [])) >= 2)

    repo_counts: dict[str, int] = {}
    for q in questions:
        r = q["repository"]
        repo_counts[r] = repo_counts.get(r, 0) + 1

    errors: list[str] = []
    if total_q != 60:
        errors.append(f"Expected 60 questions, found {total_q}")
    if answerable_q != 54:
        errors.append(f"Expected 54 answerable questions, found {answerable_q}")
    if unanswerable_q != 6:
        errors.append(f"Expected 6 unanswerable questions, found {unanswerable_q}")
    if total_spans != 71:
        errors.append(f"Expected 71 evidence spans, found {total_spans}")
    if multi_ev != 17:
        errors.append(f"Expected 17 multi-evidence questions, found {multi_ev}")
    for r_name, cnt in repo_counts.items():
        if cnt != 20:
            errors.append(f"Expected 20 questions for {r_name}, found {cnt}")

    commit_verifications: dict[str, Any] = {}
    for r in repos:
        r_name = r["name"]
        alias = r_name.split("/")[-1]
        expected_sha = r["commit_sha"].strip().lower()
        repo_dir = os.path.join(CACHE_DIR, alias)
        sha_file = os.path.join(repo_dir, ".commit_sha")
        if not os.path.exists(repo_dir):
            errors.append(f"Missing cached repository at {repo_dir}")
            continue
        if not os.path.exists(sha_file):
            errors.append(f"Missing .commit_sha marker at {sha_file}")
            continue
        with open(sha_file, "r", encoding="utf-8") as f_sha:
            actual_sha = f_sha.read().strip().lower()
        if actual_sha != expected_sha:
            errors.append(f"Commit SHA mismatch for {r_name}: {actual_sha} != {expected_sha}")
        else:
            commit_verifications[r_name] = {
                "verified": True,
                "commit_sha": actual_sha,
                "marker_path": sha_file,
            }

    if errors:
        raise ValueError("Preflight checks failed:\n  - " + "\n  - ".join(errors))

    return {
        "total_questions": total_q,
        "answerable_questions": answerable_q,
        "unanswerable_questions": unanswerable_q,
        "total_evidence_spans": total_spans,
        "multi_evidence_questions": multi_ev,
    }, commit_verifications


class DirectONNXEmbedder:
    """Unified CPU ONNX Runtime embedder supporting exact V0 Chroma parity and HF ONNX models."""

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.cfg = model_cfg
        self.experiment_id = model_cfg["experiment_id"]
        self.pooling = model_cfg["pooling"]
        self.max_seq_length = int(model_cfg["max_seq_length"])
        self.padding_strategy = model_cfg["padding_strategy"]
        self.normalize_embeddings = bool(model_cfg["normalize_embeddings"])
        self.expected_dimension = int(model_cfg["expected_dimension"])
        self.document_prefix = model_cfg.get("document_prefix", "")
        self.query_prefix = model_cfg.get("query_prefix", "")

        self.model_path, self.tokenizer_path, self.model_version = self._resolve_artifacts()
        self.model_file_size_mb = round(os.path.getsize(self.model_path) / (1024.0 * 1024.0), 2)

        t_load_start = time.perf_counter()
        self.tokenizer = self._init_tokenizer()
        self.raw_len_tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.raw_len_tokenizer.no_truncation()
        self.raw_len_tokenizer.no_padding()

        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}
        self.model_load_duration_seconds = round(time.perf_counter() - t_load_start, 4)
        self.warmup_duration_seconds = 0.0

    def _resolve_artifacts(self) -> tuple[str, str, str]:
        source = self.cfg["source"]
        if source == "chroma_s3_onnx_cache":
            from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

            loader = ONNXMiniLM_L6_V2()
            loader._download_model_if_not_exists()
            base = os.path.join(str(loader.DOWNLOAD_PATH), loader.EXTRACTED_FOLDER_NAME)
            model_path = os.path.join(base, "model.onnx")
            tokenizer_path = os.path.join(base, "tokenizer.json")
            version = f"chroma_onnx_sha256:{loader._MODEL_SHA256[:12]}"
            return model_path, tokenizer_path, version

        if source == "huggingface_hub":
            hf_repo = self.cfg["hf_repo"]
            onnx_rel = self.cfg["onnx_relative_path"]
            model_path = hf_hub_download(hf_repo, onnx_rel)
            tokenizer_path = hf_hub_download(hf_repo, "tokenizer.json")
            try:
                hf_sha = HfApi().model_info(hf_repo).sha or "unknown"
            except Exception:
                hf_sha = get_file_sha256(model_path)[:12]
            return model_path, tokenizer_path, f"hf_commit:{hf_sha[:12]}"

        raise ValueError(f"Unsupported model source: {source}")

    def _init_tokenizer(self) -> Tokenizer:
        tok = Tokenizer.from_file(self.tokenizer_path)
        tok.enable_truncation(max_length=self.max_seq_length)
        if self.padding_strategy == "fixed_256":
            tok.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
        else:
            tok.no_padding()
        return tok

    def warmup(self) -> float:
        """Execute warm-up inference outside steady-state benchmark timers."""
        t0 = time.perf_counter()
        warmup_docs = [
            "def initialize_router(app, prefix=''):\n    return app",
            "class RequestContext:\n    def __enter__(self):\n        return self",
            "template <typename T> void format_value(T&& val) {}",
        ]
        warmup_queries = [
            "How does RequestContext initialize request locals?",
            "Where is format_value defined?",
        ]
        doc_vecs, _ = self.encode_texts(
            warmup_docs, prefix=self.document_prefix, batch_size=32, track_truncation=False
        )
        q_vecs, _ = self.encode_texts(
            warmup_queries, prefix=self.query_prefix, batch_size=32, track_truncation=False
        )
        assert doc_vecs.shape == (len(warmup_docs), self.expected_dimension)
        assert q_vecs.shape == (len(warmup_queries), self.expected_dimension)
        assert np.all(np.isfinite(doc_vecs)) and np.all(np.isfinite(q_vecs))
        self.warmup_duration_seconds = round(time.perf_counter() - t0, 4)
        return self.warmup_duration_seconds

    def _pool_and_normalize(
        self, last_hidden_state: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        if self.pooling == "cls":
            pooled = last_hidden_state[:, 0]
        elif self.pooling == "mean":
            mask_exp = np.broadcast_to(
                np.expand_dims(attention_mask, -1), last_hidden_state.shape
            )
            pooled = np.sum(last_hidden_state * mask_exp, axis=1) / np.clip(
                mask_exp.sum(axis=1), a_min=1e-9, a_max=None
            )
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling}")

        if self.normalize_embeddings:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            pooled = pooled / norms

        return pooled.astype(np.float32)

    def _run_batch(self, encoded_batch: list[Any]) -> np.ndarray:
        if self.padding_strategy == "fixed_256":
            input_ids = np.array([e.ids for e in encoded_batch], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded_batch], dtype=np.int64)
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
        else:
            batch_max_len = max(len(e.ids) for e in encoded_batch)
            b_size = len(encoded_batch)
            input_ids = np.zeros((b_size, batch_max_len), dtype=np.int64)
            attention_mask = np.zeros((b_size, batch_max_len), dtype=np.int64)
            token_type_ids = np.zeros((b_size, batch_max_len), dtype=np.int64)
            for row, enc in enumerate(encoded_batch):
                seq_len = len(enc.ids)
                input_ids[row, :seq_len] = enc.ids
                attention_mask[row, :seq_len] = enc.attention_mask
                token_type_ids[row, :seq_len] = enc.type_ids

        ort_inputs: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self.input_names:
            ort_inputs["token_type_ids"] = token_type_ids

        last_hidden = self.session.run(None, ort_inputs)[0]
        return self._pool_and_normalize(last_hidden, attention_mask)

    def encode_texts(
        self,
        texts: list[str],
        prefix: str = "",
        batch_size: int = 32,
        track_truncation: bool = True,
        rss_tracker: list[float] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Encode texts in batches, returning float32 vectors plus norm/timing stats."""
        prefixed_texts = [f"{prefix}{t}" if prefix else t for t in texts]

        truncated_count = 0
        if track_truncation:
            for pt in prefixed_texts:
                if len(self.raw_len_tokenizer.encode(pt).ids) > self.max_seq_length:
                    truncated_count += 1

        t_start = time.perf_counter()
        encoded = [self.tokenizer.encode(pt) for pt in prefixed_texts]

        if self.padding_strategy == "dynamic_batch" and len(encoded) > batch_size:
            order = sorted(range(len(encoded)), key=lambda idx: len(encoded[idx].ids))
        else:
            order = list(range(len(encoded)))

        all_vectors = np.zeros((len(encoded), self.expected_dimension), dtype=np.float32)
        batches_count = 0

        for i in range(0, len(order), batch_size):
            batch_indices = order[i : i + batch_size]
            batch_encoded = [encoded[j] for j in batch_indices]
            batch_vecs = self._run_batch(batch_encoded)
            for row_i, orig_idx in enumerate(batch_indices):
                all_vectors[orig_idx] = batch_vecs[row_i]
            batches_count += 1
            if rss_tracker is not None and batches_count % 10 == 0:
                rss_tracker.append(get_rss_mb())

        duration_s = time.perf_counter() - t_start

        if not np.all(np.isfinite(all_vectors)):
            raise ValueError(f"Non-finite embedding produced by {self.experiment_id}")
        if all_vectors.shape[1] != self.expected_dimension:
            raise ValueError(
                f"Dimension mismatch for {self.experiment_id}: {all_vectors.shape[1]} != {self.expected_dimension}"
            )

        norms = np.linalg.norm(all_vectors, axis=1)
        stats = {
            "duration_seconds": round(duration_s, 4),
            "texts_count": len(texts),
            "batches_count": batches_count,
            "avg_ms_per_item": round((duration_s * 1000.0) / max(1, len(texts)), 3),
            "dimension": int(all_vectors.shape[1]),
            "truncated_count": truncated_count if track_truncation else None,
            "truncated_pct": round((truncated_count / len(texts)) * 100.0, 2) if (track_truncation and texts) else None,
            "norm_stats": {
                "min": round(float(np.min(norms)), 6),
                "mean": round(float(np.mean(norms)), 6),
                "max": round(float(np.max(norms)), 6),
            },
        }
        return all_vectors, stats

    def encode_single_query(self, query_text: str) -> tuple[np.ndarray, float, float]:
        """Encode one raw query using the configured query prefix and measure pure inference time."""
        prefixed = f"{self.query_prefix}{query_text}" if self.query_prefix else query_text
        t0 = time.perf_counter()
        enc = self.tokenizer.encode(prefixed)
        vec = self._run_batch([enc])[0]
        dur_s = time.perf_counter() - t0
        if not np.all(np.isfinite(vec)) or vec.shape[0] != self.expected_dimension:
            raise ValueError(f"Invalid query vector for {self.experiment_id}")
        norm_val = float(np.linalg.norm(vec))
        return vec, dur_s, norm_val


def make_collection_name(exp_id: str, repo_alias: str, run_short: str) -> str:
    """Create a fresh, sanitized <=63-char collection name isolated per experiment and run."""
    raw = f"d2_{exp_id}_{repo_alias}_{run_short}"
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in raw)[:63]
    if len(safe) < 3:
        safe = safe.ljust(3, "_")
    return safe


def compute_breakdowns(
    answerable_records: list[dict[str, Any]],
    repos_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute repository, category, query-style, and evidence-complexity breakdowns."""
    by_repo = {}
    for r_info in repos_data:
        r_name = r_info["name"]
        subset = [r for r in answerable_records if r["repository"] == r_name]
        by_repo[r_name] = aggregate_retrieval_metrics(subset)

    by_category = {}
    for cat in sorted({r["category"] for r in answerable_records}):
        subset = [r for r in answerable_records if r["category"] == cat]
        by_category[cat] = aggregate_retrieval_metrics(subset)

    by_style = {}
    for style in sorted({r["query_style"] for r in answerable_records}):
        subset = [r for r in answerable_records if r["query_style"] == style]
        by_style[style] = aggregate_retrieval_metrics(subset)

    single_ev = [r for r in answerable_records if r["total_evidence_spans_count"] == 1]
    multi_ev = [r for r in answerable_records if r["total_evidence_spans_count"] >= 2]
    by_evidence = {
        "single_evidence_queries": aggregate_retrieval_metrics(single_ev),
        "multi_evidence_queries": aggregate_retrieval_metrics(multi_ev),
    }

    return {
        "by_repository": by_repo,
        "by_category": by_category,
        "by_query_style": by_style,
        "by_evidence_spans": by_evidence,
    }


def run_single_model_experiment(
    model_cfg: dict[str, Any],
    suite_cfg: dict[str, Any],
    repos_data: list[dict[str, Any]],
    questions_data: list[dict[str, Any]],
    gt_map: dict[str, dict[str, Any]],
    corpus_chunks: dict[str, list[dict[str, Any]]],
    corpus_chunks_by_file: dict[str, dict[str, list[tuple[int, int]]]],
    dataset_fingerprints: dict[str, str],
    commit_verifications: dict[str, Any],
    chroma_client: chromadb.ClientAPI,
    run_id: str,
    run_short: str,
    env_metadata: dict[str, Any],
    shared_doc_cache: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Execute fresh indexing and 60-question evaluation for one embedding configuration."""
    exp_id = model_cfg["experiment_id"]
    top_k = int(suite_cfg["top_k"])
    batch_size = int(suite_cfg["batch_size"])
    chroma_batch_size = int(suite_cfg["chroma_insert_batch_size"])

    print("\n" + "=" * 78, flush=True)
    print(f"MODEL EXPERIMENT: {exp_id} ({model_cfg['model_name']})", flush=True)
    print("=" * 78, flush=True)

    rss_before_load = get_rss_mb()
    embedder = DirectONNXEmbedder(model_cfg)
    rss_after_load = get_rss_mb()
    rss_samples = [rss_after_load]
    warmup_dur = embedder.warmup()
    rss_samples.append(get_rss_mb())

    print(
        f"[{exp_id}] Loaded {embedder.model_path} ({embedder.model_file_size_mb} MB) in "
        f"{embedder.model_load_duration_seconds}s | Warm-up: {warmup_dur}s | "
        f"Pooling: {embedder.pooling} | MaxLen: {embedder.max_seq_length} | Padding: {embedder.padding_strategy}",
        flush=True,
    )

    exp_start = time.perf_counter()

    # Stage A & B: Document embedding (fresh or explicit in-run invariant passage vectors for BGE instruction ablation)
    repo_collections: dict[str, Any] = {}
    collection_names_map: dict[str, str] = {}
    indexing_by_repo: dict[str, Any] = {}
    generated_doc_vectors: dict[str, np.ndarray] = {}

    total_doc_embed_s = 0.0
    total_chroma_insert_s = 0.0
    total_doc_batches = 0
    total_truncated_chunks = 0
    total_chunks_count = 0
    all_norm_mins: list[float] = []
    all_norm_maxs: list[float] = []
    weighted_norm_sum = 0.0

    reused_doc_source: str | None = None
    if (
        shared_doc_cache is not None
        and model_cfg["group"] == "E1"
        and exp_id != "E1a_bge_small_en_v15_no_prefix"
        and "E1a_bge_small_en_v15_no_prefix" in shared_doc_cache
    ):
        reused_doc_source = "E1a_bge_small_en_v15_no_prefix"
        print(
            f"[{exp_id}] Query-instruction ablation shares BGE's identical zero-prefix passage vectors "
            f"freshly generated in this run by {reused_doc_source}; inserting into fresh isolated collections.",
            flush=True,
        )

    for repo_info in repos_data:
        repo_name = repo_info["name"]
        alias = repo_name.split("/")[-1]
        chunks = corpus_chunks[repo_name]
        docs = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"chunk_{j}" for j in range(len(chunks))]

        col_name = make_collection_name(exp_id, alias, run_short)
        collection_names_map[repo_name] = col_name

        # Guarantee clean fresh collection
        try:
            chroma_client.delete_collection(name=col_name)
        except Exception:
            pass

        collection = chroma_client.create_collection(
            name=col_name,
            metadata={"hnsw:space": suite_cfg["distance_metric"]},
        )

        if reused_doc_source is not None:
            doc_vecs = shared_doc_cache[reused_doc_source][repo_name]
            emb_stats = {
                "duration_seconds": 0.0,
                "texts_count": len(docs),
                "batches_count": 0,
                "avg_ms_per_item": 0.0,
                "dimension": int(doc_vecs.shape[1]),
                "truncated_count": None,
                "truncated_pct": None,
                "norm_stats": {
                    "min": round(float(np.min(np.linalg.norm(doc_vecs, axis=1))), 6),
                    "mean": round(float(np.mean(np.linalg.norm(doc_vecs, axis=1))), 6),
                    "max": round(float(np.max(np.linalg.norm(doc_vecs, axis=1))), 6),
                },
            }
        else:
            print(f"[{exp_id}][{repo_name}] Embedding {len(docs)} chunks...", flush=True)
            doc_vecs, emb_stats = embedder.encode_texts(
                docs,
                prefix=embedder.document_prefix,
                batch_size=batch_size,
                track_truncation=True,
                rss_tracker=rss_samples,
            )

        generated_doc_vectors[repo_name] = doc_vecs

        # Chroma insertion using precomputed embeddings ONLY
        t_ins_start = time.perf_counter()
        for start in range(0, len(chunks), chroma_batch_size):
            end = start + chroma_batch_size
            collection.add(
                embeddings=doc_vecs[start:end].tolist(),
                documents=docs[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
        chroma_insert_s = round(time.perf_counter() - t_ins_start, 4)

        repo_collections[repo_name] = collection
        total_doc_embed_s += emb_stats["duration_seconds"]
        total_chroma_insert_s += chroma_insert_s
        total_doc_batches += emb_stats["batches_count"]
        total_chunks_count += len(chunks)
        if emb_stats["truncated_count"] is not None:
            total_truncated_chunks += emb_stats["truncated_count"]
        all_norm_mins.append(emb_stats["norm_stats"]["min"])
        all_norm_maxs.append(emb_stats["norm_stats"]["max"])
        weighted_norm_sum += emb_stats["norm_stats"]["mean"] * len(chunks)

        indexing_by_repo[repo_name] = {
            "repository": repo_name,
            "commit_sha": repo_info["commit_sha"],
            "collection_name": col_name,
            "chunks_count": len(chunks),
            "batches_count": emb_stats["batches_count"],
            "document_embedding_duration_seconds": emb_stats["duration_seconds"],
            "avg_embedding_ms_per_chunk": emb_stats["avg_ms_per_item"],
            "truncated_chunks_count": emb_stats["truncated_count"],
            "truncated_chunks_pct": emb_stats["truncated_pct"],
            "doc_vector_norm_stats": emb_stats["norm_stats"],
            "chroma_insertion_duration_seconds": chroma_insert_s,
            "total_indexing_duration_seconds": round(emb_stats["duration_seconds"] + chroma_insert_s, 4),
            "reused_same_run_doc_vectors_from": reused_doc_source,
        }
        print(
            f"[{exp_id}][{repo_name}] Indexed {len(chunks)} chunks | "
            f"Embed: {emb_stats['duration_seconds']:.2f}s | Chroma add: {chroma_insert_s:.2f}s | "
            f"Norms: [{emb_stats['norm_stats']['min']:.6f}, {emb_stats['norm_stats']['max']:.6f}]",
            flush=True,
        )

    # Query Evaluation across all 60 questions
    evaluation_records: list[dict[str, Any]] = []
    failed_queries = 0
    query_norms: list[float] = []

    print(f"[{exp_id}] Evaluating 60 queries (Top-K={top_k}, distance=l2)...", flush=True)
    for idx, q_item in enumerate(questions_data, start=1):
        qid = q_item["question_id"]
        repo = q_item["repository"]
        category = q_item["category"]
        query_style = q_item.get("query_style", "identifier")
        raw_question = q_item["question"]
        is_answerable = category != "unanswerable"
        gt_entry = gt_map[qid]
        collection = repo_collections[repo]

        query_failed = False
        error_msg: str | None = None
        formatted_retrieved: list[dict[str, Any]] = []
        q_emb_dur_s = 0.0
        vec_search_dur_s = 0.0

        try:
            q_vec, q_emb_dur_s, q_norm = embedder.encode_single_query(raw_question)
            query_norms.append(q_norm)

            t_search_start = time.perf_counter()
            raw_res = collection.query(query_embeddings=[q_vec.tolist()], n_results=top_k)
            vec_search_dur_s = time.perf_counter() - t_search_start

            docs_list = raw_res.get("documents", [[]])[0]
            metas_list = raw_res.get("metadatas", [[]])[0]
            ids_list = raw_res.get("ids", [[]])[0]
            dists_list = raw_res.get("distances", [[]])[0]

            for rank_i in range(len(docs_list)):
                ch_meta = metas_list[rank_i]
                dist_val = dists_list[rank_i] if dists_list else None
                formatted_retrieved.append(
                    {
                        "rank": rank_i + 1,
                        "chunk_id": ids_list[rank_i] if ids_list else f"chunk_{rank_i}",
                        "l2_distance": round(float(dist_val), 6) if dist_val is not None else None,
                        "file_path": ch_meta.get("file_path"),
                        "start_line": ch_meta.get("start_line"),
                        "end_line": ch_meta.get("end_line"),
                        "chunk_type": ch_meta.get("chunk_type"),
                        "content": docs_list[rank_i],
                    }
                )
        except Exception as exc:
            query_failed = True
            failed_queries += 1
            error_msg = f"{type(exc).__name__}: {exc}"

        rss_samples.append(get_rss_mb())
        total_ret_dur_s = q_emb_dur_s + vec_search_dur_s
        q_emb_ms = round(q_emb_dur_s * 1000.0, 2)
        vec_search_ms = round(vec_search_dur_s * 1000.0, 2)
        total_ret_ms = round(total_ret_dur_s * 1000.0, 2)

        l2_dists = [c["l2_distance"] for c in formatted_retrieved if c.get("l2_distance") is not None]
        top1_l2 = l2_dists[0] if l2_dists else None
        mean_top5_l2 = round(sum(l2_dists) / len(l2_dists), 6) if l2_dists else None

        if is_answerable:
            gt_spans = gt_entry.get("evidence", [])
            total_rel_repo = count_relevant_chunks_in_repo(
                evidence_spans=gt_spans,
                chunks_by_file=corpus_chunks_by_file[repo],
            )
            qm = compute_query_metrics(
                retrieved_chunks=formatted_retrieved,
                evidence_spans=gt_spans,
                total_relevant_chunks_in_repo=total_rel_repo,
                top_k=top_k,
                failed=query_failed,
            )
            record = {
                "question_id": qid,
                "repository": repo,
                "category": category,
                "query_style": query_style,
                "answerable": True,
                "raw_question": raw_question,
                "conversation_id": q_item.get("conversation_id"),
                "turn_number": q_item.get("turn_number"),
                "query_embedding_latency_ms": q_emb_ms,
                "vector_search_latency_ms": vec_search_ms,
                "retrieval_latency_ms": total_ret_ms,
                "query_embedding_duration_seconds": round(q_emb_dur_s, 6),
                "vector_search_duration_seconds": round(vec_search_dur_s, 6),
                "total_retrieval_duration_seconds": round(total_ret_dur_s, 6),
                "n_results_returned": len(formatted_retrieved),
                "top1_l2_distance": top1_l2,
                "mean_top5_l2_distance": mean_top5_l2,
                "per_chunk_relevance": qm["per_chunk_relevance"],
                "first_relevant_rank": qm["first_relevant_rank"],
                "matched_evidence_spans_count": qm["matched_evidence_spans_count"],
                "total_evidence_spans_count": qm["total_evidence_spans_count"],
                "total_relevant_chunks_in_repo": qm["total_relevant_chunks_in_repo"],
                "dcg_at_5": qm["dcg_at_5"],
                "failed": query_failed,
                "metrics": qm["metrics"],
                "retrieved_chunks": formatted_retrieved,
                "error": error_msg,
            }
        else:
            record = {
                "question_id": qid,
                "repository": repo,
                "category": category,
                "query_style": query_style,
                "answerable": False,
                "raw_question": raw_question,
                "conversation_id": q_item.get("conversation_id"),
                "turn_number": q_item.get("turn_number"),
                "query_embedding_latency_ms": q_emb_ms,
                "vector_search_latency_ms": vec_search_ms,
                "retrieval_latency_ms": total_ret_ms,
                "query_embedding_duration_seconds": round(q_emb_dur_s, 6),
                "vector_search_duration_seconds": round(vec_search_dur_s, 6),
                "total_retrieval_duration_seconds": round(total_ret_dur_s, 6),
                "n_results_returned": len(formatted_retrieved),
                "top1_l2_distance": top1_l2,
                "mean_top5_l2_distance": mean_top5_l2,
                "per_chunk_relevance": [False] * len(formatted_retrieved),
                "first_relevant_rank": None,
                "justification": gt_entry.get("justification"),
                "failed": query_failed,
                "retrieved_chunks": formatted_retrieved,
                "error": error_msg,
            }
        evaluation_records.append(record)

    exp_total_dur_s = round(time.perf_counter() - exp_start, 4)

    answerable_records = [r for r in evaluation_records if r["answerable"]]
    unanswerable_records = [r for r in evaluation_records if not r["answerable"]]

    overall_metrics = aggregate_retrieval_metrics(answerable_records)
    breakdowns = compute_breakdowns(answerable_records, repos_data)

    # Distance analysis
    ans_top1 = [r["top1_l2_distance"] for r in answerable_records if r["top1_l2_distance"] is not None]
    unans_top1 = [r["top1_l2_distance"] for r in unanswerable_records if r["top1_l2_distance"] is not None]
    ans_mean5 = [r["mean_top5_l2_distance"] for r in answerable_records if r["mean_top5_l2_distance"] is not None]
    unans_mean5 = [r["mean_top5_l2_distance"] for r in unanswerable_records if r["mean_top5_l2_distance"] is not None]

    distance_analysis = {
        "distance_metric": "l2 (squared Euclidean distance on unit-normalized vectors)",
        "answerable_questions_top1_l2": calc_distance_distribution(ans_top1),
        "unanswerable_questions_top1_l2": calc_distance_distribution(unans_top1),
        "answerable_questions_mean_top5_l2": calc_distance_distribution(ans_mean5),
        "unanswerable_questions_mean_top5_l2": calc_distance_distribution(unans_mean5),
    }

    # Separated query latency summaries
    ans_total_lat = [r["retrieval_latency_ms"] for r in answerable_records]
    all_total_lat = [r["retrieval_latency_ms"] for r in evaluation_records]
    ans_emb_lat = [r["query_embedding_latency_ms"] for r in answerable_records]
    all_emb_lat = [r["query_embedding_latency_ms"] for r in evaluation_records]
    ans_vec_lat = [r["vector_search_latency_ms"] for r in answerable_records]
    all_vec_lat = [r["vector_search_latency_ms"] for r in evaluation_records]

    latency_summary = {
        "p95_convention": "nearest rank index math.ceil(0.95 * n) - 1",
        "answerable_queries_latency_ms": compute_latency_stats(ans_total_lat),
        "all_queries_latency_ms": compute_latency_stats(all_total_lat),
        "query_embedding_only_ms": {
            "answerable_queries": compute_latency_stats(ans_emb_lat),
            "all_queries": compute_latency_stats(all_emb_lat),
        },
        "vector_search_only_ms": {
            "answerable_queries": compute_latency_stats(ans_vec_lat),
            "all_queries": compute_latency_stats(all_vec_lat),
        },
    }

    performance_metrics = {
        "model_file_size_mb": embedder.model_file_size_mb,
        "cold_start_model_load_seconds": embedder.model_load_duration_seconds,
        "warmup_duration_seconds_excluded": embedder.warmup_duration_seconds,
        "document_embedding_duration_seconds": round(total_doc_embed_s, 4),
        "chroma_insertion_duration_seconds": round(total_chroma_insert_s, 4),
        "total_indexing_duration_seconds": round(total_doc_embed_s + total_chroma_insert_s, 4),
        "total_chunks_indexed": total_chunks_count,
        "total_embedding_batches": total_doc_batches,
        "avg_doc_embedding_ms_per_chunk": round((total_doc_embed_s * 1000.0) / total_chunks_count, 3) if reused_doc_source is None else None,
        "truncated_chunks_count": total_truncated_chunks if reused_doc_source is None else None,
        "truncated_chunks_pct": round((total_truncated_chunks / total_chunks_count) * 100.0, 2) if (reused_doc_source is None and total_chunks_count) else None,
        "total_query_embedding_duration_seconds": round(sum(r["query_embedding_duration_seconds"] for r in evaluation_records), 4),
        "total_vector_search_duration_seconds": round(sum(r["vector_search_duration_seconds"] for r in evaluation_records), 4),
        "total_retrieval_duration_seconds": round(sum(r["total_retrieval_duration_seconds"] for r in evaluation_records), 4),
        "total_experiment_execution_seconds": exp_total_dur_s,
        "memory_rss_mb": {
            "rss_before_load_mb": rss_before_load,
            "rss_after_load_mb": rss_after_load,
            "model_load_rss_delta_mb": round(rss_after_load - rss_before_load, 2),
            "max_sampled_rss_during_eval_mb": round(max(rss_samples), 2),
        },
        "reused_same_run_doc_vectors_from": reused_doc_source,
    }

    summary_payload = {
        "experiment_metadata": {
            "experiment_id": exp_id,
            "group": model_cfg["group"],
            "display_name": model_cfg["display_name"],
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "VALID_COMPLETED" if failed_queries == 0 else "INVALID_FAILED",
            "model_identifier": model_cfg["model_name"],
            "model_version": embedder.model_version,
            "backend": model_cfg["backend"],
            "embedding_dimension": embedder.expected_dimension,
            "pooling": embedder.pooling,
            "max_seq_length": embedder.max_seq_length,
            "padding_strategy": embedder.padding_strategy,
            "normalize_embeddings": embedder.normalize_embeddings,
            "document_prefix": embedder.document_prefix,
            "query_prefix": embedder.query_prefix,
            "doc_vector_norm_verification": {
                "min": round(min(all_norm_mins), 6),
                "mean": round(weighted_norm_sum / max(1, total_chunks_count), 6),
                "max": round(max(all_norm_maxs), 6),
            },
            "query_vector_norm_verification": {
                "min": round(min(query_norms), 6) if query_norms else None,
                "mean": round(sum(query_norms) / len(query_norms), 6) if query_norms else None,
                "max": round(max(query_norms), 6) if query_norms else None,
            },
            "top_k": top_k,
            "distance_metric": "l2 (squared Euclidean distance)",
            "total_chunks": total_chunks_count,
            "total_queries": len(evaluation_records),
            "answerable_queries": len(answerable_records),
            "unanswerable_queries": len(unanswerable_records),
            "successful_queries": len(evaluation_records) - failed_queries,
            "failed_queries": failed_queries,
            "dedicated_collections": collection_names_map,
            "pinned_commit_verifications": commit_verifications,
            "dataset_fingerprints": dataset_fingerprints,
            "environment": env_metadata,
        },
        "indexing_summary_by_repo": indexing_by_repo,
        "performance_metrics": performance_metrics,
        "overall_retrieval_metrics": overall_metrics,
        "latency_summary": latency_summary,
        "breakdowns": breakdowns,
        "top1_l2_distance_analysis": distance_analysis,
    }

    print(
        f"[{exp_id}] DONE | Hit@1={overall_metrics['hit_rate_at_1']:.4f} | "
        f"Hit@5={overall_metrics['hit_rate_at_5']:.4f} | MRR@5={overall_metrics['mrr_at_5']:.4f} | "
        f"NDCG@5={overall_metrics['ndcg_at_5']:.4f} | SpanRec@5={overall_metrics['span_recall_at_5']:.4f} | "
        f"Query P50={latency_summary['answerable_queries_latency_ms']['median_p50']}ms | Failures={failed_queries}",
        flush=True,
    )

    del embedder
    gc.collect()

    return summary_payload, evaluation_records, generated_doc_vectors


def build_error_analysis(
    all_per_model_records: dict[str, list[dict[str, Any]]],
    baseline_id: str = "E0_v0_minilm_l6_v2",
) -> dict[str, Any]:
    """Compare solved/missed/regressed answerable queries across all models vs baseline."""
    if baseline_id not in all_per_model_records:
        return {}

    base_map = {r["question_id"]: r for r in all_per_model_records[baseline_id] if r["answerable"]}
    answerable_qids = list(base_map.keys())

    transitions: dict[str, Any] = {}
    for exp_id, recs in all_per_model_records.items():
        if exp_id == baseline_id:
            continue
        cur_map = {r["question_id"]: r for r in recs if r["answerable"]}
        solved_over_v0 = []
        regressed_vs_v0 = []
        both_hit = []
        both_miss = []

        for qid in answerable_qids:
            v0_r = base_map[qid]
            m_r = cur_map[qid]
            v0_hit = v0_r["metrics"]["hit_rate_at_5"] == 1.0
            m_hit = m_r["metrics"]["hit_rate_at_5"] == 1.0
            entry = {
                "question_id": qid,
                "repository": m_r["repository"],
                "category": m_r["category"],
                "query_style": m_r["query_style"],
                "question": m_r["raw_question"],
                "v0_first_rank": v0_r["first_relevant_rank"],
                "model_first_rank": m_r["first_relevant_rank"],
                "v0_span_recall": v0_r["metrics"]["span_recall_at_5"],
                "model_span_recall": m_r["metrics"]["span_recall_at_5"],
            }
            if not v0_hit and m_hit:
                solved_over_v0.append(entry)
            elif v0_hit and not m_hit:
                regressed_vs_v0.append(entry)
            elif v0_hit and m_hit:
                both_hit.append(entry)
            else:
                both_miss.append(entry)

        transitions[exp_id] = {
            "solved_by_model_missed_by_v0_count": len(solved_over_v0),
            "solved_by_model_missed_by_v0": solved_over_v0,
            "regressed_by_model_solved_by_v0_count": len(regressed_vs_v0),
            "regressed_by_model_solved_by_v0": regressed_vs_v0,
            "both_hit_count": len(both_hit),
            "both_miss_count": len(both_miss),
        }

    missed_by_all = []
    for qid in answerable_qids:
        if all(
            next(r for r in recs if r["question_id"] == qid)["metrics"]["hit_rate_at_5"] == 0.0
            for recs in all_per_model_records.values()
        ):
            r0 = base_map[qid]
            missed_by_all.append(
                {
                    "question_id": qid,
                    "repository": r0["repository"],
                    "category": r0["category"],
                    "query_style": r0["query_style"],
                    "question": r0["raw_question"],
                }
            )

    return {
        "baseline_experiment_id": baseline_id,
        "transitions_vs_baseline": transitions,
        "missed_by_all_models_count": len(missed_by_all),
        "missed_by_all_models": missed_by_all,
    }


def main(selected_ids: list[str] | None = None) -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PER_MODEL_DIR, exist_ok=True)
    os.makedirs(CHROMA_EXP_DIR, exist_ok=True)

    run_start = time.perf_counter()
    now_utc = datetime.now(timezone.utc)
    run_id = f"d2_emb_{now_utc.strftime('%Y%m%d_%H%M%S')}"
    run_short = now_utc.strftime("%H%M%S")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        suite_cfg = json.load(f)
    with open(REPOS_FILE, "r", encoding="utf-8") as f:
        repos_data = json.load(f)
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    preflight_stats, commit_verifications = run_preflight_checks(
        repos_data, questions_data, ground_truth_data
    )
    gt_map = {item["question_id"]: item for item in ground_truth_data}

    dataset_fingerprints = {
        "repositories.json_sha256": get_file_sha256(REPOS_FILE),
        "evaluation_questions.json_sha256": get_file_sha256(QUESTIONS_FILE),
        "ground_truth.json_sha256": get_file_sha256(GROUND_TRUTH_FILE),
        "embedding_models.json_sha256": get_file_sha256(CONFIG_FILE),
    }

    env_metadata = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "processor": platform.processor(),
        "cpu_logical_cores": os.cpu_count(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024.0**3), 2),
        "ort_providers": ort.get_available_providers(),
        "packages": {
            "chromadb": chromadb.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "psutil": psutil.__version__,
        },
        "chunker_configuration": {
            "max_lines": 100,
            "sliding_window_size": 50,
            "sliding_window_overlap": 10,
            "parsers": "Python AST with regex and sliding-window fallback",
        },
    }

    # Build identical V0 chunk corpus ONCE
    print("Building fixed V0 chunk corpus once across pinned repositories...", flush=True)
    corpus_chunks: dict[str, list[dict[str, Any]]] = {}
    corpus_chunks_by_file: dict[str, dict[str, list[tuple[int, int]]]] = {}
    t_corpus_extract = 0.0
    t_corpus_chunk = 0.0
    for r_info in repos_data:
        r_name = r_info["name"]
        alias = r_name.split("/")[-1]
        repo_path = os.path.join(CACHE_DIR, alias)

        t0 = time.perf_counter()
        source_files = get_source_files(repo_path)
        t_corpus_extract += time.perf_counter() - t0

        t1 = time.perf_counter()
        chunks: list[dict[str, Any]] = []
        for sf in source_files:
            chunks.extend(chunk_code(sf))
        t_corpus_chunk += time.perf_counter() - t1

        corpus_chunks[r_name] = chunks
        corpus_chunks_by_file[r_name] = build_chunks_by_file_index(chunks)
        print(f"  Corpus [{r_name}]: {len(source_files)} files -> {len(chunks)} chunks", flush=True)

    # Fresh Chroma persistent client isolated strictly for Day 2 experiments
    chroma_client = chromadb.PersistentClient(path=CHROMA_EXP_DIR)

    models_to_run = suite_cfg["models"]
    if selected_ids:
        models_to_run = [m for m in models_to_run if m["experiment_id"] in selected_ids]
        if not models_to_run:
            raise ValueError(f"No models matched filter: {selected_ids}")

    all_summaries: dict[str, dict[str, Any]] = {}
    all_records: dict[str, list[dict[str, Any]]] = {}
    shared_doc_cache: dict[str, dict[str, np.ndarray]] = {}
    any_failures = False

    for m_cfg in models_to_run:
        exp_id = m_cfg["experiment_id"]
        summary_data, eval_records, doc_vecs = run_single_model_experiment(
            model_cfg=m_cfg,
            suite_cfg=suite_cfg,
            repos_data=repos_data,
            questions_data=questions_data,
            gt_map=gt_map,
            corpus_chunks=corpus_chunks,
            corpus_chunks_by_file=corpus_chunks_by_file,
            dataset_fingerprints=dataset_fingerprints,
            commit_verifications=commit_verifications,
            chroma_client=chroma_client,
            run_id=run_id,
            run_short=run_short,
            env_metadata=env_metadata,
            shared_doc_cache=shared_doc_cache,
        )
        if exp_id == "E1a_bge_small_en_v15_no_prefix":
            shared_doc_cache[exp_id] = doc_vecs

        all_summaries[exp_id] = summary_data
        all_records[exp_id] = eval_records
        if summary_data["experiment_metadata"]["failed_queries"] > 0:
            any_failures = True

        # Persist incrementally so no experiment work is lost
        with open(os.path.join(PER_MODEL_DIR, f"{exp_id}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        with open(os.path.join(PER_MODEL_DIR, f"{exp_id}_evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(eval_records, f, indent=2)

    error_analysis = build_error_analysis(all_records)

    comparison_table = []
    for exp_id, s in all_summaries.items():
        om = s["overall_retrieval_metrics"]
        pm = s["performance_metrics"]
        ls = s["latency_summary"]["answerable_queries_latency_ms"]
        comparison_table.append(
            {
                "experiment_id": exp_id,
                "model_name": s["experiment_metadata"]["model_identifier"],
                "dimension": s["experiment_metadata"]["embedding_dimension"],
                "pooling": s["experiment_metadata"]["pooling"],
                "max_seq_length": s["experiment_metadata"]["max_seq_length"],
                "query_prefix": s["experiment_metadata"]["query_prefix"],
                "document_prefix": s["experiment_metadata"]["document_prefix"],
                "hit_rate_at_1": om["hit_rate_at_1"],
                "hit_rate_at_3": om["hit_rate_at_3"],
                "hit_rate_at_5": om["hit_rate_at_5"],
                "precision_at_5": om["precision_at_5"],
                "span_recall_at_5": om["span_recall_at_5"],
                "mrr_at_5": om["mrr_at_5"],
                "ndcg_at_5": om["ndcg_at_5"],
                "query_p50_ms": ls["median_p50"],
                "query_p95_ms": ls["p95"],
                "doc_embedding_s": pm["document_embedding_duration_seconds"],
                "chroma_insert_s": pm["chroma_insertion_duration_seconds"],
                "model_size_mb": pm["model_file_size_mb"],
                "failed_queries": s["experiment_metadata"]["failed_queries"],
            }
        )

    total_suite_dur = round(time.perf_counter() - run_start, 2)

    summary_out = {
        "suite_metadata": {
            "run_id": run_id,
            "timestamp_utc": now_utc.isoformat(),
            "status": "VALID_COMPLETED" if not any_failures else "INVALID_FAILED",
            "total_suite_duration_seconds": total_suite_dur,
            "corpus_extraction_duration_seconds": round(t_corpus_extract, 4),
            "corpus_chunking_duration_seconds": round(t_corpus_chunk, 4),
            "preflight_stats": preflight_stats,
            "dataset_fingerprints": dataset_fingerprints,
            "pinned_commit_verifications": commit_verifications,
            "environment": env_metadata,
        },
        "comparison_table": comparison_table,
        "models": all_summaries,
        "error_analysis": error_analysis,
    }

    comparison_out = {
        "suite_metadata": summary_out["suite_metadata"],
        "comparison_table": comparison_table,
        "error_analysis": error_analysis,
        "per_model_results": {
            exp_id: {
                "summary": all_summaries[exp_id],
                "query_records": all_records[exp_id],
            }
            for exp_id in all_summaries
        },
    }

    with open(os.path.join(RESULTS_DIR, "embedding_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "embedding_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_out, f, indent=2)

    print("\n" + "=" * 78, flush=True)
    print(f"DAY 2 EMBEDDING COMPARISON COMPLETE in {total_suite_dur:.2f}s", flush=True)
    print("=" * 78, flush=True)
    for row in comparison_table:
        print(
            f"{row['experiment_id']:<34} | Hit@1={row['hit_rate_at_1']:.4f} | Hit@5={row['hit_rate_at_5']:.4f} | "
            f"MRR@5={row['mrr_at_5']:.4f} | NDCG@5={row['ndcg_at_5']:.4f} | P50={row['query_p50_ms']}ms",
            flush=True,
        )

    return 1 if any_failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 2 Controlled Embedding Comparison Runner")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional list of experiment_id values to run (defaults to all in embedding_models.json)",
    )
    args = parser.parse_args()
    sys.exit(main(selected_ids=args.models))
