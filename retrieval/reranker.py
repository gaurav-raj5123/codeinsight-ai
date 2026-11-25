"""ONNX-Only Cross-Encoder Reranker for CodeInsight AI Assistant (Day 5).

Executes CPU cross-encoder inference cleanly via `onnxruntime`, `tokenizers`,
and `huggingface_hub` (zero PyTorch or sentence-transformers dependencies).

Pinned default model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Artifact: onnx/model.onnx (86.80 MB, Apache-2.0)
  - Architecture: BertForSequenceClassification (max_position_embeddings=512)
  - Inputs: input_ids, attention_mask, token_type_ids
  - Output: raw unnormalized relevance logits of shape (batch_size, 1)

Important score-semantics note:
  Raw logits are sorted descending. Any sigmoid(logit) score is an uncalibrated
  exploratory monotonic transform, NOT a calibrated relevance probability.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Callable, Sequence

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CROSS_ENCODER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
DEFAULT_ONNX_RELATIVE_PATH = "onnx/model.onnx"
DEFAULT_MAX_SEQ_LENGTH = 512
MIN_CANDIDATE_TOKEN_RESERVE = 64


def stable_sigmoid(logit: float) -> float:
    """Numerically stable scalar sigmoid function."""
    x = float(logit)
    if x >= 0.0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


class ONNXCrossEncoderReranker:
    """CPU ONNX Runtime cross-encoder using native query-document pair encoding."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        revision: str = DEFAULT_CROSS_ENCODER_REVISION,
        onnx_relative_path: str = DEFAULT_ONNX_RELATIVE_PATH,
        max_length: int = DEFAULT_MAX_SEQ_LENGTH,
        min_candidate_reserve: int = MIN_CANDIDATE_TOKEN_RESERVE,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.onnx_relative_path = onnx_relative_path
        self.max_length = int(max_length)
        self.min_candidate_reserve = int(min_candidate_reserve)
        self.batch_size = int(batch_size)

        t_dl0 = time.perf_counter()
        self.model_path = self._download_pinned(onnx_relative_path)
        self.tokenizer_path = self._download_pinned("tokenizer.json")
        self.download_or_locate_seconds = round(time.perf_counter() - t_dl0, 4)
        self.model_size_mb = round(os.path.getsize(self.model_path) / (1024.0 * 1024.0), 2)

        t_init0 = time.perf_counter()
        # Native pair tokenizer with candidate-first ('only_second') truncation
        self.pair_tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.pair_tokenizer.enable_truncation(max_length=self.max_length, strategy="only_second")
        self.pair_tokenizer.no_padding()

        # Raw untruncated probe tokenizer for exact per-side truncation telemetry
        self.raw_tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.raw_tokenizer.no_truncation()
        self.raw_tokenizer.no_padding()

        empty_pair = self.raw_tokenizer.encode("", "")
        self.special_tokens_count = len(empty_pair.ids)  # 3 for BERT: [CLS] [SEP] [SEP]
        self.cls_token_id = self.raw_tokenizer.token_to_id("[CLS]")
        self.sep_token_id = self.raw_tokenizer.token_to_id("[SEP]")
        self.pad_token_id = self.raw_tokenizer.token_to_id("[PAD]") or 0

        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )
        self.providers = list(self.session.get_providers())
        self.input_signature = [
            (inp.name, inp.type, inp.shape) for inp in self.session.get_inputs()
        ]
        self.output_signature = [
            (out.name, out.type, out.shape) for out in self.session.get_outputs()
        ]
        self.input_names = {inp.name for inp in self.session.get_inputs()}
        self.supports_token_type_ids = "token_type_ids" in self.input_names
        self.model_load_seconds = round(time.perf_counter() - t_init0, 4)
        self.warmup_seconds = 0.0

    def _download_pinned(self, filename: str) -> str:
        try:
            return hf_hub_download(
                self.model_name,
                filename,
                revision=self.revision,
                local_files_only=True,
            )
        except Exception:
            return hf_hub_download(
                self.model_name,
                filename,
                revision=self.revision,
            )

    def warmup(self) -> float:
        """Run one steady-state warmup pair outside benchmark timers."""
        t0 = time.perf_counter()
        _ = self.score_pairs(
            "Where is Flask defined?",
            ["class Flask:\n    def __init__(self, import_name): pass"],
        )
        self.warmup_seconds = round(time.perf_counter() - t0, 4)
        return self.warmup_seconds

    def get_model_info(self) -> dict[str, Any]:
        """Return verifiable ONNX artifact metadata and signatures."""
        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "onnx_artifact": self.onnx_relative_path,
            "model_size_mb": self.model_size_mb,
            "max_length": self.max_length,
            "special_tokens_per_pair": self.special_tokens_count,
            "cpu_providers": self.providers,
            "input_signature": [list(item) for item in self.input_signature],
            "output_signature": [list(item) for item in self.output_signature],
            "supports_token_type_ids": self.supports_token_type_ids,
            "score_semantics": "single_raw_relevance_logit_descending (ms-marco-MiniLM-L-6-v2)",
            "download_or_locate_seconds": self.download_or_locate_seconds,
            "model_load_seconds": self.model_load_seconds,
            "warmup_seconds": self.warmup_seconds,
        }

    def encode_pair(
        self,
        query: str,
        candidate_text: str,
    ) -> tuple[dict[str, list[int]], dict[str, Any]]:
        """Encode (query, candidate_text) as `[CLS] query [SEP] candidate [SEP]`.

        Enforces `only_second` pair truncation up to `max_length` while protecting
        against `TruncationException` on exceptionally long queries by capping query
        content tokens to `max_length - special_tokens_count - min_candidate_reserve`.
        """
        q_str = query if isinstance(query, str) else str(query or "")
        c_str = candidate_text if isinstance(candidate_text, str) else str(candidate_text or "")

        q_raw_ids = self.raw_tokenizer.encode(q_str, add_special_tokens=False).ids
        c_raw_ids = self.raw_tokenizer.encode(c_str, add_special_tokens=False).ids

        usable_tokens = max(4, self.max_length - self.special_tokens_count)
        max_q_tokens = max(1, usable_tokens - min(self.min_candidate_reserve, usable_tokens // 2))
        query_truncated = len(q_raw_ids) > max_q_tokens
        effective_q_len = min(len(q_raw_ids), max_q_tokens)
        avail_c_tokens = max(0, usable_tokens - effective_q_len)
        candidate_truncated = len(c_raw_ids) > avail_c_tokens
        pair_truncated = query_truncated or candidate_truncated
        recovered_error = False

        if not query_truncated:
            try:
                enc = self.pair_tokenizer.encode(q_str, c_str)
                encoded_dict = {
                    "input_ids": list(enc.ids),
                    "attention_mask": list(enc.attention_mask),
                    "token_type_ids": list(enc.type_ids),
                    "tokens": list(enc.tokens),
                }
                if len(encoded_dict["input_ids"]) <= self.max_length:
                    return encoded_dict, {
                        "query_raw_tokens": len(q_raw_ids),
                        "candidate_raw_tokens": len(c_raw_ids),
                        "final_pair_tokens": len(encoded_dict["input_ids"]),
                        "query_truncated": query_truncated,
                        "candidate_truncated": candidate_truncated,
                        "pair_truncated": pair_truncated,
                        "truncation_error_recovered": False,
                    }
            except Exception:
                recovered_error = True

        # Deterministic safe assembly when query exceeds max_q_tokens or pair_tokenizer errored
        kept_q = q_raw_ids[:max_q_tokens]
        kept_c = c_raw_ids[: max(0, usable_tokens - len(kept_q))]
        cls_id = self.cls_token_id if self.cls_token_id is not None else 101
        sep_id = self.sep_token_id if self.sep_token_id is not None else 102

        input_ids = [cls_id] + kept_q + [sep_id] + kept_c + [sep_id]
        attention_mask = [1] * len(input_ids)
        # Segment 0 for [CLS] query [SEP], Segment 1 for candidate [SEP]
        token_type_ids = [0] * (len(kept_q) + 2) + [1] * (len(kept_c) + 1)

        return (
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "tokens": None,
            },
            {
                "query_raw_tokens": len(q_raw_ids),
                "candidate_raw_tokens": len(c_raw_ids),
                "final_pair_tokens": len(input_ids),
                "query_truncated": query_truncated,
                "candidate_truncated": candidate_truncated,
                "pair_truncated": pair_truncated,
                "truncation_error_recovered": recovered_error,
            },
        )

    def _run_onnx_batch(self, batch_encoded: list[dict[str, list[int]]]) -> np.ndarray:
        max_len = max(len(item["input_ids"]) for item in batch_encoded)
        bsz = len(batch_encoded)
        input_ids = np.full((bsz, max_len), self.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((bsz, max_len), dtype=np.int64)
        token_type_ids = np.zeros((bsz, max_len), dtype=np.int64)

        for row, item in enumerate(batch_encoded):
            seq_len = len(item["input_ids"])
            input_ids[row, :seq_len] = item["input_ids"]
            attention_mask[row, :seq_len] = item["attention_mask"]
            token_type_ids[row, :seq_len] = item["token_type_ids"]

        feeds: dict[str, np.ndarray] = {}
        if "input_ids" in self.input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = attention_mask
        if self.supports_token_type_ids:
            feeds["token_type_ids"] = token_type_ids

        raw_out = self.session.run(None, feeds)[0]
        arr = np.asarray(raw_out, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[:, 0]
        if arr.ndim == 1:
            return arr
        raise ValueError(f"Unexpected cross-encoder output shape {arr.shape}")

    def score_pairs(
        self,
        query: str,
        candidate_texts: Sequence[str],
    ) -> tuple[list[float], dict[str, Any]]:
        """Score a sequence of candidate texts against a query cleanly with telemetry."""
        if not candidate_texts:
            return [], {
                "pairs_count": 0,
                "pairs_truncated_count": 0,
                "query_truncated": False,
                "candidate_inputs_truncated_count": 0,
                "truncation_errors_recovered": 0,
                "tokenization_ms": 0.0,
                "inference_ms": 0.0,
                "total_scoring_ms": 0.0,
            }

        t_tok0 = time.perf_counter()
        encoded_items: list[dict[str, list[int]]] = []
        pairs_truncated = 0
        query_truncated_any = False
        candidates_truncated = 0
        errors_recovered = 0

        for cand_txt in candidate_texts:
            enc, tmeta = self.encode_pair(query, cand_txt)
            encoded_items.append(enc)
            if tmeta["pair_truncated"]:
                pairs_truncated += 1
            if tmeta["query_truncated"]:
                query_truncated_any = True
            if tmeta["candidate_truncated"]:
                candidates_truncated += 1
            if tmeta["truncation_error_recovered"]:
                errors_recovered += 1
        tokenization_ms = (time.perf_counter() - t_tok0) * 1000.0

        t_inf0 = time.perf_counter()
        all_logits: list[float] = []
        for start in range(0, len(encoded_items), self.batch_size):
            batch_logits = self._run_onnx_batch(encoded_items[start : start + self.batch_size])
            all_logits.extend(float(v) for v in batch_logits)
        inference_ms = (time.perf_counter() - t_inf0) * 1000.0

        if len(all_logits) != len(candidate_texts) or not all(
            math.isfinite(s) for s in all_logits
        ):
            raise ValueError("Non-finite or mismatched cross-encoder logit outputs produced")

        telemetry = {
            "pairs_count": len(candidate_texts),
            "pairs_truncated_count": pairs_truncated,
            "query_truncated": query_truncated_any,
            "candidate_inputs_truncated_count": candidates_truncated,
            "truncation_errors_recovered": errors_recovered,
            "tokenization_ms": round(tokenization_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "total_scoring_ms": round(tokenization_ms + inference_ms, 2),
        }
        return all_logits, telemetry

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_k: int = 5,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Rerank deduplicated candidates descending by raw cross-encoder logit."""
        t0 = time.perf_counter()
        if not candidates:
            return [], {
                "pairs_count": 0,
                "pairs_truncated_count": 0,
                "query_truncated": False,
                "candidate_inputs_truncated_count": 0,
                "truncation_errors_recovered": 0,
                "tokenization_ms": 0.0,
                "inference_ms": 0.0,
                "total_rerank_ms": 0.0,
            }

        # Deduplicate defensively on chunk ID while preserving incoming original rank
        deduped: list[tuple[int, dict[str, Any]]] = []
        seen_ids: set[str] = set()
        for orig_rank, ch in enumerate(candidates, start=1):
            cid = str(ch.get("id") or f"cand_{orig_rank}")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            deduped.append((orig_rank, ch))

        texts = [str(ch.get("content", "")) for _, ch in deduped]
        logits, score_meta = self.score_pairs(query, texts)

        scored: list[dict[str, Any]] = []
        for (orig_rank, ch), logit in zip(deduped, logits):
            item = dict(ch)
            item["reranker_raw_logit"] = round(float(logit), 6)
            item["reranker_sigmoid_score"] = round(stable_sigmoid(logit), 6)
            item["reranker_score_interpretation"] = "raw_unnormalized_relevance_logit"
            item["original_retrieval_rank"] = int(
                ch.get("rrf_rank") or ch.get("dense_rank") or ch.get("bm25_rank") or orig_rank
            )
            scored.append(item)

        # Deterministic descending tie-breaking: raw logit desc -> original rank asc -> id asc
        scored.sort(
            key=lambda x: (
                -round(float(x["reranker_raw_logit"]), 6),
                int(x["original_retrieval_rank"]),
                str(x.get("id", "")),
            )
        )

        for r_idx, item in enumerate(scored, start=1):
            item["rerank_rank"] = r_idx

        total_ms = (time.perf_counter() - t0) * 1000.0
        score_meta["total_rerank_ms"] = round(total_ms, 2)
        return scored[:top_k], {**score_meta, "all_reranked": scored}


class HybridRerankRetriever:
    """Two-stage Hybrid (Dense + BM25) + Cross-Encoder Reranking orchestrator."""

    def __init__(
        self,
        hybrid_retriever: Any,
        reranker: ONNXCrossEncoderReranker,
        candidate_pool_mode: str = "candidate_union",
    ) -> None:
        if candidate_pool_mode not in ("rrf_top_20", "candidate_union"):
            raise ValueError(
                f"Unsupported candidate_pool_mode {candidate_pool_mode!r}; expected 'rrf_top_20' or 'candidate_union'"
            )
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.candidate_pool_mode = candidate_pool_mode

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        fallback_to_rrf_on_error: bool = True,
        return_diagnostics: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        """Retrieve candidates from Stage-1 Hybrid and rerank to Final Top-K."""
        stage1_top_k = top_k
        rrf_top5, h_diag = self.hybrid_retriever.retrieve(
            query, top_k=stage1_top_k, candidate_k=candidate_k, return_diagnostics=True
        )

        if self.candidate_pool_mode == "rrf_top_20":
            pool = h_diag["rrf_top_20"]
        else:
            pool = h_diag["candidate_union"]

        t_pool0 = time.perf_counter()
        # Ensure clean deduplicated candidate ordering preserving stage-1 RRF/appearance
        seen_cids: set[str] = set()
        dedup_pool: list[dict[str, Any]] = []
        for ch in pool:
            cid = str(ch.get("id", ""))
            if cid and cid not in seen_cids:
                seen_cids.add(cid)
                dedup_pool.append(ch)
        pool_prep_ms = (time.perf_counter() - t_pool0) * 1000.0

        rerank_failed = False
        rerank_error: str | None = None
        fallback_used = False

        try:
            reranked_top_k, r_meta = self.reranker.rerank(query, dedup_pool, top_k=top_k)
        except Exception as exc:
            rerank_failed = True
            rerank_error = str(exc)
            if not fallback_to_rrf_on_error:
                raise
            fallback_used = True
            reranked_top_k = rrf_top5[:top_k]
            r_meta = {
                "pairs_count": 0,
                "pairs_truncated_count": 0,
                "query_truncated": False,
                "candidate_inputs_truncated_count": 0,
                "truncation_errors_recovered": 0,
                "tokenization_ms": 0.0,
                "inference_ms": 0.0,
                "total_rerank_ms": 0.0,
                "all_reranked": reranked_top_k,
            }

        if not return_diagnostics:
            return reranked_top_k

        t1 = h_diag["timings_ms"]
        total_pipeline_ms = float(t1["total_hybrid_ms"]) + pool_prep_ms + float(r_meta["total_rerank_ms"])
        diag = {
            "candidate_pool_mode": self.candidate_pool_mode,
            "candidate_pool_count": len(dedup_pool),
            "stage1_hybrid": h_diag,
            "stage1_rrf_top5": rrf_top5,
            "reranker_meta": {k: v for k, v in r_meta.items() if k != "all_reranked"},
            "all_reranked": r_meta["all_reranked"],
            "rerank_failed": rerank_failed,
            "rerank_error": rerank_error,
            "fallback_to_rrf_used": fallback_used,
            "timings_ms": {
                **t1,
                "candidate_pool_prep_ms": round(pool_prep_ms, 3),
                "reranker_tokenization_ms": r_meta["tokenization_ms"],
                "reranker_inference_ms": r_meta["inference_ms"],
                "reranker_total_ms": r_meta["total_rerank_ms"],
                "total_pipeline_ms": round(total_pipeline_ms, 2),
            },
        }
        return reranked_top_k, diag
