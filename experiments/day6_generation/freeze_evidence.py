"""Freeze Day 5 Selected Retrieval Evidence (`D5-E3a_rerank_rrf20`) for Day 6 Experiments.

Strictly hydrates all 60 evaluation queries' exact Top-5 chunks directly from
`experiments/day4_hybrid/chroma_day4_db/` without re-running any retrieval step
(embeddings, BM25, dense, RRF, cross-encoder, or query rewriting).

Also freezes deterministic repo commit SHAs, cached `directory_tree` strings,
Turn-2 `previous_context`, verified cohorts (Cohort A/B/C), and locks the
deterministic stratified 18-query ablation subset and 24-query judge subset.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import repo_loader

DAY5_EVAL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "day5_rerank_rewrite"
    / "results"
    / "per_experiment"
    / "D5-E3a_rerank_rrf20_evaluation.json"
)
DAY5_SUMMARY_PATH = (
    PROJECT_ROOT / "experiments" / "day5_rerank_rewrite" / "results" / "day5_summary.json"
)
CHROMA_DB_PATH = PROJECT_ROOT / "experiments" / "day4_hybrid" / "chroma_day4_db"
QUESTIONS_PATH = PROJECT_ROOT / "experiments" / "datasets" / "evaluation_questions.json"
GROUND_TRUTH_PATH = PROJECT_ROOT / "experiments" / "datasets" / "ground_truth.json"
REPOS_PATH = PROJECT_ROOT / "experiments" / "datasets" / "repositories.json"
CACHE_ROOT = PROJECT_ROOT / "experiments" / "datasets" / "cache"

FROZEN_OUTPUT_PATH = SCRIPT_DIR / "frozen_retrieval_evidence.json"
ABLATION_IDS_PATH = SCRIPT_DIR / "ablation_question_ids.json"
JUDGE_IDS_PATH = SCRIPT_DIR / "judge_question_ids.json"

REPO_COLLECTION_MAP = {
    "pallets/flask": ("day4_v1_flask", "flask"),
    "expressjs/express": ("day4_v1_express", "express"),
    "fmtlib/fmt": ("day4_v1_fmt", "fmt"),
}

# Pre-committed deterministic stratified ablation subset (N=18: 6 per repo; 9 Cohort A, 6 Cohort B, 3 Cohort C)
DETERMINISTIC_ABLATION_IDS = [
    # pallets/flask (3 A, 2 B, 1 C)
    "flask_001",
    "flask_006",
    "flask_012",
    "flask_002",
    "flask_017",
    "flask_019",
    # expressjs/express (3 A, 2 B, 1 C)
    "express_004",
    "express_010",
    "express_016",
    "express_008",
    "express_018",
    "express_020",
    # fmtlib/fmt (3 A, 2 B, 1 C)
    "fmt_005",
    "fmt_009",
    "fmt_011",
    "fmt_003",
    "fmt_013",
    "fmt_020",
]

# Pre-committed deterministic stratified Layer-2 Judge subset (N=24: 12 Cohort A, 6 Cohort B, 6 Cohort C)
DETERMINISTIC_JUDGE_IDS = [
    # pallets/flask (4 A, 2 B, 2 C)
    "flask_001",
    "flask_006",
    "flask_009",
    "flask_014",
    "flask_002",
    "flask_017",
    "flask_019",
    "flask_020",
    # expressjs/express (4 A, 2 B, 2 C)
    "express_002",
    "express_004",
    "express_010",
    "express_015",
    "express_008",
    "express_011",
    "express_019",
    "express_020",
    # fmtlib/fmt (4 A, 2 B, 2 C)
    "fmt_005",
    "fmt_009",
    "fmt_011",
    "fmt_017",
    "fmt_003",
    "fmt_013",
    "fmt_019",
    "fmt_020",
]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_evidence() -> dict[str, Any]:
    if not DAY5_EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing Day 5 evaluation artifact: {DAY5_EVAL_PATH}")
    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(f"Missing Day 4/5 ChromaDB directory: {CHROMA_DB_PATH}")

    day5_summary = json.loads(DAY5_SUMMARY_PATH.read_text(encoding="utf-8"))
    selected_pipeline = day5_summary.get("selected_best_fixed_pipeline")
    if selected_pipeline != "D5-E3a_rerank_rrf20":
        raise RuntimeError(
            f"Unexpected selected Day 5 pipeline {selected_pipeline!r}; expected 'D5-E3a_rerank_rrf20'"
        )

    day5_records = json.loads(DAY5_EVAL_PATH.read_text(encoding="utf-8"))
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    repos_meta = json.loads(REPOS_PATH.read_text(encoding="utf-8"))

    q_by_id = {q["question_id"]: q for q in questions}
    gt_by_id = {g["question_id"]: g for g in ground_truth}
    repo_by_name = {r["name"]: r for r in repos_meta}

    if len(day5_records) != 60 or len(questions) != 60 or len(ground_truth) != 60:
        raise RuntimeError(
            f"Expected 60 questions everywhere; got d5={len(day5_records)}, q={len(questions)}, gt={len(ground_truth)}"
        )

    # Verify cached repo snapshots + commit SHAs + build identical directory trees
    repository_trees: dict[str, str] = {}
    repository_metadata: dict[str, Any] = {}
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collections = {}

    day5_corpus_checks = (
        day5_summary.get("provenance", {})
        .get("controls", {})
        .get("day4_corpus_verifications", {})
    )

    for repo_name, (col_name, cache_dir_name) in REPO_COLLECTION_MAP.items():
        snap_dir = CACHE_ROOT / cache_dir_name
        sha_marker = snap_dir / ".commit_sha"
        if not snap_dir.exists() or not sha_marker.exists():
            raise FileNotFoundError(f"Missing cached snapshot or .commit_sha in {snap_dir}")
        actual_sha = sha_marker.read_text(encoding="utf-8").strip()
        expected_sha = repo_by_name[repo_name]["commit_sha"]
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Commit SHA mismatch for {repo_name}: cache={actual_sha}, declared={expected_sha}"
            )
        col = client.get_collection(col_name)
        collections[repo_name] = col
        tree_str = repo_loader.generate_directory_tree(str(snap_dir))
        repository_trees[repo_name] = tree_str
        repository_metadata[repo_name] = {
            "commit_sha": actual_sha,
            "collection_name": col_name,
            "chunks_count": col.count(),
            "day5_fingerprint_sha256": day5_corpus_checks.get(repo_name, {}).get(
                "corpus_fingerprint_sha256"
            ),
            "directory_tree_chars": len(tree_str),
            "directory_tree_lines": len(tree_str.splitlines()),
        }

    total_chunks_checked = 0
    missing_ids: list[str] = []
    metadata_mismatches: list[str] = []
    cohort_counter: Counter[str] = Counter()
    frozen_records: list[dict[str, Any]] = []

    for row in day5_records:
        qid = row["question_id"]
        repo_name = row["repository"]
        q_item = q_by_id[qid]
        gt_item = gt_by_id[qid]

        if row["raw_question"] != q_item["question"]:
            raise RuntimeError(f"Question text mismatch for {qid}")
        if bool(row["answerable"]) != bool(gt_item["answerable"]):
            raise RuntimeError(f"Answerability mismatch for {qid}")

        answerable = bool(gt_item["answerable"])
        if not answerable:
            hit_at_5 = None
            cohort = "cohort_c_unanswerable"
        else:
            hit_at_5 = bool((row.get("metrics") or {}).get("hit_rate_at_5", 0.0) == 1.0)
            cohort = "cohort_a_hit" if hit_at_5 else "cohort_b_miss"
        cohort_counter[cohort] += 1

        col = collections[repo_name]
        top_k = sorted(row["top_k_results"], key=lambda x: int(x["rank"]))
        expected_ids = [item["chunk_id"] for item in top_k]
        hydrated = col.get(ids=expected_ids, include=["documents", "metadatas"])
        hyd_map = {
            cid: (doc, meta)
            for cid, doc, meta in zip(
                hydrated["ids"], hydrated["documents"], hydrated["metadatas"]
            )
        }

        hydrated_chunks: list[dict[str, Any]] = []
        for idx, item in enumerate(top_k, start=1):
            total_chunks_checked += 1
            cid = item["chunk_id"]
            if int(item["rank"]) != idx:
                raise RuntimeError(f"{qid}: unexpected non-contiguous rank {item['rank']} vs {idx}")
            if cid not in hyd_map:
                missing_ids.append(f"{qid}:{cid}")
                continue
            doc_text, meta = hyd_map[cid]
            if not doc_text or not meta:
                metadata_mismatches.append(f"{qid}:{cid}:empty_doc_or_meta")
                continue

            if (
                meta.get("file_path") != item["file_path"]
                or meta.get("start_line") != item["start_line"]
                or meta.get("end_line") != item["end_line"]
            ):
                metadata_mismatches.append(f"{qid}:{cid}:line_or_path_mismatch")

            hydrated_chunks.append(
                {
                    "evidence_id": f"EVIDENCE_{idx}",
                    "rank": idx,
                    "chunk_id": cid,
                    "content": doc_text,
                    "metadata": dict(meta),
                    "file_path": meta.get("file_path", item["file_path"]),
                    "symbol_name": meta.get("symbol_name", item.get("symbol_name", "")),
                    "start_line": meta.get("start_line", item["start_line"]),
                    "end_line": meta.get("end_line", item["end_line"]),
                    "content_sha256": meta.get("content_sha256", _sha256_text(doc_text)),
                    "reranker_raw_logit": item.get("reranker_raw_logit"),
                    "reranker_sigmoid_score": item.get("reranker_sigmoid_score"),
                    "original_retrieval_rank": item.get("original_retrieval_rank"),
                    "retrieval_sources": item.get("retrieval_sources", []),
                    "dense_rank": item.get("dense_rank"),
                    "bm25_rank": item.get("bm25_rank"),
                    "rrf_rank": item.get("rrf_rank"),
                }
            )

        frozen_records.append(
            {
                "question_id": qid,
                "question": q_item["question"],
                "repository": repo_name,
                "commit_sha": repo_by_name[repo_name]["commit_sha"],
                "category": q_item["category"],
                "query_style": q_item["query_style"],
                "turn_number": int(q_item.get("turn_number", 1)),
                "conversation_id": q_item.get("conversation_id"),
                "previous_context": q_item.get("previous_context"),
                "answerable": answerable,
                "ground_truth_evidence": gt_item["evidence"],
                "retrieval_hit_at_5": hit_at_5,
                "cohort": cohort,
                "day5_experiment_id": "D5-E3a_rerank_rrf20",
                "retrieved_chunk_ids": expected_ids,
                "retrieved_chunks": hydrated_chunks,
            }
        )

    if missing_ids or metadata_mismatches:
        raise RuntimeError(
            f"Hydration verification failed: missing={missing_ids}, mismatches={metadata_mismatches}"
        )

    payload = {
        "schema_version": "day6_frozen_evidence_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "source_day5_experiment": "D5-E3a_rerank_rrf20",
        "source_day5_evaluation_file": str(DAY5_EVAL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_chroma_db": str(CHROMA_DB_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "total_questions": len(frozen_records),
        "cohort_counts": {
            "cohort_a_hit": int(cohort_counter["cohort_a_hit"]),
            "cohort_b_miss": int(cohort_counter["cohort_b_miss"]),
            "cohort_c_unanswerable": int(cohort_counter["cohort_c_unanswerable"]),
        },
        "hydration_verification": {
            "total_chunks_checked": total_chunks_checked,
            "hydrated_matched_count": total_chunks_checked - len(missing_ids) - len(metadata_mismatches),
            "missing_ids_count": len(missing_ids),
            "metadata_mismatch_count": len(metadata_mismatches),
            "verified": True,
        },
        "repository_metadata": repository_metadata,
        "repository_trees": repository_trees,
        "records": frozen_records,
    }

    FROZEN_OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Build stratified subset artifacts with full verification
    rec_by_id = {r["question_id"]: r for r in frozen_records}

    def _subset_summary(ids: list[str], target_name: str, expected_len: int) -> dict[str, Any]:
        if len(ids) != expected_len or len(set(ids)) != expected_len:
            raise RuntimeError(f"{target_name}: expected {expected_len} unique IDs, got {len(ids)}")
        items = [rec_by_id[qid] for qid in ids]
        repo_dist = dict(Counter(r["repository"] for r in items))
        cohort_dist = dict(Counter(r["cohort"] for r in items))
        cat_dist = dict(Counter(r["category"] for r in items))
        style_dist = dict(Counter(r["query_style"] for r in items))
        turn2_cnt = sum(1 for r in items if r["turn_number"] == 2)
        multi_file_cnt = sum(1 for r in items if len(r["ground_truth_evidence"]) >= 2)
        return {
            "name": target_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_policy": "deterministic_pre_experiment_stratified",
            "count": len(ids),
            "question_ids": ids,
            "breakdown": {
                "by_repository": repo_dist,
                "by_cohort": cohort_dist,
                "by_category": cat_dist,
                "by_query_style": style_dist,
                "turn_2_count": turn2_cnt,
                "multi_evidence_count": multi_file_cnt,
            },
        }

    ablation_payload = _subset_summary(
        DETERMINISTIC_ABLATION_IDS, "day6_ablation_g1_g2_n18", expected_len=18
    )
    judge_payload = _subset_summary(
        DETERMINISTIC_JUDGE_IDS, "day6_blinded_judge_n24", expected_len=24
    )

    if ablation_payload["breakdown"]["by_repository"] != {
        "pallets/flask": 6,
        "expressjs/express": 6,
        "fmtlib/fmt": 6,
    }:
        raise RuntimeError(f"Invalid ablation repo breakdown: {ablation_payload}")

    if judge_payload["breakdown"]["by_cohort"] != {
        "cohort_a_hit": 12,
        "cohort_b_miss": 6,
        "cohort_c_unanswerable": 6,
    }:
        raise RuntimeError(f"Invalid judge cohort breakdown: {judge_payload}")

    ABLATION_IDS_PATH.write_text(json.dumps(ablation_payload, indent=2), encoding="utf-8")
    JUDGE_IDS_PATH.write_text(json.dumps(judge_payload, indent=2), encoding="utf-8")

    return {
        "frozen_path": str(FROZEN_OUTPUT_PATH),
        "frozen_size_kb": round(FROZEN_OUTPUT_PATH.stat().st_size / 1024, 1),
        "cohorts": payload["cohort_counts"],
        "hydration": payload["hydration_verification"],
        "ablation_breakdown": ablation_payload["breakdown"],
        "judge_breakdown": judge_payload["breakdown"],
    }


if __name__ == "__main__":
    summary = freeze_evidence()
    print(json.dumps(summary, indent=2))
