"""Day 6 Checkpointed Benchmark Runner (G0 -> G3 -> G4 -> Layer 2 Judge -> G1 -> G2).

Guarantees:
  - Uses ONLY frozen Day 5 evidence (`experiments/day6_generation/frozen_retrieval_evidence.json`).
  - Zero retrieval re-execution (no embeddings, BM25, dense, RRF, cross-encoder, or rewrite).
  - Atomic per-query checkpointing after every single API call with resumability:
    skips completed items and retries only missing or retryable transient failures.
  - G4 is strictly deterministic offline post-processing of saved G3 outputs (0 API calls).
  - Conservative inter-call pacing (`2.0s` default) + `45.0s` timeout + bounded retries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generation.evaluation import (
    aggregate_judge_results,
    aggregate_layer1_metrics,
    evaluate_layer1_record,
    judge_single_candidate,
)
from generation.grounded_generator import (
    DEFAULT_GEMINI_MODEL,
    apply_g4_postprocessing,
    is_daily_quota_error,
    is_retryable_gemini_error,
    run_generation_experiment_item,
)

FROZEN_PATH = SCRIPT_DIR / "frozen_retrieval_evidence.json"
ABLATION_IDS_PATH = SCRIPT_DIR / "ablation_question_ids.json"
JUDGE_IDS_PATH = SCRIPT_DIR / "judge_question_ids.json"
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
RESULTS_DIR = SCRIPT_DIR / "results"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass
        raise


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "items_by_key" in raw:
        return dict(raw["items_by_key"])
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"Unexpected checkpoint structure in {path}")


def save_checkpoint(path: Path, experiment_name: str, items_by_key: dict[str, dict[str, Any]]) -> None:
    payload = {
        "experiment": experiment_name,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_saved": len(items_by_key),
        "api_success_count": sum(1 for v in items_by_key.values() if v.get("api_success")),
        "items_by_key": items_by_key,
    }
    atomic_write_json(path, payload)


def should_skip_checkpoint_item(existing: dict[str, Any] | None) -> bool:
    """Return True if completed; False if missing or failed with retryable error."""
    if not existing:
        return False
    if existing.get("api_success") is True:
        return True
    err = str(existing.get("error") or "")
    return not is_retryable_gemini_error(err)


def run_generation_stage(
    experiment_id: str,
    frozen_data: dict[str, Any],
    target_ids: list[str] | None = None,
    checkpoint_path: Path | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout_seconds: float = 45.0,
    max_retries: int = 2,
    base_backoff_seconds: float = 2.0,
    pacing_seconds: float = 2.0,
    stop_on_daily_quota: bool = True,
    priority_first_ids: list[str] | None = None,
    llm_call_fn: Callable[[str], tuple[str, dict[str, Any]] | str] | None = None,
) -> list[dict[str, Any]]:
    records = frozen_data["records"]
    trees = frozen_data["repository_trees"]
    id_filter = set(target_ids) if target_ids is not None else None
    selected = [r for r in records if (id_filter is None or r["question_id"] in id_filter)]
    if priority_first_ids:
        order_idx = {qid: i for i, qid in enumerate(priority_first_ids)}
        selected.sort(key=lambda r: (0, order_idx[r["question_id"]]) if r["question_id"] in order_idx else (1, 0))

    ckpt_file = checkpoint_path or (CHECKPOINTS_DIR / f"{experiment_id.lower()}_checkpoint.json")
    ckpt_map = load_checkpoint(ckpt_file)

    executed_calls_this_run = 0
    for idx, rec in enumerate(selected):
        qid = rec["question_id"]
        if should_skip_checkpoint_item(ckpt_map.get(qid)):
            continue

        if executed_calls_this_run > 0 and pacing_seconds > 0:
            time.sleep(pacing_seconds)

        res = run_generation_experiment_item(
            experiment_id=experiment_id,  # type: ignore[arg-type]
            question=rec["question"],
            retrieved_chunks=rec["retrieved_chunks"],
            directory_tree=trees.get(rec["repository"], ""),
            chat_history=None,
            previous_context=rec.get("previous_context"),
            repository_name=rec["repository"],
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            base_backoff_seconds=base_backoff_seconds,
            llm_call_fn=llm_call_fn,
        )
        executed_calls_this_run += 1

        # Enrich checkpoint record with required provenance & frozen chunk IDs
        res_full = {
            "question_id": qid,
            "repository": rec["repository"],
            "commit_sha": rec["commit_sha"],
            "cohort": rec["cohort"],
            "answerable": rec["answerable"],
            "turn_number": rec["turn_number"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency": res["generation_latency_seconds"],
            "error_status": res["error"] or "ok",
            "frozen_evidence_chunk_ids": rec["retrieved_chunk_ids"],
            **res,
        }
        ckpt_map[qid] = res_full
        save_checkpoint(ckpt_file, experiment_id, ckpt_map)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(selected) or not res["api_success"]:
            ok_cnt = sum(1 for r2 in selected if ckpt_map.get(r2["question_id"], {}).get("api_success"))
            print(
                f"[{experiment_id}] {idx + 1}/{len(selected)} processed | ok={ok_cnt} | "
                f"last={qid} lat={res['generation_latency_seconds']}s err={res['error']}",
                flush=True,
            )

        if stop_on_daily_quota and not res["api_success"] and is_daily_quota_error(res.get("error") or ""):
            print(
                f"[{experiment_id}] Daily quota exhausted at {qid}; halting stage cleanly and preserving checkpoint.",
                flush=True,
            )
            break

    return [ckpt_map[r["question_id"]] for r in selected if r["question_id"] in ckpt_map]


def run_g4_offline_stage(
    frozen_data: dict[str, Any],
    g3_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run G4 purely offline over saved G3 records (ZERO live Gemini calls)."""
    by_qid = {r["question_id"]: r for r in frozen_data["records"]}
    g4_records: list[dict[str, Any]] = []
    cit_report_entries: list[dict[str, Any]] = []

    for g3_rec in g3_results:
        qid = g3_rec["question_id"]
        frozen_item = by_qid[qid]
        g4_post = apply_g4_postprocessing(g3_rec, frozen_item["retrieved_chunks"])
        merged = {
            "question_id": qid,
            "repository": frozen_item["repository"],
            "commit_sha": frozen_item["commit_sha"],
            "cohort": frozen_item["cohort"],
            "answerable": frozen_item["answerable"],
            "turn_number": frozen_item["turn_number"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frozen_evidence_chunk_ids": frozen_item["retrieved_chunk_ids"],
            "generation_latency_seconds": g3_rec.get("generation_latency_seconds"),
            "retry_count": g3_rec.get("retry_count", 0),
            "api_calls_made": 0,  # G4 itself performs ZERO API calls
            "api_success": g3_rec.get("api_success", False),
            **g4_post,
        }
        g4_records.append(merged)
        cit_report_entries.append(
            {
                "question_id": qid,
                "cohort": frozen_item["cohort"],
                "validation_status": g4_post["validation_status"],
                "abstained": g4_post["abstained"],
                "confidence_tier": g4_post["confidence_tier"],
                **g4_post["citation_validation_report"],
            }
        )

    cit_summary = {
        "total_questions": len(cit_report_entries),
        "additional_live_api_calls": 0,
        "by_validation_status": {},
        "entries": cit_report_entries,
    }
    for e in cit_report_entries:
        st = e["validation_status"]
        cit_summary["by_validation_status"][st] = cit_summary["by_validation_status"].get(st, 0) + 1

    return g4_records, cit_summary


def run_judge_stage(
    frozen_data: dict[str, Any],
    judge_ids: list[str],
    g0_results: list[dict[str, Any]],
    g4_results: list[dict[str, Any]],
    checkpoint_path: Path | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout_seconds: float = 45.0,
    max_retries: int = 2,
    base_backoff_seconds: float = 2.0,
    pacing_seconds: float = 2.0,
    llm_call_fn: Callable[[str], tuple[str, dict[str, Any]] | str] | None = None,
) -> dict[str, Any]:
    """Run blinded Layer 2 structured judge on 24 stratified questions (G0 vs G4 = 48 calls)."""
    by_qid = {r["question_id"]: r for r in frozen_data["records"]}
    g0_map = {r["question_id"]: r for r in g0_results}
    g4_map = {r["question_id"]: r for r in g4_results}

    ckpt_file = checkpoint_path or (CHECKPOINTS_DIR / "judge_checkpoint.json")
    ckpt_map = load_checkpoint(ckpt_file)

    # Deterministic alternation of G0 vs G4 evaluation order per question
    tasks: list[tuple[str, str, str]] = []
    for qid in judge_ids:
        digest = int(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:4], 16)
        order = ("G0", "G4") if (digest % 2 == 0) else ("G4", "G0")
        for variant in order:
            key = f"{qid}__{variant}"
            tasks.append((key, qid, variant))

    calls_run = 0
    for idx, (key, qid, variant) in enumerate(tasks):
        if should_skip_checkpoint_item(ckpt_map.get(key)):
            continue

        frozen_item = by_qid[qid]
        src_rec = g0_map.get(qid) if variant == "G0" else g4_map.get(qid)
        if not src_rec or not src_rec.get("api_success"):
            ckpt_map[key] = {
                "question_id": qid,
                "repository": frozen_item["repository"],
                "cohort": frozen_item["cohort"],
                "variant": variant,
                "api_success": False,
                "parse_success": False,
                "error": "Candidate generation missing or failed; skipped judge call.",
                "api_calls_made": 0,
                "retry_count": 0,
                "verdict": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_checkpoint(ckpt_file, "judge_g0_vs_g4", ckpt_map)
            continue

        cand_answer = (
            src_rec.get("raw_response", "")
            if variant == "G0"
            else src_rec.get("rendered_answer", src_rec.get("raw_g3_response_preserved", ""))
        )

        if calls_run > 0 and pacing_seconds > 0:
            time.sleep(pacing_seconds)

        verdict_rec = judge_single_candidate(
            frozen_item=frozen_item,
            candidate_answer=cand_answer,
            variant_label=variant,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            base_backoff_seconds=base_backoff_seconds,
            llm_call_fn=llm_call_fn,
        )
        verdict_rec["timestamp"] = datetime.now(timezone.utc).isoformat()
        calls_run += 1
        ckpt_map[key] = verdict_rec
        save_checkpoint(ckpt_file, "judge_g0_vs_g4", ckpt_map)

        if (idx + 1) % 8 == 0 or (idx + 1) == len(tasks) or not verdict_rec["api_success"]:
            v_ok = (verdict_rec.get("verdict") or {}).get("faithfulness")
            print(
                f"[Judge] {idx + 1}/{len(tasks)} | {key} ok={verdict_rec['api_success']} "
                f"faithfulness={v_ok} lat={verdict_rec['judge_latency_seconds']}s err={verdict_rec.get('error')}",
                flush=True,
            )
        if not verdict_rec["api_success"] and is_daily_quota_error(verdict_rec.get("error") or ""):
            print(
                f"[Judge] Daily quota exhausted at {key}; halting judge stage cleanly.",
                flush=True,
            )
            break

    g0_verdicts = [ckpt_map[f"{qid}__G0"] for qid in judge_ids if f"{qid}__G0" in ckpt_map]
    g4_verdicts = [ckpt_map[f"{qid}__G4"] for qid in judge_ids if f"{qid}__G4" in ckpt_map]
    return {
        "judge_subset_size": len(judge_ids),
        "judge_question_ids": judge_ids,
        "G0_verdicts": g0_verdicts,
        "G4_verdicts": g4_verdicts,
        "G0_summary": aggregate_judge_results(g0_verdicts),
        "G4_summary": aggregate_judge_results(g4_verdicts),
    }


def build_and_save_reports(
    frozen_data: dict[str, Any],
    ablation_ids: list[str],
    judge_ids: list[str],
    checkpoint_prefix: str = "",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    by_qid = {r["question_id"]: r for r in frozen_data["records"]}
    prefix = f"{checkpoint_prefix}_" if checkpoint_prefix else ""

    g0_ckpt = load_checkpoint(CHECKPOINTS_DIR / f"{prefix}g0_checkpoint.json")
    g1_ckpt = load_checkpoint(CHECKPOINTS_DIR / f"{prefix}g1_checkpoint.json")
    g2_ckpt = load_checkpoint(CHECKPOINTS_DIR / f"{prefix}g2_checkpoint.json")
    g3_ckpt = load_checkpoint(CHECKPOINTS_DIR / f"{prefix}g3_checkpoint.json")
    judge_ckpt = load_checkpoint(CHECKPOINTS_DIR / f"{prefix}judge_checkpoint.json")

    all_ids = [r["question_id"] for r in frozen_data["records"]]

    g0_rows = [g0_ckpt[q] for q in all_ids if q in g0_ckpt]
    g1_rows = [g1_ckpt[q] for q in ablation_ids if q in g1_ckpt]
    g2_rows = [g2_ckpt[q] for q in ablation_ids if q in g2_ckpt]
    g3_rows = [g3_ckpt[q] for q in all_ids if q in g3_ckpt]

    g4_rows, cit_summary = run_g4_offline_stage(frozen_data, g3_rows) if g3_rows else ([], {})

    # Save per-experiment outputs cleanly
    if g0_rows:
        atomic_write_json(out_dir / "g0_results.json", g0_rows)
    if g1_rows:
        atomic_write_json(out_dir / "g1_results.json", g1_rows)
    if g2_rows:
        atomic_write_json(out_dir / "g2_results.json", g2_rows)
    if g3_rows:
        atomic_write_json(out_dir / "g3_results.json", g3_rows)
    if g4_rows:
        atomic_write_json(out_dir / "g4_results.json", g4_rows)
        atomic_write_json(out_dir / "citation_validation_report.json", cit_summary)

    # Layer 1 evaluation per experiment
    l1_g0 = [evaluate_layer1_record(by_qid[r["question_id"]], r) for r in g0_rows]
    l1_g1 = [evaluate_layer1_record(by_qid[r["question_id"]], r) for r in g1_rows]
    l1_g2 = [evaluate_layer1_record(by_qid[r["question_id"]], r) for r in g2_rows]
    l1_g3 = [evaluate_layer1_record(by_qid[r["question_id"]], r) for r in g3_rows]
    g3_by_id = {r["question_id"]: r for r in g3_rows}
    l1_g4 = [
        evaluate_layer1_record(by_qid[r["question_id"]], g3_by_id[r["question_id"]], g4_record=r)
        for r in g4_rows
    ]

    # Paired G0 vs G1 vs G2 vs G4 strictly on N=18 ablation subset (fair same-sample view!)
    ablation_set = set(ablation_ids)
    l1_g0_n18 = [x for x in l1_g0 if x["question_id"] in ablation_set]
    l1_g4_n18 = [x for x in l1_g4 if x["question_id"] in ablation_set]

    layer1_payload = {
        "G0_N60": aggregate_layer1_metrics(l1_g0) if l1_g0 else {},
        "G3_N60": aggregate_layer1_metrics(l1_g3) if l1_g3 else {},
        "G4_N60": aggregate_layer1_metrics(l1_g4) if l1_g4 else {},
        "ablation_N18_same_subset": {
            "G0_N18": aggregate_layer1_metrics(l1_g0_n18) if l1_g0_n18 else {},
            "G1_N18": aggregate_layer1_metrics(l1_g1) if l1_g1 else {},
            "G2_N18": aggregate_layer1_metrics(l1_g2) if l1_g2 else {},
            "G4_N18": aggregate_layer1_metrics(l1_g4_n18) if l1_g4_n18 else {},
        },
        "per_query": {
            "G0": l1_g0,
            "G1": l1_g1,
            "G2": l1_g2,
            "G3": l1_g3,
            "G4": l1_g4,
        },
    }
    atomic_write_json(out_dir / "layer1_evaluation.json", layer1_payload)

    # Layer 2 judge aggregation
    g0_verdicts = [judge_ckpt[f"{q}__G0"] for q in judge_ids if f"{q}__G0" in judge_ckpt]
    g4_verdicts = [judge_ckpt[f"{q}__G4"] for q in judge_ids if f"{q}__G4" in judge_ckpt]
    layer2_payload = {
        "judge_subset_size": len(judge_ids),
        "judge_question_ids": judge_ids,
        "G0_summary": aggregate_judge_results(g0_verdicts) if g0_verdicts else {},
        "G4_summary": aggregate_judge_results(g4_verdicts) if g4_verdicts else {},
        "G0_verdicts": g0_verdicts,
        "G4_verdicts": g4_verdicts,
    }
    atomic_write_json(out_dir / "layer2_blinded_judge.json", layer2_payload)

    # Actual API call accounting
    def _api_stats(rows: list[dict[str, Any]], planned: int) -> dict[str, Any]:
        return {
            "planned_calls": planned,
            "sample_size_processed": len(rows),
            "succeeded": sum(1 for r in rows if r.get("api_success")),
            "failed": sum(1 for r in rows if not r.get("api_success")),
            "actual_api_calls_made": sum(int(r.get("api_calls_made", 0)) for r in rows),
            "total_retries": sum(int(r.get("retry_count", 0)) for r in rows),
        }

    api_ledger = {
        "G0": _api_stats(g0_rows, 60),
        "G3": _api_stats(g3_rows, 60),
        "G4": {"planned_calls": 0, "actual_api_calls_made": 0, "sample_size_processed": len(g4_rows)},
        "Judge": _api_stats(g0_verdicts + g4_verdicts, 48),
        "G1": _api_stats(g1_rows, 18),
        "G2": _api_stats(g2_rows, 18),
    }
    api_ledger["total_planned_live_calls"] = 60 + 60 + 0 + 48 + 18 + 18
    api_ledger["total_actual_live_calls"] = (
        api_ledger["G0"]["actual_api_calls_made"]
        + api_ledger["G3"]["actual_api_calls_made"]
        + api_ledger["Judge"]["actual_api_calls_made"]
        + api_ledger["G1"]["actual_api_calls_made"]
        + api_ledger["G2"]["actual_api_calls_made"]
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_evidence_source": frozen_data["source_day5_experiment"],
        "cohorts": frozen_data["cohort_counts"],
        "api_budget_and_actuals": api_ledger,
        "layer1_summary": {
            k: v for k, v in layer1_payload.items() if k != "per_query"
        },
        "layer2_judge_summary": {
            "G0": layer2_payload["G0_summary"],
            "G4": layer2_payload["G4_summary"],
        },
        "citation_validation_counts": cit_summary.get("by_validation_status", {}),
    }
    atomic_write_json(out_dir / "day6_summary.json", summary)
    return summary


def main() -> None:
    load_dotenv(str(PROJECT_ROOT / ".env"))
    parser = argparse.ArgumentParser(description="Day 6 Checkpointed Generation Benchmark Runner")
    parser.add_argument(
        "--stage",
        choices=["all", "g0", "g3", "g4", "judge", "g1", "g2", "summarize"],
        default="all",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--checkpoint-prefix", type=str, default="")
    parser.add_argument("--priority-ablation-first", action="store_true")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--pacing", type=float, default=2.0)
    args = parser.parse_args()

    frozen_data = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    ablation_ids = json.loads(ABLATION_IDS_PATH.read_text(encoding="utf-8"))["question_ids"]
    judge_ids = json.loads(JUDGE_IDS_PATH.read_text(encoding="utf-8"))["question_ids"]
    prefix = f"{args.checkpoint_prefix}_" if args.checkpoint_prefix else ""
    prio_ids = ablation_ids if args.priority_ablation_first else None

    stages = (
        ["g0", "g3", "g4", "judge", "g1", "g2", "summarize"]
        if args.stage == "all"
        else [args.stage]
    )

    g0_rows: list[dict[str, Any]] = []
    g3_rows: list[dict[str, Any]] = []
    g4_rows: list[dict[str, Any]] = []

    for st in stages:
        if st == "g0":
            print(f"=== STAGE G0 (N=60 Baseline, model={args.model}) ===", flush=True)
            g0_rows = run_generation_stage(
                "G0",
                frozen_data,
                checkpoint_path=CHECKPOINTS_DIR / f"{prefix}g0_checkpoint.json",
                model=args.model,
                timeout_seconds=args.timeout,
                max_retries=args.retries,
                pacing_seconds=args.pacing,
                priority_first_ids=prio_ids,
            )
        elif st == "g3":
            print(f"=== STAGE G3 (N=60 Single-Call Grounded JSON, model={args.model}) ===", flush=True)
            g3_rows = run_generation_stage(
                "G3",
                frozen_data,
                checkpoint_path=CHECKPOINTS_DIR / f"{prefix}g3_checkpoint.json",
                model=args.model,
                timeout_seconds=args.timeout,
                max_retries=args.retries,
                pacing_seconds=args.pacing,
                priority_first_ids=prio_ids,
            )
        elif st == "g4":
            print("=== STAGE G4 (N=60 Offline Citation Validation, 0 calls) ===", flush=True)
            if not g3_rows:
                g3_ckpt = load_checkpoint(CHECKPOINTS_DIR / f"{prefix}g3_checkpoint.json")
                g3_rows = [g3_ckpt[r["question_id"]] for r in frozen_data["records"] if r["question_id"] in g3_ckpt]
            g4_rows, _ = run_g4_offline_stage(frozen_data, g3_rows)
        elif st == "judge":
            print(f"=== STAGE LAYER 2 BLINDED JUDGE (N=24 -> 48 calls, model={args.model}) ===", flush=True)
            if not g0_rows:
                g0_ckpt = load_checkpoint(CHECKPOINTS_DIR / f"{prefix}g0_checkpoint.json")
                g0_rows = [g0_ckpt[r["question_id"]] for r in frozen_data["records"] if r["question_id"] in g0_ckpt]
            if not g4_rows:
                g3_ckpt = load_checkpoint(CHECKPOINTS_DIR / f"{prefix}g3_checkpoint.json")
                g3_rows = [g3_ckpt[r["question_id"]] for r in frozen_data["records"] if r["question_id"] in g3_ckpt]
                g4_rows, _ = run_g4_offline_stage(frozen_data, g3_rows)
            run_judge_stage(
                frozen_data,
                judge_ids,
                g0_rows,
                g4_rows,
                checkpoint_path=CHECKPOINTS_DIR / f"{prefix}judge_checkpoint.json",
                model=args.model,
                timeout_seconds=args.timeout,
                max_retries=args.retries,
                pacing_seconds=args.pacing,
            )
        elif st == "g1":
            print(f"=== STAGE G1 (N=18 Grounded Only Ablation, model={args.model}) ===", flush=True)
            run_generation_stage(
                "G1",
                frozen_data,
                target_ids=ablation_ids,
                checkpoint_path=CHECKPOINTS_DIR / f"{prefix}g1_checkpoint.json",
                model=args.model,
                timeout_seconds=args.timeout,
                max_retries=args.retries,
                pacing_seconds=args.pacing,
            )
        elif st == "g2":
            print(f"=== STAGE G2 (N=18 Grounded + Citations Ablation, model={args.model}) ===", flush=True)
            run_generation_stage(
                "G2",
                frozen_data,
                target_ids=ablation_ids,
                checkpoint_path=CHECKPOINTS_DIR / f"{prefix}g2_checkpoint.json",
                model=args.model,
                timeout_seconds=args.timeout,
                max_retries=args.retries,
                pacing_seconds=args.pacing,
            )
        elif st == "summarize":
            out_dir = RESULTS_DIR / args.checkpoint_prefix if args.checkpoint_prefix else RESULTS_DIR
            summary = build_and_save_reports(
                frozen_data,
                ablation_ids,
                judge_ids,
                checkpoint_prefix=args.checkpoint_prefix,
                output_dir=out_dir,
            )
            print("=== DAY 6 SUMMARY WRITTEN ===", flush=True)
            print(json.dumps(summary["api_budget_and_actuals"], indent=2), flush=True)


if __name__ == "__main__":
    main()
