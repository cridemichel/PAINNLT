#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ENV_NAME = "MLCG_MPS_EMPTY_CACHE_EVERY_FORCE_CALLS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the TEL22 MPS emptyCache A/B diagnostic.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--candidate-cadence", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate_cadence <= 0:
        parser.error("--candidate-cadence must be positive")
    return args


def load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not payload.get("run", {}).get("complete"):
        raise RuntimeError(f"incomplete memory diagnostic: {path}")
    return payload


def require_attestation(log_path: Path, cadence: int) -> None:
    marker = (
        f"[PaiNN] MPS diagnostic emptyCache cadence: {cadence} "
        "successful force calls"
    )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if marker not in text:
        raise RuntimeError(
            f"runtime attestation missing from {log_path}: expected {marker!r}; "
            "synchronize and rebuild ESPResSo"
        )


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def memory_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    analysis = report["memory_analysis"]
    footprint = analysis.get("physical_footprint") or {}
    return {
        "peak_footprint_mib": footprint.get("peak_mib"),
        "last_footprint_mib": footprint.get("at_last_step_mib"),
        "footprint_growth_after_warmup_mib": footprint.get("peak_minus_warmup_mib"),
        "footprint_slope_mib_per_1000_steps": footprint.get(
            "slope_mib_per_1000_steps"
        ),
        "peak_rss_mib": analysis.get("peak_rss_mib"),
        "last_rss_mib": analysis.get("rss_at_last_step_mib"),
        "rss_slope_mib_per_1000_steps": analysis.get("rss_slope_mib_per_1000_steps"),
        "elapsed_seconds": report.get("sampling", {}).get("elapsed_seconds"),
    }


def build_report(
    baseline: dict[str, Any], candidate: dict[str, Any], candidate_cadence: int
) -> dict[str, Any]:
    baseline_env = baseline.get("environment", {}).get(ENV_NAME)
    candidate_env = candidate.get("environment", {}).get(ENV_NAME)
    if baseline_env != "0":
        raise RuntimeError(f"baseline must record {ENV_NAME}=0, got {baseline_env!r}")
    if candidate_env != str(candidate_cadence):
        raise RuntimeError(
            f"candidate must record {ENV_NAME}={candidate_cadence}, got {candidate_env!r}"
        )

    baseline_hashes = {
        role: item["sha256"] for role, item in baseline["provenance_inputs"].items()
    }
    candidate_hashes = {
        role: item["sha256"] for role, item in candidate["provenance_inputs"].items()
    }
    if baseline_hashes != candidate_hashes:
        raise RuntimeError("baseline and candidate provenance hashes differ")

    base = memory_metrics(baseline)
    cand = memory_metrics(candidate)
    peak_reduction = base["peak_footprint_mib"] - cand["peak_footprint_mib"]
    peak_reduction_fraction = ratio(peak_reduction, base["peak_footprint_mib"])
    slope_ratio = ratio(
        cand["footprint_slope_mib_per_1000_steps"],
        base["footprint_slope_mib_per_1000_steps"],
    )
    growth_ratio = ratio(
        cand["footprint_growth_after_warmup_mib"],
        base["footprint_growth_after_warmup_mib"],
    )
    elapsed_ratio = ratio(cand["elapsed_seconds"], base["elapsed_seconds"])

    meaningful_peak_reduction = (
        peak_reduction_fraction is not None
        and peak_reduction_fraction >= 0.10
        and peak_reduction >= 2048.0
    )
    tail_improves = (
        (slope_ratio is not None and slope_ratio <= 0.50)
        or (growth_ratio is not None and growth_ratio <= 0.50)
    )
    effective = meaningful_peak_reduction and tail_improves
    performance_penalty = (
        elapsed_ratio - 1.0 if elapsed_ratio is not None else None
    )

    if not effective:
        decision = "do_not_promote_empty_cache_no_material_memory_benefit"
    elif performance_penalty is not None and performance_penalty > 0.25:
        decision = "memory_effective_but_tune_cadence_for_performance"
    else:
        decision = "empty_cache_candidate_worth_longer_validation"

    return {
        "schema_version": 1,
        "kind": "tel22_mps_empty_cache_ab_summary",
        "candidate_cadence_force_calls": candidate_cadence,
        "baseline": base,
        "candidate": cand,
        "comparison": {
            "peak_footprint_reduction_mib": peak_reduction,
            "peak_footprint_reduction_fraction": peak_reduction_fraction,
            "footprint_slope_ratio_candidate_over_baseline": slope_ratio,
            "footprint_growth_ratio_candidate_over_baseline": growth_ratio,
            "elapsed_ratio_candidate_over_baseline": elapsed_ratio,
            "performance_penalty_fraction": performance_penalty,
        },
        "decision": decision,
        "gates": {
            "meaningful_peak_reduction": meaningful_peak_reduction,
            "tail_improves": tail_improves,
            "effective": effective,
        },
        "provenance_sha256": baseline_hashes,
        "caution": (
            "This A/B isolates periodic MPS allocator emptyCache calls. It does not prove "
            "that retained driver memory is a leak, and a useful cadence still requires a "
            "longer performance/stability validation."
        ),
    }


def main() -> int:
    args = parse_args()
    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)
    require_attestation(args.baseline_log, 0)
    require_attestation(args.candidate_log, args.candidate_cadence)
    report = build_report(baseline, candidate, args.candidate_cadence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("[MPS EMPTYCACHE A/B]")
    print("decision:", report["decision"])
    print(
        "peak footprint reduction: "
        f"{report['comparison']['peak_footprint_reduction_mib']:.1f} MiB"
    )
    print("report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
