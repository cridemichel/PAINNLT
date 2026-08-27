#!/usr/bin/env python3
"""Summarize TEL22 Morse stabilizer A/B/C robustness comparison.

A = production: all 180 Morse a=0.30
B = selective: fixed top-18 local-curvature Morse a=0.255
C = uniform: all 180 Morse a=0.255

All arms are priors-only, production-like marker/non-bonded switched Morse,
10 ps, full dt grid. A and B are reused from test 21; only C is newly run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MOD_PATH = HERE / "21_summarize_morse_top10_a0p85_robustness.py"
spec = importlib.util.spec_from_file_location("morse_robust", MOD_PATH)
robust = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(robust)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def distance_to_one(value: float) -> float:
    return abs(float(value) - 1.0)


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "exponent_p": metrics["exponent_p"],
        "abs_p_minus_2": metrics["abs_p_minus_2"],
        "loglog_r2": metrics["loglog_r2"],
        "c2_spread_max_over_min": metrics["c2_spread_max_over_min"],
        "local_exponent_range": metrics["local_exponent_range"],
        "c2_small_over_coarse_median": metrics["c2_small_over_coarse_median"],
        "fine_fit_dt_0p001_to_0p002": metrics["fine_fit_dt_0p001_to_0p002"],
        "coarse_fit_dt_0p002_to_0p005": metrics["coarse_fit_dt_0p002_to_0p005"],
        "max_relative_block_mean_drift": metrics["max_relative_block_mean_drift"],
        "runs": metrics["runs"],
        "report": metrics["report"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-report", type=Path, required=True)
    ap.add_argument("--selective-report", type=Path, required=True)
    ap.add_argument("--uniform-report", type=Path, required=True)
    ap.add_argument("--selective-manifest", type=Path, required=True)
    ap.add_argument("--uniform-manifest", type=Path, required=True)
    ap.add_argument("--test21-summary", type=Path, required=True)
    ap.add_argument("--production-priors-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    selective_manifest = load(args.selective_manifest)
    if selective_manifest.get("kind") != "tel22_morse_top10_local_curvature_a_refinement_inputs":
        raise ValueError("Unexpected selective/test-20 manifest kind")
    selective_meta = selective_manifest.get("variants", {}).get("top10_a0p850")
    if not isinstance(selective_meta, dict):
        raise ValueError("Selective manifest lacks top10_a0p850")
    if int(selective_meta.get("selected_count", -1)) != 18:
        raise ValueError("Selective branch is not exactly 18 Morse contacts")
    if abs(float(selective_meta.get("a_scale", -1.0)) - 0.85) > 1e-15:
        raise ValueError("Selective branch is not a=0.85")

    uniform_manifest = load(args.uniform_manifest)
    if uniform_manifest.get("kind") != "tel22_morse_uniform_a0p85_inputs":
        raise ValueError("Unexpected uniform-a manifest kind")
    if int(uniform_manifest.get("morse_count", -1)) != 180:
        raise ValueError("Uniform branch does not change exactly 180 Morse contacts")
    if abs(float(uniform_manifest.get("a_scale", -1.0)) - 0.85) > 1e-15:
        raise ValueError("Uniform branch is not a=0.85")

    test21 = load(args.test21_summary)
    if test21.get("kind") != "tel22_morse_top10_a0p85_robustness_10ps_fullgrid":
        raise ValueError("Unexpected test-21 summary kind")
    if test21.get("interpretation") != "a0p85_5ps_gain_not_robust_on_10ps_full_grid":
        raise ValueError("Test-21 prerequisite no longer has the expected non-robust selective result")

    full = robust.summarize_report(args.full_report, args.production_priors_sha256)
    selective = robust.summarize_report(args.selective_report, str(selective_meta["priors_sha256"]))
    uniform = robust.summarize_report(args.uniform_report, str(uniform_manifest["priors_sha256"]))

    # Verify that the reused A/B reports are exactly the ones summarized by test 21.
    for label, new, old in (
        ("full", full, test21["full_reference"]),
        ("selective", selective, test21["candidate_a0p85"]),
    ):
        for key in ("exponent_p", "c2_spread_max_over_min", "local_exponent_range"):
            if abs(float(new[key]) - float(old[key])) > 1e-12:
                raise ValueError(f"Reused {label} report disagrees with test-21 summary for {key}")

    full_vs_selective = robust.compare(full, selective)
    full_vs_uniform = robust.compare(full, uniform)
    selective_vs_uniform = robust.compare(selective, uniform)

    uniform_improves_core = (
        uniform["c2_spread_max_over_min"] < full["c2_spread_max_over_min"]
        and uniform["local_exponent_range"] < full["local_exponent_range"]
        and uniform["abs_p_minus_2"] < full["abs_p_minus_2"]
        and distance_to_one(uniform["c2_small_over_coarse_median"])
            < distance_to_one(full["c2_small_over_coarse_median"])
        and uniform["max_relative_block_mean_drift"] <= 1.0e-4
    )
    uniform_improves_regularity = (
        uniform["c2_spread_max_over_min"] < full["c2_spread_max_over_min"]
        and uniform["local_exponent_range"] < full["local_exponent_range"]
        and uniform["max_relative_block_mean_drift"] <= 1.0e-4
    )

    # This is a numerical-stabilizer decision, not a physical fit. Prefer C only
    # when it improves the full production-Morse reference on multiple independent
    # integration diagnostics. Do not promote B: test 21 already established that
    # its 5 ps gain is not robust at 10 ps/full-grid.
    if uniform_improves_core:
        interpretation = "uniform_a0p85_supports_next_painn_closure"
        recommended = "C_uniform_a0p85"
    elif uniform_improves_regularity:
        interpretation = "uniform_a0p85_regularizes_some_metrics_but_is_mixed"
        recommended = "review_before_painn_closure"
    else:
        interpretation = "uniform_a0p85_does_not_outperform_production_morse_on_10ps_full_grid"
        recommended = "A_production"

    out = {
        "schema_version": 1,
        "kind": "tel22_morse_stabilizer_abc_10ps_fullgrid",
        "scope": (
            "PaiNN disabled; production-like marker/non-bonded switched Morse runtime; "
            "10 ps at dt=0.001,0.0015,0.002,0.003,0.004,0.005 ps; identical physical checkpoint state. "
            "A and B reuse test-21 reports; C is the newly computed uniform-a branch."
        ),
        "stabilizer_context": (
            "TEL22 Morse terms are empirical structural/numerical stabilizers. The comparison tests whether a single "
            "uniform a=0.255 is a cleaner and more robust stabilizer choice than either production a=0.30 or the "
            "snapshot-ranked selective top-18 softening."
        ),
        "arms": {
            "A_production": {"morse_count": 180, "softened_count": 0, "a_default": 0.30, **compact(full)},
            "B_selective_top18_a0p85": {
                "morse_count": 180, "softened_count": 18, "a_softened": 0.255, "a_other": 0.30, **compact(selective)
            },
            "C_uniform_a0p85": {
                "morse_count": 180, "softened_count": 180, "a_uniform": 0.255, "k_at_r0_ratio": 0.7225, **compact(uniform)
            },
        },
        "comparisons": {
            "B_over_A": full_vs_selective,
            "C_over_A": full_vs_uniform,
            "C_over_B": selective_vs_uniform,
        },
        "decision_checks": {
            "B_selective_known_nonrobust_from_test21": True,
            "C_improves_C2_spread_vs_A": uniform["c2_spread_max_over_min"] < full["c2_spread_max_over_min"],
            "C_improves_local_exponent_range_vs_A": uniform["local_exponent_range"] < full["local_exponent_range"],
            "C_improves_abs_p_minus_2_vs_A": uniform["abs_p_minus_2"] < full["abs_p_minus_2"],
            "C_improves_small_over_coarse_closeness_to_1_vs_A": (
                distance_to_one(uniform["c2_small_over_coarse_median"])
                < distance_to_one(full["c2_small_over_coarse_median"])
            ),
            "C_drift_pass": uniform["max_relative_block_mean_drift"] <= 1.0e-4,
        },
        "interpretation": interpretation,
        "recommended_numerical_stabilizer_for_next_step": recommended,
        "next_step_if_C_supported": (
            "Run the short PaiNN closure with the chosen uniform priors before rebuilding the complete TEL22 residual/training pipeline."
        ),
        "caution": (
            "Numerical-stabilizer selection does not establish production accuracy. Any changed-prior full TEL22 model must rebuild "
            "the residual target and retrain PaiNN before production claims."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[TEL22 MORSE STABILIZER A/B/C -- 10 ps FULL GRID]")
    for key in ("A_production", "B_selective_top18_a0p85", "C_uniform_a0p85"):
        row = out["arms"][key]
        print(
            f"{key:28s}: p={row['exponent_p']:.8f} R2={row['loglog_r2']:.8f} "
            f"C2spread={row['c2_spread_max_over_min']:.6f} localRange={row['local_exponent_range']:.6f} "
            f"C2small/coarse={row['c2_small_over_coarse_median']:.6f}"
        )
    print(f"[INTERPRETATION] {interpretation}")
    print(f"[NEXT] {recommended}")
    print(f"[REPORT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
