#!/usr/bin/env python3
"""Summarize 10 ps full-grid TEL22 Morse a=0.85 numerical-stabilizer validation."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

EXPECTED_DTS = [0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]
COARSE_DTS = [0.002, 0.003, 0.004, 0.005]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_power(rows: list[dict[str, float]]) -> dict[str, float]:
    if len(rows) < 3:
        raise ValueError("Power-law fit needs at least three points")
    xs = [math.log(float(r["dt_ps"])) for r in rows]
    ys = [math.log(float(r["sigma_E"])) for r in rows]
    xm = statistics.fmean(xs)
    ym = statistics.fmean(ys)
    sxx = sum((x - xm) ** 2 for x in xs)
    if sxx <= 0.0:
        raise ValueError("Degenerate dt grid")
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sxx
    intercept = ym - slope * xm
    ss_tot = sum((y - ym) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else 1.0 - ss_res / ss_tot
    return {"exponent_p": slope, "abs_p_minus_2": abs(slope - 2.0), "loglog_r2": r2}


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: r["dt_ps"])
    c2 = [r["C2_sigma_over_dt2"] for r in rows]
    local = []
    for lo, hi in zip(rows, rows[1:]):
        p = math.log(hi["sigma_E"] / lo["sigma_E"]) / math.log(hi["dt_ps"] / lo["dt_ps"])
        local.append({"dt_low_ps": lo["dt_ps"], "dt_high_ps": hi["dt_ps"], "local_exponent_p": p})
    lv = [x["local_exponent_p"] for x in local]
    fit = fit_power(rows)
    return {
        **fit,
        "c2_spread_max_over_min": max(c2) / min(c2),
        "local_exponent_range": max(lv) - min(lv),
        "adjacent_local_exponents": local,
        "runs": rows,
    }


def summarize_report(path: Path, expected_priors_sha: str) -> dict[str, Any]:
    report = load(path)
    definition = report.get("definition", {})
    if definition.get("hamiltonian_mode") != "conservative_classical_model_provenance_ml_disabled":
        raise ValueError(f"{path} is not a priors-only --disable-ml report")
    runtime = report.get("pair_specific_morse_runtime", definition.get("pair_specific_morse_runtime"))
    if runtime != "marker-nonbonded":
        raise ValueError(f"{path}: Morse runtime={runtime!r}, expected marker-nonbonded")
    if report.get("inputs_sha256", {}).get("priors") != expected_priors_sha:
        raise ValueError(f"{path}: priors hash mismatch")
    runs_in = sorted(report.get("runs", []), key=lambda x: float(x["dt_ps"]))
    dts = [float(x["dt_ps"]) for x in runs_in]
    if len(dts) != len(EXPECTED_DTS) or any(abs(a - b) > 1e-12 for a, b in zip(dts, EXPECTED_DTS)):
        raise ValueError(f"{path}: unexpected dt grid {dts}")
    rows: list[dict[str, float]] = []
    for x in runs_in:
        dt = float(x["dt_ps"])
        if abs(float(x["duration_ps"]) - 10.0) > 0.5 * dt + 1e-12:
            raise ValueError(f"{path}: run at dt={dt} is not 10 ps")
        sigma = float(x["sigma_E"])
        rows.append({
            "dt_ps": dt,
            "sigma_E": sigma,
            "C2_sigma_over_dt2": sigma / (dt * dt),
            "relative_block_mean_drift": float(x["relative_block_mean_drift"]),
        })
    metrics = summarize_rows(rows)
    fine_rows = [r for r in rows if r["dt_ps"] <= 0.002 + 1e-15]
    coarse_rows = [r for r in rows if r["dt_ps"] >= 0.002 - 1e-15]
    coarse_c2 = [r["C2_sigma_over_dt2"] for r in coarse_rows]
    metrics.update({
        "report": str(path.resolve()),
        "fine_fit_dt_0p001_to_0p002": fit_power(fine_rows),
        "coarse_fit_dt_0p002_to_0p005": summarize_rows(coarse_rows),
        "c2_small_over_coarse_median": rows[0]["C2_sigma_over_dt2"] / statistics.median(coarse_c2),
        "max_relative_block_mean_drift": max(r["relative_block_mean_drift"] for r in rows),
    })
    return metrics


def compare(full: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    paired = []
    for a, b in zip(full["runs"], candidate["runs"]):
        paired.append({
            "dt_ps": a["dt_ps"],
            "sigma_full": a["sigma_E"],
            "sigma_candidate": b["sigma_E"],
            "sigma_ratio_candidate_over_full": b["sigma_E"] / a["sigma_E"],
            "C2_full": a["C2_sigma_over_dt2"],
            "C2_candidate": b["C2_sigma_over_dt2"],
        })
    return {
        "delta_p": candidate["exponent_p"] - full["exponent_p"],
        "delta_abs_p_minus_2": candidate["abs_p_minus_2"] - full["abs_p_minus_2"],
        "delta_r2": candidate["loglog_r2"] - full["loglog_r2"],
        "c2_spread_ratio": candidate["c2_spread_max_over_min"] / full["c2_spread_max_over_min"],
        "local_exponent_range_ratio": candidate["local_exponent_range"] / full["local_exponent_range"],
        "c2_small_over_coarse_ratio": candidate["c2_small_over_coarse_median"] / full["c2_small_over_coarse_median"],
        "paired_runs": paired,
    }


def coarse_from_five_ps(row: dict[str, Any]) -> dict[str, Any]:
    runs = [{
        "dt_ps": float(x["dt_ps"]),
        "sigma_E": float(x["sigma_E"]),
        "C2_sigma_over_dt2": float(x["C2_sigma_over_dt2"]),
        "relative_block_mean_drift": float(x["relative_block_mean_drift"]),
    } for x in row["runs"] if float(x["dt_ps"]) in COARSE_DTS]
    if [x["dt_ps"] for x in sorted(runs, key=lambda x: x["dt_ps"])] != COARSE_DTS:
        raise ValueError("5 ps reference lacks expected coarse grid")
    return summarize_rows(runs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-report", type=Path, required=True)
    ap.add_argument("--candidate-report", type=Path, required=True)
    ap.add_argument("--candidate-input-manifest", type=Path, required=True)
    ap.add_argument("--five-ps-summary", type=Path, required=True)
    ap.add_argument("--production-priors-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    manifest = load(args.candidate_input_manifest)
    if manifest.get("kind") != "tel22_morse_top10_local_curvature_a_refinement_inputs":
        raise ValueError("Unexpected test-20 candidate manifest kind")
    candidate_meta = manifest.get("variants", {}).get("top10_a0p850")
    if not isinstance(candidate_meta, dict) or abs(float(candidate_meta.get("a_scale", -1.0)) - 0.85) > 1e-15:
        raise ValueError("Missing valid test-20 a=0.85 candidate")
    if int(candidate_meta.get("selected_count", -1)) != 18:
        raise ValueError("a=0.85 candidate does not contain exactly 18 selected Morse contacts")

    full = summarize_report(args.full_report, args.production_priors_sha256)
    candidate = summarize_report(args.candidate_report, str(candidate_meta["priors_sha256"]))
    comp = compare(full, candidate)

    five = load(args.five_ps_summary)
    if five.get("kind") != "tel22_morse_top10_local_curvature_a_refinement_coarse_5ps":
        raise ValueError("Unexpected 5 ps refinement summary kind")
    five_full = coarse_from_five_ps(five["full_reference"])
    five_candidate = coarse_from_five_ps(five["variants"]["top10_a0p850"])
    ten_full_coarse = full["coarse_fit_dt_0p002_to_0p005"]
    ten_candidate_coarse = candidate["coarse_fit_dt_0p002_to_0p005"]

    five_improves = (
        five_candidate["c2_spread_max_over_min"] < five_full["c2_spread_max_over_min"]
        and five_candidate["local_exponent_range"] < five_full["local_exponent_range"]
    )
    ten_improves = (
        candidate["c2_spread_max_over_min"] < full["c2_spread_max_over_min"]
        and candidate["local_exponent_range"] < full["local_exponent_range"]
    )
    strong = (
        ten_improves
        and comp["c2_spread_ratio"] <= 0.90
        and comp["local_exponent_range_ratio"] <= 0.80
        and candidate["max_relative_block_mean_drift"] <= 1.0e-4
    )
    if strong and five_improves:
        interpretation = "supports_a0p85_as_robust_morse_numerical_stabilizer_candidate"
    elif ten_improves and five_improves:
        interpretation = "a0p85_improvement_persists_but_is_partial_on_10ps_full_grid"
    else:
        interpretation = "a0p85_5ps_gain_not_robust_on_10ps_full_grid"

    out = {
        "schema_version": 1,
        "kind": "tel22_morse_top10_a0p85_robustness_10ps_fullgrid",
        "scope": "PaiNN disabled; production-like marker/non-bonded switched Morse runtime in both arms; full production Morse versus the exact test-20 top-10%-local-curvature a=0.85 candidate; 10 ps at dt=0.001,0.0015,0.002,0.003,0.004,0.005 ps; identical physical checkpoint state.",
        "stabilizer_context": "TEL22 Morse priors are treated here as numerical/structural stabilizers rather than physically calibrated interaction parameters; the validation target is integration robustness, not inference of a molecular Morse parameter.",
        "full_reference": full,
        "candidate_a0p85": {**candidate, "a_scale": 0.85, "scaled_a": float(candidate_meta["scaled_a"]), "selected_count": 18},
        "comparison_10ps_fullgrid": comp,
        "duration_robustness": {
            "five_ps_full_coarse": five_full,
            "five_ps_candidate_coarse": five_candidate,
            "ten_ps_full_coarse": ten_full_coarse,
            "ten_ps_candidate_coarse": ten_candidate_coarse,
            "five_ps_candidate_improves_c2_and_local_range": five_improves,
            "ten_ps_candidate_improves_c2_and_local_range": ten_improves,
        },
        "candidate_input_manifest": str(args.candidate_input_manifest.resolve()),
        "five_ps_summary": str(args.five_ps_summary.resolve()),
        "interpretation": interpretation,
        "caution": "This validates a numerical stabilizer choice with PaiNN disabled. If promoted to the full TEL22 Hamiltonian, the residual dataset/model must be regenerated or retrained against the modified priors before making production-accuracy claims.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[TEL22 MORSE a=0.85 ROBUSTNESS -- 10 ps FULL DT GRID]")
    print(
        f"full      : p={full['exponent_p']:.8f} R2={full['loglog_r2']:.8f} "
        f"C2spread={full['c2_spread_max_over_min']:.6f} localRange={full['local_exponent_range']:.6f} "
        f"fineP={full['fine_fit_dt_0p001_to_0p002']['exponent_p']:.6f}"
    )
    print(
        f"a0p85     : p={candidate['exponent_p']:.8f} R2={candidate['loglog_r2']:.8f} "
        f"C2spread={candidate['c2_spread_max_over_min']:.6f} localRange={candidate['local_exponent_range']:.6f} "
        f"fineP={candidate['fine_fit_dt_0p001_to_0p002']['exponent_p']:.6f}"
    )
    print(
        f"[RATIOS] C2spread={comp['c2_spread_ratio']:.4f} localRange={comp['local_exponent_range_ratio']:.4f} "
        f"small/coarse={comp['c2_small_over_coarse_ratio']:.4f} dR2={comp['delta_r2']:+.6f}"
    )
    print(
        "[DURATION ROBUSTNESS] "
        f"5ps_improves={five_improves} 10ps_improves={ten_improves} "
        f"candidate C2spread 5ps={five_candidate['c2_spread_max_over_min']:.4f} "
        f"10ps-coarse={ten_candidate_coarse['c2_spread_max_over_min']:.4f}"
    )
    print(f"[INTERPRETATION] {interpretation}")
    print(f"[REPORT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
