#!/usr/bin/env python3
"""Compare marker/non-bonded and bonded-analytic pair-specific Morse runtimes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_DTS = [0.002, 0.003, 0.004, 0.005]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_report(path: Path, expected_runtime: str) -> dict[str, Any]:
    report = load_json(path)
    definition = report.get("definition", {})
    if definition.get("hamiltonian_mode") != "conservative_classical_model_provenance_ml_disabled":
        raise ValueError(f"{path} is not a priors-only --disable-ml report")
    runtime = report.get(
        "pair_specific_morse_runtime", definition.get("pair_specific_morse_runtime")
    )
    if expected_runtime == "marker-nonbonded":
        # The reusable reference predates the explicit runtime field.
        if runtime not in (None, "marker-nonbonded"):
            raise ValueError(f"{path} Morse runtime is {runtime!r}, expected marker-nonbonded")
    elif runtime != expected_runtime:
        raise ValueError(f"{path} Morse runtime is {runtime!r}, expected {expected_runtime}")

    runs = sorted(report["runs"], key=lambda row: float(row["dt_ps"]))
    dts = [float(row["dt_ps"]) for row in runs]
    if len(runs) != 4 or any(abs(a - b) > 1.0e-12 for a, b in zip(dts, EXPECTED_DTS)):
        raise ValueError(f"Unexpected dt grid in {path}: {dts}")
    if any(
        abs(float(row["duration_ps"]) - 5.0) > 0.5 * float(row["dt_ps"]) + 1.0e-12
        for row in runs
    ):
        raise ValueError(f"{path} does not represent the requested 5 ps window")

    rows: list[dict[str, Any]] = []
    for row in runs:
        dt = float(row["dt_ps"])
        sigma = float(row["sigma_E"])
        # Older reusable NVE reports predate the explicit block_mean_drift field.
        # Recover the same absolute quantity from the stored block means when
        # possible; otherwise leave it unavailable.  This diagnostic is not
        # used for the Morse runtime A/B attribution.
        if "block_mean_drift" in row:
            block_mean_drift_abs = abs(float(row["block_mean_drift"]))
        elif "first_block_mean_E" in row and "last_block_mean_E" in row:
            block_mean_drift_abs = abs(
                float(row["last_block_mean_E"]) - float(row["first_block_mean_E"])
            )
        else:
            block_mean_drift_abs = None
        rows.append({
            "dt_ps": dt,
            "sigma_E": sigma,
            "C2_sigma_over_dt2": sigma / (dt * dt),
            "block_mean_drift_abs": block_mean_drift_abs,
            "relative_block_mean_drift": float(row["relative_block_mean_drift"]),
        })
    local = []
    for low, high in zip(rows, rows[1:]):
        local.append({
            "dt_low_ps": low["dt_ps"],
            "dt_high_ps": high["dt_ps"],
            "local_exponent_p": math.log(high["sigma_E"] / low["sigma_E"]) / math.log(high["dt_ps"] / low["dt_ps"]),
        })
    scaling = report["certification"]["scaling"]
    p = float(scaling["exponent_p"])
    c2 = [row["C2_sigma_over_dt2"] for row in rows]
    local_values = [x["local_exponent_p"] for x in local]
    abs_block_drifts = [
        float(row["block_mean_drift_abs"])
        for row in rows
        if row["block_mean_drift_abs"] is not None
    ]
    return {
        "report": str(path.resolve()),
        "pair_specific_morse_runtime": expected_runtime,
        "exponent_p": p,
        "abs_p_minus_2": abs(p - 2.0),
        "loglog_r2": float(scaling["loglog_r2"]),
        "c2_spread_max_over_min": max(c2) / min(c2),
        "local_exponent_range": max(local_values) - min(local_values),
        "max_abs_block_mean_drift": max(abs_block_drifts) if abs_block_drifts else None,
        "max_relative_block_mean_drift": max(row["relative_block_mean_drift"] for row in rows),
        "runs": rows,
        "adjacent_local_exponents": local,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-report", type=Path, required=True)
    parser.add_argument("--bonded-report", type=Path, required=True)
    parser.add_argument("--reference-validation", type=Path, required=True)
    parser.add_argument("--no-morse-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validation = load_json(args.reference_validation)
    if not validation.get("pass", False):
        raise ValueError("Marker/non-bonded reference validation did not pass")
    marker = summarize_report(args.marker_report, "marker-nonbonded")
    bonded = summarize_report(args.bonded_report, "bonded-analytic")

    paired = []
    rel_sigma = []
    for ref, cand in zip(marker["runs"], bonded["runs"]):
        if abs(ref["dt_ps"] - cand["dt_ps"]) > 1.0e-12:
            raise ValueError("Runtime A/B dt grids do not align")
        ratio = cand["sigma_E"] / ref["sigma_E"]
        rel = abs(ratio - 1.0)
        rel_sigma.append(rel)
        paired.append({
            "dt_ps": ref["dt_ps"],
            "sigma_marker_nonbonded": ref["sigma_E"],
            "sigma_bonded_analytic": cand["sigma_E"],
            "sigma_ratio_bonded_over_marker": ratio,
            "relative_sigma_difference": rel,
            "C2_marker_nonbonded": ref["C2_sigma_over_dt2"],
            "C2_bonded_analytic": cand["C2_sigma_over_dt2"],
        })

    comparison = {
        "delta_p_bonded_minus_marker": bonded["exponent_p"] - marker["exponent_p"],
        "delta_abs_p_minus_2_bonded_minus_marker": bonded["abs_p_minus_2"] - marker["abs_p_minus_2"],
        "c2_spread_ratio_bonded_over_marker": bonded["c2_spread_max_over_min"] / marker["c2_spread_max_over_min"],
        "local_exponent_range_ratio_bonded_over_marker": bonded["local_exponent_range"] / marker["local_exponent_range"],
        "max_relative_sigma_difference": max(rel_sigma),
        "paired_runs": paired,
    }

    no_morse = None
    if args.no_morse_summary is not None and args.no_morse_summary.is_file():
        ref = load_json(args.no_morse_summary)
        no_morse = {
            "summary": str(args.no_morse_summary.resolve()),
            "exponent_p": float(ref["exponent_p"]),
            "c2_spread_max_over_min": float(ref["c2_spread_max_over_min"]),
        }

    spread_ratio = comparison["c2_spread_ratio_bonded_over_marker"]
    local_ratio = comparison["local_exponent_range_ratio_bonded_over_marker"]
    if comparison["max_relative_sigma_difference"] <= 0.02:
        interpretation = "morse_curvature_not_marker_nonbonded_runtime_explains_coarse_structure"
    elif spread_ratio <= 0.85 and local_ratio <= 0.70:
        interpretation = "marker_nonbonded_hybrid_runtime_materially_contributes_to_coarse_nonideality"
    elif spread_ratio < 1.0 or local_ratio < 1.0:
        interpretation = "bonded_runtime_improves_some_scaling_metrics_but_does_not_fully_explain_nonideality"
    else:
        interpretation = "bonded_runtime_changes_dynamics_without_improving_coarse_scaling"

    out = {
        "schema_version": 1,
        "kind": "tel22_priors_only_morse_runtime_ab_coarse_5ps",
        "scope": (
            "Same production TEL22 priors, same equilibrated checkpoint and particle set, PaiNN disabled, "
            "same 180 explicit Morse endpoint pairs and same D/a/r0/r_cut, same WCA/harmonic priors, same "
            "dt grid and 5 ps windows. Reference uses technical markers plus non-bonded Morse/hybrid "
            "decomposition. Candidate keeps the technical markers inert for checkpoint parity and applies "
            "analytic MorseBond directly to the physical COM/site endpoints using the regular decomposition."
        ),
        "energy_gauge_note": (
            "Analytic MorseBond energy differs from the production unswitched Morse energy by +D per contact. "
            "This constant does not affect forces, sigma_E, C2, p, or absolute block-mean energy differences. "
            "Relative drift denominators are gauge-dependent and are reported but not used for A/B attribution."
        ),
        "marker_nonbonded": marker,
        "bonded_analytic": bonded,
        "comparison": comparison,
        "no_morse_control": no_morse,
        "reference_validation": str(args.reference_validation.resolve()),
        "interpretation": interpretation,
        "caution": (
            "This is a diagnostic runtime substitution, not a production parameter change. MorseBond returns "
            "a broken-bond signal at r>=r_cut; the existing switch A/B showed the sampled TEL22 trajectories "
            "remain far below the 15 nm cutoff, but production replacement requires a separate safety decision."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[TEL22 MORSE RUNTIME A/B -- 5 ps COARSE DT]")
    print(f"marker/nonbonded : p={marker['exponent_p']:.8f} R2={marker['loglog_r2']:.8f} C2spread={marker['c2_spread_max_over_min']:.6f}")
    print(f"bonded analytic  : p={bonded['exponent_p']:.8f} R2={bonded['loglog_r2']:.8f} C2spread={bonded['c2_spread_max_over_min']:.6f}")
    print(f"max rel sigma delta : {comparison['max_relative_sigma_difference']:.6e}")
    print(f"C2 spread ratio     : {spread_ratio:.6f}")
    print(f"local-p range ratio : {local_ratio:.6f}")
    print(f"delta |p-2|         : {comparison['delta_abs_p_minus_2_bonded_minus_marker']:+.6f}")
    print(f"[INTERPRETATION] {interpretation}")
    print(f"[REPORT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
