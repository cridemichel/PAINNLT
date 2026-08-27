#!/usr/bin/env python3
"""Compare TEL22 switched vs stock-shifted Morse with identical marker machinery."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

EXPECTED_DTS = [0.002, 0.003, 0.004, 0.005]
DEFAULT_R_CUT = 15.0
DEFAULT_SWITCH_FRACTION = 0.75


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_report(path: Path, expected_mode: str) -> dict[str, Any]:
    report = load_json(path)
    definition = report.get("definition", {})
    if definition.get("hamiltonian_mode") != "conservative_classical_model_provenance_ml_disabled":
        raise ValueError(f"{path} is not a priors-only --disable-ml report")
    mode = report.get("morse_switch_mode", definition.get("morse_switch_mode"))
    if expected_mode == "switched":
        # Existing reference reports predate the explicit mode field.
        if mode not in (None, "switched"):
            raise ValueError(f"{path} Morse mode is {mode!r}, expected switched")
    elif mode != expected_mode:
        raise ValueError(f"{path} Morse mode is {mode!r}, expected {expected_mode}")

    runs = sorted(report["runs"], key=lambda row: float(row["dt_ps"]))
    dts = [float(row["dt_ps"]) for row in runs]
    if len(runs) != 4 or any(abs(a - b) > 1.0e-12 for a, b in zip(dts, EXPECTED_DTS)):
        raise ValueError(f"Unexpected dt grid in {path}: {dts}")
    if any(abs(float(row["duration_ps"]) - 5.0) > 0.5 * float(row["dt_ps"]) + 1.0e-12 for row in runs):
        raise ValueError(f"{path} does not represent the requested 5 ps window")

    rows: list[dict[str, float]] = []
    for row in runs:
        dt = float(row["dt_ps"])
        sigma = float(row["sigma_E"])
        rows.append({
            "dt_ps": dt,
            "sigma_E": sigma,
            "C2_sigma_over_dt2": sigma / (dt * dt),
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
    return {
        "report": str(path.resolve()),
        "morse_switch_mode": expected_mode,
        "exponent_p": p,
        "abs_p_minus_2": abs(p - 2.0),
        "loglog_r2": float(scaling["loglog_r2"]),
        "c2_spread_max_over_min": max(c2) / min(c2),
        "max_relative_block_mean_drift": max(row["relative_block_mean_drift"] for row in rows),
        "runs": rows,
        "adjacent_local_exponents": local,
    }


def morse_inventory(priors_path: Path) -> dict[str, Any]:
    priors = load_json(priors_path)
    entries = [
        item for item in priors.get("bonds", [])
        if str(item.get("type", "harmonic")).lower() == "morse"
    ]
    if not entries:
        raise ValueError("Production priors contain no pair-specific Morse entries")
    r0 = [float(item["r0"]) for item in entries]
    rcut = [float(item.get("r_cut", DEFAULT_R_CUT)) for item in entries]
    rswitch = [
        float(item.get("r_switch", r + DEFAULT_SWITCH_FRACTION * (c - r)))
        for item, r, c in zip(entries, r0, rcut)
    ]
    return {
        "entries": len(entries),
        "explicit_r_switch_entries": sum("r_switch" in item for item in entries),
        "explicit_r_cut_entries": sum("r_cut" in item for item in entries),
        "r0_nm": {"min": min(r0), "median": statistics.median(r0), "max": max(r0)},
        "r_switch_nm": {"min": min(rswitch), "median": statistics.median(rswitch), "max": max(rswitch)},
        "r_cut_nm": {"min": min(rcut), "median": statistics.median(rcut), "max": max(rcut)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switched-report", type=Path, required=True)
    parser.add_argument("--stock-shifted-report", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--reference-validation", type=Path, required=True)
    parser.add_argument("--no-morse-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validation = load_json(args.reference_validation)
    if not validation.get("pass", False):
        raise ValueError("Switched reference validation did not pass")
    switched = summarize_report(args.switched_report, "switched")
    stock = summarize_report(args.stock_shifted_report, "stock-shifted")
    inventory = morse_inventory(args.priors)

    paired = []
    rel_sigma = []
    for a, b in zip(switched["runs"], stock["runs"]):
        if abs(a["dt_ps"] - b["dt_ps"]) > 1.0e-12:
            raise ValueError("Switched and stock-shifted dt grids do not align")
        ratio = b["sigma_E"] / a["sigma_E"]
        rel = abs(ratio - 1.0)
        rel_sigma.append(rel)
        paired.append({
            "dt_ps": a["dt_ps"],
            "sigma_switched": a["sigma_E"],
            "sigma_stock_shifted": b["sigma_E"],
            "sigma_ratio_stock_shifted_over_switched": ratio,
            "relative_sigma_difference": rel,
            "C2_switched": a["C2_sigma_over_dt2"],
            "C2_stock_shifted": b["C2_sigma_over_dt2"],
        })

    comparison = {
        "delta_p_stock_shifted_minus_switched": stock["exponent_p"] - switched["exponent_p"],
        "delta_abs_p_minus_2_stock_shifted_minus_switched": stock["abs_p_minus_2"] - switched["abs_p_minus_2"],
        "c2_spread_ratio_stock_shifted_over_switched": stock["c2_spread_max_over_min"] / switched["c2_spread_max_over_min"],
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

    if comparison["max_relative_sigma_difference"] <= 5.0e-3:
        interpretation = "switching_not_responsible_marker_nonbonded_path_remains_candidate"
    elif (
        comparison["c2_spread_ratio_stock_shifted_over_switched"] <= 0.85
        and comparison["delta_abs_p_minus_2_stock_shifted_minus_switched"] < 0.0
    ):
        interpretation = "switching_materially_contributes_to_coarse_nonideality"
    else:
        interpretation = "switching_changes_dynamics_but_does_not_cleanly_explain_coarse_nonideality"

    out = {
        "schema_version": 1,
        "kind": "tel22_priors_only_morse_switch_ab_coarse_5ps",
        "scope": (
            "Same production TEL22 priors, same equilibrated mechanical checkpoint, PaiNN disabled, "
            "same pair-specific marker mapping, same Morse D/a/r0/r_cut, same hybrid/non-bonded path, "
            "same dt grid and 5 ps windows. The only intended runtime change is switched C2-tail Morse "
            "versus ESPResSo stock-shifted Morse selected by switch_start=-1."
        ),
        "morse_inventory": inventory,
        "switched": switched,
        "stock_shifted": stock,
        "comparison": comparison,
        "no_morse_control": no_morse,
        "reference_validation": str(args.reference_validation.resolve()),
        "interpretation": interpretation,
        "caution": (
            "Stock-shifted Morse has the ESPResSo cutoff energy shift and a nonzero force immediately below "
            "cutoff. This is a diagnostic A/B, not a proposed production replacement."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[TEL22 MORSE SWITCH A/B -- 5 ps COARSE DT]")
    print(f"switched      : p={switched['exponent_p']:.8f} R2={switched['loglog_r2']:.8f} C2spread={switched['c2_spread_max_over_min']:.6f}")
    print(f"stock-shifted : p={stock['exponent_p']:.8f} R2={stock['loglog_r2']:.8f} C2spread={stock['c2_spread_max_over_min']:.6f}")
    print(f"max rel sigma delta : {comparison['max_relative_sigma_difference']:.6e}")
    print(f"C2 spread ratio     : {comparison['c2_spread_ratio_stock_shifted_over_switched']:.6f}")
    print(f"delta |p-2|         : {comparison['delta_abs_p_minus_2_stock_shifted_minus_switched']:+.6f}")
    print(f"Morse switch range  : {inventory['r_switch_nm']['min']:.3f}..{inventory['r_switch_nm']['max']:.3f} nm")
    print(f"[INTERPRETATION] {interpretation}")
    print(f"[REPORT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
