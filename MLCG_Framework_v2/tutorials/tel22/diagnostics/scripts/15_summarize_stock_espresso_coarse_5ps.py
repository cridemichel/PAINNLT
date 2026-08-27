#!/usr/bin/env python3
"""Summarize TEL22 stock-ESPResSo (PaiNN off, custom Morse off) coarse NVE."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_DTS = [0.002, 0.003, 0.004, 0.005]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-priors-only", type=Path)
    args = parser.parse_args()

    report = load_json(args.report)
    definition = report.get("definition", {})
    if definition.get("hamiltonian_mode") != "conservative_classical_model_provenance_ml_disabled":
        raise ValueError("Report is not a classical --disable-ml certification")

    inputs = load_json(args.inputs)
    variant = inputs.get("variants", {}).get("no_morse", {})
    if int(variant.get("remaining_morse_entries", -1)) != 0:
        raise ValueError("Input manifest does not certify Morse-free priors")
    if int(variant.get("remaining_morse_markers", -1)) != 0:
        raise ValueError("Input manifest does not certify a marker-free checkpoint")

    runs = sorted(report["runs"], key=lambda row: float(row["dt_ps"]))
    if len(runs) != 4:
        raise ValueError(f"Expected exactly four coarse-dt runs, found {len(runs)}")
    actual_dts = [float(row["dt_ps"]) for row in runs]
    if any(abs(a - b) > 1.0e-12 for a, b in zip(actual_dts, EXPECTED_DTS)):
        raise ValueError(f"Unexpected dt grid: {actual_dts}")
    if any(abs(float(row["duration_ps"]) - 5.0) > 0.5 * float(row["dt_ps"]) + 1.0e-12 for row in runs):
        raise ValueError("Report does not represent the requested 5 ps physical window")

    scaling = report["certification"]["scaling"]
    p = float(scaling["exponent_p"])
    r2 = float(scaling["loglog_r2"])
    rows: list[dict[str, float]] = []
    c2_values: list[float] = []
    for row in runs:
        dt = float(row["dt_ps"])
        sigma = float(row["sigma_E"])
        c2 = sigma / (dt * dt)
        c2_values.append(c2)
        rows.append({
            "dt_ps": dt,
            "sigma_E": sigma,
            "C2_sigma_over_dt2": c2,
            "relative_block_mean_drift": float(row["relative_block_mean_drift"]),
        })

    adjacent: list[dict[str, float]] = []
    for low, high in zip(rows, rows[1:]):
        p_local = math.log(high["sigma_E"] / low["sigma_E"]) / math.log(high["dt_ps"] / low["dt_ps"])
        adjacent.append({
            "dt_low_ps": low["dt_ps"],
            "dt_high_ps": high["dt_ps"],
            "local_exponent_p": p_local,
        })

    spread = max(c2_values) / min(c2_values)
    abs_p = abs(p - 2.0)
    if abs_p <= 0.05 and r2 >= 0.99 and spread <= 1.20:
        interpretation = "stock_espresso_strongly_consistent_with_second_order"
    elif abs_p <= 0.10 and r2 >= 0.98 and spread <= 1.40:
        interpretation = "stock_espresso_compatible_with_second_order_finite_window_variation"
    else:
        interpretation = "stock_espresso_coarse_scaling_nonideal_review_local_structure"

    comparison = None
    if args.reference_priors_only is not None:
        ref = load_json(args.reference_priors_only)
        comparison = {
            "reference_summary": str(args.reference_priors_only.resolve()),
            "reference_kind": ref.get("kind"),
            "delta_p_stock_minus_priors_only": p - float(ref["exponent_p"]),
            "delta_abs_p_minus_2_stock_minus_priors_only": abs_p - float(ref["abs_p_minus_2"]),
            "c2_spread_ratio_stock_over_priors_only": spread / float(ref["c2_spread_max_over_min"]),
            "max_drift_ratio_stock_over_priors_only": (
                max(row["relative_block_mean_drift"] for row in rows)
                / float(ref["max_relative_block_mean_drift"])
            ),
        }

    out = {
        "schema_version": 1,
        "kind": "tel22_stock_espresso_no_ml_no_custom_morse_coarse_5ps",
        "scope": (
            "Derived TEL22 Hamiltonian with trained PaiNN disabled and all custom pair-specific Morse priors removed. "
            "Production harmonic bonds, harmonic angles, WCA pair parameters and topology exclusions are retained. "
            "WCA is evaluated through stock ESPResSo LennardJones. Four coarse timesteps, 5 ps each."
        ),
        "stock_interaction_inventory": {
            "painn_active": False,
            "custom_morse_active": False,
            "conservative_splines_active": False,
            "dihedrals_active": False,
            "harmonic_bonds": 210,
            "harmonic_angles": 200,
            "wca_runtime": "espressomd.interactions.LennardJones",
        },
        "input_manifest": str(args.inputs.resolve()),
        "report": str(args.report.resolve()),
        "exponent_p": p,
        "abs_p_minus_2": abs_p,
        "loglog_r2": r2,
        "c2_spread_max_over_min": spread,
        "max_relative_block_mean_drift": max(row["relative_block_mean_drift"] for row in rows),
        "runs": rows,
        "adjacent_local_exponents": adjacent,
        "comparison_to_priors_only_with_morse": comparison,
        "interpretation": interpretation,
        "caution": (
            "This is an integration/control diagnostic. Removing Morse changes the physical Hamiltonian; "
            "the result should not be interpreted as a reparameterized TEL22 production model."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[TEL22 STOCK ESPRESSO 5 ps COARSE-DT SUMMARY]")
    print(f"p             : {p:.8f}")
    print(f"|p-2|         : {abs_p:.8f}")
    print(f"R2            : {r2:.8f}")
    print(f"C2 spread     : {spread:.6f}")
    print(f"max drift     : {out['max_relative_block_mean_drift']:.3e}")
    print("local p       : " + ", ".join(
        f"{row['dt_low_ps']:g}->{row['dt_high_ps']:g}:{row['local_exponent_p']:.4f}" for row in adjacent
    ))
    if comparison is not None:
        print(f"delta |p-2| vs priors-only+Morse : {comparison['delta_abs_p_minus_2_stock_minus_priors_only']:+.6f}")
        print(f"C2 spread ratio vs priors-only   : {comparison['c2_spread_ratio_stock_over_priors_only']:.6f}")
    print(f"[INTERPRETATION] {interpretation}")
    print(f"[REPORT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
