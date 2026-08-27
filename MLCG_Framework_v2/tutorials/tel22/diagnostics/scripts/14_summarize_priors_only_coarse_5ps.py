#!/usr/bin/env python3
"""Summarize the TEL22 priors-only 5 ps coarse-dt NVE consistency check."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = json.loads(args.report.read_text(encoding="utf-8"))
    definition = report.get("definition", {})
    if definition.get("hamiltonian_mode") != "conservative_classical_model_provenance_ml_disabled":
        raise ValueError("Report is not a priors-only (--disable-ml) certification")

    runs = sorted(report["runs"], key=lambda row: float(row["dt_ps"]))
    if len(runs) != 4:
        raise ValueError(f"Expected exactly four coarse-dt runs, found {len(runs)}")
    expected_dts = [0.002, 0.003, 0.004, 0.005]
    actual_dts = [float(row["dt_ps"]) for row in runs]
    if any(abs(a - b) > 1.0e-12 for a, b in zip(actual_dts, expected_dts)):
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
        rows.append(
            {
                "dt_ps": dt,
                "sigma_E": sigma,
                "C2_sigma_over_dt2": c2,
                "relative_block_mean_drift": float(row["relative_block_mean_drift"]),
            }
        )

    adjacent: list[dict[str, float]] = []
    for low, high in zip(rows, rows[1:]):
        p_local = math.log(high["sigma_E"] / low["sigma_E"]) / math.log(high["dt_ps"] / low["dt_ps"])
        adjacent.append(
            {
                "dt_low_ps": low["dt_ps"],
                "dt_high_ps": high["dt_ps"],
                "local_exponent_p": p_local,
            }
        )

    spread = max(c2_values) / min(c2_values)
    abs_p = abs(p - 2.0)
    if abs_p <= 0.05 and r2 >= 0.99 and spread <= 1.20:
        interpretation = "strongly_consistent_with_second_order_coarse_scaling"
    elif abs_p <= 0.10 and r2 >= 0.98 and spread <= 1.40:
        interpretation = "compatible_with_second_order_with_finite_window_variation"
    else:
        interpretation = "coarse_scaling_remains_nonideal_review_c2_and_local_exponents"

    out = {
        "schema_version": 1,
        "kind": "tel22_priors_only_native_espresso_coarse_5ps",
        "scope": (
            "Production TEL22 priors and identical equilibrated checkpoint; trained model retained only "
            "for provenance and disabled with --disable-ml. ESPResSo integrates the classical conservative "
            "Hamiltonian at native precision. Four coarse timesteps, 5 ps each."
        ),
        "precision_note": (
            "--ml-precision is irrelevant to force arithmetic because PaiNN is disabled. This is not an "
            "FP32-vs-FP64 ML comparison; the classical ESPResSo path uses its native numerical precision."
        ),
        "report": str(args.report.resolve()),
        "exponent_p": p,
        "abs_p_minus_2": abs_p,
        "loglog_r2": r2,
        "c2_spread_max_over_min": spread,
        "max_relative_block_mean_drift": max(row["relative_block_mean_drift"] for row in rows),
        "runs": rows,
        "adjacent_local_exponents": adjacent,
        "interpretation": interpretation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[TEL22 PRIORS-ONLY 5 ps COARSE-DT SUMMARY]")
    print(f"p             : {p:.8f}")
    print(f"|p-2|         : {abs_p:.8f}")
    print(f"R2            : {r2:.8f}")
    print(f"C2 spread     : {spread:.6f}")
    print(f"max drift     : {out['max_relative_block_mean_drift']:.3e}")
    print("local p       : " + ", ".join(
        f"{row['dt_low_ps']:g}->{row['dt_high_ps']:g}:{row['local_exponent_p']:.4f}" for row in adjacent
    ))
    print(f"[INTERPRETATION] {interpretation}")
    print(f"[REPORT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
