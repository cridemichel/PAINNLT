#!/usr/bin/env python3
"""Compare full TEL22 NVE scaling with the same Hamiltonian with PaiNN disabled."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def metrics(report: dict[str, Any]) -> dict[str, Any]:
    cert = report["certification"]
    scaling = cert["scaling"]
    runs = sorted(report["runs"], key=lambda row: float(row["dt_ps"]))
    c2 = [float(row["sigma_E"]) / float(row["dt_ps"]) ** 2 for row in runs]
    coarse = c2[-3:] if len(c2) >= 3 else c2
    worst = max(runs, key=lambda row: float(row["relative_block_mean_drift"]))
    return {
        "pass": bool(cert["pass"]),
        "scaling_pass": bool(cert["scaling_pass"]),
        "drift_pass": bool(cert["drift_pass"]),
        "exponent_p": float(scaling["exponent_p"]),
        "abs_p_minus_2": abs(float(scaling["exponent_p"]) - 2.0),
        "loglog_r2": float(scaling["loglog_r2"]),
        "c2_spread_max_over_min": max(c2) / min(c2),
        "dt_min_ps": float(runs[0]["dt_ps"]),
        "sigma_E_dt_min": float(runs[0]["sigma_E"]),
        "c2_dt_min": c2[0],
        "c2_coarse_median": statistics.median(coarse),
        "small_dt_c2_over_coarse_median": c2[0] / statistics.median(coarse),
        "max_relative_block_mean_drift": float(worst["relative_block_mean_drift"]),
        "max_drift_dt_ps": float(worst["dt_ps"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--priors-only", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.full, args.priors_only):
        if not path.is_file():
            raise FileNotFoundError(path)

    full_report = json.loads(args.full.read_text(encoding="utf-8"))
    priors_report = json.loads(args.priors_only.read_text(encoding="utf-8"))
    if full_report.get("definition", {}).get("hamiltonian_mode") != "model_active":
        raise ValueError("Full report is not a model_active TEL22 certification")
    if priors_report.get("definition", {}).get("hamiltonian_mode") != "conservative_classical_model_provenance_ml_disabled":
        raise ValueError("Priors-only report was not generated with --disable-ml")

    full = metrics(full_report)
    priors = metrics(priors_report)
    delta_p = priors["exponent_p"] - full["exponent_p"]
    improvement = full["abs_p_minus_2"] - priors["abs_p_minus_2"]
    floor_ratio = (
        priors["small_dt_c2_over_coarse_median"] / full["small_dt_c2_over_coarse_median"]
    )

    if priors["abs_p_minus_2"] <= 0.03 and improvement >= 0.05 and floor_ratio <= 0.9:
        interpretation = "strong_evidence_trained_painn_contributes_to_fp32_nonideal_scaling"
    elif improvement >= 0.04:
        interpretation = "evidence_trained_painn_contributes_to_fp32_nonideal_scaling"
    elif improvement <= -0.04:
        interpretation = "disabling_painn_worsens_scaling"
    else:
        interpretation = "priors_only_does_not_materially_separate_the_global_exponent"

    out = {
        "schema_version": 1,
        "kind": "tel22_nve_full_vs_priors_only_comparison",
        "scope": (
            "Same production TEL22 priors, topology, model provenance, checkpoint mechanical state, "
            "dt grid and ESPResSo runtime; PaiNN active in full branch and disabled via --disable-ml "
            "in priors-only branch."
        ),
        "full_report": str(args.full.resolve()),
        "priors_only_report": str(args.priors_only.resolve()),
        "full": full,
        "priors_only": priors,
        "comparison": {
            "delta_p_priors_only_minus_full": delta_p,
            "improvement_in_abs_p_minus_2": improvement,
            "sigma_E_dt_min_ratio_priors_only_over_full": priors["sigma_E_dt_min"] / full["sigma_E_dt_min"],
            "small_dt_c2_ratio_priors_only_over_full": floor_ratio,
        },
        "interpretation": interpretation,
        "caution": (
            "This A/B attributes any difference to activating the trained TEL22 PaiNN residual in the "
            "full Hamiltonian (including its coupling to the fixed priors). It does not imply a defect "
            "of the PaiNN architecture, C4 Toxvaerd smoothing, LibTorch, or ESPResSo in general."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[TEL22 FULL vs PRIORS-ONLY NVE]")
    print("variant       p         R2        C2spread  C2small/coarse  max_drift")
    for name, item in (("full", full), ("priors_only", priors)):
        print(
            f"{name:12s} {item['exponent_p']:.6f}  {item['loglog_r2']:.6f}  "
            f"{item['c2_spread_max_over_min']:.3f}     "
            f"{item['small_dt_c2_over_coarse_median']:.3f}           "
            f"{item['max_relative_block_mean_drift']:.3e}"
        )
    print(f"[PAINN A/B] {interpretation}")
    print(f"[REPORT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
