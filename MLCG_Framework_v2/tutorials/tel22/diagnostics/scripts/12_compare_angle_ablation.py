#!/usr/bin/env python3
"""Compare baseline and no-angle TEL22 NVE scaling reports."""
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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--no-angles", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.baseline, args.no_angles, args.inputs):
        if not path.is_file():
            raise FileNotFoundError(path)

    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    base = metrics(json.loads(args.baseline.read_text(encoding="utf-8")))
    no_angles = metrics(json.loads(args.no_angles.read_text(encoding="utf-8")))
    delta_p = no_angles["exponent_p"] - base["exponent_p"]
    improvement = base["abs_p_minus_2"] - no_angles["abs_p_minus_2"]

    if no_angles["abs_p_minus_2"] <= 0.03 and improvement >= 0.05:
        interpretation = "strong_evidence_angles_contribute_to_nonideal_scaling"
    elif improvement >= 0.03:
        interpretation = "evidence_angles_contribute_to_nonideal_scaling"
    elif improvement <= -0.03:
        interpretation = "removing_angles_worsens_scaling"
    else:
        interpretation = "angle_ablation_does_not_materially_change_global_exponent"

    out = {
        "schema_version": 1,
        "kind": "tel22_nve_angle_prior_ablation_comparison",
        "scope": inputs["scope"],
        "removed_angle_entries": int(inputs["no_angles"]["removed_angle_entries"]),
        "baseline_report": str(args.baseline.resolve()),
        "no_angles_report": str(args.no_angles.resolve()),
        "baseline": base,
        "no_angles": no_angles,
        "comparison": {
            "delta_p_no_angles_minus_baseline": delta_p,
            "improvement_in_abs_p_minus_2": improvement,
            "sigma_E_dt_min_ratio_no_angles_over_baseline": no_angles["sigma_E_dt_min"] / base["sigma_E_dt_min"],
            "small_dt_c2_ratio_no_angles_over_baseline": (
                no_angles["small_dt_c2_over_coarse_median"] / base["small_dt_c2_over_coarse_median"]
            ),
        },
        "interpretation": interpretation,
        "warning": inputs["warning"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[TEL22 ANGLE NVE ABLATION]")
    print("variant      p         R2        C2spread  C2small/coarse  max_drift")
    for name, item in (("baseline", base), ("no_angles", no_angles)):
        print(
            f"{name:11s} {item['exponent_p']:.6f}  {item['loglog_r2']:.6f}  "
            f"{item['c2_spread_max_over_min']:.3f}     "
            f"{item['small_dt_c2_over_coarse_median']:.3f}           "
            f"{item['max_relative_block_mean_drift']:.3e}"
        )
    print(f"[ANGLE] {interpretation}")
    print(f"[REPORT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
