#!/usr/bin/env python3
"""Compare the completed TEL22 30 K FP32 and FP64 NVE certifications."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    cert = report["certification"]
    scaling = cert["scaling"]
    runs = sorted(report["runs"], key=lambda item: float(item["dt_ps"]))
    c2 = [float(run["sigma_E"]) / float(run["dt_ps"]) ** 2 for run in runs]
    coarse = median(c2[-min(3, len(c2)):])
    worst = max(runs, key=lambda item: float(item["relative_block_mean_drift"]))
    return {
        "precision": str(report.get("ml_precision", report.get("precision", "unknown"))),
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
        "c2_coarse_median": coarse,
        "small_dt_c2_over_coarse_median": c2[0] / coarse,
        "max_relative_block_mean_drift": float(worst["relative_block_mean_drift"]),
        "max_drift_dt_ps": float(worst["dt_ps"]),
    }


def build_comparison(fp32: dict[str, Any], fp64: dict[str, Any]) -> dict[str, Any]:
    m32 = _metrics(fp32)
    m64 = _metrics(fp64)
    improvement = m32["abs_p_minus_2"] - m64["abs_p_minus_2"]
    # Diagnostic, deliberately not a new certification gate.  The threshold only labels whether
    # FP64 is visibly closer to second order than FP32; raw metrics remain authoritative.
    visibly_closer = improvement >= 0.05
    fp64_near_second_order = m64["abs_p_minus_2"] <= 0.10 and m64["loglog_r2"] >= 0.97
    conclusion = (
        "supports_precision_specific_tel22_effect"
        if visibly_closer and fp64_near_second_order
        else "inconclusive_from_30K_precision_pair"
    )
    return {
        "kind": "tel22_30K_fp32_fp64_nve_comparison",
        "scope": "Same 30 K iso-configurational checkpoint construction and TEL22 Hamiltonian; only ML inference precision differs.",
        "fp32": m32,
        "fp64": m64,
        "delta_p_fp64_minus_fp32": m64["exponent_p"] - m32["exponent_p"],
        "improvement_in_abs_p_minus_2_fp64_vs_fp32": improvement,
        "sigma_E_dt_min_fp64_over_fp32": m64["sigma_E_dt_min"] / m32["sigma_E_dt_min"],
        "small_dt_c2_ratio_fp64_over_fp32": (
            m64["small_dt_c2_over_coarse_median"] / m32["small_dt_c2_over_coarse_median"]
        ),
        "diagnostic_flags": {
            "fp64_near_second_order": fp64_near_second_order,
            "fp64_visibly_closer_to_second_order_than_fp32": visibly_closer,
        },
        "interpretation": conclusion,
        "note": (
            "This comparison does not identify which TEL22 component creates the precision sensitivity; "
            "it only tests whether the 30 K non-ideal exponent is precision dependent."
        ),
    }


def print_comparison(payload: dict[str, Any]) -> None:
    print("\n[TEL22 30 K FP32 vs FP64 NVE CLOSURE CHECK]")
    print("precision   p          R2         C2spread   C2small/coarse  max_drift")
    for key in ("fp32", "fp64"):
        row = payload[key]
        print(
            f"{key:<10} {row['exponent_p']:<10.6f} {row['loglog_r2']:<10.6f} "
            f"{row['c2_spread_max_over_min']:<10.3f} "
            f"{row['small_dt_c2_over_coarse_median']:<15.3f} "
            f"{row['max_relative_block_mean_drift']:.3e}"
        )
    print(f"delta p (FP64-FP32)       : {payload['delta_p_fp64_minus_fp32']:+.6f}")
    print(f"improvement |p-2|         : {payload['improvement_in_abs_p_minus_2_fp64_vs_fp32']:+.6f}")
    print(f"sigma(dt_min) FP64/FP32   : {payload['sigma_E_dt_min_fp64_over_fp32']:.3f}")
    print(f"interpretation            : {payload['interpretation']}")


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=here / "results")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = args.results_dir.expanduser().resolve()
    fp32_path = results / "T30K_float32" / "nve_certification_report.json"
    fp64_path = results / "T30K_float64" / "nve_certification_report.json"
    payload = build_comparison(_load(fp32_path), _load(fp64_path))
    output = (args.output or (results / "T30K_fp32_vs_fp64_closure.json")).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print_comparison(payload)
    print(f"[REPORT] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
