#!/usr/bin/env python3
"""Compare analytic-reference and neural-potential TorchMD NVE certification reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def median_ms(report: dict) -> float:
    return float(statistics.median(float(row["ms_per_step"]) for row in report["runs"]))


def max_drift(report: dict) -> float:
    return max(float(row["relative_block_mean_drift"]) for row in report["runs"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("analytic_report", type=Path)
    parser.add_argument("neural_report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    analytic = load(args.analytic_report)
    neural = load(args.neural_report)
    for field in ("particles", "device", "precision", "duration_ps", "dts_ps"):
        if analytic.get(field) != neural.get(field):
            raise ValueError(f"reports are not matched: {field} differs ({analytic.get(field)!r} != {neural.get(field)!r})")

    a_cert = analytic["certification"]
    n_cert = neural["certification"]
    a_ms = median_ms(analytic)
    n_ms = median_ms(neural)
    summary = {
        "matched": True,
        "particles": analytic["particles"],
        "device": analytic["device"],
        "precision": analytic["precision"],
        "duration_ps": analytic["duration_ps"],
        "dts_ps": analytic["dts_ps"],
        "analytic": {
            "pass": bool(a_cert["pass"]),
            "p": float(a_cert["scaling"]["exponent_p"]),
            "r2": float(a_cert["scaling"]["loglog_r2"]),
            "c2_spread": float(a_cert["c2_spread_max_over_min"]),
            "max_drift": max_drift(analytic),
            "median_ms_per_step": a_ms,
        },
        "neural": {
            "pass": bool(n_cert["pass"]),
            "p": float(n_cert["scaling"]["exponent_p"]),
            "r2": float(n_cert["scaling"]["loglog_r2"]),
            "c2_spread": float(n_cert["c2_spread_max_over_min"]),
            "max_drift": max_drift(neural),
            "median_ms_per_step": n_ms,
        },
        "delta_p_neural_minus_analytic": float(n_cert["scaling"]["exponent_p"] - a_cert["scaling"]["exponent_p"]),
        "neural_over_analytic_median_runtime": float(n_ms / a_ms),
    }

    print("[TORCHMD NVE ANALYTIC vs NEURAL]")
    print(f"device / precision : {summary['device']} / {summary['precision']}")
    print(
        f"analytic            : p={summary['analytic']['p']:.6f}  "
        f"R2={summary['analytic']['r2']:.6f}  C2spread={summary['analytic']['c2_spread']:.3f}  "
        f"drift={summary['analytic']['max_drift']:.3e}"
    )
    print(
        f"neural              : p={summary['neural']['p']:.6f}  "
        f"R2={summary['neural']['r2']:.6f}  C2spread={summary['neural']['c2_spread']:.3f}  "
        f"drift={summary['neural']['max_drift']:.3e}"
    )
    print(f"delta p             : {summary['delta_p_neural_minus_analytic']:+.6f}")
    print(f"runtime ratio       : {summary['neural_over_analytic_median_runtime']:.3f}x (diagnostic only)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"report              : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
