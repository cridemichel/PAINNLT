#!/usr/bin/env python3
"""Compare synthetic PaiNN NVE FP32/FP64 reports and flag a precision-selective scaling loss."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str):
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("kind") != "torchmd_tel22_sized_synthetic_painn_nve_certification":
        raise ValueError(f"not a synthetic-PaiNN report: {p}")
    return p, data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("float32_report")
    ap.add_argument("float64_report")
    ap.add_argument("--output", default="results/painn_fp32_vs_fp64.json")
    args = ap.parse_args()
    p32, a = load(args.float32_report)
    p64, b = load(args.float64_report)
    if a["precision"] != "float32" or b["precision"] != "float64":
        raise ValueError("first report must be float32 and second float64")
    keys = ("particles", "duration_ps", "dts_ps", "initial_state_seed")
    for key in keys:
        if a[key] != b[key]:
            raise ValueError(f"report mismatch in {key}")
    if a["synthetic_painn"]["parameter_sha256_canonical_float64"] != b["synthetic_painn"]["parameter_sha256_canonical_float64"]:
        raise ValueError("FP32/FP64 reports use different PaiNN parameters")
    p32v = float(a["certification"]["scaling"]["exponent_p"])
    p64v = float(b["certification"]["scaling"]["exponent_p"])
    out = {
        "kind": "torchmd_synthetic_painn_fp32_vs_fp64",
        "float32_report": str(p32),
        "float64_report": str(p64),
        "p_float32": p32v,
        "p_float64": p64v,
        "delta_p_float32_minus_float64": p32v - p64v,
        "r2_float32": float(a["certification"]["scaling"]["loglog_r2"]),
        "r2_float64": float(b["certification"]["scaling"]["loglog_r2"]),
        "c2_spread_float32": float(a["certification"]["c2_spread_max_over_min"]),
        "c2_spread_float64": float(b["certification"]["c2_spread_max_over_min"]),
        "precision_selective_loss_indicator": bool(p32v < 1.95 and p64v >= 1.95),
    }
    op = Path(args.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[TORCHMD SYNTHETIC PAINN FP32 vs FP64]")
    print(f"FP32 : p={p32v:.6f}  R2={out['r2_float32']:.6f}  C2spread={out['c2_spread_float32']:.3f}")
    print(f"FP64 : p={p64v:.6f}  R2={out['r2_float64']:.6f}  C2spread={out['c2_spread_float64']:.3f}")
    print(f"delta p (32-64) : {out['delta_p_float32_minus_float64']:+.6f}")
    print(f"precision-selective loss indicator: {out['precision_selective_loss_indicator']}")
    print(f"report: {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
