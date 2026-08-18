#!/usr/bin/env python3
"""Compare TorchMD and MLCG/ESPResSo reports for the shared synthetic PaiNN case."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

KCAL_TO_KJ = 4.184


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def median_runtime(report: dict) -> float:
    return float(statistics.median(float(r["ms_per_step"]) for r in report["runs"]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("torchmd_report")
    p.add_argument("mlcg_report")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    t = load(args.torchmd_report)
    m = load(args.mlcg_report)
    if t["precision"] != m["precision"]:
        raise ValueError("precision mismatch")
    tdt = [float(x) for x in t["dts_ps"]]
    mdt = [float(x) for x in m["dts_ps"]]
    if tdt != mdt:
        raise ValueError("dt-grid mismatch")
    if int(t["particles"]) != int(m["particles"]):
        raise ValueError("particle-count mismatch")
    if int(t["graph"]["directed_edges"]) != int(m["graph"]["directed_edges"]):
        raise ValueError("edge-count mismatch")
    t_fingerprint = t["synthetic_painn"]["parameter_sha256_canonical_float64"]
    m_fingerprint = m["painn"]["parameter_sha256_canonical_float64"]
    if t_fingerprint != m_fingerprint:
        raise ValueError(
            "PaiNN fingerprint mismatch: TorchMD and MLCG reports do not use the same canonical weights"
        )

    ts = t["certification"]["scaling"]
    ms = m["certification"]["scaling"]
    t_runtime = median_runtime(t)
    m_runtime = median_runtime(m)
    parity_rows = [r["static_parity"] for r in m["runs"]]
    sigma_rows = []
    for tr, mr in zip(t["runs"], m["runs"]):
        t_sigma = float(tr["sigma_E"])
        m_sigma_kcal = float(mr["sigma_E"]) / KCAL_TO_KJ
        sigma_rows.append({
            "dt_ps": float(tr["dt_ps"]),
            "torchmd_sigma_kcal_mol": t_sigma,
            "mlcg_sigma_kcal_mol_equivalent": m_sigma_kcal,
            "relative_difference": abs(m_sigma_kcal - t_sigma) / max(abs(t_sigma), 1.0e-30),
        })

    out = {
        "kind": "torchmd_vs_mlcg_exact_synthetic_painn_comparison",
        "precision": t["precision"],
        "torchmd": {
            "p": float(ts["exponent_p"]),
            "r2": float(ts["loglog_r2"]),
            "c2_spread": float(t["certification"]["c2_spread_max_over_min"]),
            "max_drift": max(float(r["relative_block_mean_drift"]) for r in t["runs"]),
            "median_ms_per_step": t_runtime,
        },
        "mlcg": {
            "p": float(ms["exponent_p"]),
            "r2": float(ms["loglog_r2"]),
            "c2_spread": float(m["certification"]["c2_spread_max_over_min"]),
            "max_drift": max(float(r["relative_block_mean_drift"]) for r in m["runs"]),
            "median_ms_per_step": m_runtime,
        },
        "parameter_sha256_canonical_float64": t_fingerprint,
        "delta_p_mlcg_minus_torchmd": float(ms["exponent_p"] - ts["exponent_p"]),
        "runtime_ratio_mlcg_over_torchmd": m_runtime / t_runtime,
        "static_parity": {
            "all_pass": all(bool(x["pass"]) for x in parity_rows),
            "max_energy_relative_error": max(float(x["energy_relative_error"]) for x in parity_rows),
            "max_force_relative_rms_error": max(float(x["force_relative_rms_error"]) for x in parity_rows),
        },
        "sigma_comparison": sigma_rows,
    }
    print("[TORCHMD vs MLCG -- EXACT SYNTHETIC PAINN]")
    print(f"precision : {out['precision']}")
    print(
        "TorchMD   : p={p:.6f}  R2={r2:.6f}  C2spread={c2:.3f}  drift={drift:.3e}  {ms:.3f} ms/step".format(
            p=out["torchmd"]["p"], r2=out["torchmd"]["r2"], c2=out["torchmd"]["c2_spread"],
            drift=out["torchmd"]["max_drift"], ms=out["torchmd"]["median_ms_per_step"]
        )
    )
    print(
        "MLCG      : p={p:.6f}  R2={r2:.6f}  C2spread={c2:.3f}  drift={drift:.3e}  {ms:.3f} ms/step".format(
            p=out["mlcg"]["p"], r2=out["mlcg"]["r2"], c2=out["mlcg"]["c2_spread"],
            drift=out["mlcg"]["max_drift"], ms=out["mlcg"]["median_ms_per_step"]
        )
    )
    print(f"delta p (MLCG-TorchMD): {out['delta_p_mlcg_minus_torchmd']:+.6f}")
    print(f"runtime MLCG/TorchMD : {out['runtime_ratio_mlcg_over_torchmd']:.3f}x")
    print(
        "static parity          : {status}  max dErel={de:.3e}  max dFrmsrel={df:.3e}".format(
            status="PASS" if out["static_parity"]["all_pass"] else "FAIL",
            de=out["static_parity"]["max_energy_relative_error"],
            df=out["static_parity"]["max_force_relative_rms_error"],
        )
    )
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"report                  : {args.output}")
    return 0 if out["static_parity"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
