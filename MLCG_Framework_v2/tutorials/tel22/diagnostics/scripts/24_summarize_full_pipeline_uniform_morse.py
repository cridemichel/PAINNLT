#!/usr/bin/env python3
"""Summarize the isolated TEL22 full rebuild with uniform Morse a=0.255."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_metrics(report_path: Path) -> dict[str, Any]:
    r = json.loads(report_path.read_text())
    runs = sorted(r["runs"], key=lambda x: float(x["dt_ps"]))
    s = r["certification"]["scaling"]
    c2 = [float(x["sigma_E"]) / float(x["dt_ps"]) ** 2 for x in runs]
    coarse = [v for x, v in zip(runs, c2) if float(x["dt_ps"]) >= 0.003]
    coarse_med = sorted(coarse)[len(coarse)//2] if coarse else float("nan")
    local = []
    for lo, hi in zip(runs[:-1], runs[1:]):
        p = math.log(float(hi["sigma_E"]) / float(lo["sigma_E"])) / math.log(float(hi["dt_ps"]) / float(lo["dt_ps"]))
        local.append({"dt_low_ps": float(lo["dt_ps"]), "dt_high_ps": float(hi["dt_ps"]), "local_exponent_p": p})
    return {
        "report": str(report_path.resolve()),
        "report_sha256": sha256(report_path),
        "exponent_p": float(s["exponent_p"]),
        "abs_p_minus_2": abs(float(s["exponent_p"]) - 2.0),
        "loglog_r2": float(s["loglog_r2"]),
        "c2_spread_max_over_min": max(c2) / min(c2),
        "c2_small_over_coarse_median": c2[0] / coarse_med,
        "local_exponent_range": max(x["local_exponent_p"] for x in local) - min(x["local_exponent_p"] for x in local),
        "adjacent_local_exponents": local,
        "max_relative_block_mean_drift": max(float(x["relative_block_mean_drift"]) for x in runs),
        "runs": [
            {
                "dt_ps": float(x["dt_ps"]),
                "sigma_E": float(x["sigma_E"]),
                "C2_sigma_over_dt2": float(x["sigma_E"]) / float(x["dt_ps"]) ** 2,
                "relative_block_mean_drift": float(x["relative_block_mean_drift"]),
            }
            for x in runs
        ],
    }


def training_metrics(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty training log: {path}")
    def fv(row: dict[str, str], key: str) -> float:
        v = float(row[key])
        if not math.isfinite(v): raise ValueError(f"nonfinite {key} in training log")
        return v
    best = min(rows, key=lambda x: fv(x, "Val_Loss"))
    last = rows[-1]
    return {
        "log": str(path.resolve()),
        "log_sha256": sha256(path),
        "epochs_recorded": len(rows),
        "best_epoch": int(best["Epoch"]),
        "best_val_loss": fv(best, "Val_Loss"),
        "best_val_mae_force": fv(best, "Val_MAE_F"),
        "best_val_mae_torque": fv(best, "Val_MAE_T"),
        "last_epoch": int(last["Epoch"]),
        "last_val_loss": fv(last, "Val_Loss"),
    }


def nvt_metrics(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        raise ValueError(f"NVT energy log too short: {path}")
    keys = ("Time_ps", "E_tot", "E_kin", "E_class", "E_ml", "min_dist", "f_max", "torque_max")
    vals: dict[str, list[float]] = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            v = float(row[k])
            if not math.isfinite(v):
                raise ValueError(f"nonfinite NVT {k}")
            vals[k].append(v)
    return {
        "energy_csv": str(path.resolve()),
        "energy_csv_sha256": sha256(path),
        "samples": len(rows),
        "duration_ps": max(vals["Time_ps"]) - min(vals["Time_ps"]),
        "min_distance_nm": min(vals["min_dist"]),
        "max_force": max(vals["f_max"]),
        "max_torque": max(vals["torque_max"]),
        "mean_kinetic_energy": sum(vals["E_kin"]) / len(rows),
        "mean_ml_energy": sum(vals["E_ml"]) / len(rows),
        "finite_stability_pass": min(vals["min_dist"]) > 0.05 and max(vals["f_max"]) < 1.0e6 and max(vals["torque_max"]) < 1.0e6,
    }


def compare(new: dict[str, Any], old: dict[str, Any]) -> dict[str, float]:
    return {
        "delta_p_new_minus_old": new["exponent_p"] - old["exponent_p"],
        "delta_abs_p_minus_2": new["abs_p_minus_2"] - old["abs_p_minus_2"],
        "r2_delta": new["loglog_r2"] - old["loglog_r2"],
        "c2_spread_ratio": new["c2_spread_max_over_min"] / old["c2_spread_max_over_min"],
        "small_over_coarse_ratio": new["c2_small_over_coarse_median"] / old["c2_small_over_coarse_median"],
        "dtmin_sigma_ratio": new["runs"][0]["sigma_E"] / old["runs"][0]["sigma_E"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-manifest", required=True, type=Path)
    ap.add_argument("--build-manifest", required=True, type=Path)
    ap.add_argument("--training-log", required=True, type=Path)
    ap.add_argument("--model-manifest", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--nvt-energy", required=True, type=Path)
    ap.add_argument("--fp32-report", required=True, type=Path)
    ap.add_argument("--fp64-report", type=Path, default=None)
    ap.add_argument("--old-full-report", type=Path, default=None)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    inp = json.loads(args.input_manifest.read_text())
    build = json.loads(args.build_manifest.read_text())
    if inp.get("kind") != "tel22_full_pipeline_uniform_morse_inputs" or int(inp.get("morse_count", -1)) != 180:
        raise ValueError("unexpected input manifest")
    if abs(float(inp.get("candidate_uniform_a", -1.0)) - 0.255) > 1e-15:
        raise ValueError("pipeline input manifest is not uniform a=0.255")
    if build.get("kind") != "tel22_full_pipeline_residual_build":
        raise ValueError("unexpected residual build manifest")

    train = training_metrics(args.training_log)
    nvt = nvt_metrics(args.nvt_energy)
    fp32 = fit_metrics(args.fp32_report)
    fp64 = fit_metrics(args.fp64_report) if args.fp64_report and args.fp64_report.is_file() else None
    old = fit_metrics(args.old_full_report) if args.old_full_report and args.old_full_report.is_file() else None

    model_manifest = json.loads(args.model_manifest.read_text())
    model_info = {
        "manifest": str(args.model_manifest.resolve()),
        "manifest_sha256": sha256(args.model_manifest),
        "model_sha256": model_manifest.get("model_sha256"),
        "dataset_sha256": model_manifest.get("dataset_sha256"),
        "config_sha256": model_manifest.get("config_sha256"),
        "best_validation_loss": model_manifest.get("best_validation_loss"),
    }

    checks = {
        "uniform_morse_180_a0p255": int(inp["morse_count"]) == 180 and abs(float(inp["candidate_uniform_a"]) - 0.255) < 1e-15,
        "residual_dataset_bound_to_candidate_priors": build.get("candidate_priors_sha256") == inp.get("candidate_priors_sha256"),
        "model_bound_to_rebuilt_dataset": model_manifest.get("dataset_sha256") == build.get("dataset_sha256"),
        "nvt_finite_stability": bool(nvt["finite_stability_pass"]),
        "fp32_drift_pass": fp32["max_relative_block_mean_drift"] <= 1e-4,
        "fp32_near_second_order": fp32["abs_p_minus_2"] <= 0.10,
        "fp32_r2_strong": fp32["loglog_r2"] >= 0.99,
        "fp32_c2_regular": fp32["c2_spread_max_over_min"] <= 1.35,
        "fp32_small_dt_regular": 0.80 <= fp32["c2_small_over_coarse_median"] <= 1.20,
    }

    if all(checks[k] for k in ("fp32_drift_pass", "fp32_near_second_order", "fp32_r2_strong", "fp32_c2_regular", "fp32_small_dt_regular")):
        interpretation = "retrained_full_tel22_supports_second_order_fp32_scaling"
    elif old is not None and fp32["abs_p_minus_2"] < old["abs_p_minus_2"] and fp32["c2_spread_max_over_min"] < old["c2_spread_max_over_min"]:
        interpretation = "retrained_full_tel22_improves_fp32_scaling_but_not_all_regularity_gates"
    else:
        interpretation = "retrained_full_tel22_still_nonideal_review_learned_residual"

    precision = None
    if fp64 is not None:
        precision = compare(fp64, fp32)
        precision["fp64_improves_abs_p_minus_2"] = fp64["abs_p_minus_2"] < fp32["abs_p_minus_2"]
        precision["fp64_dtmin_sigma_over_fp32"] = fp64["runs"][0]["sigma_E"] / fp32["runs"][0]["sigma_E"]

    out = {
        "schema_version": 1,
        "kind": "tel22_full_pipeline_uniform_morse_a0p255",
        "scope": "Isolated full TEL22 rebuild from the same atomistic reference trajectory: exact production priors except all 180 empirical Morse stabilizers use a=0.255; residual dataset rebuilt, PaiNN retrained from scratch, fresh equilibration, production-like NVT smoke, then full FP32 NVE scaling certification (plus FP64 closure when enabled).",
        "input_manifest": inp,
        "residual_build_manifest": build,
        "training": train,
        "model": model_info,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
        "nvt": nvt,
        "nve_fp32": fp32,
        "nve_fp64": fp64,
        "old_full_reference": old,
        "comparison_new_fp32_to_old_full": compare(fp32, old) if old is not None else None,
        "precision_closure_fp64_vs_fp32": precision,
        "decision_checks": checks,
        "interpretation": interpretation,
        "production_promotion_allowed": False,
        "next_step": "Review force/structure quality and NVE metrics; promote the uniform Morse prior only after this candidate full Hamiltonian is accepted. This report does not overwrite production artifacts.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")

    print("\n[TEL22 FULL PIPELINE -- UNIFORM MORSE a=0.255]")
    print(f"training  : best_val={train['best_val_loss']:.6g} epoch={train['best_epoch']} val_MAE_F={train['best_val_mae_force']:.6g}")
    print(f"NVT smoke : finite={nvt['finite_stability_pass']} min_dist={nvt['min_distance_nm']:.6g} maxF={nvt['max_force']:.6g}")
    print(f"FP32 NVE  : p={fp32['exponent_p']:.8f} R2={fp32['loglog_r2']:.8f} C2spread={fp32['c2_spread_max_over_min']:.6f} small/coarse={fp32['c2_small_over_coarse_median']:.6f}")
    if fp64 is not None:
        print(f"FP64 NVE  : p={fp64['exponent_p']:.8f} R2={fp64['loglog_r2']:.8f} C2spread={fp64['c2_spread_max_over_min']:.6f} small/coarse={fp64['c2_small_over_coarse_median']:.6f}")
    if old is not None:
        print(f"OLD FP32  : p={old['exponent_p']:.8f} R2={old['loglog_r2']:.8f} C2spread={old['c2_spread_max_over_min']:.6f}")
    print(f"[INTERPRETATION] {interpretation}")
    print(f"[REPORT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
