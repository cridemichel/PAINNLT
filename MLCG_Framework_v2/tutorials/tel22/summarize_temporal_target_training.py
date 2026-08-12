#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def num(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def summarize_case(case_dir: Path, temporal_report: dict):
    report_path = case_dir / "dataset_report.json"
    log_path = case_dir / "run" / "cg_training_log.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(log_path.open()))
    if not rows:
        raise RuntimeError(f"empty training log: {log_path}")
    best = min(rows, key=lambda r: num(r, "Val_Loss"))
    final = rows[-1]
    window = float(report["temporal_target"]["window_ps"])
    key = report["temporal_target"]["window_key"]
    diag = temporal_report.get("windows", {}).get(key, {})
    nearest = diag.get("nearest_same_copy_gap", {})
    noise_f = nearest.get("force_half_pair_difference_mse_fraction_of_target_mse")
    noise_t = nearest.get("torque_half_pair_difference_mse_fraction_of_target_mse")
    base_cmp = temporal_report.get("primary_comparison_vs_baseline", {}).get(key, {})
    best_f = num(best, "Val_Loss_F_Norm")
    best_t = num(best, "Val_Loss_T_Norm")
    zero_f = num(best, "Val_Zero_F_Norm")
    zero_t = num(best, "Val_Zero_T_Norm")
    return {
        "window_ps": window,
        "window_key": key,
        "selected_frames": int(report["sampling"]["selected_frames"]),
        "train_frames": int(report["split"]["train_frames"]),
        "validation_frames": int(report["split"]["validation_frames"]),
        "best_epoch": int(best["Epoch"]),
        "best_val_total": num(best, "Val_Loss"),
        "best_val_force_norm": best_f,
        "best_val_torque_norm": best_t,
        "zero_val_force_norm": zero_f,
        "zero_val_torque_norm": zero_t,
        "force_fractional_improvement_vs_zero": 1.0 - best_f / zero_f if zero_f > 0 else math.nan,
        "torque_fractional_improvement_vs_zero": 1.0 - best_t / zero_t if zero_t > 0 else math.nan,
        "train_force_norm_at_best": num(best, "Train_Loss_F_Norm"),
        "train_torque_norm_at_best": num(best, "Train_Loss_T_Norm"),
        "force_generalization_gap_at_best": best_f - num(best, "Train_Loss_F_Norm"),
        "torque_generalization_gap_at_best": best_t - num(best, "Train_Loss_T_Norm"),
        "final_epoch": int(final["Epoch"]),
        "final_train_force_norm": num(final, "Train_Loss_F_Norm"),
        "final_val_force_norm": num(final, "Val_Loss_F_Norm"),
        "final_train_torque_norm": num(final, "Train_Loss_T_Norm"),
        "final_val_torque_norm": num(final, "Val_Loss_T_Norm"),
        "target_force_rms": report["target_scale"]["force_component_rms_kj_mol_nm"],
        "target_torque_rms": report["target_scale"]["torque_component_rms_kj_mol_multisite_only"],
        "diagnostic_force_noise_proxy": noise_f,
        "diagnostic_torque_noise_proxy": noise_t,
        "diagnostic_force_noise_ratio_vs_1ps": base_cmp.get("force_normalized_noise_floor_ratio_vs_baseline"),
        "diagnostic_torque_noise_ratio_vs_1ps": base_cmp.get("torque_normalized_noise_floor_ratio_vs_baseline"),
        "best_val_force_minus_noise_proxy": (
            best_f - float(noise_f) if noise_f is not None else math.nan
        ),
        "best_val_torque_minus_noise_proxy": (
            best_t - float(noise_t) if noise_t is not None else math.nan
        ),
        "dataset_report": str(report_path),
        "training_csv": str(log_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize instantaneous vs temporal-target PaiNN training")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--temporal-report", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    temporal_report = json.loads(args.temporal_report.read_text(encoding="utf-8"))
    items = []
    if not args.root.exists():
        raise SystemExit(f"root does not exist: {args.root}")
    for child in args.root.iterdir():
        if not child.is_dir() or not child.name.endswith("ps"):
            continue
        if (child / "dataset_report.json").exists() and (child / "run" / "cg_training_log.csv").exists():
            items.append(summarize_case(child, temporal_report))
    if not items:
        raise SystemExit(f"no completed temporal-target training runs under {args.root}")
    items.sort(key=lambda x: x["window_ps"])

    report = {
        "definition": {
            "purpose": "controlled PaiNN comparison of instantaneous and temporally averaged DNA-self targets",
            "split": "same common center-frame pool and deterministic stratified temporal holdout for every window",
            "architecture": "same PaiNN/training hyperparameters for every temporal window",
            "noise_proxy_guardrail": "nearest-pair half-MSE is a diagnostic proxy measured over the common center pool, not a rigorous validation lower bound",
        },
        "runs": items,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    fields = [k for k in items[0].keys() if k not in ("dataset_report", "training_csv")]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow({k: item[k] for k in fields})

    print("======================================================")
    print(" TEL22 TEMPORAL TARGET TRAINING SUMMARY")
    print("======================================================")
    print("window Ntrain Nval bestEp  ValF     proxyF   ValF-proxy  ValT     proxyT")
    for x in items:
        pf = x["diagnostic_force_noise_proxy"]
        pt = x["diagnostic_torque_noise_proxy"]
        print(
            f"{x['window_ps']:5g}ps {x['train_frames']:6d} {x['validation_frames']:4d} "
            f"{x['best_epoch']:6d} {x['best_val_force_norm']:.6f} "
            f"{float(pf):.6f} {x['best_val_force_minus_noise_proxy']:+.6f} "
            f"{x['best_val_torque_norm']:.6f} {float(pt):.6f}"
        )
    print(f"JSON: {args.output_json}")
    print(f"CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
