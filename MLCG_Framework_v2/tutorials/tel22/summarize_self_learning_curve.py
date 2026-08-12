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


def summarize_run(run_dir: Path, report_path: Path):
    rows = list(csv.DictReader((run_dir / "cg_training_log.csv").open()))
    if not rows:
        raise RuntimeError(f"empty training log: {run_dir}")
    report = json.loads(report_path.read_text())
    best = min(rows, key=lambda r: num(r, "Val_Loss"))
    final = rows[-1]
    zero_f = num(best, "Val_Zero_F_Norm")
    zero_t = num(best, "Val_Zero_T_Norm")
    best_f = num(best, "Val_Loss_F_Norm")
    best_t = num(best, "Val_Loss_T_Norm")
    return {
        "target_mode": report["inputs"]["target_mode"],
        "selected_frames": int(report["sampling"]["selected_frames"]),
        "train_frames": int(report["split"]["train_frames"]),
        "validation_frames": int(report["split"]["validation_frames"]),
        "best_epoch": int(best["Epoch"]),
        "best_val_total": num(best, "Val_Loss"),
        "best_val_force_norm": best_f,
        "best_val_torque_norm": best_t,
        "zero_val_force_norm": zero_f,
        "zero_val_torque_norm": zero_t,
        "force_fractional_improvement": 1.0 - best_f / zero_f if zero_f > 0 else math.nan,
        "torque_fractional_improvement": 1.0 - best_t / zero_t if zero_t > 0 else math.nan,
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
        "dataset_report": str(report_path),
        "training_csv": str(run_dir / "cg_training_log.csv"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    items = []
    for mode in ("total", "residual"):
        mode_dir = root / mode
        if not mode_dir.exists():
            continue
        for child in mode_dir.iterdir():
            if not child.is_dir() or not child.name.isdigit():
                continue
            report = child / "dataset_report.json"
            run = child / "run"
            if report.exists() and (run / "cg_training_log.csv").exists():
                items.append(summarize_run(run, report))
    if not items:
        raise SystemExit(f"no completed learning-curve runs found under {root}")
    items.sort(key=lambda x: (x["target_mode"], x["selected_frames"]))

    by_mode = {}
    for item in items:
        by_mode.setdefault(item["target_mode"], []).append(item)
    report = {
        "definition": {
            "purpose": "TEL22 isolated DNA-self learning curve",
            "split": "deterministic stratified temporal holdout prepared by the dataset builder; trainer consumes exact validation tail",
            "comparison": "same PaiNN/training hyperparameters across sample sizes and target modes",
        },
        "runs": items,
        "learning_curve": {
            mode: [
                {
                    "selected_frames": x["selected_frames"],
                    "train_frames": x["train_frames"],
                    "validation_frames": x["validation_frames"],
                    "best_val_force_norm": x["best_val_force_norm"],
                    "best_val_torque_norm": x["best_val_torque_norm"],
                    "force_fractional_improvement": x["force_fractional_improvement"],
                    "torque_fractional_improvement": x["torque_fractional_improvement"],
                }
                for x in vals
            ]
            for mode, vals in by_mode.items()
        },
    }

    outj = Path(args.output_json)
    outc = Path(args.output_csv)
    outj.parent.mkdir(parents=True, exist_ok=True)
    outc.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")

    fields = [k for k in items[0].keys() if k not in ("dataset_report", "training_csv")]
    with outc.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for item in items:
            w.writerow({k: item[k] for k in fields})

    print("======================================================")
    print(" TEL22 SELF LEARNING CURVE SUMMARY")
    print("======================================================")
    print("target     Nsel Ntrain Nval bestEp  ValF     ValT     dFzero   dTzero")
    for x in items:
        print(
            f"{x['target_mode']:<10s} {x['selected_frames']:4d} {x['train_frames']:6d} "
            f"{x['validation_frames']:4d} {x['best_epoch']:6d} "
            f"{x['best_val_force_norm']:.6f} {x['best_val_torque_norm']:.6f} "
            f"{100*x['force_fractional_improvement']:7.2f}% "
            f"{100*x['torque_fractional_improvement']:7.2f}%"
        )
    print(f"JSON: {outj}")
    print(f"CSV:  {outc}")


if __name__ == "__main__":
    main()
