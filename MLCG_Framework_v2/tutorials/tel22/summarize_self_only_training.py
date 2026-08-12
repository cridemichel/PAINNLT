#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--dataset-report", required=True)
    ap.add_argument("--conditional-report")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.training_csv).open()))
    if not rows:
        raise SystemExit("training CSV has no epochs")
    best = min(rows, key=lambda r: f(r, "Val_Loss"))
    zero_f = f(best, "Val_Zero_F_Norm")
    zero_t = f(best, "Val_Zero_T_Norm")
    best_f = f(best, "Val_Loss_F_Norm")
    best_t = f(best, "Val_Loss_T_Norm")

    report = {
        "best_epoch_by_total_validation_loss": int(best["Epoch"]),
        "best_validation": {
            "total": f(best, "Val_Loss"),
            "force_norm": best_f,
            "torque_norm": best_t,
            "force_mae": f(best, "Val_MAE_F"),
            "torque_mae": f(best, "Val_MAE_T"),
        },
        "zero_predictor_validation": {
            "force_norm": zero_f,
            "torque_norm": zero_t,
        },
        "fractional_improvement_vs_zero_predictor": {
            "force": (1.0 - best_f / zero_f) if zero_f > 0 else math.nan,
            "torque": (1.0 - best_t / zero_t) if zero_t > 0 else math.nan,
        },
        "dataset": json.loads(Path(args.dataset_report).read_text()),
    }

    if args.conditional_report and Path(args.conditional_report).exists():
        cond = json.loads(Path(args.conditional_report).read_text())
        self_pairs = cond.get("pair_analyses", {}).get("self", {})
        report["prior_self_pair_diagnostic"] = {
            name: {
                "force_nearest_vs_random": item.get("nearest_vs_random_force_half_mse_ratio"),
                "torque_nearest_vs_random": item.get("nearest_vs_random_torque_half_mse_ratio"),
                "nearest_force_half_mse_fraction_of_target_mse": item.get("nearest", {}).get(
                    "force_half_pair_difference_mse_fraction_of_target_mse"
                ),
                "nearest_torque_half_mse_fraction_of_target_mse": item.get("nearest", {}).get(
                    "torque_half_pair_difference_mse_fraction_of_target_mse"
                ),
            }
            for name, item in self_pairs.items()
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")

    imp = report["fractional_improvement_vs_zero_predictor"]
    print("======================================================")
    print(" TEL22 SELF-ONLY TRAINING ABLATION SUMMARY")
    print("======================================================")
    print(
        f"best epoch={report['best_epoch_by_total_validation_loss']} | "
        f"Val F={best_f:.6f} vs zero={zero_f:.6f} | "
        f"improvement={100.0 * imp['force']:.2f}%"
    )
    print(
        f"Val T={best_t:.6f} vs zero={zero_t:.6f} | "
        f"improvement={100.0 * imp['torque']:.2f}%"
    )
    print(f"JSON: {out}")


if __name__ == "__main__":
    main()
