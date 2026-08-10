#!/usr/bin/env python3
"""Summarize TEL22 tiny-set hyperparameter grid runs.

Each case is represented by <prefix>_<label>.json and
<prefix>_<label>_training_log.csv in the current directory.
The selected epoch can use either the actual weighted validation loss or the
weight-independent diagnostic score Val_F + Val_T.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


def as_float(row, key):
    return float(row[key])


def selection_score(row, metric):
    if metric == "total":
        return as_float(row, "Val_Loss")
    if metric == "balanced_ft":
        return as_float(row, "Val_Loss_F_Norm") + as_float(row, "Val_Loss_T_Norm")
    raise ValueError(metric)


def summarize_case(config_path: Path, csv_path: Path, metric: str):
    with config_path.open() as handle:
        cfg = json.load(handle)
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty training log: {csv_path}")

    best = min(rows, key=lambda row: selection_score(row, metric))
    final = rows[-1]
    best_f = as_float(best, "Val_Loss_F_Norm")
    best_t = as_float(best, "Val_Loss_T_Norm")
    return {
        "case": config_path.stem,
        "hidden_channels": int(cfg["hidden_channels"]),
        "n_layers": int(cfg["n_layers"]),
        "num_rbf": int(cfg["num_rbf"]),
        "cutoff_nm": float(cfg["cutoff"]),
        "torque_weight": float(cfg.get("torque_weight", 0.0)),
        "batch_size": int(cfg["batch_size"]),
        "selected_epoch": int(best["Epoch"]),
        "selection_score": selection_score(best, metric),
        "val_weighted_loss": as_float(best, "Val_Loss"),
        "val_F": best_f,
        "val_T": best_t,
        "val_F_plus_T": best_f + best_t,
        "val_MAE_F": as_float(best, "Val_MAE_F"),
        "val_MAE_T": as_float(best, "Val_MAE_T"),
        "final_val_weighted_loss": as_float(final, "Val_Loss"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="File prefix, e.g. ablation or torque_D_both")
    parser.add_argument("--output", required=True, help="Summary CSV path")
    parser.add_argument(
        "--epoch-metric",
        choices=("total", "balanced_ft"),
        default="total",
        help="Metric used both to select the epoch inside each run and rank cases",
    )
    args = parser.parse_args()

    configs = sorted(Path(p) for p in glob.glob(f"{args.prefix}_*.json"))
    if not configs:
        raise SystemExit(f"No {args.prefix}_*.json files found")

    summaries = []
    for config_path in configs:
        csv_path = config_path.with_name(config_path.stem + "_training_log.csv")
        if csv_path.is_file():
            summaries.append(summarize_case(config_path, csv_path, args.epoch_metric))

    if not summaries:
        raise SystemExit(f"No completed {args.prefix} training logs found")

    summaries.sort(key=lambda row: row["selection_score"])
    fields = list(summaries[0].keys())
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    print()
    print(f"Summary: {args.output} | epoch/ranking metric: {args.epoch_metric}")
    print("case                       h   L  rbf  cutoff   wT    epoch   score     val_F    val_T")
    print("------------------------- --- --- ---- ------- ----- ------- --------- -------- --------")
    for row in summaries:
        print(
            f"{row['case']:<25} {row['hidden_channels']:>3} {row['n_layers']:>3} "
            f"{row['num_rbf']:>4} {row['cutoff_nm']:>7.4f} {row['torque_weight']:>5.2f} "
            f"{row['selected_epoch']:>7} {row['selection_score']:>9.5f} "
            f"{row['val_F']:>8.5f} {row['val_T']:>8.5f}"
        )
    print()
    print(f"Best case by {args.epoch_metric}: {summaries[0]['case']}")
    print("NOTE: train and validation are the same tiny set; this ranks representability/optimization, not production generalization.")


if __name__ == "__main__":
    main()
