#!/usr/bin/env python3
"""Summarize the TEL22 full-dataset cutoff generalization A/B test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def f(row, key):
    return float(row[key])


def summarize(case_dir: Path):
    config_path = case_dir / "config.json"
    log_path = case_dir / "cg_training_log.csv"
    if not config_path.is_file() or not log_path.is_file():
        return None
    with config_path.open() as handle:
        cfg = json.load(handle)
    with log_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty log: {log_path}")

    # Select on the actual weighted validation objective used for early stopping.
    best = min(rows, key=lambda r: f(r, "Val_Loss"))
    final = rows[-1]
    zero_f = f(best, "Val_Zero_F_Norm")
    zero_t = f(best, "Val_Zero_T_Norm")
    zero_total = f(best, "Val_Zero_Total")
    val_f = f(best, "Val_Loss_F_Norm")
    val_t = f(best, "Val_Loss_T_Norm")
    val_total = f(best, "Val_Loss")

    def improvement(value, baseline):
        return 1.0 - value / baseline if baseline > 0 else float("nan")

    return {
        "case": case_dir.name,
        "cutoff_nm": float(cfg["cutoff"]),
        "epochs_run": len(rows),
        "best_epoch": int(best["Epoch"]),
        "val_loss": val_total,
        "val_F": val_f,
        "val_T": val_t,
        "zero_val_loss": zero_total,
        "zero_val_F": zero_f,
        "zero_val_T": zero_t,
        "improvement_total_vs_zero": improvement(val_total, zero_total),
        "improvement_F_vs_zero": improvement(val_f, zero_f),
        "improvement_T_vs_zero": improvement(val_t, zero_t),
        "val_MAE_F": f(best, "Val_MAE_F"),
        "val_MAE_T": f(best, "Val_MAE_T"),
        "final_val_loss": f(final, "Val_Loss"),
        "final_val_F": f(final, "Val_Loss_F_Norm"),
        "final_val_T": f(final, "Val_Loss_T_Norm"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    summaries = [summarize(p) for p in sorted(run_root.iterdir()) if p.is_dir()]
    summaries = [x for x in summaries if x is not None]
    if not summaries:
        raise SystemExit("No completed cutoff-test runs found")
    summaries.sort(key=lambda x: x["val_loss"])

    out = Path(args.output)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print()
    print("FULL-DATASET PHYSICAL-VALIDATION SUMMARY")
    print("case                 cutoff best  valLoss zeroLoss  dTotal    valF   zeroF     dF      valT  zeroT     dT")
    print("------------------- ------- ---- -------- -------- -------- ------- ------- -------- ------- ------- --------")
    for r in summaries:
        print(
            f"{r['case']:<19} {r['cutoff_nm']:>7.4f} {r['best_epoch']:>4} "
            f"{r['val_loss']:>8.5f} {r['zero_val_loss']:>8.5f} {r['improvement_total_vs_zero']:>8.3%} "
            f"{r['val_F']:>7.4f} {r['zero_val_F']:>7.4f} {r['improvement_F_vs_zero']:>8.3%} "
            f"{r['val_T']:>7.4f} {r['zero_val_T']:>7.4f} {r['improvement_T_vs_zero']:>8.3%}"
        )
    print()
    print(f"Best physical-validation loss: {summaries[0]['case']}")
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()
