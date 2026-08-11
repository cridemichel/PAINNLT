#!/usr/bin/env python3
"""Summarize matched TEL22 Morse-prior ON/OFF training runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def fv(row, key):
    return float(row[key])


def summarize(case_dir: Path):
    cfg_path = case_dir / "config.json"
    log_path = case_dir / "cg_training_log.csv"
    if not cfg_path.is_file() or not log_path.is_file():
        return None
    cfg = json.loads(cfg_path.read_text())
    with log_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"empty training log: {log_path}")
    best = min(rows, key=lambda r: fv(r, "Val_Loss"))
    final = rows[-1]
    zf = fv(best, "Val_Zero_F_Norm")
    zt = fv(best, "Val_Zero_T_Norm")
    ztot = fv(best, "Val_Zero_Total")
    vf = fv(best, "Val_Loss_F_Norm")
    vt = fv(best, "Val_Loss_T_Norm")
    vtot = fv(best, "Val_Loss")

    def imp(v, z):
        return 1.0 - v / z if z > 0 else float("nan")

    return {
        "case": case_dir.name,
        "epochs_run": len(rows),
        "best_epoch": int(best["Epoch"]),
        "val_loss": vtot,
        "val_F": vf,
        "val_T": vt,
        "zero_val_loss": ztot,
        "zero_val_F": zf,
        "zero_val_T": zt,
        "improvement_total_vs_zero": imp(vtot, ztot),
        "improvement_F_vs_zero": imp(vf, zf),
        "improvement_T_vs_zero": imp(vt, zt),
        "val_MAE_F": fv(best, "Val_MAE_F"),
        "val_MAE_T": fv(best, "Val_MAE_T"),
        "final_val_loss": fv(final, "Val_Loss"),
        "final_val_F": fv(final, "Val_Loss_F_Norm"),
        "final_val_T": fv(final, "Val_Loss_T_Norm"),
        "batch_size": int(cfg["batch_size"]),
        "torque_weight": float(cfg["torque_weight"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    rows = []
    for p in sorted(args.run_root.iterdir()):
        if p.is_dir():
            s = summarize(p)
            if s:
                rows.append(s)
    if not rows:
        raise SystemExit("No completed Morse-ablation training runs found")
    rows.sort(key=lambda r: r["case"])

    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("MORSE ON/OFF PHYSICAL-VALIDATION SUMMARY")
    print("case       best   valLoss  zeroLoss   dTotal     valF    zeroF       dF      valT    zeroT       dT")
    print("---------  ----  --------  --------  --------  -------  -------  --------  -------  -------  --------")
    for r in rows:
        print(
            f"{r['case']:<9}  {r['best_epoch']:>4}  {r['val_loss']:>8.5f}  {r['zero_val_loss']:>8.5f}  "
            f"{r['improvement_total_vs_zero']:>8.3%}  {r['val_F']:>7.4f}  {r['zero_val_F']:>7.4f}  "
            f"{r['improvement_F_vs_zero']:>8.3%}  {r['val_T']:>7.4f}  {r['zero_val_T']:>7.4f}  "
            f"{r['improvement_T_vs_zero']:>8.3%}"
        )
    print(f"\nCSV: {args.output}")


if __name__ == "__main__":
    main()
