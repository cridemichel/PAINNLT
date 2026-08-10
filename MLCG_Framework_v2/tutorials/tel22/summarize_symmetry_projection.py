#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def f(row, key):
    return float(row[key])


def load_best(case_dir):
    path = case_dir / "cg_training_log.csv"
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"empty training log: {path}")
    best = min(rows, key=lambda r: f(r, "Val_Loss"))
    zero = f(best, "Val_Zero_Total")
    zf = f(best, "Val_Zero_F_Norm")
    zt = f(best, "Val_Zero_T_Norm")
    vf = f(best, "Val_Loss_F_Norm")
    vt = f(best, "Val_Loss_T_Norm")
    return {
        "case": case_dir.name,
        "best_epoch": int(best["Epoch"]),
        "val_loss": f(best, "Val_Loss"),
        "val_F": vf,
        "val_T": vt,
        "val_MAE_F": f(best, "Val_MAE_F"),
        "val_MAE_T": f(best, "Val_MAE_T"),
        "zero_val_loss": zero,
        "zero_val_F": zf,
        "zero_val_T": zt,
        "improvement_total_vs_zero": (zero - f(best, "Val_Loss")) / zero if zero else 0.0,
        "improvement_F_vs_zero": (zf - vf) / zf if zf else 0.0,
        "improvement_T_vs_zero": (zt - vt) / zt if zt else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--projection-report", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.run_root)
    rows = [load_best(root / "A_raw"), load_best(root / "B_projected")]
    fields = list(rows[0].keys())
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with open(args.projection_report) as fh:
        rep = json.load(fh)

    print("\n=== SYMMETRY-PROJECTION GENERALIZATION SUMMARY ===")
    for r in rows:
        print(
            f"{r['case']:12s} epoch={r['best_epoch']:3d} "
            f"val={r['val_loss']:.6g} zero={r['zero_val_loss']:.6g} "
            f"impr={100*r['improvement_total_vs_zero']:+.3f}% | "
            f"F={100*r['improvement_F_vs_zero']:+.3f}% "
            f"T={100*r['improvement_T_vs_zero']:+.3f}%"
        )
    print("\nProjection removed from raw target variance:")
    print("  force global-translation mode: "
          f"{100*rep['force']['global_translation_fraction_of_raw_mse']:.6f}%")
    print("  torque global-rotation correction: "
          f"{100*rep['torque']['global_rotation_correction_fraction_of_raw_torque_mse']:.6f}%")
    print(f"\nCSV: {args.output}")


if __name__ == "__main__":
    main()
