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


def summarize_run(case_dir: Path, agg: dict):
    report_path = case_dir / "dataset_report.json"
    log_path = case_dir / "run" / "cg_training_log.csv"
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(log_path.open()))
    if not rows:
        raise RuntimeError(f"empty training log: {log_path}")
    best = min(rows, key=lambda r: num(r, "Val_Loss"))
    final = rows[-1]
    variant = rep["aggforce_target"]["variant"]
    diag = agg["force_noise_diagnostic"][variant]
    valf = num(best, "Val_Loss_F_Norm")
    zero = num(best, "Val_Zero_F_Norm")
    proxy = float(diag["nearest_force_half_pair_difference_mse_fraction_of_target_mse"])
    return {
        "variant": variant,
        "selected_frames": int(rep["sampling"]["selected_frames"]),
        "train_frames": int(rep["split"]["train_frames"]),
        "validation_frames": int(rep["split"]["validation_frames"]),
        "best_epoch": int(best["Epoch"]),
        "best_val_force_norm": valf,
        "zero_val_force_norm": zero,
        "force_fractional_improvement_vs_zero": 1.0 - valf / zero if zero > 0 else math.nan,
        "train_force_norm_at_best": num(best, "Train_Loss_F_Norm"),
        "force_generalization_gap_at_best": valf - num(best, "Train_Loss_F_Norm"),
        "final_epoch": int(final["Epoch"]),
        "final_train_force_norm": num(final, "Train_Loss_F_Norm"),
        "final_val_force_norm": num(final, "Val_Loss_F_Norm"),
        "target_force_rms": rep["target_scale"]["force_component_rms_kj_mol_nm"],
        "diagnostic_force_noise_proxy": proxy,
        "diagnostic_noise_proxy_ratio_vs_current": (
            1.0 if variant == "current" else float(diag["normalized_noise_proxy_ratio_vs_current"])
        ),
        "diagnostic_absolute_half_pair_mse_ratio_vs_current": (
            1.0 if variant == "current" else float(diag["absolute_half_pair_mse_ratio_vs_current"])
        ),
        "best_val_force_minus_noise_proxy": valf - proxy,
        "dataset_report": str(report_path),
        "training_csv": str(log_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize TEL22 aggforce force-map training ablation")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--aggforce-report", type=Path, required=True)
    ap.add_argument("--temporal-training-summary", type=Path, default=None)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    agg = json.loads(args.aggforce_report.read_text(encoding="utf-8"))
    runs = []
    for variant in ("current", "constraint_aware", "optimized"):
        case = args.root / variant
        if (case / "dataset_report.json").exists() and (case / "run" / "cg_training_log.csv").exists():
            runs.append(summarize_run(case, agg))
    if not runs:
        raise SystemExit(f"no completed 03r training runs under {args.root}")

    prior_1ps = None
    if args.temporal_training_summary and args.temporal_training_summary.exists():
        old = json.loads(args.temporal_training_summary.read_text(encoding="utf-8"))
        for item in old.get("runs", []):
            if math.isclose(float(item.get("window_ps", -1)), 1.0, abs_tol=1e-12, rel_tol=0.0):
                prior_1ps = {
                    "note": "direct reference: 03q 1ps uses the same 991-frame pool, controlled split, PaiNN architecture, and torque_weight=0.5; 03r changes only the DA/DT translational force estimator",
                    "best_val_force_norm": item.get("best_val_force_norm"),
                    "noise_proxy": item.get("diagnostic_force_noise_proxy"),
                    "train_frames": item.get("train_frames"),
                    "validation_frames": item.get("validation_frames"),
                }
                break

    out = {
        "definition": {
            "purpose": "TEL22 instantaneous force-map variance-reduction ablation using aggforce",
            "training": "03r keeps the same torque_weight=0.5 as 03q; the trainer masks one-site DA/DT torque, while multi-site DG force/torque targets remain unchanged",
            "map_fit": "aggforce maps are fit only on training-center atomistic samples and only for exact single-site COM DA/DT residues; rigid multi-site DG is unchanged",
            "noise_proxy_guardrail": "nearest-pair half-MSE is a diagnostic proxy, not a rigorous validation lower bound",
        },
        "force_map_diagnostic": agg,
        "runs": runs,
        "prior_03q_1ps_reference": prior_1ps,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    fields = [k for k in runs[0] if k not in ("dataset_report", "training_csv")]
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in runs:
            w.writerow({k: r[k] for k in fields})

    print("======================================================")
    print(" TEL22 AGGFORCE TRAINING SUMMARY")
    print("======================================================")
    print("variant             bestEp   ValF      proxyF    ValF-proxy  proxy/current")
    for r in runs:
        print(
            f"{r['variant']:18s} {r['best_epoch']:6d} {r['best_val_force_norm']:.6f} "
            f"{r['diagnostic_force_noise_proxy']:.6f} {r['best_val_force_minus_noise_proxy']:+.6f} "
            f"{r['diagnostic_noise_proxy_ratio_vs_current']:.4f}"
        )
    print(f"JSON: {args.output_json}")
    print(f"CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
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


def summarize_run(case_dir: Path, agg: dict):
    report_path = case_dir / "dataset_report.json"
    log_path = case_dir / "run" / "cg_training_log.csv"
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(log_path.open()))
    if not rows:
        raise RuntimeError(f"empty training log: {log_path}")
    best = min(rows, key=lambda r: num(r, "Val_Loss"))
    final = rows[-1]
    variant = rep["aggforce_target"]["variant"]
    diag = agg["force_noise_diagnostic"][variant]
    valf = num(best, "Val_Loss_F_Norm")
    zero = num(best, "Val_Zero_F_Norm")
    proxy = float(diag["nearest_force_half_pair_difference_mse_fraction_of_target_mse"])
    return {
        "variant": variant,
        "selected_frames": int(rep["sampling"]["selected_frames"]),
        "train_frames": int(rep["split"]["train_frames"]),
        "validation_frames": int(rep["split"]["validation_frames"]),
        "best_epoch": int(best["Epoch"]),
        "best_val_force_norm": valf,
        "zero_val_force_norm": zero,
        "force_fractional_improvement_vs_zero": 1.0 - valf / zero if zero > 0 else math.nan,
        "train_force_norm_at_best": num(best, "Train_Loss_F_Norm"),
        "force_generalization_gap_at_best": valf - num(best, "Train_Loss_F_Norm"),
        "final_epoch": int(final["Epoch"]),
        "final_train_force_norm": num(final, "Train_Loss_F_Norm"),
        "final_val_force_norm": num(final, "Val_Loss_F_Norm"),
        "target_force_rms": rep["target_scale"]["force_component_rms_kj_mol_nm"],
        "diagnostic_force_noise_proxy": proxy,
        "diagnostic_noise_proxy_ratio_vs_current": (
            1.0 if variant == "current" else float(diag["normalized_noise_proxy_ratio_vs_current"])
        ),
        "diagnostic_absolute_half_pair_mse_ratio_vs_current": (
            1.0 if variant == "current" else float(diag["absolute_half_pair_mse_ratio_vs_current"])
        ),
        "best_val_force_minus_noise_proxy": valf - proxy,
        "dataset_report": str(report_path),
        "training_csv": str(log_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize TEL22 aggforce force-map training ablation")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--aggforce-report", type=Path, required=True)
    ap.add_argument("--temporal-training-summary", type=Path, default=None)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    agg = json.loads(args.aggforce_report.read_text(encoding="utf-8"))
    runs = []
    for variant in ("current", "constraint_aware", "optimized"):
        case = args.root / variant
        if (case / "dataset_report.json").exists() and (case / "run" / "cg_training_log.csv").exists():
            runs.append(summarize_run(case, agg))
    if not runs:
        raise SystemExit(f"no completed 03r training runs under {args.root}")

    prior_1ps = None
    if args.temporal_training_summary and args.temporal_training_summary.exists():
        old = json.loads(args.temporal_training_summary.read_text(encoding="utf-8"))
        for item in old.get("runs", []):
            if math.isclose(float(item.get("window_ps", -1)), 1.0, abs_tol=1e-12, rel_tol=0.0):
                prior_1ps = {
                    "note": "direct reference: 03q 1ps uses the same 991-frame pool, controlled split, PaiNN architecture, and torque_weight=0.5; 03r changes only the DA/DT translational force estimator",
                    "best_val_force_norm": item.get("best_val_force_norm"),
                    "noise_proxy": item.get("diagnostic_force_noise_proxy"),
                    "train_frames": item.get("train_frames"),
                    "validation_frames": item.get("validation_frames"),
                }
                break

    out = {
        "definition": {
            "purpose": "TEL22 instantaneous force-map variance-reduction ablation using aggforce",
            "training": "03r keeps the same torque_weight=0.5 as 03q; the trainer masks one-site DA/DT torque, while multi-site DG force/torque targets remain unchanged",
            "map_fit": "aggforce maps are fit only on training-center atomistic samples and only for exact single-site COM DA/DT residues; rigid multi-site DG is unchanged",
            "noise_proxy_guardrail": "nearest-pair half-MSE is a diagnostic proxy, not a rigorous validation lower bound",
        },
        "force_map_diagnostic": agg,
        "runs": runs,
        "prior_03q_1ps_reference": prior_1ps,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    fields = [k for k in runs[0] if k not in ("dataset_report", "training_csv")]
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in runs:
            w.writerow({k: r[k] for k in fields})

    print("======================================================")
    print(" TEL22 AGGFORCE TRAINING SUMMARY")
    print("======================================================")
    print("variant             bestEp   ValF      proxyF    ValF-proxy  proxy/current")
    for r in runs:
        print(
            f"{r['variant']:18s} {r['best_epoch']:6d} {r['best_val_force_norm']:.6f} "
            f"{r['diagnostic_force_noise_proxy']:.6f} {r['best_val_force_minus_noise_proxy']:+.6f} "
            f"{r['diagnostic_noise_proxy_ratio_vs_current']:.4f}"
        )
    print(f"JSON: {args.output_json}")
    print(f"CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
