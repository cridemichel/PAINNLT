#!/usr/bin/env python3
"""Validate Ala2 training artifacts and summarize force-matching skill."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


REQUIRED_COLUMNS = {
    "Epoch",
    "Train_Loss",
    "Val_Loss",
    "Train_Loss_F_Norm",
    "Val_Loss_F_Norm",
    "Train_MAE_F",
    "Val_MAE_F",
    "Val_Zero_F_Norm",
    "Val_Zero_Total",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} at epoch {row.get('Epoch')}")
    return value


def classify_skill(skill: float) -> str:
    if skill < 0.0:
        return "negative"
    if skill < 0.05:
        return "weak"
    if skill < 0.10:
        return "moderate"
    return "strong"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    paths = {
        "dataset": run_dir / "ala2_dataset.bin",
        "conversion": run_dir / "ala2_conversion_report.json",
        "priors": run_dir / "ala2_priors.json",
        "config": run_dir / "ala2_training_config_50ep.json",
        "model": run_dir / "ala2_model.pt",
        "manifest": run_dir / "ala2_model.pt.manifest.json",
        "training_csv": run_dir / "cg_training_log.csv",
        "training_stdout": run_dir / "training_stdout.log",
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {name}: {path}")

    conversion = json.loads(paths["conversion"].read_text())
    config = json.loads(paths["config"].read_text())
    priors = json.loads(paths["priors"].read_text())
    manifest = json.loads(paths["manifest"].read_text())

    if conversion["source"]["input_shape"] != [10000, 5, 3]:
        raise ValueError("Conversion report does not describe the official 10000x5x3 arrays")
    if conversion["split"] != {
        "mode": "tail",
        "train_frames": 8000,
        "validation_frames": 2000,
    }:
        raise ValueError(f"Unexpected conversion split: {conversion['split']}")
    if config.get("validation_split_mode") != "tail":
        raise ValueError("Ala2 benchmark requires validation_split_mode=tail")
    if int(config.get("validation_tail_frames", -1)) != 2000:
        raise ValueError("Ala2 benchmark requires validation_tail_frames=2000")
    if float(config.get("torque_weight", -1.0)) != 0.0:
        raise ValueError("Ala2 benchmark must use force-only training")
    if int(config.get("num_species", 0)) <= 7:
        raise ValueError("num_species must include atomic-number bead types 6 and 7")
    if abs(float(config.get("cutoff", 0.0)) - 0.5) > 1.0e-12:
        raise ValueError("Ala2 benchmark cutoff must be 0.5 nm (5 Angstrom)")

    prior_mode = conversion["prior_mode"]
    expected_bonds = 4 if prior_mode == "harmonic" else 0
    expected_angles = 3 if prior_mode == "harmonic" else 0
    if len(priors.get("bonds", [])) != expected_bonds:
        raise ValueError(f"Expected {expected_bonds} bond priors")
    if len(priors.get("angles", [])) != expected_angles:
        raise ValueError(f"Expected {expected_angles} angle priors")
    if priors.get("morse_type_pairs", []) or priors.get("dihedrals", []):
        raise ValueError("Ala2 CGnet benchmark must not contain Morse or dihedral priors")

    dataset_sha = sha256_file(paths["dataset"])
    if dataset_sha != conversion["output"]["dataset_sha256"]:
        raise ValueError("Dataset SHA-256 disagrees with conversion report")
    if dataset_sha != manifest.get("dataset_sha256"):
        raise ValueError("Dataset SHA-256 disagrees with model manifest")

    with paths["training_csv"].open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Training CSV has no epochs")
    max_epochs = int(config["epochs"])
    if len(rows) > max_epochs:
        raise ValueError(f"Training CSV has {len(rows)} rows but config allows {max_epochs}")

    for row in rows:
        for key in REQUIRED_COLUMNS:
            finite_float(row, key)
    best = min(rows, key=lambda row: finite_float(row, "Val_Loss_F_Norm"))
    first = rows[0]
    baseline_mse = finite_float(best, "Val_Zero_F_Norm")
    best_mse = finite_float(best, "Val_Loss_F_Norm")
    if baseline_mse <= 0.0:
        raise ValueError("Non-positive zero-predictor validation MSE")

    stats = conversion["statistics"]
    train_rms = float(stats["target_train"]["component_rms"])
    validation_rms = float(stats["target_validation"]["component_rms"])
    independently_expected_baseline = (validation_rms / train_rms) ** 2
    if not math.isclose(baseline_mse, independently_expected_baseline, rel_tol=2.0e-4):
        raise ValueError(
            "Trainer zero baseline is inconsistent with converted residual targets: "
            f"trainer={baseline_mse}, expected={independently_expected_baseline}"
        )

    baseline_mae = float(stats["target_validation"]["component_mae"])
    best_mae = finite_float(best, "Val_MAE_F")
    mse_skill = 1.0 - best_mse / baseline_mse
    mae_skill = 1.0 - best_mae / baseline_mae
    learning_signal = classify_skill(mse_skill)

    report = {
        "schema_version": 1,
        "status": "pass",
        "scope": "training_diagnostic_not_thermodynamic_certification",
        "prior_mode": prior_mode,
        "frames": {"train": 8000, "validation_tail": 2000},
        "epochs": {
            "configured": max_epochs,
            "completed": len(rows),
            "best": int(float(best["Epoch"])),
        },
        "validation": {
            "zero_predictor_force_mse_normalized": baseline_mse,
            "first_epoch_force_mse_normalized": finite_float(first, "Val_Loss_F_Norm"),
            "best_force_mse_normalized": best_mse,
            "explained_residual_force_variance": mse_skill,
            "zero_predictor_force_mae_kj_mol_nm": baseline_mae,
            "best_force_mae_kj_mol_nm": best_mae,
            "force_mae_relative_improvement": mae_skill,
            "learning_signal": learning_signal,
        },
        "best_epoch_train_validation_gap": (
            best_mse - finite_float(best, "Train_Loss_F_Norm")
        ),
        "artifacts": {
            name: str(path) for name, path in paths.items()
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"[PASS] Ala2 benchmark artifacts are consistent; learning_signal={learning_signal}, "
        f"MSE skill={100.0 * mse_skill:.2f}%, MAE improvement={100.0 * mae_skill:.2f}%."
    )


if __name__ == "__main__":
    main()
