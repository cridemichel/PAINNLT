#!/usr/bin/env python3
"""Fail-closed validation of the local TEL22 40-epoch pipeline smoke run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

from validate_antiparallel_topology import validate_topology_file
from prepare_variant_a_topology import validate_variant_a_topology_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    require(path.is_file() and path.stat().st_size > 0, f"Missing or empty artifact: {path}")
    return path


def validate_training_log(path: Path, expected_epochs: int) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == expected_epochs, f"Expected {expected_epochs} training rows, found {len(rows)}")
    epochs = [int(row["Epoch"]) for row in rows]
    require(epochs == list(range(1, expected_epochs + 1)), f"Non-contiguous training epochs: {epochs}")
    for row in rows:
        for key, value in row.items():
            if key == "Epoch" or value in (None, ""):
                continue
            require(math.isfinite(float(value)), f"Non-finite training metric at epoch {row['Epoch']}: {key}={value}")
    best_row = min(rows, key=lambda row: float(row["Val_Loss"]))
    zero_total = float(best_row["Val_Zero_Total"])
    require(zero_total > 0.0, "Validation zero-predictor baseline must be positive")
    return {
        "epochs": len(rows),
        "best_epoch": int(best_row["Epoch"]),
        "best_train_loss": float(best_row["Train_Loss"]),
        "best_val_loss": float(best_row["Val_Loss"]),
        "best_val_force_loss_normalized": float(best_row["Val_Loss_F_Norm"]),
        "best_val_torque_loss_normalized": float(best_row["Val_Loss_T_Norm"]),
        "best_val_force_mae": float(best_row["Val_MAE_F"]),
        "best_val_torque_mae": float(best_row["Val_MAE_T"]),
        "val_zero_total": zero_total,
        "best_relative_skill": 1.0 - float(best_row["Val_Loss"]) / zero_total,
        "initial_train_loss": float(rows[0]["Train_Loss"]),
        "final_train_loss": float(rows[-1]["Train_Loss"]),
        "initial_val_loss": float(rows[0]["Val_Loss"]),
        "final_val_loss": float(rows[-1]["Val_Loss"]),
    }


def validate_energy_log(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, "The MD energy log has no samples")
    numeric_values = 0
    for row_index, row in enumerate(rows, start=1):
        for key, value in row.items():
            if value in (None, ""):
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            numeric_values += 1
            require(math.isfinite(number), f"Non-finite MD value at row {row_index}: {key}={value}")
    require(numeric_values > 0, "The MD energy log has no numeric values")
    return {"samples": len(rows), "numeric_values_checked": numeric_values}


def validate_run(
    run_dir: Path,
    expected_epochs: int,
    *,
    topology_mode: str = "antiparallel",
    config_name: str = "tel22_training_config_pipeline40.json",
) -> dict:
    require(topology_mode in {"antiparallel", "variant-a"}, f"Unknown topology mode: {topology_mode}")
    names = {
        "dataset": "tel22_dataset.bin",
        "priors": "cg_priors.json",
        "rb_info": "rigid_bodies_info.json",
        "config": config_name,
        "model": "tel22_model.pt",
        "model_manifest": "tel22_model.pt.manifest.json",
        "training_log": "cg_training_log.csv",
        "equilibrated": "equilibrated.npz",
        "energy": "energy.csv",
        "trajectory": "cg_trajectory.vtf",
        "final_checkpoint": "production_final.npz",
    }
    files = {key: require_file(run_dir / name) for key, name in names.items()}

    with files["dataset"].open("rb") as handle:
        frame_count_raw = handle.read(4)
    require(len(frame_count_raw) == 4, "Dataset header is truncated")
    frame_count = struct.unpack("i", frame_count_raw)[0]
    require(frame_count > 1, f"Dataset has too few frames: {frame_count}")

    config = json.loads(files["config"].read_text(encoding="utf-8"))
    require(int(config["epochs"]) == expected_epochs, "Training config epoch count mismatch")
    require(
        int(config["early_stopping_patience"]) > expected_epochs,
        "Pipeline-test config must not early-stop before the requested epoch count",
    )

    if topology_mode == "antiparallel":
        priors_summary = validate_topology_file(files["priors"], r0_mode="numeric")
        report_kind = "tel22_antiparallel_local_pipeline_smoke"
    else:
        priors_summary = validate_variant_a_topology_file(files["priors"])
        report_kind = "tel22_variant_a_local_pipeline_smoke"
    json.loads(files["rb_info"].read_text(encoding="utf-8"))

    manifest = json.loads(files["model_manifest"].read_text(encoding="utf-8"))
    require(int(manifest["effective_config"]["epochs"]) == expected_epochs, "Model manifest epoch count mismatch")
    require(manifest.get("dataset_sha256") == sha256_file(files["dataset"]), "Model/dataset SHA256 mismatch")
    require(manifest.get("config_sha256") == sha256_file(files["config"]), "Model/config SHA256 mismatch")
    require(manifest.get("model_sha256") == sha256_file(files["model"]), "Model SHA256 mismatch")

    training = validate_training_log(files["training_log"], expected_epochs)
    energy = validate_energy_log(files["energy"])
    artifact_hashes = {key: sha256_file(path) for key, path in files.items()}
    return {
        "schema_version": 1,
        "kind": report_kind,
        "status": "PASS",
        "scope": "functional smoke test; not a thermodynamic or NVE certification",
        "run_dir": str(run_dir.resolve()),
        "dataset_frames": frame_count,
        "topology": priors_summary,
        "training": training,
        "short_md": energy,
        "artifact_sha256": artifact_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-epochs", type=int, default=40)
    parser.add_argument(
        "--topology-mode",
        choices=("antiparallel", "variant-a"),
        default="antiparallel",
    )
    parser.add_argument("--config-name", default="tel22_training_config_pipeline40.json")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = validate_run(
        args.run_dir,
        args.expected_epochs,
        topology_mode=args.topology_mode,
        config_name=args.config_name,
    )
    report_path = args.report or args.run_dir / "pipeline_test_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[PASS] Local TEL22 pipeline smoke validated; report: {report_path}")


if __name__ == "__main__":
    main()
