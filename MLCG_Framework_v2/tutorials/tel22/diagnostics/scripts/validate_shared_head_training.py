#!/usr/bin/env python3
"""Validate and summarize the controlled TEL22 shared-head training test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path


ARCHITECTURE = "tel22_shared_geometry_tanh_v1"
FEATURE_ORDER = (
    "tel22_per_copy_backbone21_angles20_dihedral_cos19_sin19_"
    "tetrad36_stacking16_v1"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    names = {
        "dataset": "tel22_dataset.bin",
        "priors": "cg_priors.json",
        "rigid_bodies": "rigid_bodies_info.json",
        "config": "tel22_training_config_shared_head_15ep.json",
        "model": "tel22_shared_head.pt",
        "manifest": "tel22_shared_head.pt.manifest.json",
        "training_log": "cg_training_log.csv",
        "training_stdout": "training_stdout.log",
        "cutoff_preflight": "tel22_shared_cutoff_report.json",
    }
    files = {key: run_dir / value for key, value in names.items()}
    for key, path in files.items():
        require(path.is_file() and path.stat().st_size > 0, f"Missing {key}: {path}")

    with files["dataset"].open("rb") as handle:
        frame_count = struct.unpack("i", handle.read(4))[0]
    require(frame_count > 1, "Dataset must contain at least two frames")

    priors = json.loads(files["priors"].read_text(encoding="utf-8"))
    bonds = priors.get("bonds", [])
    require(not priors.get("morse_contacts"), "Variant-A priors still contain morse_contacts")
    require(
        not any(str(item.get("type", "")).lower() == "morse" for item in bonds),
        "Variant-A priors still contain Morse bonds",
    )

    config = json.loads(files["config"].read_text(encoding="utf-8"))
    cutoff_preflight = json.loads(files["cutoff_preflight"].read_text(encoding="utf-8"))
    require(config["architecture_variant"] == ARCHITECTURE, "Wrong architecture variant")
    require(int(config["ordered_geometry_nodes"]) == 82, "Expected 82 sites/copy")
    require(int(config["ordered_geometry_copies"]) == 10, "Expected ten shared copies")
    require(float(config["ordered_geometry_energy_scale_kj_mol"]) == 2.4943387854,
            "TEL22 energy scale must be native kBT, not the Ala2 kcal-to-kJ factor")
    require(cutoff_preflight["status"] == "PASS", "Cutoff preflight did not pass")
    require(math.isclose(float(config["cutoff"]),
                         float(cutoff_preflight["selected_cutoff_nm"]), abs_tol=1.0e-12),
            "Training config does not use the preflight-selected cutoff")
    require(config.get("ordered_geometry_edge_policy") ==
            "augment_required_topology_edges_v1",
            "Training config does not enable TEL22 topology-edge augmentation")
    require(cutoff_preflight.get("edge_policy") ==
            "augment_required_topology_edges_v1",
            "Cutoff preflight reports the wrong TEL22 edge policy")

    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    effective = manifest["effective_config"]
    architecture = manifest["architecture"]
    require(architecture["variant"] == ARCHITECTURE, "Manifest architecture mismatch")
    require(int(architecture["ordered_geometry_copies"]) == 10, "Manifest copy count mismatch")
    require(bool(architecture["ordered_geometry_head_only"]), "PaiNN branch must be disabled")
    require(effective.get("ordered_geometry_edge_policy") ==
            "augment_required_topology_edges_v1",
            "Manifest does not record TEL22 topology-edge augmentation")
    require(int(effective["ordered_geometry_feature_count"]) == 131,
            "Expected 131 topology-aware features per copy")
    require(effective["ordered_geometry_feature_order"] == FEATURE_ORDER,
            "Feature contract mismatch")

    with files["training_log"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, "Empty training log")
    require(len(rows) <= int(config["epochs"]), "More epochs than requested")
    for row in rows:
        for key, value in row.items():
            if key == "Epoch" or value in (None, ""):
                continue
            require(math.isfinite(float(value)), f"Non-finite {key} at epoch {row['Epoch']}")
    best = min(rows, key=lambda row: float(row["Val_Loss"]))
    zero_total = float(best["Val_Zero_Total"])
    best_loss = float(best["Val_Loss"])
    relative_skill = 1.0 - best_loss / zero_total
    if relative_skill >= 0.10:
        screening_verdict = "material_training_signal"
    elif relative_skill >= 0.03:
        screening_verdict = "weak_training_signal"
    else:
        screening_verdict = "no_material_gain_over_zero_predictor"

    report = {
        "schema_version": 1,
        "kind": "tel22_shared_geometry_training_screen",
        "status": "PASS",
        "scope": "training diagnostic only; no thermodynamic validation",
        "architecture": {
            "variant": ARCHITECTURE,
            "copies": 10,
            "sites_per_copy": 82,
            "features_per_copy": 131,
            "shared_weights": True,
            "painn_branch_enabled": False,
            "energy_aggregation": "sum_over_copies",
        },
        "dataset_frames": frame_count,
        "cutoff_preflight": cutoff_preflight,
        "training": {
            "epochs_completed": len(rows),
            "best_epoch": int(best["Epoch"]),
            "best_val_loss": best_loss,
            "zero_predictor_val_loss": zero_total,
            "relative_skill_vs_zero": relative_skill,
            "best_val_force_mae_kj_mol_nm": float(best["Val_MAE_F"]),
            "best_val_torque_mae_kj_mol": float(best["Val_MAE_T"]),
            "screening_verdict": screening_verdict,
            "thresholds_are_diagnostic_not_literature_acceptance_criteria": True,
        },
        "artifact_sha256": {key: sha256_file(path) for key, path in files.items()},
    }
    report_path = args.report or run_dir / "tel22_shared_head_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[PASS] TEL22 shared-head training validated: {report_path}")


if __name__ == "__main__":
    main()
