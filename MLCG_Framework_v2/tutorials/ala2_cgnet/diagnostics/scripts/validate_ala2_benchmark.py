#!/usr/bin/env python3
"""Validate Ala2 training artifacts and summarize force-matching skill."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


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
    parser.add_argument("--expected-cutoff", type=float, default=0.5)
    parser.add_argument("--expected-spectral-strength", type=float, default=0.0)
    parser.add_argument(
        "--expected-architecture-variant",
        default="painn_canonical_context_silu_v2",
    )
    parser.add_argument("--require-all-to-all", action="store_true")
    parser.add_argument("--require-cgnet-matched-controls", action="store_true")
    parser.add_argument("--require-ordered-geometry", action="store_true")
    parser.add_argument(
        "--expected-ordered-energy-scale-kj-mol", type=float, default=4.184
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    paths = {
        "dataset": run_dir / "ala2_dataset.bin",
        "conversion": run_dir / "ala2_conversion_report.json",
        "priors": run_dir / "ala2_priors.json",
        "reference": run_dir / "ala2_reference.npz",
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
    if config.get("architecture_variant") != args.expected_architecture_variant:
        raise ValueError(
            "Unexpected architecture variant: "
            f"expected={args.expected_architecture_variant}, "
            f"got={config.get('architecture_variant')}"
        )
    manifest_architecture = manifest.get("architecture", {})
    if manifest_architecture.get("variant") != config.get("architecture_variant"):
        raise ValueError("Model manifest/config architecture variant mismatch")
    if abs(float(config.get("cutoff", 0.0)) - args.expected_cutoff) > 1.0e-12:
        raise ValueError(
            f"Ala2 benchmark cutoff must be {args.expected_cutoff} nm; "
            f"got {config.get('cutoff')}"
        )
    spectral_strength = float(config.get("spectral_projection_strength", 0.0))
    if abs(spectral_strength - args.expected_spectral_strength) > 1.0e-12:
        raise ValueError(
            "Unexpected dense-layer spectral projection strength: "
            f"expected={args.expected_spectral_strength}, got={spectral_strength}"
        )
    with np.load(paths["reference"], allow_pickle=False) as reference:
        coordinates = np.asarray(reference["coordinates_nm"], dtype=np.float64)
    maximum_pair_distance = max(
        float(np.max(np.linalg.norm(coordinates[:, j] - coordinates[:, i], axis=1)))
        for i in range(coordinates.shape[1])
        for j in range(i + 1, coordinates.shape[1])
    )
    all_to_all = float(config["cutoff"]) + 1.0e-12 >= maximum_pair_distance
    if args.require_all_to_all and not all_to_all:
        raise ValueError(
            f"Cutoff {config['cutoff']} nm is not all-to-all: the reference subset "
            f"contains a pair at {maximum_pair_distance} nm"
        )
    if args.require_cgnet_matched_controls:
        expected_controls = {
            "hidden_channels": 160,
            "n_layers": 5,
            "batch_size": 512,
            "epochs": 5,
        }
        for key, expected in expected_controls.items():
            if int(config.get(key, -1)) != expected:
                raise ValueError(f"CGnet-matched control {key} must be {expected}")
        expected_float_controls = {
            "learning_rate": 0.003,
            "epoch_lr_decay_factor": 0.3,
            "weight_decay": 0.0,
            "grad_clip_norm": 0.0,
        }
        for key, expected in expected_float_controls.items():
            if not math.isclose(float(config.get(key, math.nan)), expected, abs_tol=1.0e-12):
                raise ValueError(f"CGnet-matched control {key} must be {expected}")
    ordered_nodes = int(config.get("ordered_geometry_nodes", 0))
    ordered_head_layers = int(config.get("ordered_geometry_head_layers", 0))
    ordered_head_width = int(config.get("ordered_geometry_head_width", 0))
    ordered_energy_scale = float(
        config.get("ordered_geometry_energy_scale_kj_mol", 0.0)
    )
    ordered_feature_count = (
        ordered_nodes * (ordered_nodes - 1) // 2
        + max(ordered_nodes - 2, 0)
        + 2 * max(ordered_nodes - 3, 0)
    )
    if args.require_ordered_geometry:
        if (ordered_nodes, ordered_head_layers, ordered_head_width) != (5, 5, 160):
            raise ValueError(
                "Ordered Ala2 diagnostic requires nodes=5, head_layers=5, head_width=160"
            )
        if ordered_feature_count != 17:
            raise ValueError("Ordered Ala2 geometry head must contain 17 features")
        if not math.isclose(
            ordered_energy_scale,
            args.expected_ordered_energy_scale_kj_mol,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Ordered geometry energy scale must match the kcal/mol to kJ/mol "
                f"conversion: expected={args.expected_ordered_energy_scale_kj_mol}, "
                f"got={ordered_energy_scale}"
            )
        manifest_ordered_scale = float(
            manifest_architecture.get("ordered_geometry_energy_scale_kj_mol", math.nan)
        )
        if not math.isclose(
            manifest_ordered_scale,
            ordered_energy_scale,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Model manifest/config ordered energy scale mismatch")
        effective = manifest.get("effective_config", {})
        manifest_feature_count = int(effective.get("ordered_geometry_feature_count", -1))
        feature_mean = np.asarray(
            effective.get("ordered_geometry_feature_mean", []), dtype=np.float64
        )
        feature_std = np.asarray(
            effective.get("ordered_geometry_feature_std", []), dtype=np.float64
        )
        if manifest_feature_count != ordered_feature_count:
            raise ValueError(
                "Manifest ordered feature count disagrees with the architecture: "
                f"{manifest_feature_count} != {ordered_feature_count}"
            )
        if feature_mean.shape != (ordered_feature_count,):
            raise ValueError("Manifest must contain all 17 ordered feature means")
        if feature_std.shape != (ordered_feature_count,):
            raise ValueError("Manifest must contain all 17 ordered feature standard deviations")
        if not np.all(np.isfinite(feature_mean)) or not np.all(np.isfinite(feature_std)):
            raise ValueError("Manifest ordered feature statistics must be finite")
        if np.any(feature_std < 1.0e-6):
            raise ValueError("Manifest ordered feature standard deviations violate the 1e-6 floor")
        expected_feature_order = (
            "all_pair_distances_lexicographic_then_consecutive_angles_then_"
            "consecutive_dihedral_cos_sin_v1"
        )
        if effective.get("ordered_geometry_feature_order") != expected_feature_order:
            raise ValueError("Unexpected ordered feature convention in model manifest")
        if effective.get("ordered_geometry_normalization") != (
            "population_mean_std_training_split_only_floor_1e-6_v1"
        ):
            raise ValueError("Ordered feature normalization is not train-only population scaling")
        if "sin=dot(cross(n1,n2),unit(b1))" not in str(
            effective.get("ordered_geometry_dihedral_convention", "")
        ):
            raise ValueError("Manifest does not record the signed-dihedral convention")
    else:
        feature_mean = np.asarray([], dtype=np.float64)
        feature_std = np.asarray([], dtype=np.float64)

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
        "model_diagnostic": {
            "comparison_scope": (
                "ordered_distance_angle_signed_dihedral_head_plus_painn"
                if ordered_nodes
                else "cgnet_matched_transferable_controls_not_architecture_equivalence"
            ),
            "architecture_variant": str(config["architecture_variant"]),
            "cutoff_nm": float(config["cutoff"]),
            "maximum_pair_distance_nm": maximum_pair_distance,
            "all_to_all_for_every_reference_frame": all_to_all,
            "hidden_channels": int(config["hidden_channels"]),
            "interaction_layers": int(config["n_layers"]),
            "initial_learning_rate": float(config["learning_rate"]),
            "epoch_lr_decay_factor": float(config.get("epoch_lr_decay_factor", 1.0)),
            "spectral_projection_strength": spectral_strength,
            "spectral_projection_power_iterations": int(
                config.get("spectral_projection_power_iterations", 0)
            ),
            "ordered_geometry": {
                "nodes": ordered_nodes,
                "feature_count": ordered_feature_count,
                "distance_features": ordered_nodes * (ordered_nodes - 1) // 2,
                "consecutive_angle_features": max(ordered_nodes - 2, 0),
                "signed_dihedral_sin_cos_features": 2 * max(ordered_nodes - 3, 0),
                "head_layers": ordered_head_layers,
                "head_width": ordered_head_width,
                "energy_scale_kj_mol": ordered_energy_scale,
                "energy_scale_source": "one_kcal_per_mol_equals_4p184_kj_per_mol",
                "normalization": "training_split_only_buffers_in_model",
                "energy_mode": "conservative_scalar_sum_with_painn",
                "feature_mean": feature_mean.tolist(),
                "feature_std": feature_std.tolist(),
                "feature_std_min": (
                    float(np.min(feature_std)) if feature_std.size else None
                ),
                "feature_std_max": (
                    float(np.max(feature_std)) if feature_std.size else None
                ),
            },
        },
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
