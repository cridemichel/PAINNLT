#!/usr/bin/env python3
"""Create or finalize a self-describing PaiNN model sidecar manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PAINN_ARCHITECTURE_VARIANT = "painn_canonical_context_silu_v2"
PAINN_ORDERED_GEOMETRY_VARIANT = "painn_ordered_geometry_tanh_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()

    for path in (args.model, args.config, args.dataset):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = json.loads(args.config.read_text())
    variant = str(config.get("architecture_variant", ""))
    if variant not in {PAINN_ARCHITECTURE_VARIANT, PAINN_ORDERED_GEOMETRY_VARIANT}:
        raise ValueError(
            f"Unsupported config architecture_variant: {variant!r}"
        )
    architecture = {
        "variant": variant,
        "num_species": int(config["num_species"]),
        "hidden_channels": int(config["hidden_channels"]),
        "n_layers": int(config["n_layers"]),
        "num_rbf": int(config["num_rbf"]),
        "cutoff": float(config["cutoff"]),
        "toxvaerd_alpha": float(config.get("toxvaerd_alpha", 0.1)),
    }
    if variant == PAINN_ORDERED_GEOMETRY_VARIANT:
        ordered_energy_scale = float(config["ordered_geometry_energy_scale_kj_mol"])
        if not math.isfinite(ordered_energy_scale) or ordered_energy_scale <= 0.0:
            raise ValueError("ordered_geometry_energy_scale_kj_mol must be positive and finite")
        architecture.update({
            "ordered_geometry_nodes": int(config["ordered_geometry_nodes"]),
            "ordered_geometry_head_layers": int(config["ordered_geometry_head_layers"]),
            "ordered_geometry_head_width": int(config["ordered_geometry_head_width"]),
            "ordered_geometry_energy_scale_kj_mol": ordered_energy_scale,
        })
    manifest_path = Path(f"{args.model}.manifest.json")
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    effective_config = dict(config)
    previous_effective = manifest.get("effective_config", {})
    if variant == PAINN_ORDERED_GEOMETRY_VARIANT:
        required_statistics = {
            "ordered_geometry_feature_count",
            "ordered_geometry_feature_mean",
            "ordered_geometry_feature_std",
            "ordered_geometry_feature_order",
            "ordered_geometry_dihedral_convention",
            "ordered_geometry_normalization",
        }
        missing_statistics = sorted(required_statistics - set(previous_effective))
        if missing_statistics:
            raise ValueError(
                "Ordered-geometry trainer manifest is missing fitted feature metadata: "
                f"{missing_statistics}. Do not recreate this manifest from config alone."
            )
    for key, value in previous_effective.items():
        if key.startswith("ordered_geometry_feature_") or key in {
            "ordered_geometry_dihedral_convention",
            "ordered_geometry_normalization",
        }:
            effective_config[key] = value
    manifest.update({
        "schema_version": 3,
        "framework": "MLCG_Framework_v2",
        "energy_gauge": "isolated_species_zero_v1",
        "architecture": architecture,
        "effective_config": effective_config,
        "model_path": str(args.model),
        "model_file_size_bytes": args.model.stat().st_size,
        "model_sha256": sha256_file(args.model),
        "dataset_path": str(args.dataset),
        "dataset_file_size_bytes": args.dataset.stat().st_size,
        "dataset_sha256": sha256_file(args.dataset),
        "config_path": str(args.config),
        "config_file_size_bytes": args.config.stat().st_size,
        "config_sha256": sha256_file(args.config),
        "split_seed": int(config.get("split_seed", 42)),
        "validation_fraction": float(config.get("validation_fraction", 0.2)),
        "physical_validation_only": bool(config.get("physical_validation_only", True)),
        "include_decoys_in_train": bool(config.get("include_decoys_in_train", False)),
        "shuffle_each_epoch": bool(config.get("shuffle_each_epoch", True)),
        "force_units": "kJ mol^-1 nm^-1",
        "torque_units": "kJ mol^-1",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Finalized model manifest: {manifest_path}")


if __name__ == "__main__":
    main()
