#!/usr/bin/env python3
"""Create or finalize a self-describing PaiNN model sidecar manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


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
    architecture = {
        "num_species": int(config["num_species"]),
        "hidden_channels": int(config["hidden_channels"]),
        "n_layers": int(config["n_layers"]),
        "num_rbf": int(config["num_rbf"]),
        "cutoff": float(config["cutoff"]),
        "toxvaerd_alpha": float(config.get("toxvaerd_alpha", 0.1)),
    }
    manifest_path = Path(f"{args.model}.manifest.json")
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "schema_version": 1,
        "framework": "MLCG_Framework_v2",
        "architecture": architecture,
        "effective_config": config,
        "model_path": str(args.model),
        "model_file_size_bytes": args.model.stat().st_size,
        "model_sha256": sha256_file(args.model),
        "dataset_path": str(args.dataset),
        "dataset_file_size_bytes": args.dataset.stat().st_size,
        "dataset_sha256": sha256_file(args.dataset),
        "config_path": str(args.config),
        "config_file_size_bytes": args.config.stat().st_size,
        "config_sha256": sha256_file(args.config),
        "split_seed": 42,
        "validation_fraction": 0.2,
        "force_units": "kJ mol^-1 nm^-1",
        "torque_units": "kJ mol^-1",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Finalized model manifest: {manifest_path}")


if __name__ == "__main__":
    main()
