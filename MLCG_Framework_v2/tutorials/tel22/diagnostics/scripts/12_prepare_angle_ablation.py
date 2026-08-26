#!/usr/bin/env python3
"""Prepare a provenance-safe TEL22 checkpoint/priors pair with angular priors removed.

The production files are never edited. All checkpoint arrays are preserved exactly;
only metadata provenance is rebound to the derived no-angle priors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_hashes(*, dataset: Path, config: Path, priors: Path, rb_info: Path, model: Path) -> dict[str, str | None]:
    manifest = Path(str(model) + ".manifest.json")
    return {
        "dataset_sha256": sha256_file(dataset),
        "config_sha256": sha256_file(config),
        "priors_sha256": sha256_file(priors),
        "rb_info_sha256": sha256_file(rb_info),
        "model_sha256": sha256_file(model),
        "model_manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
    }


def read_metadata(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value)
    if arr.shape != ():
        raise ValueError("checkpoint metadata_json must be a scalar string")
    metadata = json.loads(str(arr.item()))
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata_json must decode to an object")
    return metadata


def validate_source_checkpoint(
    checkpoint: Path,
    *,
    source_priors: Path,
    dataset: Path,
    config: Path,
    rb_info: Path,
    model: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(checkpoint, allow_pickle=False) as src:
        if "metadata_json" not in src.files:
            raise ValueError("Source checkpoint has no metadata_json provenance")
        arrays = {name: np.asarray(src[name]).copy() for name in src.files if name != "metadata_json"}
        metadata = read_metadata(src["metadata_json"])

    required = {"pos", "v", "box_l"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError("Source checkpoint lacks required fields: " + ", ".join(missing))

    recorded = metadata.get("input_hashes")
    if not isinstance(recorded, dict):
        raise ValueError("Source checkpoint metadata has no input_hashes map")
    current = expected_hashes(
        dataset=dataset, config=config, priors=source_priors, rb_info=rb_info, model=model
    )
    mismatches = [
        f"{key}: checkpoint={recorded.get(key)}, current={value}"
        for key, value in current.items() if recorded.get(key) != value
    ]
    if mismatches:
        raise ValueError("Source checkpoint provenance mismatch:\n  - " + "\n  - ".join(mismatches))
    return arrays, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rb-info", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.priors, args.config, args.dataset, args.rb_info, args.model, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    model_manifest = Path(str(args.model) + ".manifest.json")
    if not model_manifest.is_file():
        raise FileNotFoundError(model_manifest)

    source_priors = json.loads(args.priors.read_text(encoding="utf-8"))
    angles = source_priors.get("angles", [])
    if not isinstance(angles, list):
        raise ValueError("cg_priors.json field 'angles' must be a list")
    if not angles:
        raise ValueError("Production TEL22 contains no angular priors; no-angle ablation would be a no-op")

    arrays, metadata = validate_source_checkpoint(
        args.checkpoint,
        source_priors=args.priors,
        dataset=args.dataset,
        config=args.config,
        rb_info=args.rb_info,
        model=args.model,
    )

    out_dir = args.output_dir
    priors_path = out_dir / "cg_priors_no_angles.json"
    checkpoint_path = out_dir / "equilibrated_no_angles.npz"
    manifest_path = out_dir / "angle_ablation_inputs.json"
    for path in (priors_path, checkpoint_path, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    no_angles = json.loads(json.dumps(source_priors))
    no_angles["angles"] = []
    priors_path.write_text(json.dumps(no_angles, indent=2) + "\n", encoding="utf-8")

    hashes = expected_hashes(
        dataset=args.dataset,
        config=args.config,
        priors=priors_path,
        rb_info=args.rb_info,
        model=args.model,
    )
    out_metadata = json.loads(json.dumps(metadata))
    out_metadata["input_hashes"] = hashes
    out_metadata["diagnostic_angle_ablation"] = {
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "source_priors_sha256": sha256_file(args.priors),
        "removed_angle_entries": len(angles),
        "mechanical_state_policy": "all checkpoint arrays preserved exactly; only provenance metadata rebound",
    }

    np.savez_compressed(
        checkpoint_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(out_metadata, sort_keys=True)),
    )

    # Fail closed if the derivation accidentally changed any mechanical array.
    with np.load(checkpoint_path, allow_pickle=False) as derived:
        for name, original in arrays.items():
            candidate = np.asarray(derived[name])
            if candidate.dtype != original.dtype or candidate.shape != original.shape or not np.array_equal(candidate, original):
                raise RuntimeError(f"Derived checkpoint changed mechanical field {name!r}")

    report = {
        "schema_version": 1,
        "kind": "tel22_nve_angle_prior_ablation_inputs",
        "scope": "diagnostic_only; trained PaiNN and all non-angle priors remain unchanged",
        "source": {
            "priors": str(args.priors.resolve()),
            "priors_sha256": sha256_file(args.priors),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "angle_entries": len(angles),
        },
        "no_angles": {
            "priors": str(priors_path.resolve()),
            "priors_sha256": sha256_file(priors_path),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "removed_angle_entries": len(angles),
            "remaining_angle_entries": 0,
        },
        "warning": (
            "The trained PaiNN residual is intentionally not retrained after removing angular priors; "
            "this is a numerical ablation, not a reparameterized production model."
        ),
    }
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[TEL22 ANGLE ABLATION INPUTS]")
    print(f"Angular priors removed : {len(angles)}")
    print(f"Derived priors         : {priors_path}")
    print(f"Derived checkpoint     : {checkpoint_path}")
    print("Mechanical state       : byte-identical numeric arrays")
    print(f"[REPORT] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
