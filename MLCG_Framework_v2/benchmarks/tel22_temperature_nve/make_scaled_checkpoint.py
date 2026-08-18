#!/usr/bin/env python3
"""Create an iso-configurational TEL22 checkpoint with rescaled velocities."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

R_KJ_MOL_K = 0.00831446261815324


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_from_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if "metadata_json" not in arrays:
        raise ValueError("Checkpoint lacks provenance metadata_json")
    raw = np.asarray(arrays["metadata_json"])
    if raw.shape != ():
        raise ValueError("Checkpoint metadata_json must be a scalar string")
    metadata = json.loads(str(raw.item()))
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata_json must decode to an object")
    return metadata


def _rms_real(values: np.ndarray, real_mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=float)[real_mask]
    if selected.size == 0:
        raise ValueError("Checkpoint contains no real particles")
    return float(np.sqrt(np.mean(np.sum(selected * selected, axis=1))))


def rescale_checkpoint(
    source: Path,
    output: Path,
    *,
    source_temperature_k: float,
    target_temperature_k: float,
) -> dict[str, Any]:
    if source_temperature_k <= 0.0 or target_temperature_k <= 0.0:
        raise ValueError("Temperatures must be positive")
    if source.resolve() == output.resolve():
        raise ValueError("Output checkpoint must differ from source checkpoint")

    with np.load(source, allow_pickle=False) as checkpoint:
        arrays = {name: np.array(checkpoint[name], copy=True) for name in checkpoint.files}

    if "v" not in arrays:
        raise ValueError("Checkpoint has no velocity array 'v'")
    velocity = np.asarray(arrays["v"])
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError(f"Checkpoint v must have shape (N, 3), got {velocity.shape}")

    if "particle_is_virtual" in arrays:
        virtual = np.asarray(arrays["particle_is_virtual"], dtype=bool)
        if virtual.shape != (velocity.shape[0],):
            raise ValueError("particle_is_virtual shape does not match v")
        real_mask = ~virtual
    else:
        real_mask = np.ones(velocity.shape[0], dtype=bool)

    scale = math.sqrt(target_temperature_k / source_temperature_k)
    v_before = _rms_real(velocity, real_mask)
    arrays["v"] = np.asarray(velocity * scale, dtype=velocity.dtype)
    v_after = _rms_real(arrays["v"], real_mask)

    omega_before = None
    omega_after = None
    if "omega" in arrays:
        omega = np.asarray(arrays["omega"])
        if omega.shape != velocity.shape:
            raise ValueError("Checkpoint omega shape does not match v")
        omega_before = _rms_real(omega, real_mask)
        arrays["omega"] = np.asarray(omega * scale, dtype=omega.dtype)
        omega_after = _rms_real(arrays["omega"], real_mask)

    metadata = _metadata_from_arrays(arrays)
    source_hash = sha256_file(source)
    metadata["created_with_kT_kJ_mol"] = float(R_KJ_MOL_K * target_temperature_k)
    metadata["temperature_sweep_diagnostic"] = {
        "method": "iso_configurational_velocity_and_body_omega_rescale",
        "source_checkpoint_sha256": source_hash,
        "source_temperature_K": float(source_temperature_k),
        "target_temperature_K": float(target_temperature_k),
        "velocity_scale_sqrt_T_ratio": float(scale),
        "positions_unchanged": True,
        "orientations_unchanged": True,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    return {
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": source_hash,
        "output_checkpoint": str(output.resolve()),
        "output_checkpoint_sha256": sha256_file(output),
        "source_temperature_K": float(source_temperature_k),
        "target_temperature_K": float(target_temperature_k),
        "velocity_scale": float(scale),
        "real_particles": int(np.count_nonzero(real_mask)),
        "velocity_rms_before": v_before,
        "velocity_rms_after": v_after,
        "omega_rms_before": omega_before,
        "omega_rms_after": omega_after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-temperature-k", required=True, type=float)
    parser.add_argument("--target-temperature-k", required=True, type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite")

    summary = rescale_checkpoint(
        source,
        output,
        source_temperature_k=args.source_temperature_k,
        target_temperature_k=args.target_temperature_k,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
