#!/usr/bin/env python3
"""Small structural self-test for the TEL22 angle-ablation preparer."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare_angle", HERE / "12_prepare_angle_ablation.py")
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name, data in {
            "dataset.bin": b"dataset",
            "config.json": b"config",
            "rb.json": b"rb",
            "model.pt": b"model",
            "model.pt.manifest.json": b"manifest",
            "priors.json": json.dumps({"angles": [{"type": "harmonic", "k": 1.0}], "bonds": []}).encode(),
        }.items():
            (root / name).write_bytes(data)
        hashes = MOD.expected_hashes(
            dataset=root / "dataset.bin", config=root / "config.json", priors=root / "priors.json",
            rb_info=root / "rb.json", model=root / "model.pt",
        )
        metadata = {"schema_version": 3, "input_hashes": hashes}
        arrays = {
            "pos": np.arange(12, dtype=np.float64).reshape(4, 3),
            "v": np.arange(12, dtype=np.float64).reshape(4, 3) / 10,
            "box_l": np.asarray([5.0, 5.0, 5.0]),
            "particle_ids": np.arange(4, dtype=np.int64),
        }
        np.savez_compressed(root / "state.npz", **arrays, metadata_json=np.asarray(json.dumps(metadata)))
        got, meta = MOD.validate_source_checkpoint(
            root / "state.npz", source_priors=root / "priors.json", dataset=root / "dataset.bin",
            config=root / "config.json", rb_info=root / "rb.json", model=root / "model.pt",
        )
        assert meta["input_hashes"] == hashes
        for key in arrays:
            assert np.array_equal(got[key], arrays[key])
    print("[PASS] TEL22 angle-ablation structural self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
