#!/usr/bin/env python3
"""Prepare a uniform TEL22 Morse a=0.85 diagnostic input.

All 180 production pair-specific Morse stabilizers retain D, r0, cutoff,
endpoints and topology. Only Morse ``a`` is scaled from 0.30 to 0.255.
The derived checkpoint preserves every mechanical array exactly and updates
only provenance metadata so the priors hash is internally consistent.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "19_prepare_morse_top10_a_softening.py"
spec = importlib.util.spec_from_file_location("morse_a_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

SCALE = 0.85
EXPECTED_CONTACTS = 180
EXPECTED_D = 50.0
EXPECTED_A = 0.3
SCALED_A = EXPECTED_A * SCALE


def build_uniform_variant(priors: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
    out = copy.deepcopy(priors)
    changed: list[int] = []
    for idx, before in enumerate(priors.get("bonds", [])):
        after = out["bonds"][idx]
        is_morse = str(before.get("type", "harmonic")).lower() == "morse"
        if is_morse:
            if float(before["D"]) != EXPECTED_D or float(before["a"]) != EXPECTED_A:
                raise ValueError(
                    f"Production Morse parameters changed at bond[{idx}]: "
                    f"D={before.get('D')} a={before.get('a')}"
                )
            after["a"] = SCALED_A
            changed.append(idx)
        allowed = {"a"} if is_morse else set()
        keys = set(before) | set(after)
        fields = {key for key in keys if before.get(key) != after.get(key)}
        if fields != allowed:
            raise AssertionError(f"Unexpected changed fields for bond[{idx}]: {sorted(fields)}")
    if len(changed) != EXPECTED_CONTACTS:
        raise ValueError(f"Expected {EXPECTED_CONTACTS} Morse contacts, found {len(changed)}")
    return out, changed


def write_checkpoint(
    source: Path,
    target: Path,
    *,
    hashes: dict[str, str | None],
    source_priors: Path,
    target_priors: Path,
    changed_indices: list[int],
) -> None:
    with np.load(source, allow_pickle=False) as src:
        if "metadata_json" not in src.files:
            raise ValueError("Source checkpoint lacks metadata_json")
        arrays = {name: np.asarray(src[name]).copy() for name in src.files if name != "metadata_json"}
        metadata = base.scalar_json(src["metadata_json"])
    if not isinstance(metadata.get("input_hashes"), dict):
        raise ValueError("Source checkpoint metadata has no input_hashes map")
    metadata = json.loads(json.dumps(metadata))
    metadata["input_hashes"] = hashes
    metadata["diagnostic_morse_uniform_a0p85"] = {
        "a_scale": SCALE,
        "original_a": EXPECTED_A,
        "scaled_a": SCALED_A,
        "morse_count": len(changed_indices),
        "changed_bond_indices": changed_indices,
        "source_checkpoint_sha256": base.sha256_file(source),
        "source_priors_sha256": base.sha256_file(source_priors),
        "target_priors_sha256": base.sha256_file(target_priors),
        "mechanical_state_policy": "all checkpoint arrays preserved exactly; provenance metadata only is updated",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def validate_existing(manifest_path: Path, source_priors: Path) -> dict[str, Any]:
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("kind") != "tel22_morse_uniform_a0p85_inputs":
        raise ValueError("Unexpected existing uniform-a manifest kind")
    if m.get("source", {}).get("priors_sha256") != base.sha256_file(source_priors):
        raise ValueError("Existing uniform-a manifest source-priors hash mismatch")
    if abs(float(m.get("a_scale", -1.0)) - SCALE) > 1e-15:
        raise ValueError("Existing uniform-a manifest scale mismatch")
    if int(m.get("morse_count", -1)) != EXPECTED_CONTACTS:
        raise ValueError("Existing uniform-a manifest Morse-count mismatch")
    if len(m.get("changed_bond_indices", [])) != EXPECTED_CONTACTS:
        raise ValueError("Existing uniform-a manifest changed-index count mismatch")
    for key in ("priors", "checkpoint"):
        path = Path(m[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        if m.get(key + "_sha256") != base.sha256_file(path):
            raise ValueError(f"Existing uniform-a {key} hash mismatch")
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priors", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--rb-info", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args()

    for path in (args.priors, args.config, args.dataset, args.rb_info, args.model, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    model_manifest = Path(str(args.model) + ".manifest.json")
    if not model_manifest.is_file():
        raise FileNotFoundError(model_manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "morse_uniform_a0p85_inputs.json"
    if args.reuse_existing and manifest_path.is_file():
        m = validate_existing(manifest_path, args.priors)
        print(f"[REUSE INPUTS] {manifest_path}")
        print(f"[UNIFORM a0.85] priors={m['priors']} checkpoint={m['checkpoint']}")
        return 0
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite or --reuse-existing")

    priors = json.loads(args.priors.read_text(encoding="utf-8"))
    if priors.get("morse_type_pairs", []):
        raise ValueError("This diagnostic requires morse_type_pairs=[]")
    derived, changed = build_uniform_variant(priors)

    source_hashes = base.expected_hashes(
        dataset=args.dataset, config=args.config, priors=args.priors, rb_info=args.rb_info, model=args.model
    )
    with np.load(args.checkpoint, allow_pickle=False) as chk:
        if "metadata_json" not in chk.files:
            raise ValueError("Source checkpoint lacks metadata_json")
        recorded = base.scalar_json(chk["metadata_json"]).get("input_hashes")
    if not isinstance(recorded, dict):
        raise ValueError("Source checkpoint metadata has no input_hashes map")
    bad = [key for key, value in source_hashes.items() if recorded.get(key) != value]
    if bad:
        raise ValueError("Source checkpoint provenance mismatch for: " + ", ".join(bad))

    priors_path = args.output_dir / "cg_priors.json"
    priors_path.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    checkpoint_path = args.output_dir / "equilibrated.npz"
    hashes = base.expected_hashes(
        dataset=args.dataset, config=args.config, priors=priors_path, rb_info=args.rb_info, model=args.model
    )
    write_checkpoint(
        args.checkpoint,
        checkpoint_path,
        hashes=hashes,
        source_priors=args.priors,
        target_priors=priors_path,
        changed_indices=changed,
    )

    manifest = {
        "schema_version": 1,
        "kind": "tel22_morse_uniform_a0p85_inputs",
        "a_scale": SCALE,
        "original_a": EXPECTED_A,
        "scaled_a": SCALED_A,
        "D_preserved": EXPECTED_D,
        "k_at_r0_ratio": SCALE * SCALE,
        "morse_count": len(changed),
        "changed_bond_indices": changed,
        "source": {
            "priors": str(args.priors.resolve()),
            "priors_sha256": base.sha256_file(args.priors),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": base.sha256_file(args.checkpoint),
        },
        "priors": str(priors_path.resolve()),
        "priors_sha256": base.sha256_file(priors_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": base.sha256_file(checkpoint_path),
        "policy": "Uniformly scale Morse a on all 180 structural/numerical stabilizer contacts; D, r0, topology, particle set, WCA/harmonics and checkpoint mechanical arrays are preserved.",
        "caution": "Diagnostic stabilizer comparison only; this is not yet a production TEL22 reparameterization.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[UNIFORM MORSE] contacts={len(changed)} a={EXPECTED_A:.6g}->{SCALED_A:.6g} k(r0)_ratio={SCALE*SCALE:.4f}")
    print(f"[PRIORS] {priors_path}")
    print(f"[CHECKPOINT] {checkpoint_path}")
    print(f"[MANIFEST] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
