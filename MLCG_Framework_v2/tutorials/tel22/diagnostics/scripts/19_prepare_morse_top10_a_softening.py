#!/usr/bin/env python3
"""Prepare TEL22 top-10%-local-curvature Morse-a softening diagnostics.

The selected 18 contacts are taken verbatim from diagnostic test 18's
checkpoint-local-curvature ranking. Production D and r0 are preserved; only
Morse a is scaled by 0.90, 0.80, or 0.70. Derived checkpoints preserve every
mechanical array and update only provenance metadata for the derived priors.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCALES = (0.90, 0.80, 0.70)
SOURCE_VARIANT = "top_10pct_zeroD"
EXPECTED_CONTACTS = 180
EXPECTED_SELECTED = 18


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar_json(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError("checkpoint metadata_json must be scalar")
    return json.loads(str(value.item()))


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


def morse_entries(priors: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    return [
        (idx, entry)
        for idx, entry in enumerate(priors.get("bonds", []))
        if str(entry.get("type", "harmonic")).lower() == "morse"
    ]


def variant_name(scale: float) -> str:
    return f"top10_a{scale:.2f}".replace(".", "p")


def selected_indices_from_ranking(manifest: dict[str, Any], source_priors_sha256: str) -> list[int]:
    if manifest.get("kind") != "tel22_morse_checkpoint_local_curvature_quantile_inputs":
        raise ValueError("Unexpected curvature-ranking manifest kind")
    source = manifest.get("source", {})
    if source.get("priors_sha256") != source_priors_sha256:
        raise ValueError("Curvature-ranking manifest source priors hash does not match production priors")
    if int(source.get("morse_entries", -1)) != EXPECTED_CONTACTS:
        raise ValueError("Curvature-ranking manifest no longer describes 180 Morse contacts")
    variants = manifest.get("variants", {})
    if SOURCE_VARIANT not in variants:
        raise ValueError(f"Curvature-ranking manifest lacks {SOURCE_VARIANT}")
    selected = [int(x) for x in variants[SOURCE_VARIANT].get("selected_bond_indices", [])]
    if len(selected) != EXPECTED_SELECTED or len(set(selected)) != EXPECTED_SELECTED:
        raise ValueError(f"Expected exactly {EXPECTED_SELECTED} unique top-10% indices, got {len(selected)}")
    ranked = manifest.get("ranked_contacts", [])
    ranked_top = [int(x["bond_index"]) for x in ranked[:EXPECTED_SELECTED]]
    if set(selected) != set(ranked_top):
        raise ValueError("Stored top-10% subset is inconsistent with the first 18 ranked contacts")
    return sorted(selected)


def build_scaled_variant(priors: dict[str, Any], selected: list[int], scale: float) -> dict[str, Any]:
    if not (0.0 < scale < 1.0):
        raise ValueError(f"Invalid a scale {scale}")
    out = copy.deepcopy(priors)
    selected_set = set(selected)
    for idx, before in enumerate(priors.get("bonds", [])):
        after = out["bonds"][idx]
        if idx in selected_set:
            if str(before.get("type", "harmonic")).lower() != "morse":
                raise ValueError(f"Selected bond[{idx}] is not Morse")
            after["a"] = float(before["a"]) * scale
        # Fail closed: only selected Morse a may differ.
        allowed = {"a"} if idx in selected_set else set()
        keys = set(before) | set(after)
        changed = {k for k in keys if before.get(k) != after.get(k)}
        if changed != allowed:
            raise AssertionError(f"Unexpected changed fields for bond[{idx}]: {sorted(changed)}")
    return out


def write_derived_checkpoint(source: Path, target: Path, *, hashes: dict[str, str | None], scale: float, source_priors: Path, target_priors: Path, selected: list[int]) -> None:
    with np.load(source, allow_pickle=False) as src:
        if "metadata_json" not in src.files:
            raise ValueError("Source checkpoint lacks metadata_json")
        arrays = {name: np.asarray(src[name]).copy() for name in src.files if name != "metadata_json"}
        metadata = scalar_json(src["metadata_json"])
    if not isinstance(metadata.get("input_hashes"), dict):
        raise ValueError("Source checkpoint metadata has no input_hashes map")
    metadata = json.loads(json.dumps(metadata))
    metadata["input_hashes"] = hashes
    metadata["diagnostic_morse_top10_a_softening"] = {
        "a_scale": scale,
        "selected_count": len(selected),
        "selected_bond_indices": selected,
        "source_checkpoint_sha256": sha256_file(source),
        "source_priors_sha256": sha256_file(source_priors),
        "target_priors_sha256": sha256_file(target_priors),
        "mechanical_state_policy": "all checkpoint arrays preserved exactly; provenance metadata only is updated",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def validate_existing(manifest_path: Path, *, source_priors: Path, ranking_manifest: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("kind") != "tel22_morse_top10_local_curvature_a_softening_inputs":
        raise ValueError("Unexpected existing input manifest kind")
    if m.get("source", {}).get("priors_sha256") != sha256_file(source_priors):
        raise ValueError("Existing input manifest source priors hash mismatch")
    if m.get("source", {}).get("ranking_manifest_sha256") != sha256_file(ranking_manifest):
        raise ValueError("Existing input manifest ranking hash mismatch")
    if [float(x) for x in m.get("a_scales", [])] != list(SCALES):
        raise ValueError("Existing input manifest scale grid mismatch")
    if len(m.get("selected_bond_indices", [])) != EXPECTED_SELECTED:
        raise ValueError("Existing input manifest selected-count mismatch")
    for scale in SCALES:
        name = variant_name(scale)
        row = m.get("variants", {}).get(name)
        if not isinstance(row, dict):
            raise ValueError(f"Existing input manifest lacks {name}")
        for key in ("priors", "checkpoint"):
            path = Path(row[key])
            if not path.is_file():
                raise FileNotFoundError(path)
            if row.get(key + "_sha256") != sha256_file(path):
                raise ValueError(f"Existing {name} {key} hash mismatch")
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priors", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--rb-info", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--ranking-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args()

    required = (args.priors, args.config, args.dataset, args.rb_info, args.model, args.checkpoint, args.ranking_manifest)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    model_manifest = Path(str(args.model) + ".manifest.json")
    if not model_manifest.is_file():
        raise FileNotFoundError(model_manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "morse_top10_a_softening_inputs.json"
    if args.reuse_existing and manifest_path.is_file():
        m = validate_existing(manifest_path, source_priors=args.priors, ranking_manifest=args.ranking_manifest)
        print(f"[REUSE INPUTS] {manifest_path}")
        for name, row in m["variants"].items():
            print(f"[{name}] priors={row['priors']} checkpoint={row['checkpoint']}")
        return 0
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite or --reuse-existing")

    priors = json.loads(args.priors.read_text(encoding="utf-8"))
    entries = morse_entries(priors)
    if len(entries) != EXPECTED_CONTACTS:
        raise ValueError(f"Expected {EXPECTED_CONTACTS} production Morse entries, found {len(entries)}")
    if priors.get("morse_type_pairs", []):
        raise ValueError("This diagnostic requires morse_type_pairs=[]")

    ranking = json.loads(args.ranking_manifest.read_text(encoding="utf-8"))
    selected = selected_indices_from_ranking(ranking, sha256_file(args.priors))
    selected_set = set(selected)
    selected_rows = [row for row in ranking["ranked_contacts"] if int(row["bond_index"]) in selected_set]
    selected_rows.sort(key=lambda row: int(row["rank"]))
    if [int(row["rank"]) for row in selected_rows] != list(range(1, EXPECTED_SELECTED + 1)):
        raise ValueError("Selected contacts are not exactly local-curvature ranks 1..18")

    # This test is intentionally tied to the diagnosed production parameterization.
    for idx, entry in entries:
        if float(entry["D"]) != 50.0 or float(entry["a"]) != 0.3:
            raise ValueError(f"Production Morse parameters changed at bond[{idx}]; review softening protocol")
    source_hashes = expected_hashes(dataset=args.dataset, config=args.config, priors=args.priors, rb_info=args.rb_info, model=args.model)
    with np.load(args.checkpoint, allow_pickle=False) as chk:
        if "metadata_json" not in chk.files:
            raise ValueError("Source checkpoint lacks metadata_json")
        recorded = scalar_json(chk["metadata_json"]).get("input_hashes")
    if not isinstance(recorded, dict):
        raise ValueError("Source checkpoint metadata has no input_hashes map")
    bad = [k for k, v in source_hashes.items() if recorded.get(k) != v]
    if bad:
        raise ValueError("Source checkpoint provenance mismatch for: " + ", ".join(bad))

    variants: dict[str, Any] = {}
    for scale in SCALES:
        name = variant_name(scale)
        derived = build_scaled_variant(priors, selected, scale)
        outdir = args.output_dir / name
        outdir.mkdir(parents=True, exist_ok=True)
        priors_path = outdir / "cg_priors.json"
        priors_path.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
        checkpoint_path = outdir / "equilibrated.npz"
        hashes = expected_hashes(dataset=args.dataset, config=args.config, priors=priors_path, rb_info=args.rb_info, model=args.model)
        write_derived_checkpoint(
            args.checkpoint, checkpoint_path, hashes=hashes, scale=scale,
            source_priors=args.priors, target_priors=priors_path, selected=selected,
        )
        variants[name] = {
            "a_scale": scale,
            "selected_count": len(selected),
            "selected_bond_indices": selected,
            "original_a": 0.3,
            "scaled_a": 0.3 * scale,
            "D_preserved": 50.0,
            "k_at_r0_ratio": scale * scale,
            "priors": str(priors_path.resolve()),
            "priors_sha256": sha256_file(priors_path),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }

    out = {
        "schema_version": 1,
        "kind": "tel22_morse_top10_local_curvature_a_softening_inputs",
        "a_scales": list(SCALES),
        "source": {
            "priors": str(args.priors.resolve()),
            "priors_sha256": sha256_file(args.priors),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "ranking_manifest": str(args.ranking_manifest.resolve()),
            "ranking_manifest_sha256": sha256_file(args.ranking_manifest),
            "ranking_metric": ranking.get("ranking_metric"),
        },
        "selected_count": len(selected),
        "selected_bond_indices": selected,
        "selected_contacts": selected_rows,
        "variants": variants,
        "policy": "Only Morse a is scaled on the fixed top-10%-local-curvature subset; D, r0, topology, particle set, WCA/harmonics and checkpoint mechanical arrays are preserved.",
        "caution": "Diagnostic Hamiltonian softening only; variants are not reparameterized production models.",
    }
    manifest_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[TOP10 SUBSET] contacts={len(selected)} ranks=1..18 indices={selected}")
    for name, row in variants.items():
        print(f"[{name}] a={row['scaled_a']:.6g} k(r0)_ratio={row['k_at_r0_ratio']:.4f} priors={row['priors']}")
    print(f"[MANIFEST] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
