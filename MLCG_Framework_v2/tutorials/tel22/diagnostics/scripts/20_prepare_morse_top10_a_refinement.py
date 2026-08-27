#!/usr/bin/env python3
"""Prepare a refined TEL22 Morse-a sweep around the test-19 a=0.90 optimum.

The fixed 18-contact top-10% local-curvature subset is taken verbatim from
 test 18. Production D and r0 are preserved; only Morse a is scaled on those
contacts. The a=0.90 point is deliberately not generated here because test 20
reuses the already-completed test-19 report as its center reference.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "19_prepare_morse_top10_a_softening.py"
spec = importlib.util.spec_from_file_location("morse_top10_a_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

SCALE_SPECS = (
    ("top10_a0p950", 0.950),
    ("top10_a0p925", 0.925),
    ("top10_a0p875", 0.875),
    ("top10_a0p850", 0.850),
)
CENTER_SCALE = 0.900
EXPECTED_SELECTED = 18
EXPECTED_CONTACTS = 180


def write_refined_checkpoint(
    source: Path,
    target: Path,
    *,
    hashes: dict[str, str | None],
    scale: float,
    source_priors: Path,
    target_priors: Path,
    selected: list[int],
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
    metadata["diagnostic_morse_top10_a_refinement"] = {
        "a_scale": scale,
        "selected_count": len(selected),
        "selected_bond_indices": selected,
        "source_checkpoint_sha256": base.sha256_file(source),
        "source_priors_sha256": base.sha256_file(source_priors),
        "target_priors_sha256": base.sha256_file(target_priors),
        "mechanical_state_policy": "all checkpoint arrays preserved exactly; provenance metadata only is updated",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def validate_existing(manifest_path: Path, *, source_priors: Path, ranking_manifest: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("kind") != "tel22_morse_top10_local_curvature_a_refinement_inputs":
        raise ValueError("Unexpected existing input manifest kind")
    src = m.get("source", {})
    if src.get("priors_sha256") != base.sha256_file(source_priors):
        raise ValueError("Existing refinement manifest source-priors hash mismatch")
    if src.get("ranking_manifest_sha256") != base.sha256_file(ranking_manifest):
        raise ValueError("Existing refinement manifest ranking hash mismatch")
    expected = [scale for _name, scale in SCALE_SPECS]
    if [float(x) for x in m.get("candidate_a_scales", [])] != expected:
        raise ValueError("Existing refinement manifest scale grid mismatch")
    if int(m.get("selected_count", -1)) != EXPECTED_SELECTED:
        raise ValueError("Existing refinement manifest selected-count mismatch")
    for name, scale in SCALE_SPECS:
        row = m.get("variants", {}).get(name)
        if not isinstance(row, dict) or abs(float(row.get("a_scale", -1.0)) - scale) > 1e-15:
            raise ValueError(f"Existing refinement manifest lacks valid {name}")
        for key in ("priors", "checkpoint"):
            path = Path(row[key])
            if not path.is_file():
                raise FileNotFoundError(path)
            if row.get(key + "_sha256") != base.sha256_file(path):
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
    manifest_path = args.output_dir / "morse_top10_a_refinement_inputs.json"
    if args.reuse_existing and manifest_path.is_file():
        m = validate_existing(manifest_path, source_priors=args.priors, ranking_manifest=args.ranking_manifest)
        print(f"[REUSE INPUTS] {manifest_path}")
        for name, row in m["variants"].items():
            print(f"[{name}] priors={row['priors']} checkpoint={row['checkpoint']}")
        return 0
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite or --reuse-existing")

    priors = json.loads(args.priors.read_text(encoding="utf-8"))
    entries = base.morse_entries(priors)
    if len(entries) != EXPECTED_CONTACTS:
        raise ValueError(f"Expected {EXPECTED_CONTACTS} production Morse entries, found {len(entries)}")
    if priors.get("morse_type_pairs", []):
        raise ValueError("This diagnostic requires morse_type_pairs=[]")

    ranking = json.loads(args.ranking_manifest.read_text(encoding="utf-8"))
    selected = base.selected_indices_from_ranking(ranking, base.sha256_file(args.priors))
    if len(selected) != EXPECTED_SELECTED:
        raise ValueError(f"Expected {EXPECTED_SELECTED} selected contacts, got {len(selected)}")
    selected_set = set(selected)
    selected_rows = [row for row in ranking["ranked_contacts"] if int(row["bond_index"]) in selected_set]
    selected_rows.sort(key=lambda row: int(row["rank"]))
    if [int(row["rank"]) for row in selected_rows] != list(range(1, EXPECTED_SELECTED + 1)):
        raise ValueError("Selected contacts are not exactly local-curvature ranks 1..18")

    for idx, entry in entries:
        if float(entry["D"]) != 50.0 or float(entry["a"]) != 0.3:
            raise ValueError(f"Production Morse parameters changed at bond[{idx}]; review refinement protocol")

    source_hashes = base.expected_hashes(
        dataset=args.dataset, config=args.config, priors=args.priors, rb_info=args.rb_info, model=args.model
    )
    with np.load(args.checkpoint, allow_pickle=False) as chk:
        if "metadata_json" not in chk.files:
            raise ValueError("Source checkpoint lacks metadata_json")
        recorded = base.scalar_json(chk["metadata_json"]).get("input_hashes")
    if not isinstance(recorded, dict):
        raise ValueError("Source checkpoint metadata has no input_hashes map")
    bad = [k for k, v in source_hashes.items() if recorded.get(k) != v]
    if bad:
        raise ValueError("Source checkpoint provenance mismatch for: " + ", ".join(bad))

    variants: dict[str, Any] = {}
    for name, scale in SCALE_SPECS:
        derived = base.build_scaled_variant(priors, selected, scale)
        outdir = args.output_dir / name
        outdir.mkdir(parents=True, exist_ok=True)
        priors_path = outdir / "cg_priors.json"
        priors_path.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
        checkpoint_path = outdir / "equilibrated.npz"
        hashes = base.expected_hashes(
            dataset=args.dataset, config=args.config, priors=priors_path, rb_info=args.rb_info, model=args.model
        )
        write_refined_checkpoint(
            args.checkpoint,
            checkpoint_path,
            hashes=hashes,
            scale=scale,
            source_priors=args.priors,
            target_priors=priors_path,
            selected=selected,
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
            "priors_sha256": base.sha256_file(priors_path),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": base.sha256_file(checkpoint_path),
        }

    out = {
        "schema_version": 1,
        "kind": "tel22_morse_top10_local_curvature_a_refinement_inputs",
        "center_reference_a_scale": CENTER_SCALE,
        "candidate_a_scales": [scale for _name, scale in SCALE_SPECS],
        "source": {
            "priors": str(args.priors.resolve()),
            "priors_sha256": base.sha256_file(args.priors),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": base.sha256_file(args.checkpoint),
            "ranking_manifest": str(args.ranking_manifest.resolve()),
            "ranking_manifest_sha256": base.sha256_file(args.ranking_manifest),
            "ranking_metric": ranking.get("ranking_metric"),
        },
        "selected_count": len(selected),
        "selected_bond_indices": selected,
        "selected_contacts": selected_rows,
        "variants": variants,
        "policy": "Refine Morse a around scale 0.90 on the fixed top-10%-local-curvature subset; D, r0, topology, particle set, WCA/harmonics and checkpoint mechanical arrays are preserved.",
        "caution": "Diagnostic Hamiltonian refinement only; variants are not reparameterized production models.",
    }
    manifest_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[TOP10 SUBSET] contacts={len(selected)} ranks=1..18 indices={selected}")
    for name, row in variants.items():
        print(f"[{name}] a={row['scaled_a']:.6g} k(r0)_ratio={row['k_at_r0_ratio']:.6f} priors={row['priors']}")
    print(f"[MANIFEST] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
