#!/usr/bin/env python3
"""Prepare provenance-safe TEL22 NVE prior-ablation inputs.

This diagnostic never edits production priors/checkpoints.  It derives variants
that selectively remove pair-specific/type-pair Morse and/or dihedral priors.
When Morse endpoint markers disappear, the corresponding technical virtual
particles are stripped from the checkpoint tail while the real mechanical state
is kept byte-for-byte numerically identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

VARIANTS = (
    ("baseline", False, False),
    ("no_morse", True, False),
    ("no_dihedrals", False, True),
    ("no_morse_no_dihedrals", True, True),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dataset_counts(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        raw = handle.read(12)
    if len(raw) != 12:
        raise ValueError(f"Dataset header is truncated: {path}")
    frames, molecules, sites = struct.unpack("iii", raw)
    if frames <= 0 or molecules <= 0 or sites <= 0:
        raise ValueError(f"Invalid dataset counts: frames={frames}, molecules={molecules}, sites={sites}")
    return frames, molecules, sites


def morse_endpoints(priors: dict[str, Any]) -> set[tuple[int, int]]:
    endpoints: set[tuple[int, int]] = set()
    for entry in priors.get("bonds", []):
        if str(entry.get("type", "harmonic")).lower() != "morse":
            continue
        endpoints.add((int(entry["mol_i"]), int(entry.get("site_i", -1))))
        endpoints.add((int(entry["mol_j"]), int(entry.get("site_j", -1))))
    return endpoints


def count_morse(priors: dict[str, Any]) -> int:
    return sum(
        1 for entry in priors.get("bonds", [])
        if str(entry.get("type", "harmonic")).lower() == "morse"
    ) + len(priors.get("morse_type_pairs", []))


def ablate(priors: dict[str, Any], *, remove_morse: bool, remove_dihedrals: bool) -> dict[str, Any]:
    out = json.loads(json.dumps(priors))
    if remove_morse:
        out["bonds"] = [
            entry for entry in out.get("bonds", [])
            if str(entry.get("type", "harmonic")).lower() != "morse"
        ]
        out["morse_type_pairs"] = []
    if remove_dihedrals:
        out["dihedrals"] = []
    return out


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


def scalar_json(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError("checkpoint metadata_json must be scalar")
    return json.loads(str(value.item()))


def validate_source_checkpoint(
    checkpoint: Path,
    *,
    source_priors: Path,
    dataset: Path,
    config: Path,
    rb_info: Path,
    model: Path,
    base_particles: int,
    marker_count: int,
    num_species: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(checkpoint, allow_pickle=False) as src:
        required = {
            "pos", "v", "quat", "omega", "box_l", "metadata_json",
            "particle_ids", "particle_types", "particle_mol_ids", "particle_is_virtual",
        }
        missing = sorted(required.difference(src.files))
        if missing:
            raise ValueError("Source checkpoint lacks required fields: " + ", ".join(missing))
        arrays = {name: np.asarray(src[name]).copy() for name in required if name != "metadata_json"}
        metadata = scalar_json(src["metadata_json"])

    source_n = int(arrays["pos"].shape[0])
    expected_n = base_particles + marker_count
    if source_n != expected_n:
        raise ValueError(
            f"Source checkpoint particle count={source_n}; expected {base_particles} physical/runtime "
            f"particles + {marker_count} Morse markers = {expected_n}. Regenerate equilibrated.npz "
            "with the current TEL22 priors before running this diagnostic."
        )
    for name in ("v", "quat", "omega", "particle_ids", "particle_types", "particle_mol_ids", "particle_is_virtual"):
        if arrays[name].shape[0] != source_n:
            raise ValueError(f"Checkpoint field {name} does not match pos particle count")

    ids = arrays["particle_ids"].astype(np.int64, copy=False)
    if not np.array_equal(ids, np.arange(source_n, dtype=np.int64)):
        raise ValueError("TEL22 checkpoint particle IDs are not contiguous in runtime creation order")

    if marker_count:
        marker_slice = slice(base_particles, source_n)
        if not np.all(arrays["particle_is_virtual"][marker_slice].astype(bool)):
            raise ValueError("Expected Morse-marker checkpoint tail contains non-virtual particles")
        first_marker_type = int(num_species) + 2
        if np.any(arrays["particle_types"][marker_slice].astype(int) < first_marker_type):
            raise ValueError("Expected Morse-marker checkpoint tail contains physical/COM particle types")

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


def write_checkpoint(
    path: Path,
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    keep_particles: int,
    hashes: dict[str, str | None],
    source_checkpoint: Path,
    variant: str,
) -> None:
    out_meta = json.loads(json.dumps(metadata))
    out_meta["input_hashes"] = hashes
    out_meta["diagnostic_prior_ablation"] = {
        "variant": variant,
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "mechanical_state_policy": "prefix-preserving; only technical Morse marker tail may be removed",
    }
    particle_fields = {
        name: np.asarray(arrays[name])[:keep_particles].copy()
        for name in (
            "pos", "v", "quat", "omega", "particle_ids", "particle_types",
            "particle_mol_ids", "particle_is_virtual",
        )
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **particle_fields,
        box_l=np.asarray(arrays["box_l"]).copy(),
        metadata_json=np.asarray(json.dumps(out_meta, sort_keys=True)),
    )


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

    paths = [args.priors, args.config, args.dataset, args.rb_info, args.model, args.checkpoint]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    model_manifest = Path(str(args.model) + ".manifest.json")
    if not model_manifest.is_file():
        raise FileNotFoundError(model_manifest)

    source_priors = json.loads(args.priors.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, num_molecules, num_sites = read_dataset_counts(args.dataset)
    base_particles = num_molecules + num_sites
    source_markers = len(morse_endpoints(source_priors))
    source_morse = count_morse(source_priors)
    source_dihedrals = len(source_priors.get("dihedrals", []))

    arrays, metadata = validate_source_checkpoint(
        args.checkpoint,
        source_priors=args.priors,
        dataset=args.dataset,
        config=args.config,
        rb_info=args.rb_info,
        model=args.model,
        base_particles=base_particles,
        marker_count=source_markers,
        num_species=int(config["num_species"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "ablation_inputs.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite")

    source_hash = sha256_file(args.priors)
    variants: dict[str, Any] = {}
    canonical_by_semantic: dict[str, str] = {}

    def semantic_key(value: dict[str, Any]) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    for name, remove_morse, remove_dihedrals in VARIANTS:
        variant_priors = ablate(
            source_priors, remove_morse=remove_morse, remove_dihedrals=remove_dihedrals
        )
        removed_morse = source_morse - count_morse(variant_priors)
        removed_dihedrals = source_dihedrals - len(variant_priors.get("dihedrals", []))
        semantic = semantic_key(variant_priors)
        alias_of = canonical_by_semantic.get(semantic)

        if alias_of is not None:
            canonical = variants[alias_of]
            priors_path = Path(canonical["priors"])
            checkpoint_path = Path(canonical["checkpoint"])
        elif name == "baseline":
            priors_path = args.priors.resolve()
            checkpoint_path = args.checkpoint.resolve()
            canonical_by_semantic[semantic] = name
        else:
            canonical_by_semantic[semantic] = name
            variant_dir = args.output_dir / name
            priors_path = variant_dir / "cg_priors.json"
            variant_dir.mkdir(parents=True, exist_ok=True)
            priors_text = json.dumps(variant_priors, indent=2, sort_keys=False) + "\n"
            priors_path.write_text(priors_text, encoding="utf-8")
            checkpoint_path = variant_dir / "equilibrated.npz"

        target_markers = len(morse_endpoints(variant_priors))
        if target_markers not in (0, source_markers):
            raise ValueError(
                f"Variant {name} changes Morse markers from {source_markers} to {target_markers}; "
                "partial marker-topology ablation is not supported by this diagnostic."
            )

        if name != "baseline" and alias_of is None:
            hashes = expected_hashes(
                dataset=args.dataset,
                config=args.config,
                priors=priors_path,
                rb_info=args.rb_info,
                model=args.model,
            )
            keep = base_particles + target_markers
            write_checkpoint(
                checkpoint_path,
                arrays=arrays,
                metadata=metadata,
                keep_particles=keep,
                hashes=hashes,
                source_checkpoint=args.checkpoint,
                variant=name,
            )

        variants[name] = {
            "remove_morse": remove_morse,
            "remove_dihedrals": remove_dihedrals,
            "removed_morse_entries": removed_morse,
            "removed_dihedral_entries": removed_dihedrals,
            "remaining_morse_entries": count_morse(variant_priors),
            "remaining_dihedral_entries": len(variant_priors.get("dihedrals", [])),
            "remaining_morse_markers": target_markers,
            "priors": str(priors_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "alias_of": alias_of,
            "run_required": alias_of is None,
        }

    report = {
        "schema_version": 1,
        "kind": "tel22_nve_morse_dihedral_prior_ablation_inputs",
        "scope": "diagnostic_only; trained PaiNN checkpoint is unchanged while selected analytic priors are removed",
        "source": {
            "priors": str(args.priors.resolve()),
            "priors_sha256": source_hash,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "morse_entries": source_morse,
            "pair_specific_morse_markers": source_markers,
            "dihedral_entries": source_dihedrals,
            "base_runtime_particles_without_morse_markers": base_particles,
        },
        "variants": variants,
        "notes": [
            "Removing priors changes the diagnostic Hamiltonian; the trained PaiNN residual is intentionally not retrained.",
            "If production dihedrals are empty, no_dihedrals aliases baseline and no_morse_no_dihedrals aliases no_morse.",
            "Morse technical virtual markers are checkpoint-only interaction carriers and are stripped when Morse is absent.",
        ],
    }
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[TEL22 PRIOR ABLATION INPUTS]")
    print(f"Morse entries      : {source_morse}")
    print(f"Morse markers      : {source_markers}")
    print(f"Dihedral entries   : {source_dihedrals}")
    for name, item in variants.items():
        alias = f" alias={item['alias_of']}" if item["alias_of"] else ""
        print(
            f"{name:24s} run={str(item['run_required']).lower():5s} "
            f"removed_morse={item['removed_morse_entries']:3d} "
            f"removed_dihedrals={item['removed_dihedral_entries']:3d}{alias}"
        )
    if source_dihedrals == 0:
        print("[INFO] Production TEL22 has zero dihedral priors: dihedral ablation is a structural no-op.")
    print(f"[REPORT] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
