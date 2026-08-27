#!/usr/bin/env python3
"""Prepare TEL22 Morse local-curvature quantile ablations for NVE diagnostics.

All production pair-specific Morse entries are retained so the runtime technical
marker topology and particle count remain unchanged. Selected contacts are made
inert by setting D=0 in derived priors. A derived checkpoint keeps every
mechanical array identical and updates only provenance metadata for the derived
priors hash.

Because production TEL22 uses identical D and a for all 180 Morse contacts,
2*D*a^2 at r0 cannot rank them. We instead rank contacts at the equilibrated
checkpoint using the spectral magnitude of the local pair-potential Hessian:

    K_local = max(|U''(r)|, |U'(r)/r|)

for U(r)=D(exp(-2a(r-r0))-2exp(-a(r-r0))).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

FRACTIONS = (0.05, 0.10, 0.20)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def read_runtime_endpoint_map(dataset: Path) -> tuple[dict[tuple[int, int], int], np.ndarray, int, int]:
    """Return endpoint->runtime particle-id mapping from dataset creation order."""
    endpoint_pid: dict[tuple[int, int], int] = {}
    with dataset.open("rb") as f:
        raw = f.read(24)
        if len(raw) != 24:
            raise ValueError(f"Dataset header is truncated: {dataset}")
        num_frames, num_molecules, num_total_sites = struct.unpack("iii", raw[:12])
        box = np.asarray(struct.unpack("3f", raw[12:24]), dtype=float)
        if min(num_frames, num_molecules, num_total_sites) <= 0 or np.any(box <= 0):
            raise ValueError("Invalid dataset header")
        pid = 0
        sites_seen = 0
        for _ in range(num_molecules):
            header = f.read(8)
            if len(header) != 8:
                raise ValueError("Dataset first frame is truncated before molecule header")
            mol_id, num_sites = struct.unpack("ii", header)
            endpoint_pid[(int(mol_id), -1)] = pid
            pid += 1
            if len(f.read(36)) != 36:  # center + force + torque
                raise ValueError("Dataset first frame is truncated in molecule data")
            for site_idx in range(num_sites):
                rec = f.read(16)
                if len(rec) != 16:
                    raise ValueError("Dataset first frame is truncated in site data")
                endpoint_pid[(int(mol_id), int(site_idx))] = pid
                pid += 1
                sites_seen += 1
        if sites_seen != num_total_sites:
            raise ValueError(f"Dataset site count mismatch: header={num_total_sites}, parsed={sites_seen}")
    return endpoint_pid, box, num_molecules, num_total_sites


def minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return delta - box * np.rint(delta / box)


def morse_entries(priors: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    return [
        (idx, entry)
        for idx, entry in enumerate(priors.get("bonds", []))
        if str(entry.get("type", "harmonic")).lower() == "morse"
    ]


def local_curvature_metrics(entry: dict[str, Any], r: float) -> dict[str, float]:
    D = float(entry["D"])
    a = float(entry["a"])
    r0 = float(entry["r0"])
    if D < 0 or a <= 0 or r0 < 0 or r <= 0:
        raise ValueError(f"Invalid Morse/local distance: D={D}, a={a}, r0={r0}, r={r}")
    q = math.exp(-a * (r - r0))
    dU_dr = 2.0 * D * a * q * (1.0 - q)
    d2U_dr2 = 2.0 * D * a * a * q * (2.0 * q - 1.0)
    tangential = dU_dr / r
    return {
        "distance_nm": r,
        "delta_r_nm": r - r0,
        "q_exp": q,
        "dU_dr": dU_dr,
        "radial_curvature": d2U_dr2,
        "tangential_curvature": tangential,
        "spectral_curvature": max(abs(d2U_dr2), abs(tangential)),
        "k_at_r0": 2.0 * D * a * a,
    }


def write_derived_checkpoint(source: Path, target: Path, *, hashes: dict[str, str | None], variant: str, source_priors: Path, target_priors: Path) -> None:
    with np.load(source, allow_pickle=False) as src:
        if "metadata_json" not in src.files:
            raise ValueError("Source checkpoint lacks metadata_json")
        arrays = {name: np.asarray(src[name]).copy() for name in src.files if name != "metadata_json"}
        metadata = scalar_json(src["metadata_json"])
    if not isinstance(metadata.get("input_hashes"), dict):
        raise ValueError("Source checkpoint metadata has no input_hashes map")
    metadata = json.loads(json.dumps(metadata))
    metadata["input_hashes"] = hashes
    metadata["diagnostic_morse_curvature_quantile"] = {
        "variant": variant,
        "source_checkpoint_sha256": sha256_file(source),
        "source_priors_sha256": sha256_file(source_priors),
        "target_priors_sha256": sha256_file(target_priors),
        "mechanical_state_policy": "all checkpoint arrays preserved exactly; provenance metadata only is updated",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priors", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--rb-info", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for path in (args.priors, args.config, args.dataset, args.rb_info, args.model, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = Path(str(args.model) + ".manifest.json")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    priors = json.loads(args.priors.read_text(encoding="utf-8"))
    entries = morse_entries(priors)
    if len(entries) != 180:
        raise ValueError(f"Expected 180 production pair-specific Morse entries, found {len(entries)}")
    if priors.get("morse_type_pairs", []):
        raise ValueError("This diagnostic requires morse_type_pairs=[]")

    D_values = {float(e["D"]) for _, e in entries}
    a_values = {float(e["a"]) for _, e in entries}
    k0_values = {2.0 * float(e["D"]) * float(e["a"]) ** 2 for _, e in entries}
    if len(k0_values) != 1:
        raise ValueError("Production Morse k_at_r0 is no longer degenerate; review ranking policy")

    endpoint_pid, dataset_box, num_molecules, num_sites = read_runtime_endpoint_map(args.dataset)
    with np.load(args.checkpoint, allow_pickle=False) as chk:
        for field in ("pos", "box_l", "metadata_json", "particle_ids"):
            if field not in chk.files:
                raise ValueError(f"Checkpoint lacks {field}")
        positions = np.asarray(chk["pos"], dtype=float)
        box = np.asarray(chk["box_l"], dtype=float)
        ids = np.asarray(chk["particle_ids"], dtype=np.int64)
        metadata = scalar_json(chk["metadata_json"])
    if not np.array_equal(ids, np.arange(len(ids), dtype=np.int64)):
        raise ValueError("Checkpoint particle IDs are not contiguous in runtime creation order")
    if len(positions) < num_molecules + num_sites:
        raise ValueError("Checkpoint has fewer particles than physical/runtime dataset particles")
    if np.max(np.abs(box - dataset_box)) > 1.0e-4:
        raise ValueError(f"Checkpoint/dataset box mismatch: checkpoint={box.tolist()}, dataset={dataset_box.tolist()}")

    current_hashes = expected_hashes(dataset=args.dataset, config=args.config, priors=args.priors, rb_info=args.rb_info, model=args.model)
    recorded = metadata.get("input_hashes")
    if not isinstance(recorded, dict):
        raise ValueError("Source checkpoint metadata has no input_hashes map")
    bad = [k for k, v in current_hashes.items() if recorded.get(k) != v]
    if bad:
        raise ValueError("Source checkpoint provenance mismatch for: " + ", ".join(bad))

    ranked: list[dict[str, Any]] = []
    for bond_index, entry in entries:
        ep_i = (int(entry["mol_i"]), int(entry.get("site_i", -1)))
        ep_j = (int(entry["mol_j"]), int(entry.get("site_j", -1)))
        if ep_i not in endpoint_pid or ep_j not in endpoint_pid:
            raise ValueError(f"Morse bond[{bond_index}] references missing endpoint {ep_i} or {ep_j}")
        pi = positions[endpoint_pid[ep_i]]
        pj = positions[endpoint_pid[ep_j]]
        r = float(np.linalg.norm(minimum_image(pj - pi, box)))
        metrics = local_curvature_metrics(entry, r)
        ranked.append({
            "bond_index": bond_index,
            "mol_i": ep_i[0], "site_i": ep_i[1], "mol_j": ep_j[0], "site_j": ep_j[1],
            "D": float(entry["D"]), "a": float(entry["a"]), "r0_nm": float(entry["r0"]),
            **metrics,
        })
    ranked.sort(key=lambda row: (-row["spectral_curvature"], row["bond_index"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "curvature_quantile_inputs.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite")

    variants: dict[str, Any] = {}
    previous: set[int] = set()
    for fraction in FRACTIONS:
        count = int(math.ceil(fraction * len(ranked)))
        selected_rows = ranked[:count]
        selected = {int(row["bond_index"]) for row in selected_rows}
        if not previous.issubset(selected):
            raise AssertionError("Curvature quantile variants are not nested")
        previous = selected
        name = f"top_{int(round(fraction * 100)):02d}pct_zeroD"
        variant = json.loads(json.dumps(priors))
        for idx in selected:
            if str(variant["bonds"][idx].get("type", "harmonic")).lower() != "morse":
                raise AssertionError(f"Selected bond[{idx}] is not Morse")
            variant["bonds"][idx]["D"] = 0.0
        outdir = args.output_dir / name
        outdir.mkdir(parents=True, exist_ok=True)
        priors_path = outdir / "cg_priors.json"
        priors_path.write_text(json.dumps(variant, indent=2) + "\n", encoding="utf-8")
        checkpoint_path = outdir / "equilibrated.npz"
        hashes = expected_hashes(dataset=args.dataset, config=args.config, priors=priors_path, rb_info=args.rb_info, model=args.model)
        write_derived_checkpoint(args.checkpoint, checkpoint_path, hashes=hashes, variant=name, source_priors=args.priors, target_priors=priors_path)
        variants[name] = {
            "fraction": fraction,
            "selected_count": count,
            "selected_bond_indices": sorted(selected),
            "selected_rank_min": 1,
            "selected_rank_max": count,
            "spectral_curvature_threshold": float(selected_rows[-1]["spectral_curvature"]),
            "spectral_curvature_max": float(selected_rows[0]["spectral_curvature"]),
            "sum_original_D_zeroed": float(sum(row["D"] for row in selected_rows)),
            "priors": str(priors_path.resolve()),
            "priors_sha256": sha256_file(priors_path),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }

    report = {
        "schema_version": 1,
        "kind": "tel22_morse_checkpoint_local_curvature_quantile_inputs",
        "ranking_metric": "max(abs(U''(r_eq)), abs(U'(r_eq)/r_eq))",
        "ranking_formula_note": "Central-potential local Hessian eigenvalue magnitudes at the equilibrated checkpoint; used because all production Morse contacts have identical D/a and therefore identical 2*D*a^2 at r0.",
        "source": {
            "priors": str(args.priors.resolve()), "priors_sha256": sha256_file(args.priors),
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
            "morse_entries": len(entries), "unique_D": sorted(D_values), "unique_a": sorted(a_values), "unique_k_at_r0": sorted(k0_values),
        },
        "ranked_contacts": ranked,
        "variants": variants,
        "ablation_policy": "Selected Morse contacts remain in priors/topology but are made force/energy inert with D=0. Technical-marker particle count is preserved. Derived checkpoints preserve every mechanical array and update only provenance metadata.",
        "caution": "Diagnostic Hamiltonian ablation only; no candidate is a reparameterized production model.",
    }
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    values = [r["spectral_curvature"] for r in ranked]
    print(f"[MORSE CURVATURE RANKING] contacts={len(ranked)} D={sorted(D_values)} a={sorted(a_values)} k_at_r0={sorted(k0_values)}")
    print(f"[LOCAL CURVATURE] max={max(values):.9g} median={np.median(values):.9g} min={min(values):.9g}")
    for name, row in variants.items():
        print(f"[{name}] zeroD={row['selected_count']} threshold={row['spectral_curvature_threshold']:.9g} priors={row['priors']}")
    print(f"[MANIFEST] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
