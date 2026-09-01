#!/usr/bin/env python3
"""Prepare provenance-safe Ala2 runtime inputs and diverse replica starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


BOND_TUPLES = [(0, 1), (1, 2), (2, 3), (3, 4)]
ONE_THREE_TUPLES = [(0, 2), (1, 3), (2, 4)]
WCA_CUTOFF_NM = 0.22
WCA_SIGMA_NM = WCA_CUTOFF_NM / (2.0 ** (1.0 / 6.0))
KBT_300K_KJ_MOL = 0.008314462618 * 300.0
NONBONDED_TUPLES = [(0, 3), (0, 4), (1, 4)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_exact(handle, size: int) -> bytes:
    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError("Truncated MLCG binary dataset")
    return payload


def read_frame_payloads(path: Path) -> list[bytes]:
    payloads: list[bytes] = []
    with path.open("rb") as handle:
        (frame_count,) = struct.unpack("<i", read_exact(handle, 4))
        if frame_count <= 0:
            raise ValueError("Dataset contains no frames")
        for _ in range(frame_count):
            parts = [read_exact(handle, 20)]
            molecule_count, total_sites, _, _, _ = struct.unpack("<ii3f", parts[0])
            observed_sites = 0
            if molecule_count != 5 or total_sites != 5:
                raise ValueError(
                    f"Ala2 runtime expects 5 molecules and 5 sites; got "
                    f"{molecule_count} and {total_sites}"
                )
            for molecule_index in range(molecule_count):
                molecule_header = read_exact(handle, 8)
                parts.append(molecule_header)
                molecule_id, site_count = struct.unpack("<ii", molecule_header)
                if molecule_id != molecule_index or site_count != 1:
                    raise ValueError("Ala2 dataset molecule/site ordering is inconsistent")
                parts.append(read_exact(handle, 36))
                parts.append(read_exact(handle, 16 * site_count))
                observed_sites += site_count
            if observed_sites != total_sites:
                raise ValueError("Ala2 dataset site count is inconsistent")
            payloads.append(b"".join(parts))
        if handle.read(1):
            raise ValueError("Unexpected trailing bytes in MLCG binary dataset")
    return payloads


def write_single_frame_dataset(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<i", 1))
        handle.write(payload)


def runtime_priors(priors: dict) -> dict:
    result = json.loads(json.dumps(priors))
    bonds = result.get("bonds", [])
    angles = result.get("angles", [])
    if len(bonds) != 4 or len(angles) != 3:
        raise ValueError(
            "The physical A/B test requires the harmonic Ala2 baseline "
            "(four bonds and three angles)"
        )
    for bond in bonds:
        if str(bond.get("type", "harmonic")).lower() != "harmonic":
            raise ValueError("Ala2 A/B runtime accepts harmonic bond priors only")
        bond["site_i"] = 0
        bond["site_j"] = 0
        bond["exclude_wca"] = True
    for angle in angles:
        if str(angle.get("type", "harmonic")).lower() != "harmonic":
            raise ValueError("Ala2 A/B runtime accepts harmonic angle priors only")
        angle["site_i"] = 0
        angle["site_j"] = 0
        angle["site_k"] = 0
    direct_pairs = [list(pair) for pair in BOND_TUPLES]
    direct_site_pairs = [[i, j, 0, 0] for i, j in BOND_TUPLES]
    one_three_pairs = [list(pair) for pair in ONE_THREE_TUPLES]
    result["wca_exclusions"] = {
        "policy_version": 3,
        "exclude_12": True,
        "exclude_13": True,
        "direct_scope": "bonded_site_pairs_only",
        "one_three_scope": "molecule_pair_all_sites",
        "pair_source": "explicit_topology_pairs_v3",
        "direct_pairs": direct_pairs,
        "direct_site_pairs": direct_site_pairs,
        "one_three_pairs": one_three_pairs,
        "direct_pair_count": len(direct_pairs),
        "direct_site_pair_count": len(direct_site_pairs),
        "one_three_pair_count": len(one_three_pairs),
    }
    result["wca_pairs"] = {
        f"{type_i}_{type_j}": {
            "type_i": type_i,
            "type_j": type_j,
            "sigma_nm": WCA_SIGMA_NM,
            "epsilon_kjmol": KBT_300K_KJ_MOL,
            "cutoff_nm": WCA_CUTOFF_NM,
            "source": "cgnet_style_ood_excluded_volume",
        }
        for type_i, type_j in ((6, 6), (6, 7), (7, 7))
    }
    result.setdefault("morse_type_pairs", [])
    result.setdefault("dihedrals", [])
    return result


def rigid_bodies() -> dict:
    return {
        "ALA2_C": {
            "schema_version": 2,
            "body_frame": "principal_axes",
            "auto_align_sites": True,
            "mass_amu": 12.011,
            "inertia_amu_nm2": [1.0, 1.0, 1.0],
            "sites": {"C": {"type": 6, "relative_pos_nm": [0.0, 0.0, 0.0]}},
        },
        "ALA2_N": {
            "schema_version": 2,
            "body_frame": "principal_axes",
            "auto_align_sites": True,
            "mass_amu": 14.007,
            "inertia_amu_nm2": [1.0, 1.0, 1.0],
            "sites": {"N": {"type": 7, "relative_pos_nm": [0.0, 0.0, 0.0]}},
        },
    }


def minimum_nonbonded_distance(coordinates: np.ndarray) -> float:
    return min(
        float(np.min(np.linalg.norm(coordinates[:, j] - coordinates[:, i], axis=1)))
        for i, j in NONBONDED_TUPLES
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--priors", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.replicas < 2:
        raise ValueError("At least two matched replicas are required")
    frames = read_frame_payloads(args.dataset)
    if args.replicas > len(frames):
        raise ValueError("Replica count exceeds the number of dataset frames")
    minimum_nonbonded_nm = None
    if args.reference is not None:
        with np.load(args.reference, allow_pickle=False) as reference:
            coordinates = np.asarray(reference["coordinates_nm"], dtype=np.float64)
        minimum_nonbonded_nm = minimum_nonbonded_distance(coordinates)
        if minimum_nonbonded_nm <= WCA_CUTOFF_NM:
            raise ValueError(
                f"OOD WCA cutoff {WCA_CUTOFF_NM} nm changes a reference frame; "
                f"minimum nonbonded distance is {minimum_nonbonded_nm} nm"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    priors = runtime_priors(json.loads(args.priors.read_text()))
    priors_path = args.output_dir / "ala2_runtime_priors.json"
    rb_path = args.output_dir / "ala2_rigid_bodies_info.json"
    priors_path.write_text(json.dumps(priors, indent=2, sort_keys=True) + "\n")
    rb_path.write_text(json.dumps(rigid_bodies(), indent=2, sort_keys=True) + "\n")

    frame_indices = np.linspace(0, len(frames) - 1, args.replicas, dtype=int).tolist()
    replica_files = []
    for replica, frame_index in enumerate(frame_indices):
        output = args.output_dir / f"replica_{replica:02d}_dataset.bin"
        write_single_frame_dataset(output, frames[frame_index])
        replica_files.append(
            {
                "replica": replica,
                "source_frame": int(frame_index),
                "dataset": output.name,
                "sha256": sha256_file(output),
            }
        )

    report = {
        "schema_version": 1,
        "source_dataset": str(args.dataset.resolve()),
        "source_dataset_sha256": sha256_file(args.dataset),
        "source_frames": len(frames),
        "replicas": replica_files,
        "runtime_priors": priors_path.name,
        "runtime_priors_sha256": sha256_file(priors_path),
        "rigid_bodies": rb_path.name,
        "rigid_bodies_sha256": sha256_file(rb_path),
        "matched_start_policy": "evenly_spaced_reference_frames",
        "ood_wca": {
            "cutoff_nm": WCA_CUTOFF_NM,
            "sigma_nm": WCA_SIGMA_NM,
            "epsilon_kjmol": KBT_300K_KJ_MOL,
            "active_pairs": "molecular pairs separated by more than two bonds",
            "training_targets_changed": False,
            "reference_minimum_nonbonded_distance_nm": minimum_nonbonded_nm,
        },
    }
    report_path = args.report or args.output_dir / "ala2_runtime_preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[PASS] Prepared {args.replicas} matched Ala2 runtime starts.")


if __name__ == "__main__":
    main()
