#!/usr/bin/env python3
"""Convert the public CGnet Ala2 arrays to the MLCG binary dataset format."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


ANGSTROM_TO_NM = 0.1
KCAL_PER_MOL_ANGSTROM_TO_KJ_PER_MOL_NM = 41.84
R_KJ_MOL_K = 0.008314462618
WCA_CUTOFF_NM = 0.22
WCA_SIGMA_NM = WCA_CUTOFF_NM / (2.0 ** (1.0 / 6.0))
BEAD_TYPES = np.asarray([6, 7, 6, 6, 7], dtype=np.int32)
BEAD_NAMES = ["ACE_C", "ALA_N", "ALA_CA", "ALA_C", "NME_N"]
BOND_TUPLES = [(0, 1), (1, 2), (2, 3), (3, 4)]
ANGLE_TUPLES = [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
NONBONDED_TUPLES = [(0, 3), (0, 4), (1, 4)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def angles_for_triplets(coordinates: np.ndarray) -> np.ndarray:
    values = []
    for i, j, k in ANGLE_TUPLES:
        r_ji = coordinates[:, i] - coordinates[:, j]
        r_jk = coordinates[:, k] - coordinates[:, j]
        denominator = np.linalg.norm(r_ji, axis=1) * np.linalg.norm(r_jk, axis=1)
        if np.any(denominator <= 1.0e-12):
            raise ValueError(f"Degenerate angle geometry for triplet {(i, j, k)}")
        cosine = np.sum(r_ji * r_jk, axis=1) / denominator
        values.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.stack(values, axis=1)


def minimum_nonbonded_distance(coordinates: np.ndarray) -> float:
    return min(
        float(np.min(np.linalg.norm(coordinates[:, j] - coordinates[:, i], axis=1)))
        for i, j in NONBONDED_TUPLES
    )


def fit_harmonic_priors(
    coordinates: np.ndarray, temperature: float
) -> tuple[list[dict], list[dict]]:
    kbt = R_KJ_MOL_K * temperature
    bonds: list[dict] = []
    for index, (i, j) in enumerate(BOND_TUPLES):
        distances = np.linalg.norm(coordinates[:, j] - coordinates[:, i], axis=1)
        variance = float(np.var(distances))
        if variance <= 0.0:
            raise ValueError(f"Zero bond variance for {(i, j)}")
        bonds.append(
            {
                "mol_i": i,
                "mol_j": j,
                "site_i": 0,
                "site_j": 0,
                "type": "harmonic",
                "k": kbt / variance,
                "r0": float(np.mean(distances)),
                "name": f"ala2_bond_{index}_{BEAD_NAMES[i]}_{BEAD_NAMES[j]}",
                "exclude_wca": True,
            }
        )

    theta = angles_for_triplets(coordinates)
    angles: list[dict] = []
    for index, (i, j, k) in enumerate(ANGLE_TUPLES):
        variance = float(np.var(theta[:, index]))
        if variance <= 0.0:
            raise ValueError(f"Zero angle variance for {(i, j, k)}")
        angles.append(
            {
                "mol_i": i,
                "mol_j": j,
                "mol_k": k,
                "site_i": 0,
                "site_j": 0,
                "site_k": 0,
                "type": "harmonic",
                "k": kbt / variance,
                "theta0": float(np.mean(theta[:, index])),
                "name": f"ala2_angle_{index}_{BEAD_NAMES[i]}_{BEAD_NAMES[j]}_{BEAD_NAMES[k]}",
                "exclude_wca": True,
            }
        )
    return bonds, angles


def harmonic_prior_forces(
    coordinates: np.ndarray, bonds: list[dict], angles: list[dict]
) -> np.ndarray:
    """Return forces with the same signs/conventions as build_cg_dataset.py."""
    result = np.zeros_like(coordinates, dtype=np.float64)

    for bond in bonds:
        i, j = int(bond["mol_i"]), int(bond["mol_j"])
        vector_i_to_j = coordinates[:, j] - coordinates[:, i]
        distance = np.linalg.norm(vector_i_to_j, axis=1)
        if np.any(distance <= 1.0e-12):
            raise ValueError(f"Degenerate bond geometry for {(i, j)}")
        force_i = (
            float(bond["k"])
            * (distance - float(bond["r0"]))[:, None]
            * vector_i_to_j
            / distance[:, None]
        )
        result[:, i] += force_i
        result[:, j] -= force_i

    for angle in angles:
        i, j, k = int(angle["mol_i"]), int(angle["mol_j"]), int(angle["mol_k"])
        r_ji = coordinates[:, i] - coordinates[:, j]
        r_jk = coordinates[:, k] - coordinates[:, j]
        d_ji = np.linalg.norm(r_ji, axis=1)
        d_jk = np.linalg.norm(r_jk, axis=1)
        if np.any(d_ji <= 1.0e-12) or np.any(d_jk <= 1.0e-12):
            raise ValueError(f"Degenerate angle geometry for {(i, j, k)}")
        cosine = np.sum(r_ji * r_jk, axis=1) / (d_ji * d_jk)
        cosine = np.clip(cosine, -1.0, 1.0)
        theta = np.arccos(cosine)
        sine = np.sqrt(np.maximum(1.0 - cosine * cosine, 1.0e-12))
        grad_i_cos = (
            r_jk / (d_ji * d_jk)[:, None]
            - cosine[:, None] * r_ji / (d_ji * d_ji)[:, None]
        )
        grad_k_cos = (
            r_ji / (d_ji * d_jk)[:, None]
            - cosine[:, None] * r_jk / (d_jk * d_jk)[:, None]
        )
        scale = (
            float(angle["k"])
            * (theta - float(angle["theta0"]))
            / sine
        )[:, None]
        force_i = scale * grad_i_cos
        force_k = scale * grad_k_cos
        result[:, i] += force_i
        result[:, k] += force_k
        result[:, j] -= force_i + force_k

    net_prior_force = np.sum(result, axis=1)
    if float(np.max(np.abs(net_prior_force))) > 1.0e-8:
        raise RuntimeError("Harmonic prior forces violate force conservation")
    return result


def target_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "component_mae": float(np.mean(np.abs(values))),
        "component_rms": float(np.sqrt(np.mean(values * values))),
        "max_vector_norm": float(np.max(np.linalg.norm(values, axis=2))),
        "mean_net_force_norm": float(np.mean(np.linalg.norm(np.sum(values, axis=1), axis=1))),
    }


def write_dataset(
    path: Path, coordinates: np.ndarray, target_forces: np.ndarray, box_nm: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames, n_beads, _ = coordinates.shape
    if n_beads != len(BEAD_TYPES):
        raise ValueError(f"Expected {len(BEAD_TYPES)} beads, got {n_beads}")
    zero_torque = (0.0, 0.0, 0.0)
    with path.open("wb") as handle:
        handle.write(struct.pack("<i", n_frames))
        for frame_index in range(n_frames):
            handle.write(struct.pack("<ii3f", n_beads, n_beads, box_nm, box_nm, box_nm))
            for bead_index in range(n_beads):
                position = coordinates[frame_index, bead_index].astype(np.float32)
                force = target_forces[frame_index, bead_index].astype(np.float32)
                handle.write(struct.pack("<ii", bead_index, 1))
                handle.write(struct.pack("<3f", *position))
                handle.write(struct.pack("<3f", *force))
                handle.write(struct.pack("<3f", *zero_torque))
                handle.write(struct.pack("<i3f", int(BEAD_TYPES[bead_index]), *position))


def build_priors_document(
    bonds: list[dict], angles: list[dict], temperature: float, prior_mode: str
) -> dict:
    direct_pairs = [list(pair) for pair in BOND_TUPLES] if bonds else []
    direct_site_pairs = [[i, j, 0, 0] for i, j in BOND_TUPLES] if bonds else []
    one_three_pairs = [[0, 2], [1, 3], [2, 4]] if bonds else []
    return {
        "bonds": bonds,
        "morse_type_pairs": [],
        "wca": {"sigma": 0.0, "epsilon": 0.0, "overrides": {}},
        "angles": angles,
        "dihedrals": [],
        "wca_pairs": {
            f"{type_i}_{type_j}": {
                "type_i": type_i,
                "type_j": type_j,
                "sigma_nm": WCA_SIGMA_NM,
                "epsilon_kjmol": R_KJ_MOL_K * temperature,
                "cutoff_nm": WCA_CUTOFF_NM,
                "source": "cgnet_style_ood_excluded_volume",
            }
            for type_i, type_j in ((6, 6), (6, 7), (7, 7))
        },
        "wca_exclusions": {
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
        },
        "benchmark": {
            "system": "alanine_dipeptide_5bead",
            "source": "coarse-graining/cgnet examples/data",
            "bead_names": BEAD_NAMES,
            "bead_types": BEAD_TYPES.tolist(),
            "temperature_kelvin": temperature,
            "prior_mode": prior_mode,
            "fit_policy": "training_prefix_only",
            "wca_policy": (
                "active only for pairs separated by more than two bonds; "
                "cutoff lies below every such distance in the official 10000-frame subset"
            ),
        },
    }


def build_rigid_bodies_document() -> dict:
    """Return the two unique one-site templates required by ESPResSo."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument("--forces", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--priors-output", required=True, type=Path)
    parser.add_argument("--rb-info-output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--prior-mode", choices=("harmonic", "none"), default="harmonic")
    parser.add_argument("--validation-tail-frames", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--box-nm", type=float, default=4.0)
    args = parser.parse_args()

    raw_coordinates = np.load(args.coordinates, allow_pickle=False)
    raw_forces = np.load(args.forces, allow_pickle=False)
    expected_shape = (10000, 5, 3)
    if raw_coordinates.shape != expected_shape or raw_forces.shape != expected_shape:
        raise ValueError(
            f"Expected coordinate/force arrays with shape {expected_shape}; "
            f"got {raw_coordinates.shape} and {raw_forces.shape}"
        )
    if not np.isfinite(raw_coordinates).all() or not np.isfinite(raw_forces).all():
        raise ValueError("Input arrays contain non-finite values")
    if not 0 < args.validation_tail_frames < raw_coordinates.shape[0]:
        raise ValueError("validation-tail-frames must leave non-empty train and validation sets")
    if args.temperature <= 0.0 or args.box_nm <= 0.0:
        raise ValueError("temperature and box-nm must be positive")

    coordinates = raw_coordinates.astype(np.float64) * ANGSTROM_TO_NM
    forces = raw_forces.astype(np.float64) * KCAL_PER_MOL_ANGSTROM_TO_KJ_PER_MOL_NM
    # Remove only global translation; internal geometry and forces are unchanged.
    coordinates -= np.mean(coordinates, axis=1, keepdims=True)
    coordinates += args.box_nm / 2.0
    if np.min(coordinates) <= 0.0 or np.max(coordinates) >= args.box_nm:
        raise ValueError("Centered Ala2 coordinates do not fit inside the selected box")
    minimum_nonbonded_nm = minimum_nonbonded_distance(coordinates)
    if minimum_nonbonded_nm <= WCA_CUTOFF_NM:
        raise ValueError(
            f"The OOD WCA cutoff {WCA_CUTOFF_NM} nm is not inactive on the dataset; "
            f"minimum nonbonded distance is {minimum_nonbonded_nm} nm"
        )

    train_frames = coordinates.shape[0] - args.validation_tail_frames
    if args.prior_mode == "harmonic":
        bonds, angles = fit_harmonic_priors(coordinates[:train_frames], args.temperature)
        prior_forces = harmonic_prior_forces(coordinates, bonds, angles)
    else:
        bonds, angles = [], []
        prior_forces = np.zeros_like(forces)
    targets = forces - prior_forces

    write_dataset(args.output, coordinates, targets, args.box_nm)
    priors_document = build_priors_document(bonds, angles, args.temperature, args.prior_mode)
    args.priors_output.parent.mkdir(parents=True, exist_ok=True)
    args.priors_output.write_text(json.dumps(priors_document, indent=2, sort_keys=True) + "\n")

    if args.rb_info_output is not None:
        args.rb_info_output.parent.mkdir(parents=True, exist_ok=True)
        args.rb_info_output.write_text(
            json.dumps(build_rigid_bodies_document(), indent=2, sort_keys=True) + "\n"
        )

    if args.reference_output is not None:
        args.reference_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.reference_output,
            coordinates_nm=coordinates.astype(np.float32),
            forces_kj_mol_nm=forces.astype(np.float32),
            prior_forces_kj_mol_nm=prior_forces.astype(np.float32),
            residual_forces_kj_mol_nm=targets.astype(np.float32),
            bead_types=BEAD_TYPES,
        )

    report = {
        "schema_version": 1,
        "source": {
            "repository": "https://github.com/coarse-graining/cgnet",
            "coordinates_sha256": sha256_file(args.coordinates),
            "forces_sha256": sha256_file(args.forces),
            "input_shape": list(expected_shape),
            "input_coordinate_units": "angstrom",
            "input_force_units": "kcal mol^-1 angstrom^-1",
        },
        "output": {
            "dataset": str(args.output.resolve()),
            "dataset_sha256": sha256_file(args.output),
            "coordinate_units": "nm",
            "force_units": "kJ mol^-1 nm^-1",
            "box_nm": args.box_nm,
            "bead_names": BEAD_NAMES,
            "bead_types": BEAD_TYPES.tolist(),
        },
        "split": {
            "mode": "tail",
            "train_frames": train_frames,
            "validation_frames": args.validation_tail_frames,
        },
        "prior_mode": args.prior_mode,
        "prior_fit": {
            "temperature_kelvin": args.temperature,
            "bonds": bonds,
            "angles": angles,
            "fit_frames": train_frames,
            "ood_wca_cutoff_nm": WCA_CUTOFF_NM,
            "minimum_nonbonded_distance_nm": minimum_nonbonded_nm,
            "ood_wca_zero_on_all_frames": True,
        },
        "statistics": {
            "raw_train": target_statistics(forces[:train_frames]),
            "raw_validation": target_statistics(forces[train_frames:]),
            "prior_train": target_statistics(prior_forces[:train_frames]),
            "prior_validation": target_statistics(prior_forces[train_frames:]),
            "target_train": target_statistics(targets[:train_frames]),
            "target_validation": target_statistics(targets[train_frames:]),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"[PASS] Wrote {coordinates.shape[0]} Ala2 frames with prior_mode={args.prior_mode}; "
        f"train={train_frames}, validation={args.validation_tail_frames}."
    )


if __name__ == "__main__":
    main()
