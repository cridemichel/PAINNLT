"""Geometry extraction utilities for bonded DBI/IBI.

The functions in this module are deliberately independent of ESPResSo.  They
read either the framework binary dataset (the atomistic-reference target) or a
structured NPZ trajectory written by ``simulation/run_cg_md.py`` and evaluate
the same site-addressable bond/angle/dihedral coordinates in both cases.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Mapping

import numpy as np

from prior_kernels import espresso_dihedral_geometry


IBI_TYPES = {"ibi", "dbi"}


def mic_vector(pos1, pos2, box_dim):
    """Minimum-image vector from *pos1* to *pos2*."""
    delta = np.asarray(pos2, dtype=float) - np.asarray(pos1, dtype=float)
    box = np.asarray(box_dim, dtype=float)
    if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError(f"Invalid periodic box: {box_dim!r}")
    return delta - box * np.round(delta / box)


def requested_mode(prior: Mapping[str, object]) -> str | None:
    """Return ``ibi``/``dbi`` for a seed or converted tabulated prior."""
    direct = str(prior.get("type", "")).lower()
    if direct in IBI_TYPES:
        return direct
    preserved = str(prior.get("ibi_mode", "")).lower()
    return preserved if preserved in IBI_TYPES else None


def resolve_position(centers, sites, mol: int, site: int):
    if mol < 0 or mol >= len(centers):
        raise IndexError(f"Invalid molecule index {mol}")
    if site == -1:
        return centers[mol]
    if site < 0 or site >= len(sites[mol]):
        raise IndexError(f"Invalid site {site} for molecule {mol}")
    return sites[mol][site]


def _append_geometry_values(priors, centers, sites, box, bond_values, angle_values, dihedral_values):
    for idx, prior in enumerate(priors.get("bonds", [])):
        if requested_mode(prior) is None:
            continue
        i, j = int(prior["mol_i"]), int(prior["mol_j"])
        pi = resolve_position(centers, sites, i, int(prior.get("site_i", -1)))
        pj = resolve_position(centers, sites, j, int(prior.get("site_j", -1)))
        bond_values[idx].append(float(np.linalg.norm(mic_vector(pi, pj, box))))

    for idx, prior in enumerate(priors.get("angles", [])):
        if requested_mode(prior) is None:
            continue
        i, j, k = (int(prior[key]) for key in ("mol_i", "mol_j", "mol_k"))
        pi = resolve_position(centers, sites, i, int(prior.get("site_i", -1)))
        pj = resolve_position(centers, sites, j, int(prior.get("site_j", -1)))
        pk = resolve_position(centers, sites, k, int(prior.get("site_k", -1)))
        a = mic_vector(pj, pi, box)
        b = mic_vector(pj, pk, box)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na <= 1.0e-12 or nb <= 1.0e-12:
            continue
        angle_values[idx].append(
            float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))
        )

    for idx, prior in enumerate(priors.get("dihedrals", [])):
        if requested_mode(prior) is None:
            continue
        mols = [int(prior[key]) for key in ("mol_i", "mol_j", "mol_k", "mol_l")]
        site_ids = [int(prior.get(key, -1)) for key in ("site_i", "site_j", "site_k", "site_l")]
        coords = [resolve_position(centers, sites, mol, site) for mol, site in zip(mols, site_ids)]
        geom = espresso_dihedral_geometry(*coords, box)
        if geom is not None:
            dihedral_values[idx].append(float(geom[0]))


def _empty_value_maps(priors):
    return (
        {i: [] for i in range(len(priors.get("bonds", [])))},
        {i: [] for i in range(len(priors.get("angles", [])))},
        {i: [] for i in range(len(priors.get("dihedrals", [])))},
    )


def _read_exact(handle, nbytes: int, description: str) -> bytes:
    data = handle.read(nbytes)
    if len(data) != nbytes:
        raise ValueError(f"Truncated CG dataset while reading {description}")
    return data


def read_target_distributions(dataset: str | Path, priors: Mapping[str, object]):
    """Read requested bonded-coordinate distributions from a framework dataset."""
    dataset = Path(dataset)
    bond_values, angle_values, dihedral_values = _empty_value_maps(priors)

    with dataset.open("rb") as handle:
        n_frames = struct.unpack("i", _read_exact(handle, 4, "frame count"))[0]
        if n_frames <= 0:
            raise ValueError(f"Dataset has no frames: {dataset}")
        expected_molecules = None
        expected_site_counts = None

        for frame in range(n_frames):
            n_molecules = struct.unpack("i", _read_exact(handle, 4, f"frame {frame} molecule count"))[0]
            n_sites = struct.unpack("i", _read_exact(handle, 4, f"frame {frame} site count"))[0]
            if n_molecules <= 0 or n_sites < 0:
                raise ValueError(f"Invalid dataset counts in frame {frame}: molecules={n_molecules}, sites={n_sites}")
            if expected_molecules is None:
                expected_molecules = n_molecules
            elif n_molecules != expected_molecules:
                raise ValueError(
                    f"Molecule count changes at frame {frame}: {n_molecules} != {expected_molecules}"
                )
            box = np.asarray(struct.unpack("3f", _read_exact(handle, 12, f"frame {frame} box")), dtype=float)
            if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0.0):
                raise ValueError(f"Invalid periodic box in frame {frame}: {box}")
            centers = []
            sites = []
            frame_site_counts = []

            for mol in range(n_molecules):
                mol_id = struct.unpack("i", _read_exact(handle, 4, f"frame {frame} molecule {mol} id"))[0]
                if mol_id != mol:
                    raise ValueError(
                        f"Dataset molecule IDs must be contiguous/in order: frame {frame} has id {mol_id} at slot {mol}"
                    )
                n_mol_sites = struct.unpack("i", _read_exact(handle, 4, f"frame {frame} molecule {mol} site count"))[0]
                if n_mol_sites < 0:
                    raise ValueError(f"Negative site count for molecule {mol} in frame {frame}")
                frame_site_counts.append(n_mol_sites)
                center = np.asarray(struct.unpack("3f", _read_exact(handle, 12, "COM position")), dtype=float)
                _read_exact(handle, 12, "COM force")
                _read_exact(handle, 12, "COM torque")
                mol_sites = []
                for site in range(n_mol_sites):
                    _read_exact(handle, 4, f"molecule {mol} site {site} type")
                    mol_sites.append(
                        np.asarray(struct.unpack("3f", _read_exact(handle, 12, "site position")), dtype=float)
                    )
                centers.append(center)
                sites.append(mol_sites)

            if sum(frame_site_counts) != n_sites:
                raise ValueError(
                    f"Dataset site-count header mismatch in frame {frame}: header={n_sites}, parsed={sum(frame_site_counts)}"
                )
            if expected_site_counts is None:
                expected_site_counts = tuple(frame_site_counts)
            elif tuple(frame_site_counts) != expected_site_counts:
                raise ValueError(
                    f"Per-molecule site counts change at frame {frame}: {tuple(frame_site_counts)} != {expected_site_counts}"
                )

            _append_geometry_values(
                priors, centers, sites, box,
                bond_values, angle_values, dihedral_values,
            )

        # Extra bytes usually indicate a writer/reader schema mismatch.  Fail
        # closed instead of silently using a partially misparsed trajectory.
        if handle.read(1):
            raise ValueError(f"Dataset contains trailing bytes after {n_frames} frames: {dataset}")

    return bond_values, angle_values, dihedral_values


def read_sampled_distributions(sample_npz: str | Path, priors: Mapping[str, object]):
    """Read bonded distributions from a structured runtime sampling NPZ."""
    sample_npz = Path(sample_npz)
    bond_values, angle_values, dihedral_values = _empty_value_maps(priors)

    with np.load(sample_npz, allow_pickle=False) as sample:
        required = {
            "schema_version", "complete", "com", "sites",
            "site_molecule", "site_index", "box", "steps",
        }
        missing = sorted(required - set(sample.files))
        if missing:
            raise ValueError(f"Sampling trajectory is missing arrays {missing}: {sample_npz}")
        schema_version = int(np.asarray(sample["schema_version"]).reshape(()))
        complete = int(np.asarray(sample["complete"]).reshape(()))
        if schema_version != 1:
            raise ValueError(
                f"Unsupported sampling trajectory schema {schema_version}: {sample_npz}"
            )
        if complete != 1:
            raise ValueError(f"Sampling trajectory is not marked complete: {sample_npz}")

        com = np.asarray(sample["com"], dtype=float)
        flat_sites = np.asarray(sample["sites"], dtype=float)
        site_molecule = np.asarray(sample["site_molecule"], dtype=int)
        site_index = np.asarray(sample["site_index"], dtype=int)
        box = np.asarray(sample["box"], dtype=float)
        steps = np.asarray(sample["steps"], dtype=np.int64)

        if com.ndim != 3 or com.shape[2] != 3:
            raise ValueError(f"Invalid COM trajectory shape {com.shape}: {sample_npz}")
        if flat_sites.ndim != 3 or flat_sites.shape[0] != com.shape[0] or flat_sites.shape[2] != 3:
            raise ValueError(f"Invalid site trajectory shape {flat_sites.shape}: {sample_npz}")
        if site_molecule.shape != (flat_sites.shape[1],) or site_index.shape != (flat_sites.shape[1],):
            raise ValueError("Sampling site metadata does not match the site trajectory")
        if steps.shape != (com.shape[0],):
            raise ValueError("Sampling step array does not match the number of trajectory frames")
        if com.shape[0] == 0:
            raise ValueError(f"Sampling trajectory contains no frames: {sample_npz}")
        if np.any(np.diff(steps) <= 0):
            raise ValueError("Sampling steps must be strictly increasing")
        if box.shape == (3,):
            boxes = np.repeat(box[None, :], com.shape[0], axis=0)
        elif box.shape == (com.shape[0], 3):
            boxes = box
        else:
            raise ValueError(f"Invalid sampling box shape {box.shape}: {sample_npz}")

        lookup = {}
        for flat_idx, (mol, site) in enumerate(zip(site_molecule, site_index)):
            key = (int(mol), int(site))
            if key in lookup:
                raise ValueError(f"Duplicate sampled site metadata for {key}")
            lookup[key] = flat_idx

        expected_molecules = set(range(com.shape[1]))
        seen_molecules = set(int(x) for x in site_molecule)
        if not seen_molecules.issubset(expected_molecules):
            raise ValueError("Sampling site metadata references an out-of-range molecule")

        # Convert flattened physical-site storage to the molecule-major shape
        # used by the shared coordinate evaluator.
        site_counts = [0] * com.shape[1]
        for mol, site in lookup:
            site_counts[mol] = max(site_counts[mol], site + 1)
        for mol in range(com.shape[1]):
            for site in range(site_counts[mol]):
                if (mol, site) not in lookup:
                    raise ValueError(f"Sampling trajectory is missing physical site {(mol, site)}")

        for frame in range(com.shape[0]):
            sites = [
                [flat_sites[frame, lookup[(mol, site)]] for site in range(site_counts[mol])]
                for mol in range(com.shape[1])
            ]
            _append_geometry_values(
                priors, com[frame], sites, boxes[frame],
                bond_values, angle_values, dihedral_values,
            )

    return bond_values, angle_values, dihedral_values


def pool_requested(priors: Mapping[str, object], values, key: str):
    """Pool entries sharing a ``name`` while enforcing one IBI/DBI mode."""
    groups = {}
    for idx, prior in enumerate(priors.get(key, [])):
        mode = requested_mode(prior)
        if mode is None:
            continue
        name = str(prior.get("name", f"idx_{idx}"))
        group = groups.setdefault(name, {"mode": mode, "values": [], "indices": []})
        if group["mode"] != mode:
            raise ValueError(f"Group {name!r} mixes IBI and DBI entries")
        group["values"].extend(values[idx])
        group["indices"].append(idx)
    return groups
