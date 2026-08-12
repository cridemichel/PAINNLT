#!/usr/bin/env python3
"""Build a diagnostic TEL22 dataset with DNA self targets and isolated copies.

This is an *ablation dataset*, not a production replacement for force matching.
The target on each residue is taken from the single-copy GROMACS reruns created
by 03l_dna_self_vs_intercopy.sh.  The ten TEL22 copies from each sampled frame
are then translated onto a sparse periodic lattice so that no site belonging to
different copies lies inside the PaiNN cutoff.  Internal copy geometry,
orientation, force vectors and torque vectors are left unchanged.

Why this construction is useful:
  * target: F_self / tau_self, so water, ions and the other nine TEL22 copies are
    absent from the atomistic target;
  * input: all ten copies remain in one training frame, preserving a frame-level
    train/validation split, but inter-copy PaiNN edges are geometrically removed;
  * the trainer itself is unchanged.

The resulting binary uses the same on-disk format as tel22_dataset.bin.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import analyze_conditional_noise as cn
import analyze_dna_self_vs_intercopy as dsi
import analyze_force_source_decomposition as fs


def read_selected_frames(dataset: Path, raw_indices: Sequence[int]) -> List[cn.Frame]:
    wanted = {int(frame_idx): out_idx for out_idx, frame_idx in enumerate(raw_indices)}
    if len(wanted) != len(raw_indices):
        raise ValueError("raw frame indices are not unique")
    selected: List[cn.Frame | None] = [None] * len(raw_indices)
    with dataset.open("rb") as fh:
        nframes = cn.I32.unpack(cn.read_exact(fh, cn.I32.size))[0]
        if nframes <= 0:
            raise ValueError("dataset contains no frames")
        if wanted and (min(wanted) < 0 or max(wanted) >= nframes):
            raise ValueError(
                f"requested raw dataset frame outside [0,{nframes - 1}]: "
                f"min={min(wanted)}, max={max(wanted)}"
            )
        for frame_idx in range(nframes):
            frame = cn.read_frame(fh)
            out_idx = wanted.get(frame_idx)
            if out_idx is not None:
                selected[out_idx] = frame
        if fh.read(1):
            raise ValueError("unexpected trailing bytes after dataset")
    missing = [raw_indices[i] for i, frame in enumerate(selected) if frame is None]
    if missing:
        raise RuntimeError(f"failed to read requested dataset frames: {missing[:8]}")
    return [frame for frame in selected if frame is not None]


def unwrap_copy(block: Sequence[cn.Molecule], box: np.ndarray):
    """Return copy-local molecule centers/sites in one PBC-consistent image."""
    raw_centers = np.stack([m.center for m in block], axis=0)
    centers = np.empty_like(raw_centers, dtype=np.float64)
    centers[0] = raw_centers[0]
    for i in range(1, len(block)):
        centers[i] = centers[i - 1] + cn.mic(raw_centers[i] - raw_centers[i - 1], box)

    site_blocks: List[np.ndarray] = []
    all_sites: List[np.ndarray] = []
    for i, mol in enumerate(block):
        sites = []
        for _site_type, site_pos in mol.sites:
            sites.append(centers[i] + cn.mic(site_pos - mol.center, box))
        arr = np.asarray(sites, dtype=np.float64)
        site_blocks.append(arr)
        all_sites.extend(arr)

    xyz = np.asarray(all_sites, dtype=np.float64)
    origin = xyz.mean(axis=0)
    centers_local = centers - origin
    sites_local = [sites - origin for sites in site_blocks]
    radius = float(np.max(np.linalg.norm(xyz - origin, axis=1)))
    return centers_local, sites_local, radius


def min_intercopy_distance(copy_sites: Sequence[np.ndarray], box: np.ndarray) -> float:
    best = math.inf
    for i in range(len(copy_sites)):
        a = copy_sites[i]
        for j in range(i + 1, len(copy_sites)):
            b = copy_sites[j]
            delta = a[:, None, :] - b[None, :, :]
            for axis in range(3):
                length = float(box[axis])
                delta[..., axis] -= length * np.rint(delta[..., axis] / length)
            d2 = np.sum(delta * delta, axis=2)
            best = min(best, float(np.sqrt(np.min(d2))))
    return best


def write_dataset(path: Path, frames: Sequence[cn.Frame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(cn.I32.pack(len(frames)))
        for frame in frames:
            nsites_total = sum(m.nsites for m in frame.molecules)
            fh.write(cn.I32.pack(len(frame.molecules)))
            fh.write(cn.I32.pack(nsites_total))
            fh.write(cn.F32_3.pack(*np.asarray(frame.box, dtype=np.float32)))
            for expected_id, mol in enumerate(frame.molecules):
                if mol.mol_id != expected_id:
                    raise ValueError(
                        f"output molecule ids must be sequential: expected {expected_id}, got {mol.mol_id}"
                    )
                fh.write(cn.I32.pack(int(mol.mol_id)))
                fh.write(cn.I32.pack(mol.nsites))
                fh.write(cn.F32_3.pack(*np.asarray(mol.center, dtype=np.float32)))
                fh.write(cn.F32_3.pack(*np.asarray(mol.force, dtype=np.float32)))
                fh.write(cn.F32_3.pack(*np.asarray(mol.torque, dtype=np.float32)))
                for site_type, pos in mol.sites:
                    fh.write(cn.SITE.pack(int(site_type), *np.asarray(pos, dtype=np.float32)))



def _mic_vector(pos1: np.ndarray, pos2: np.ndarray, box: np.ndarray) -> np.ndarray:
    dvec = np.asarray(pos2, dtype=np.float64) - np.asarray(pos1, dtype=np.float64)
    return dvec - box * np.round(dvec / box)


def _resolve_site_position(frame: cn.Frame, mol_idx: int, site_idx: int) -> np.ndarray:
    mol = frame.molecules[mol_idx]
    if site_idx == -1:
        return np.asarray(mol.center, dtype=np.float64)
    if site_idx < 0 or site_idx >= len(mol.sites):
        raise IndexError(
            f"invalid site index {site_idx} for molecule {mol_idx}; "
            f"available sites={len(mol.sites)}"
        )
    return np.asarray(mol.sites[site_idx][1], dtype=np.float64)


def _dihedral_energy(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    pos_k: np.ndarray,
    pos_l: np.ndarray,
    box: np.ndarray,
    K: float,
    n: int,
    phi0: float,
) -> float:
    b1 = _mic_vector(pos_i, pos_j, box)
    b2 = _mic_vector(pos_j, pos_k, box)
    b3 = _mic_vector(pos_k, pos_l, box)
    m1 = np.cross(b1, b2)
    m2 = np.cross(b2, b3)
    m1_sq = float(np.dot(m1, m1))
    m2_sq = float(np.dot(m2, m2))
    if m1_sq < 1.0e-12 or m2_sq < 1.0e-12:
        phi = 0.0
    else:
        b2_norm = float(np.linalg.norm(b2))
        cos_phi = float(np.clip(np.dot(m1, m2) / math.sqrt(m1_sq * m2_sq), -1.0, 1.0))
        sin_phi = float(np.dot(b2, np.cross(m1, m2)) / (b2_norm * math.sqrt(m1_sq * m2_sq)))
        phi = math.atan2(sin_phi, cos_phi)
    return float(K) * (1.0 - math.cos(int(n) * phi - float(phi0)))


def _dihedral_forces(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    pos_k: np.ndarray,
    pos_l: np.ndarray,
    box: np.ndarray,
    K: float,
    n: int,
    phi0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray([pos_i, pos_j, pos_k, pos_l], dtype=np.float64)
    forces = np.zeros_like(positions)
    eps = 1.0e-6
    for atom in range(4):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += eps
            minus[atom, axis] -= eps
            ep = _dihedral_energy(*plus, box, K, n, phi0)
            em = _dihedral_energy(*minus, box, K, n, phi0)
            forces[atom, axis] = -(ep - em) / (2.0 * eps)
    forces -= forces.mean(axis=0, keepdims=True)
    return tuple(forces)  # type: ignore[return-value]


def _target_scale(frames: Sequence[cn.Frame]) -> Dict[str, float]:
    force_sum2 = 0.0
    force_count = 0
    torque_sum2 = 0.0
    torque_count = 0
    for frame in frames:
        for mol in frame.molecules:
            f = np.asarray(mol.force, dtype=np.float64)
            force_sum2 += float(np.dot(f, f))
            force_count += 3
            if mol.nsites > 1:
                t = np.asarray(mol.torque, dtype=np.float64)
                torque_sum2 += float(np.dot(t, t))
                torque_count += 3
    return {
        "force_component_rms_kj_mol_nm": float(math.sqrt(force_sum2 / force_count)),
        "torque_component_rms_kj_mol_multisite_only": (
            float(math.sqrt(torque_sum2 / torque_count)) if torque_count else math.nan
        ),
    }


def subtract_cg_priors(
    frames: Sequence[cn.Frame],
    priors: Dict,
    period: int,
    copies: int,
) -> Tuple[List[cn.Frame], Dict[str, object]]:
    """Return F_self-F_prior,self on already isolated copies.

    The implementation mirrors preprocessing/build_cg_dataset.py conventions.
    Cross-copy WCA candidates are omitted explicitly; this is equivalent to the
    isolated geometry because every cross-copy site distance is beyond the WCA
    cutoffs, and it reduces the cost of the 1001-frame diagnostic substantially.
    """
    if not frames:
        raise ValueError("cannot subtract priors from an empty frame list")
    nmol = len(frames[0].molecules)
    if nmol != period * copies:
        raise ValueError(f"molecule count {nmol} != period*copies {period * copies}")

    direct = {tuple(sorted(map(int, pair))) for pair in priors.get("wca_exclusions", {}).get("direct_pairs", [])}
    one_three = {tuple(sorted(map(int, pair))) for pair in priors.get("wca_exclusions", {}).get("one_three_pairs", [])}
    excluded = direct | one_three

    flat_mol = []
    flat_type = []
    for mi, mol in enumerate(frames[0].molecules):
        for site_type, _pos in mol.sites:
            flat_mol.append(mi)
            flat_type.append(int(site_type))
    flat_mol = np.asarray(flat_mol, dtype=np.int32)
    flat_type = np.asarray(flat_type, dtype=np.int32)
    pair_i, pair_j = np.triu_indices(len(flat_type), k=1)
    mol_i = flat_mol[pair_i]
    mol_j = flat_mol[pair_j]
    same_copy = (mol_i // period) == (mol_j // period)
    different_molecule = mol_i != mol_j
    allowed_topology = np.fromiter(
        (tuple(sorted((int(i), int(j)))) not in excluded for i, j in zip(mol_i, mol_j)),
        dtype=bool,
        count=len(mol_i),
    )
    keep = same_copy & different_molecule & allowed_topology
    pair_i = pair_i[keep]
    pair_j = pair_j[keep]
    mol_i = flat_mol[pair_i]
    mol_j = flat_mol[pair_j]

    wca_pairs = priors.get("wca_pairs", {})
    if not isinstance(wca_pairs, dict) or not wca_pairs:
        raise ValueError("cg_priors.json contains no pair-specific wca_pairs")
    max_type = max(
        int(np.max(flat_type)),
        max(max(int(v["type_i"]), int(v["type_j"])) for v in wca_pairs.values()),
    )
    ntypes = max_type + 1
    sigma = np.zeros((ntypes, ntypes), dtype=np.float64)
    epsilon = np.zeros_like(sigma)
    cutoff_sq = np.zeros_like(sigma)
    for info in wca_pairs.values():
        ti, tj = int(info["type_i"]), int(info["type_j"])
        sigma[ti, tj] = sigma[tj, ti] = float(info["sigma_nm"])
        epsilon[ti, tj] = epsilon[tj, ti] = float(info["epsilon_kjmol"])
        cutoff_sq[ti, tj] = cutoff_sq[tj, ti] = float(info["cutoff_nm"]) ** 2
    pair_sigma = sigma[flat_type[pair_i], flat_type[pair_j]]
    pair_epsilon = epsilon[flat_type[pair_i], flat_type[pair_j]]
    pair_cutoff_sq = cutoff_sq[flat_type[pair_i], flat_type[pair_j]]
    defined = pair_cutoff_sq > 0.0
    pair_i, pair_j = pair_i[defined], pair_j[defined]
    mol_i, mol_j = mol_i[defined], mol_j[defined]
    pair_sigma, pair_epsilon, pair_cutoff_sq = (
        pair_sigma[defined], pair_epsilon[defined], pair_cutoff_sq[defined]
    )

    prior_force_sum2 = 0.0
    prior_force_count = 0
    prior_torque_sum2 = 0.0
    prior_torque_count = 0
    active_wca_pairs = 0
    output: List[cn.Frame] = []

    for frame_idx, frame in enumerate(frames):
        if len(frame.molecules) != nmol:
            raise ValueError(f"frame {frame_idx}: molecule count changed")
        box = np.asarray(frame.box, dtype=np.float64)
        centers = np.stack([np.asarray(m.center, dtype=np.float64) for m in frame.molecules])
        prior_f = np.zeros((nmol, 3), dtype=np.float64)
        prior_t = np.zeros((nmol, 3), dtype=np.float64)

        flat_pos = np.asarray(
            [np.asarray(pos, dtype=np.float64) for m in frame.molecules for _st, pos in m.sites],
            dtype=np.float64,
        )
        if flat_pos.shape[0] != len(flat_type):
            raise ValueError(f"frame {frame_idx}: site count changed")
        if pair_i.size:
            diff = flat_pos[pair_i] - flat_pos[pair_j]
            diff -= box * np.round(diff / box)
            d2 = np.einsum("ij,ij->i", diff, diff)
            active = np.flatnonzero((d2 > 1.0e-6) & (d2 < pair_cutoff_sq))
            if active.size:
                active_wca_pairs += int(active.size)
                ii, jj = pair_i[active], pair_j[active]
                mi, mj = mol_i[active], mol_j[active]
                r = np.sqrt(d2[active])
                rhat = diff[active] / r[:, None]
                sr6 = (pair_sigma[active] / r) ** 6
                fscalar = 24.0 * pair_epsilon[active] * (2.0 * sr6 * sr6 - sr6) / r
                fvec = fscalar[:, None] * rhat
                np.add.at(prior_f, mi, fvec)
                np.add.at(prior_f, mj, -fvec)
                lever_i = flat_pos[ii] - centers[mi]
                lever_i -= box * np.round(lever_i / box)
                lever_j = flat_pos[jj] - centers[mj]
                lever_j -= box * np.round(lever_j / box)
                np.add.at(prior_t, mi, np.cross(lever_i, fvec))
                np.add.at(prior_t, mj, -np.cross(lever_j, fvec))

        for b in priors.get("bonds", []):
            i, j = int(b["mol_i"]), int(b["mol_j"])
            if i >= nmol or j >= nmol:
                continue
            si, sj = int(b.get("site_i", -1)), int(b.get("site_j", -1))
            pi = _resolve_site_position(frame, i, si)
            pj = _resolve_site_position(frame, j, sj)
            rvec = _mic_vector(pi, pj, box)
            r = float(np.linalg.norm(rvec))
            if r < 1.0e-6:
                continue
            rhat = rvec / r
            btype = str(b.get("type", "harmonic")).lower()
            if btype == "harmonic":
                fscalar = -float(b["k"]) * (r - float(b["r0"]))
            elif btype == "fene":
                diff_r = r - float(b["r0"])
                rmax = float(b["r_max"])
                if abs(diff_r) >= rmax:
                    raise ValueError(f"FENE bond {i}-{j} outside domain in frame {frame_idx}")
                fscalar = -float(b["k"]) * diff_r / (1.0 - (diff_r / rmax) ** 2)
            elif btype == "morse":
                diff_r = r - float(b["r0"])
                exp_term = math.exp(-float(b["a"]) * diff_r)
                fscalar = -2.0 * float(b["a"]) * float(b["D"]) * (1.0 - exp_term) * exp_term
            else:
                continue
            fvec = -fscalar * rhat
            prior_f[i] += fvec
            prior_f[j] -= fvec
            if si != -1:
                prior_t[i] += np.cross(_mic_vector(centers[i], pi, box), fvec)
            if sj != -1:
                prior_t[j] -= np.cross(_mic_vector(centers[j], pj, box), fvec)

        for a in priors.get("angles", []):
            atype = str(a.get("type", "harmonic")).lower()
            if atype != "harmonic":
                continue
            i, j, k = int(a["mol_i"]), int(a["mol_j"]), int(a["mol_k"])
            if max(i, j, k) >= nmol:
                continue
            si, sj, sk = int(a.get("site_i", -1)), int(a.get("site_j", -1)), int(a.get("site_k", -1))
            pi, pj, pk = (_resolve_site_position(frame, i, si), _resolve_site_position(frame, j, sj), _resolve_site_position(frame, k, sk))
            rji = _mic_vector(pj, pi, box)
            rjk = _mic_vector(pj, pk, box)
            dji, djk = float(np.linalg.norm(rji)), float(np.linalg.norm(rjk))
            if dji < 1.0e-6 or djk < 1.0e-6:
                continue
            c = float(np.clip(np.dot(rji, rjk) / (dji * djk), -1.0, 1.0))
            st = math.sqrt(max(0.0, 1.0 - c * c))
            if st < 1.0e-6:
                continue
            theta = math.acos(c)
            grad_i = rjk / (dji * djk) - c * rji / (dji * dji)
            grad_k = rji / (dji * djk) - c * rjk / (djk * djk)
            scalar = float(a["k"]) * (theta - float(a["theta0"])) / st
            fi = scalar * grad_i
            fk = scalar * grad_k
            fj = -(fi + fk)
            prior_f[i] += fi
            prior_f[j] += fj
            prior_f[k] += fk
            if si != -1:
                prior_t[i] += np.cross(_mic_vector(centers[i], pi, box), fi)
            if sj != -1:
                prior_t[j] += np.cross(_mic_vector(centers[j], pj, box), fj)
            if sk != -1:
                prior_t[k] += np.cross(_mic_vector(centers[k], pk, box), fk)

        for d in priors.get("dihedrals", []):
            if str(d.get("type", "cosine")).lower() != "cosine":
                continue
            i, j, k, l = int(d["mol_i"]), int(d["mol_j"]), int(d["mol_k"]), int(d["mol_l"])
            if max(i, j, k, l) >= nmol:
                continue
            sis = [int(d.get(name, -1)) for name in ("site_i", "site_j", "site_k", "site_l")]
            pos = [_resolve_site_position(frame, mi, si) for mi, si in zip((i, j, k, l), sis)]
            ff = _dihedral_forces(*pos, box, float(d["k"]), int(d.get("n", 1)), float(d["phi0"]))
            for mi, si, pp, fvec in zip((i, j, k, l), sis, pos, ff):
                prior_f[mi] += fvec
                if si != -1:
                    prior_t[mi] += np.cross(_mic_vector(centers[mi], pp, box), fvec)

        molecules: List[cn.Molecule] = []
        for mi, mol in enumerate(frame.molecules):
            rf = np.asarray(mol.force, dtype=np.float64) - prior_f[mi]
            rt = np.asarray(mol.torque, dtype=np.float64) - prior_t[mi]
            prior_force_sum2 += float(np.dot(prior_f[mi], prior_f[mi]))
            prior_force_count += 3
            if mol.nsites > 1:
                prior_torque_sum2 += float(np.dot(prior_t[mi], prior_t[mi]))
                prior_torque_count += 3
            molecules.append(
                cn.Molecule(
                    mol_id=mol.mol_id,
                    center=np.asarray(mol.center, dtype=np.float64),
                    force=rf,
                    torque=rt,
                    sites=[(st, np.asarray(pos, dtype=np.float64)) for st, pos in mol.sites],
                )
            )
        output.append(cn.Frame(box=box.copy(), molecules=molecules))

    prior_scale = {
        "force_component_rms_kj_mol_nm": float(math.sqrt(prior_force_sum2 / prior_force_count)),
        "torque_component_rms_kj_mol_multisite_only": (
            float(math.sqrt(prior_torque_sum2 / prior_torque_count)) if prior_torque_count else math.nan
        ),
    }
    return output, {
        "definition": "production CG priors evaluated on the isolated-copy geometry and subtracted from DNA-self generalized targets",
        "wca_scope": "same-copy only; cross-copy WCA omitted because copies are isolated beyond every WCA cutoff",
        "active_wca_site_pairs_across_frames": active_wca_pairs,
        "prior_scale": prior_scale,
        "residual_target_scale": _target_scale(output),
    }


def evenly_spaced_indices(n_available: int, n_select: int) -> np.ndarray:
    if n_select <= 0 or n_select > n_available:
        raise ValueError(f"sample-count must be in [1,{n_available}], got {n_select}")
    if n_select == n_available:
        return np.arange(n_available, dtype=np.int64)
    idx = np.rint(np.linspace(0, n_available - 1, n_select)).astype(np.int64)
    if len(np.unique(idx)) != n_select:
        raise RuntimeError("evenly spaced frame selection produced duplicate indices")
    return idx


def stratified_tail_order(n_selected: int, validation_stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if validation_stride < 2:
        raise ValueError("validation-stride must be >= 2")
    val = np.arange(validation_stride - 1, n_selected, validation_stride, dtype=np.int64)
    if len(val) == 0:
        val = np.asarray([n_selected - 1], dtype=np.int64)
    mask = np.ones(n_selected, dtype=bool)
    mask[val] = False
    train = np.flatnonzero(mask).astype(np.int64)
    if len(train) == 0:
        raise ValueError("validation split left no training frames")
    order = np.concatenate([train, val])
    return train, val, order


def build_frames(
    source_frames: Sequence[cn.Frame],
    self_forces: np.ndarray,
    self_torques: np.ndarray,
    period: int,
    copies: int,
    cutoff: float,
    margin: float,
    layout_cols: int,
):
    if self_forces.shape != self_torques.shape:
        raise ValueError("force/torque shapes differ")
    expected_shape = (len(source_frames), period * copies, 3)
    if self_forces.shape != expected_shape:
        raise ValueError(f"self target shape {self_forces.shape}, expected {expected_shape}")

    prepared = []
    max_radius = 0.0
    signature = None
    for frame_idx, frame in enumerate(source_frames):
        if len(frame.molecules) != period * copies:
            raise ValueError(
                f"frame {frame_idx}: molecule count {len(frame.molecules)} != {period * copies}"
            )
        current_signature = [cn.molecule_signature(m) for m in frame.molecules]
        if signature is None:
            signature = current_signature
        elif current_signature != signature:
            raise ValueError(f"frame {frame_idx}: molecule/site signature changed")

        copy_data = []
        for copy_idx in range(copies):
            lo = copy_idx * period
            hi = lo + period
            centers_local, sites_local, radius = unwrap_copy(frame.molecules[lo:hi], frame.box)
            max_radius = max(max_radius, radius)
            copy_data.append((centers_local, sites_local))
        prepared.append(copy_data)

    if max_radius <= 0.0:
        raise ValueError("non-positive copy radius")
    spacing = 2.0 * max_radius + cutoff + margin
    cols = max(1, min(int(layout_cols), copies))
    rows = int(math.ceil(copies / cols))
    out_box = np.asarray(
        [cols * spacing, rows * spacing, 2.0 * spacing], dtype=np.float64
    )

    output_frames: List[cn.Frame] = []
    min_distances = []
    force_sum2 = 0.0
    torque_sum2 = 0.0
    force_count = 0
    torque_count = 0
    max_translation_reconstruction_error = 0.0

    for frame_idx, (frame, copy_data) in enumerate(zip(source_frames, prepared)):
        molecules: List[cn.Molecule] = []
        flat_sites_by_copy: List[np.ndarray] = []
        for copy_idx in range(copies):
            col = copy_idx % cols
            row = copy_idx // cols
            slot = np.asarray(
                [(col + 0.5) * spacing, (row + 0.5) * spacing, spacing],
                dtype=np.float64,
            )
            lo = copy_idx * period
            centers_local, sites_local = copy_data[copy_idx]
            relocated_copy_sites = []

            for local_mol in range(period):
                src = frame.molecules[lo + local_mol]
                center = centers_local[local_mol] + slot
                sites = []
                for site_idx, (site_type, _old_pos) in enumerate(src.sites):
                    pos = sites_local[local_mol][site_idx] + slot
                    err = float(np.max(np.abs((pos - slot) - sites_local[local_mol][site_idx])))
                    max_translation_reconstruction_error = max(
                        max_translation_reconstruction_error, err
                    )
                    sites.append((site_type, pos))
                    relocated_copy_sites.append(pos)

                target_idx = lo + local_mol
                force = np.asarray(self_forces[frame_idx, target_idx], dtype=np.float64)
                torque = np.asarray(self_torques[frame_idx, target_idx], dtype=np.float64)
                force_sum2 += float(np.dot(force, force))
                force_count += 3
                if src.nsites > 1:
                    torque_sum2 += float(np.dot(torque, torque))
                    torque_count += 3
                molecules.append(
                    cn.Molecule(
                        mol_id=len(molecules),
                        center=center,
                        force=force,
                        torque=torque,
                        sites=sites,
                    )
                )
            flat_sites_by_copy.append(np.asarray(relocated_copy_sites, dtype=np.float64))

        min_dist = min_intercopy_distance(flat_sites_by_copy, out_box)
        min_distances.append(min_dist)
        if min_dist <= cutoff:
            raise RuntimeError(
                f"frame {frame_idx}: inter-copy site distance {min_dist:.6f} nm "
                f"is inside cutoff {cutoff:.6f} nm"
            )
        output_frames.append(cn.Frame(box=out_box.copy(), molecules=molecules))

    report = {
        "definition": {
            "purpose": "diagnostic self-only PaiNN learnability ablation",
            "target": "single-copy GROMACS rerun generalized force/torque assembled in original copy order",
            "geometry": "original retained CG geometry; each TEL22 copy is translated rigidly to a sparse periodic lattice",
            "intercopy_model_edges": "forbidden geometrically: every cross-copy site distance is greater than the PaiNN cutoff",
            "production_guardrail": "diagnostic only; targets are total DNA-self generalized forces, not the production residual F_ref-F_prior",
        },
        "counts": {
            "frames": len(output_frames),
            "copies_per_frame": copies,
            "residues_per_copy": period,
            "molecules_per_frame": copies * period,
            "sites_per_frame": sum(m.nsites for m in output_frames[0].molecules),
        },
        "isolation": {
            "cutoff_nm": cutoff,
            "margin_nm": margin,
            "max_copy_radius_nm": max_radius,
            "lattice_spacing_nm": spacing,
            "layout_columns": cols,
            "layout_rows": rows,
            "output_box_nm": [float(x) for x in out_box],
            "minimum_intercopy_site_distance_nm": float(min(min_distances)),
            "minimum_intercopy_site_distance_p50_nm": float(np.percentile(min_distances, 50)),
            "translation_reconstruction_max_abs_error_nm": max_translation_reconstruction_error,
        },
        "target_scale": {
            "force_component_rms_kj_mol_nm": float(math.sqrt(force_sum2 / force_count)),
            "torque_component_rms_kj_mol_multisite_only": (
                float(math.sqrt(torque_sum2 / torque_count)) if torque_count else math.nan
            ),
        },
    }
    return output_frames, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tel22_dataset.bin")
    ap.add_argument("--config", default="tel22_training_config.json")
    ap.add_argument("--raw-topology", default="md.gro")
    ap.add_argument("--raw-trr", default="md.trr")
    ap.add_argument("--copy-dir", default="dna_self_vs_intercopy")
    ap.add_argument("--copy-manifest", default="dna_self_vs_intercopy/copy_groups.json")
    ap.add_argument("--priors", default="cg_priors.json")
    ap.add_argument("--target-mode", choices=("total", "residual"), default="total")
    ap.add_argument("--sample-count", type=int, default=None)
    ap.add_argument("--validation-stride", type=int, default=0)
    ap.add_argument("--cutoff", type=float, default=None)
    ap.add_argument("--margin", type=float, default=0.50)
    ap.add_argument("--layout-cols", type=int, default=5)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    config = Path(args.config)
    raw_topology = Path(args.raw_topology)
    raw_trr = Path(args.raw_trr)
    copy_dir = Path(args.copy_dir)
    manifest_path = Path(args.copy_manifest)
    priors_path = Path(args.priors)
    required = [dataset, config, raw_topology, raw_trr, manifest_path]
    if args.target_mode == "residual":
        required.append(priors_path)
    for path in required:
        if not path.exists():
            raise SystemExit(f"required file not found: {path}")

    cfg = json.loads(config.read_text())
    cutoff = float(cfg["cutoff"] if args.cutoff is None else args.cutoff)
    if cutoff <= 0.0:
        raise SystemExit("cutoff must be > 0")
    if args.margin <= 0.0:
        raise SystemExit("margin must be > 0")
    if args.layout_cols <= 0:
        raise SystemExit("layout-cols must be > 0")
    if args.validation_stride not in (0, 1) and args.validation_stride < 2:
        raise SystemExit("validation-stride must be 0 (trainer default) or >= 2")
    if args.validation_stride == 1:
        raise SystemExit("validation-stride=1 would leave no training frames; use 0 or >=2")

    manifest: Dict = json.loads(manifest_path.read_text())
    copies = int(manifest["copies"])
    period = int(manifest["residues_per_copy"])
    if copies < 2 or period <= 0:
        raise SystemExit(f"invalid copy manifest: copies={copies}, residues_per_copy={period}")

    first_gro = copy_dir / "copy_00.gro"
    first_trr = copy_dir / "copy_00_rerun.trr"
    if not first_gro.exists() or not first_trr.exists():
        raise SystemExit(
            "single-copy rerun files are missing. Run 03n_prepare_full_self_reruns.sh first "
            f"(expected {first_gro} and {first_trr})."
        )

    reference_times_all, _f0, _t0, _sig0 = dsi.load_targets(first_gro, first_trr)
    self_f_all, self_t_all, self_sig = dsi.load_self_targets(
        copy_dir, manifest, reference_times_all
    )
    if len(self_sig) != period:
        raise RuntimeError(
            f"single-copy rerun residue signature has {len(self_sig)} residues, expected {period}"
        )

    n_available = len(reference_times_all)
    sample_count = n_available if args.sample_count is None else int(args.sample_count)
    sampled_positions = evenly_spaced_indices(n_available, sample_count)
    if int(args.validation_stride) >= 2:
        train_pos, val_pos, dataset_order = stratified_tail_order(
            sample_count, int(args.validation_stride)
        )
    else:
        train_pos = np.arange(sample_count, dtype=np.int64)
        val_pos = np.asarray([], dtype=np.int64)
        dataset_order = train_pos.copy()
    ordered_rerun_positions = sampled_positions[dataset_order]

    reference_times = np.asarray(reference_times_all)[ordered_rerun_positions]
    self_f = np.asarray(self_f_all)[ordered_rerun_positions]
    self_t = np.asarray(self_t_all)[ordered_rerun_positions]
    raw_indices = fs.raw_time_to_frame_indices(raw_topology, raw_trr, reference_times)
    source_frames = read_selected_frames(dataset, raw_indices)
    if not source_frames:
        raise RuntimeError("no source frames selected")

    detected_period = cn.detect_repeat_period(source_frames[0].molecules)
    if detected_period != period:
        raise RuntimeError(
            f"dataset repeat period {detected_period} does not match rerun manifest {period}"
        )
    if len(source_frames[0].molecules) != copies * period:
        raise RuntimeError(
            "dataset copy count does not match rerun manifest: "
            f"molecules={len(source_frames[0].molecules)}, copies={copies}, period={period}"
        )

    output_frames, report = build_frames(
        source_frames,
        self_f,
        self_t,
        period=period,
        copies=copies,
        cutoff=cutoff,
        margin=float(args.margin),
        layout_cols=int(args.layout_cols),
    )
    total_scale = dict(report["target_scale"])
    prior_report = None
    if args.target_mode == "residual":
        priors = json.loads(priors_path.read_text())
        output_frames, prior_report = subtract_cg_priors(
            output_frames, priors, period=period, copies=copies
        )
        report["definition"]["target"] = (
            "single-copy GROMACS DNA-self generalized force/torque minus the production CG priors "
            "evaluated on the isolated-copy geometry"
        )
        report["definition"]["production_guardrail"] = (
            "diagnostic self-residual: atomistic target excludes solvent/ions/other copies and the "
            "CG prior subtraction excludes cross-copy WCA by construction"
        )
        report["target_scale_total_self_before_prior_subtraction"] = total_scale
        report["prior_subtraction"] = prior_report
        report["target_scale"] = prior_report["residual_target_scale"]

    sampled_raw_positions = sampled_positions.tolist()
    train_sample_positions = sampled_positions[train_pos].tolist()
    val_sample_positions = sampled_positions[val_pos].tolist()
    report["sampling"] = {
        "available_rerun_frames": n_available,
        "selected_frames": sample_count,
        "selection": "rounded linspace across the full rerun time span",
        "selected_rerun_positions_in_chronological_order": [int(x) for x in sampled_raw_positions],
    }
    if len(val_pos):
        report["split"] = {
            "mode": "stratified_temporal_tail_v1",
            "validation_stride": int(args.validation_stride),
            "train_frames": int(len(train_pos)),
            "validation_frames": int(len(val_pos)),
            "dataset_binary_order": "all training frames first, then validation frames",
            "trainer_config": {
                "validation_split_mode": "tail",
                "validation_tail_frames": int(len(val_pos)),
            },
            "train_rerun_positions": [int(x) for x in train_sample_positions],
            "validation_rerun_positions": [int(x) for x in val_sample_positions],
        }
    else:
        report["split"] = {
            "mode": "trainer_default_random",
            "validation_stride": 0,
            "train_frames": None,
            "validation_frames": None,
            "dataset_binary_order": "chronological selected-frame order",
        }
    report["inputs"] = {
        "dataset": str(dataset),
        "raw_topology": str(raw_topology),
        "raw_trr": str(raw_trr),
        "copy_dir": str(copy_dir),
        "copy_manifest": str(manifest_path),
        "priors": str(priors_path) if args.target_mode == "residual" else None,
        "target_mode": args.target_mode,
        "rerun_times_ps_in_dataset_order": [float(x) for x in reference_times],
        "raw_dataset_frame_indices_in_dataset_order": [int(x) for x in raw_indices],
    }

    out_path = Path(args.output)
    report_path = Path(args.report)
    write_dataset(out_path, output_frames)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")

    print("======================================================")
    print(" TEL22 SELF-ONLY ISOLATED TRAINING DATASET")
    print("======================================================")
    print(
        f"target={args.target_mode} | frames={report['counts']['frames']} | "
        f"train={len(train_pos)} | val={len(val_pos)} | copies/frame={copies}"
    )
    print(
        f"cutoff={cutoff:.4f} nm | min inter-copy distance="
        f"{report['isolation']['minimum_intercopy_site_distance_nm']:.4f} nm"
    )
    print(
        f"RMS F={report['target_scale']['force_component_rms_kj_mol_nm']:.3f} kJ/(mol nm) | "
        f"T={report['target_scale']['torque_component_rms_kj_mol_multisite_only']:.3f} kJ/mol"
    )
    if prior_report is not None:
        ps = prior_report["prior_scale"]
        print(
            f"RMS prior F={ps['force_component_rms_kj_mol_nm']:.3f} | "
            f"prior T={ps['torque_component_rms_kj_mol_multisite_only']:.3f}"
        )
    print(f"dataset: {out_path}")
    print(f"report:  {report_path}")


if __name__ == "__main__":
    main()
