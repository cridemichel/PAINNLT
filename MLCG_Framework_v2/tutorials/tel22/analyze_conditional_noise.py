#!/usr/bin/env python3
"""TEL22 local conditional-noise diagnostic on the residual CG dataset.

The diagnostic is deliberately offline: it never changes the dataset or model.
It treats the ten repeated TEL22 molecules as repeated realizations of the same
22-residue topology and compares generalized force/torque targets between
geometrically similar copies after removing global translation/rotation.

Important interpretation: nearby-pair half-difference MSE is a *proxy*, not a
mathematical proof of the irreducible conditional variance. Finite geometry
mismatch and omitted inter-copy CG environment can only make that proxy less
clean. To reduce the latter contamination, a main subset keeps copies with no
site from another TEL22 copy inside the PaiNN cutoff.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

I32 = struct.Struct("=i")
F32_3 = struct.Struct("=3f")
SITE = struct.Struct("=ifff")


def read_exact(handle, n: int) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise EOFError(f"unexpected EOF: requested {n} bytes, got {len(data)}")
    return data


@dataclass
class Molecule:
    mol_id: int
    center: np.ndarray
    force: np.ndarray
    torque: np.ndarray
    sites: List[Tuple[int, np.ndarray]]

    @property
    def nsites(self) -> int:
        return len(self.sites)


@dataclass
class Frame:
    box: np.ndarray
    molecules: List[Molecule]


def read_frame(handle) -> Frame:
    nmol = I32.unpack(read_exact(handle, I32.size))[0]
    nsites_total = I32.unpack(read_exact(handle, I32.size))[0]
    if nmol <= 0 or nsites_total <= 0:
        raise ValueError(f"invalid frame header: nmol={nmol}, nsites={nsites_total}")
    box = np.asarray(F32_3.unpack(read_exact(handle, F32_3.size)), dtype=np.float64)
    molecules: List[Molecule] = []
    counted = 0
    for _ in range(nmol):
        mol_id = I32.unpack(read_exact(handle, I32.size))[0]
        nsites = I32.unpack(read_exact(handle, I32.size))[0]
        if nsites <= 0:
            raise ValueError(f"invalid nsites={nsites} for molecule {mol_id}")
        center = np.asarray(F32_3.unpack(read_exact(handle, F32_3.size)), dtype=np.float64)
        force = np.asarray(F32_3.unpack(read_exact(handle, F32_3.size)), dtype=np.float64)
        torque = np.asarray(F32_3.unpack(read_exact(handle, F32_3.size)), dtype=np.float64)
        sites: List[Tuple[int, np.ndarray]] = []
        for _s in range(nsites):
            st, x, y, z = SITE.unpack(read_exact(handle, SITE.size))
            sites.append((int(st), np.asarray((x, y, z), dtype=np.float64)))
        counted += nsites
        molecules.append(Molecule(mol_id, center, force, torque, sites))
    if counted != nsites_total:
        raise ValueError(f"site-count mismatch: header={nsites_total}, parsed={counted}")
    return Frame(box=box, molecules=molecules)


def molecule_signature(mol: Molecule) -> Tuple[int, Tuple[int, ...]]:
    return mol.nsites, tuple(st for st, _ in mol.sites)


def detect_repeat_period(molecules: Sequence[Molecule]) -> int:
    signatures = [molecule_signature(m) for m in molecules]
    n = len(signatures)
    divisors = [p for p in range(1, n + 1) if n % p == 0]
    for p in divisors:
        if p == 1:
            # A homogeneous one-residue repeat is too weak a topology inference
            # for this diagnostic. Prefer a larger genuine sequence when present.
            continue
        if all(signatures[i] == signatures[i % p] for i in range(n)):
            return p
    return n


def mic(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    out = np.asarray(delta, dtype=np.float64).copy()
    for k in range(3):
        L = float(box[k])
        if L > 0.0:
            out[..., k] -= L * np.rint(out[..., k] / L)
    return out


def unwrap_copy_geometry(block: Sequence[Molecule], box: np.ndarray) -> np.ndarray:
    """Concatenate all copy sites in sequence order in one PBC-consistent image."""
    raw_centers = np.stack([m.center for m in block], axis=0)
    centers = np.empty_like(raw_centers)
    centers[0] = raw_centers[0]
    for i in range(1, len(block)):
        step = mic(raw_centers[i] - raw_centers[i - 1], box)
        centers[i] = centers[i - 1] + step

    coords: List[np.ndarray] = []
    for i, mol in enumerate(block):
        for _st, site_pos in mol.sites:
            local = mic(site_pos - mol.center, box)
            coords.append(centers[i] + local)
    xyz = np.stack(coords, axis=0)
    xyz -= xyz.mean(axis=0, keepdims=True)
    return xyz


def kabsch_row(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return R for row vectors such that mobile @ R best matches reference."""
    h = mobile.T @ reference
    u, _s, vt = np.linalg.svd(h)
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def infer_residue_labels(block: Sequence[Molecule]) -> List[str]:
    labels = []
    for mol in block:
        first = mol.sites[0][0]
        if mol.nsites == 1 and first == 0:
            labels.append("DA")
        elif mol.nsites == 1 and first == 1:
            labels.append("DT")
        elif mol.nsites > 1 and first == 2:
            labels.append("DG")
        else:
            labels.append(f"site{first}_n{mol.nsites}")
    return labels


def cross_copy_contact_flags(frame: Frame, period: int, cutoff: float) -> np.ndarray:
    """True for copies having any site-site neighbor in another copy <= cutoff."""
    nmol = len(frame.molecules)
    ncopies = nmol // period
    all_pos: List[np.ndarray] = []
    owner_copy: List[int] = []
    for mi, mol in enumerate(frame.molecules):
        copy_idx = mi // period
        for _st, p in mol.sites:
            all_pos.append(p)
            owner_copy.append(copy_idx)
    xyz = np.stack(all_pos, axis=0)
    owner = np.asarray(owner_copy, dtype=np.int32)
    box = np.asarray(frame.box, dtype=np.float64)
    if np.any(box <= 0.0):
        return np.ones(ncopies, dtype=bool)
    wrapped = np.mod(xyz, box[None, :])
    tree = cKDTree(wrapped, boxsize=box)
    pairs = tree.query_pairs(r=float(cutoff), output_type="ndarray")
    flags = np.zeros(ncopies, dtype=bool)
    if pairs.size:
        ci = owner[pairs[:, 0]]
        cj = owner[pairs[:, 1]]
        mask = ci != cj
        if np.any(mask):
            flags[np.unique(ci[mask])] = True
            flags[np.unique(cj[mask])] = True
    return flags


def percentile_dict(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"p50": math.nan, "p90": math.nan, "p95": math.nan, "p99": math.nan, "max": math.nan}
    return {
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def rms_components(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else math.nan


def unique_pairs(pairs: Iterable[Tuple[int, int]]) -> np.ndarray:
    uniq = sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs if int(i) != int(j)})
    if not uniq:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(uniq, dtype=np.int64)


def nearest_different_copy(descriptors: np.ndarray, copy_ids: np.ndarray, subset: np.ndarray, k_search: int = 96) -> np.ndarray:
    subset = np.asarray(subset, dtype=np.int64)
    if subset.size < 2:
        return np.empty((0, 2), dtype=np.int64)
    x = descriptors[subset]
    tree = cKDTree(x)
    k = min(max(2, k_search), len(subset))
    d, idx = tree.query(x, k=k)
    if k == 1:
        idx = idx[:, None]
    pairs = []
    for row, global_i in enumerate(subset):
        for local_j in np.atleast_1d(idx[row]):
            global_j = int(subset[int(local_j)])
            if global_j == global_i:
                continue
            if copy_ids[global_j] == copy_ids[global_i]:
                continue
            pairs.append((int(global_i), global_j))
            break
    return unique_pairs(pairs)


def nearest_same_copy_gap(descriptors: np.ndarray, copy_ids: np.ndarray, frame_ids: np.ndarray, min_gap: int) -> np.ndarray:
    pairs: List[Tuple[int, int]] = []
    for copy_id in np.unique(copy_ids):
        subset = np.flatnonzero(copy_ids == copy_id)
        if subset.size < 2:
            continue
        x = descriptors[subset]
        tree = cKDTree(x)
        k = min(max(16, 2 * min_gap + 8), len(subset))
        _d, idx = tree.query(x, k=k)
        if k == 1:
            idx = idx[:, None]
        for row, gi in enumerate(subset):
            for lj in np.atleast_1d(idx[row]):
                gj = int(subset[int(lj)])
                if gj == gi:
                    continue
                if abs(int(frame_ids[gj]) - int(frame_ids[gi])) < min_gap:
                    continue
                pairs.append((int(gi), gj))
                break
    return unique_pairs(pairs)


def random_control_pairs(
    rng: np.random.Generator,
    n_pairs: int,
    candidates: np.ndarray,
    copy_ids: np.ndarray,
    frame_ids: np.ndarray,
    mode: str,
    min_gap: int,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if n_pairs <= 0 or candidates.size < 2:
        return np.empty((0, 2), dtype=np.int64)
    result = set()
    attempts = 0
    max_attempts = max(1000, n_pairs * 200)
    while len(result) < n_pairs and attempts < max_attempts:
        attempts += 1
        i = int(rng.choice(candidates))
        if mode == "different_copy":
            valid = candidates[copy_ids[candidates] != copy_ids[i]]
        elif mode == "same_copy_gap":
            valid = candidates[(copy_ids[candidates] == copy_ids[i]) & (np.abs(frame_ids[candidates] - frame_ids[i]) >= min_gap)]
        else:
            raise ValueError(mode)
        if valid.size == 0:
            continue
        j = int(rng.choice(valid))
        if i != j:
            result.add((min(i, j), max(i, j)))
    if not result:
        return np.empty((0, 2), dtype=np.int64)
    arr = np.asarray(sorted(result), dtype=np.int64)
    if len(arr) > n_pairs:
        arr = arr[:n_pairs]
    return arr


def pair_metrics(
    name: str,
    pairs: np.ndarray,
    descriptors: np.ndarray,
    forces: np.ndarray,
    torques: np.ndarray,
    residue_labels: Sequence[str],
    rigid_mask: np.ndarray,
    force_target_rms: float,
    torque_target_rms: float,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    if len(pairs) == 0:
        return {"name": name, "pairs": 0}, {}
    i = pairs[:, 0]
    j = pairs[:, 1]
    geom = np.sqrt(np.mean(np.square(descriptors[i] - descriptors[j], dtype=np.float64), axis=1))
    df = forces[i] - forces[j]
    dt = torques[i] - torques[j]
    f_pair = np.sqrt(np.mean(np.square(df, dtype=np.float64), axis=(1, 2)))
    if np.any(rigid_mask):
        t_pair = np.sqrt(np.mean(np.square(dt[:, rigid_mask, :], dtype=np.float64), axis=(1, 2)))
    else:
        t_pair = np.full(len(pairs), np.nan)

    force_half_mse_frac = float(np.mean(np.square(df)) / (2.0 * force_target_rms * force_target_rms)) if force_target_rms > 0 else math.nan
    if torque_target_rms > 0 and np.any(rigid_mask):
        torque_half_mse_frac = float(np.mean(np.square(dt[:, rigid_mask, :])) / (2.0 * torque_target_rms * torque_target_rms))
    else:
        torque_half_mse_frac = math.nan

    by_type = {}
    labels = np.asarray(residue_labels)
    for label in sorted(set(residue_labels)):
        mask = labels == label
        dfl = df[:, mask, :]
        target_rms_type = rms_components(forces[:, mask, :])
        by_type[label] = {
            "positions_per_copy": int(mask.sum()),
            "target_component_rms": target_rms_type,
            "pair_difference_component_rms": rms_components(dfl),
            "half_pair_difference_mse_fraction_of_target_mse": (
                float(np.mean(np.square(dfl)) / (2.0 * target_rms_type * target_rms_type)) if target_rms_type > 0 else math.nan
            ),
        }

    corr = math.nan
    if len(geom) >= 3 and np.std(geom) > 0 and np.std(f_pair) > 0:
        corr = float(np.corrcoef(geom, f_pair)[0, 1])

    summary = {
        "name": name,
        "pairs": int(len(pairs)),
        "geometry_rmsd_nm": percentile_dict(geom),
        "force_pair_difference_rms": percentile_dict(f_pair),
        "torque_pair_difference_rms": percentile_dict(t_pair[np.isfinite(t_pair)]),
        "force_half_pair_difference_mse_fraction_of_target_mse": force_half_mse_frac,
        "torque_half_pair_difference_mse_fraction_of_target_mse": torque_half_mse_frac,
        "pearson_geometry_rmsd_vs_force_pair_rms": corr,
        "force_by_residue_type": by_type,
    }
    arrays = {"geom": geom, "force_pair": f_pair, "torque_pair": t_pair}
    return summary, arrays


def process_dataset(dataset: Path, cutoff: float):
    descriptors: List[np.ndarray] = []
    forces: List[np.ndarray] = []
    torques: List[np.ndarray] = []
    frame_ids: List[int] = []
    copy_ids: List[int] = []
    has_external_contact: List[bool] = []

    with dataset.open("rb") as fh:
        nframes = I32.unpack(read_exact(fh, I32.size))[0]
        if nframes <= 0:
            raise ValueError("dataset contains no frames")
        first = read_frame(fh)
        period = detect_repeat_period(first.molecules)
        if len(first.molecules) % period != 0:
            raise ValueError("molecule count is not divisible by detected repeat period")
        ncopies = len(first.molecules) // period
        if ncopies < 2:
            raise ValueError("conditional-noise copy analysis requires at least two repeated copies")
        reference_block = first.molecules[:period]
        reference_xyz = unwrap_copy_geometry(reference_block, first.box)
        labels = infer_residue_labels(reference_block)
        rigid_mask = np.asarray([m.nsites > 1 for m in reference_block], dtype=bool)
        signature_ref = [molecule_signature(m) for m in first.molecules]

        def consume(frame: Frame, frame_idx: int):
            if len(frame.molecules) != period * ncopies:
                raise ValueError(f"frame {frame_idx}: molecule count changed")
            if [molecule_signature(m) for m in frame.molecules] != signature_ref:
                raise ValueError(f"frame {frame_idx}: molecule/site topology changed")
            contacts = cross_copy_contact_flags(frame, period, cutoff)
            for copy_idx in range(ncopies):
                lo = copy_idx * period
                hi = lo + period
                block = frame.molecules[lo:hi]
                xyz = unwrap_copy_geometry(block, frame.box)
                r = kabsch_row(xyz, reference_xyz)
                aligned = xyz @ r
                descriptors.append(aligned.reshape(-1).astype(np.float32))
                forces.append(np.stack([m.force for m in block], axis=0) @ r)
                torques.append(np.stack([m.torque for m in block], axis=0) @ r)
                frame_ids.append(frame_idx)
                copy_ids.append(copy_idx)
                has_external_contact.append(bool(contacts[copy_idx]))

        consume(first, 0)
        for fi in range(1, nframes):
            consume(read_frame(fh), fi)
        if fh.read(1):
            raise ValueError("unexpected trailing bytes after dataset")

    return {
        "descriptors": np.asarray(descriptors, dtype=np.float32),
        "forces": np.asarray(forces, dtype=np.float32),
        "torques": np.asarray(torques, dtype=np.float32),
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "copy_ids": np.asarray(copy_ids, dtype=np.int16),
        "contacts": np.asarray(has_external_contact, dtype=bool),
        "period": period,
        "copies": ncopies,
        "frames": nframes,
        "labels": labels,
        "rigid_mask": rigid_mask,
        "sites_per_copy": int(reference_xyz.shape[0]),
    }


def write_pair_csv(path: Path, rows: List[Dict]):
    fields = [
        "pair_set", "control", "sample_i", "sample_j",
        "frame_i", "frame_j", "copy_i", "copy_j",
        "contact_i", "contact_j", "geometry_rmsd_nm",
        "force_pair_rms", "torque_pair_rms",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="tel22_training_config.json")
    ap.add_argument("--cutoff", type=float, default=None, help="override PaiNN cutoff in nm")
    ap.add_argument("--same-copy-gap-frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    config = Path(args.config)
    if not dataset.exists():
        raise SystemExit(f"dataset not found: {dataset}")
    cutoff = args.cutoff
    if cutoff is None:
        if not config.exists():
            raise SystemExit(f"config not found and --cutoff absent: {config}")
        cfg = json.loads(config.read_text())
        cutoff = float(cfg["cutoff"])
    if cutoff <= 0.0:
        raise SystemExit("cutoff must be positive")
    if args.same_copy_gap_frames < 1:
        raise SystemExit("--same-copy-gap-frames must be >= 1")

    data = process_dataset(dataset, cutoff)
    desc = data["descriptors"]
    forces = data["forces"]
    torques = data["torques"]
    frame_ids = data["frame_ids"]
    copy_ids = data["copy_ids"]
    contacts = data["contacts"]
    rigid = data["rigid_mask"]
    labels = data["labels"]

    force_target_rms = rms_components(forces)
    torque_target_rms = rms_components(torques[:, rigid, :]) if np.any(rigid) else math.nan
    all_idx = np.arange(len(desc), dtype=np.int64)
    isolated_idx = np.flatnonzero(~contacts)

    pair_sets: Dict[str, np.ndarray] = {}
    pair_sets["nearest_different_copy_all"] = nearest_different_copy(desc, copy_ids, all_idx)
    if len(isolated_idx) >= 2:
        pair_sets["nearest_different_copy_isolated"] = nearest_different_copy(desc, copy_ids, isolated_idx)
    else:
        pair_sets["nearest_different_copy_isolated"] = np.empty((0, 2), dtype=np.int64)
    pair_sets["nearest_same_copy_gap"] = nearest_same_copy_gap(
        desc, copy_ids, frame_ids, args.same_copy_gap_frames
    )

    rng = np.random.default_rng(args.seed)
    reports = {}
    pair_rows: List[Dict] = []

    for name, pairs in pair_sets.items():
        report, arrays = pair_metrics(
            name, pairs, desc, forces, torques, labels, rigid, force_target_rms, torque_target_rms
        )
        reports[name] = {"nearest": report}

        if "different_copy" in name:
            candidates = isolated_idx if name.endswith("isolated") else all_idx
            mode = "different_copy"
        else:
            candidates = all_idx
            mode = "same_copy_gap"
        random_pairs = random_control_pairs(
            rng, len(pairs), candidates, copy_ids, frame_ids, mode, args.same_copy_gap_frames
        )
        random_report, random_arrays = pair_metrics(
            name + "_random_control", random_pairs, desc, forces, torques, labels, rigid,
            force_target_rms, torque_target_rms
        )
        reports[name]["random_control"] = random_report

        if report.get("pairs", 0) and random_report.get("pairs", 0):
            near_force = report["force_half_pair_difference_mse_fraction_of_target_mse"]
            rand_force = random_report["force_half_pair_difference_mse_fraction_of_target_mse"]
            reports[name]["nearest_vs_random_force_half_mse_ratio"] = (
                float(near_force / rand_force) if rand_force > 0 else math.nan
            )
            near_t = report["torque_half_pair_difference_mse_fraction_of_target_mse"]
            rand_t = random_report["torque_half_pair_difference_mse_fraction_of_target_mse"]
            reports[name]["nearest_vs_random_torque_half_mse_ratio"] = (
                float(near_t / rand_t) if np.isfinite(rand_t) and rand_t > 0 else math.nan
            )

        for control, pp, aa in (("nearest", pairs, arrays), ("random", random_pairs, random_arrays)):
            if len(pp) == 0:
                continue
            for row_idx, (i, j) in enumerate(pp):
                pair_rows.append({
                    "pair_set": name,
                    "control": control,
                    "sample_i": int(i),
                    "sample_j": int(j),
                    "frame_i": int(frame_ids[i]),
                    "frame_j": int(frame_ids[j]),
                    "copy_i": int(copy_ids[i]),
                    "copy_j": int(copy_ids[j]),
                    "contact_i": int(contacts[i]),
                    "contact_j": int(contacts[j]),
                    "geometry_rmsd_nm": float(aa["geom"][row_idx]),
                    "force_pair_rms": float(aa["force_pair"][row_idx]),
                    "torque_pair_rms": float(aa["torque_pair"][row_idx]),
                })

    label_counts = {label: int(sum(x == label for x in labels)) for label in sorted(set(labels))}
    report = {
        "definition": {
            "geometry": "all physical CG sites of one repeated TEL22 copy, MIC-unwrapped along residue sequence, centered, Kabsch-aligned to a fixed reference; RMSD in nm",
            "target": "residual molecular force and rigid-body torque stored in tel22_dataset.bin, rotated by the same Kabsch transform",
            "local_noise_proxy": "MSE(target_i-target_j)/(2*MSE(target)); exact conditional-noise fraction only in the zero-geometry-separation/independent-hidden-DOF limit",
            "random_control": "same pairing constraints but random geometrically unrelated samples",
            "isolation": "copy has no CG site from another TEL22 copy within the PaiNN cutoff",
        },
        "inputs": {
            "dataset": str(dataset),
            "config": str(config),
            "cutoff_nm": cutoff,
            "same_copy_min_gap_frames": args.same_copy_gap_frames,
            "random_seed": args.seed,
        },
        "counts": {
            "frames": int(data["frames"]),
            "molecules_per_frame": int(data["period"] * data["copies"]),
            "detected_residues_per_copy": int(data["period"]),
            "detected_copies_per_frame": int(data["copies"]),
            "sites_per_copy": int(data["sites_per_copy"]),
            "copy_samples": int(len(desc)),
            "isolated_copy_samples": int(len(isolated_idx)),
            "isolated_copy_fraction": float(len(isolated_idx) / len(desc)),
            "copy_samples_with_external_cg_contact": int(np.sum(contacts)),
            "residue_positions_per_copy": label_counts,
        },
        "target_scale": {
            "force_component_rms_kj_mol_nm": force_target_rms,
            "torque_component_rms_kj_mol": torque_target_rms,
        },
        "pair_analyses": reports,
        "interpretation_guardrails": [
            "A nearest/random ratio near 1 means geometry similarity, at the tested resolution, barely reduces target differences.",
            "A ratio well below 1 means nearby CG geometries carry predictive information about the target.",
            "The half-pair MSE fraction is not a rigorous lower bound when geometries are not identical.",
            "Prefer nearest_different_copy_isolated when it has enough samples, because external TEL22-TEL22 interactions are then absent inside the model cutoff.",
            "Same-copy pairs separated in time are complementary but their eliminated solvent/ion degrees of freedom may remain temporally correlated.",
        ],
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    write_pair_csv(out_csv, pair_rows)

    print("======================================================")
    print(" TEL22 CONDITIONAL-NOISE / LOCAL-NEIGHBOR DIAGNOSTIC")
    print("======================================================")
    print(f"frames={data['frames']} | copies/frame={data['copies']} | residues/copy={data['period']} | sites/copy={data['sites_per_copy']}")
    print(f"copy samples={len(desc)} | isolated={len(isolated_idx)} ({100.0*len(isolated_idx)/len(desc):.1f}%) | cutoff={cutoff:.4f} nm")
    print(f"target RMS: force={force_target_rms:.3f} kJ/(mol nm) | torque={torque_target_rms:.3f} kJ/mol")
    for name, rr in reports.items():
        near = rr["nearest"]
        rand = rr["random_control"]
        if near.get("pairs", 0) == 0:
            print(f"[{name}] no valid pairs")
            continue
        print(f"[{name}] pairs={near['pairs']} | geom P50={near['geometry_rmsd_nm']['p50']:.4f} nm | "
              f"F half-MSE/target={near['force_half_pair_difference_mse_fraction_of_target_mse']:.3f} | "
              f"random={rand.get('force_half_pair_difference_mse_fraction_of_target_mse', math.nan):.3f} | "
              f"near/random={rr.get('nearest_vs_random_force_half_mse_ratio', math.nan):.3f}")
    print(f"JSON: {out_json}")
    print(f"CSV:  {out_csv}")


if __name__ == "__main__":
    main()
