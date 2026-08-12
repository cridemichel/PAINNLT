#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import MDAnalysis as mda
import numpy as np

import analyze_conditional_noise as cn
import analyze_dna_self_vs_intercopy as dsi
import analyze_force_source_decomposition as fs


ANCHORS: Dict[str, Tuple[str, str, str]] = {
    "DA": ("N9", "C4", "N1"),
    "DT": ("N1", "C4", "C6"),
}


def _safe_unit(v: np.ndarray, label: str, eps: float = 1.0e-10) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= eps:
        raise RuntimeError(f"Degenerate orientation vector for {label}: norm={n}")
    return np.asarray(v, dtype=np.float64) / n


def residue_orientation_frame(residue, box_nm: np.ndarray) -> np.ndarray:
    """Return a deterministic right-handed 3x3 row-axis frame for DA/DT.

    Only the orientation is retained: anchor distances and other internal deformations are
    discarded by normalization/orthogonalization.  Rows are x,y,z axes in lab coordinates.
    """
    resname = str(residue.resname)
    if resname not in ANCHORS:
        raise ValueError(f"No orientation anchors defined for residue {resname}")
    names = [str(x) for x in residue.atoms.names]
    wanted = ANCHORS[resname]
    indices = []
    for atom_name in wanted:
        hits = [i for i, n in enumerate(names) if n == atom_name]
        if len(hits) != 1:
            raise RuntimeError(
                f"{resname} residue {residue.resid}: expected exactly one atom named {atom_name}; "
                f"found {len(hits)}. Available atoms: {', '.join(names)}"
            )
        indices.append(hits[0])

    pos_nm = np.asarray(residue.atoms.positions, dtype=np.float64) / 10.0
    pos_nm = fs.unwrap_residue(pos_nm, box_nm)
    a, b, c = (pos_nm[i] for i in indices)
    x = _safe_unit(b - a, f"{resname}:{wanted[0]}->{wanted[1]}")
    cvec = c - a
    y_raw = cvec - float(np.dot(cvec, x)) * x
    y = _safe_unit(y_raw, f"{resname}:{wanted[2]} plane component")
    z = _safe_unit(np.cross(x, y), f"{resname}:plane normal")
    # Recompute y to make the basis exactly orthonormal/right-handed.
    y = _safe_unit(np.cross(z, x), f"{resname}:orthogonalized y")
    frame = np.stack([x, y, z], axis=0)
    if not np.all(np.isfinite(frame)) or np.linalg.det(frame) < 0.999999:
        raise RuntimeError(f"Invalid orientation frame for {resname} residue {residue.resid}")
    return frame


def load_self_targets(copy_dir: Path, manifest: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Sequence[Tuple[str, int]]]:
    times, _f0, _t0, _sig0 = dsi.load_targets(copy_dir / "copy_00.gro", copy_dir / "copy_00_rerun.trr")
    times = np.asarray(times, dtype=np.float64)
    self_f, self_t, self_sig = dsi.load_self_targets(copy_dir, manifest, times)
    return times, np.asarray(self_f), np.asarray(self_t), self_sig


def load_cg_samples(
    dataset: Path,
    raw_indices: Sequence[int],
    target_f: np.ndarray,
    target_t: np.ndarray,
):
    wanted = {int(v): i for i, v in enumerate(raw_indices)}
    descriptors: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    forces: List[np.ndarray] = []
    torques: List[np.ndarray] = []
    frame_ids: List[int] = []
    copy_ids: List[int] = []

    with dataset.open("rb") as fh:
        nframes = cn.I32.unpack(cn.read_exact(fh, cn.I32.size))[0]
        if nframes <= 0:
            raise ValueError("dataset contains no frames")
        first = cn.read_frame(fh)
        period = cn.detect_repeat_period(first.molecules)
        ncopies = len(first.molecules) // period
        if ncopies < 2 or period * ncopies != len(first.molecules):
            raise ValueError("invalid repeated-copy topology in CG dataset")
        ref_block = first.molecules[:period]
        ref_xyz = cn.unwrap_copy_geometry(ref_block, first.box)
        labels = cn.infer_residue_labels(ref_block)
        signature_ref = [cn.molecule_signature(m) for m in first.molecules]
        current_rigid_mask = np.asarray([m.nsites > 1 for m in ref_block], dtype=bool)
        if target_f.shape[1:] != (len(first.molecules), 3) or target_t.shape != target_f.shape:
            raise ValueError(f"self target shape {target_f.shape} incompatible with dataset")

        def consume(frame, fi: int) -> None:
            if fi not in wanted:
                return
            ti = wanted[fi]
            if [cn.molecule_signature(m) for m in frame.molecules] != signature_ref:
                raise ValueError(f"frame {fi}: molecule/site topology changed")
            for ci in range(ncopies):
                lo = ci * period
                hi = lo + period
                block = frame.molecules[lo:hi]
                xyz = cn.unwrap_copy_geometry(block, frame.box)
                r = cn.kabsch_row(xyz, ref_xyz)
                descriptors.append((xyz @ r).reshape(-1).astype(np.float32))
                rotations.append(r.astype(np.float64))
                forces.append((target_f[ti, lo:hi, :] @ r).astype(np.float32))
                torques.append((target_t[ti, lo:hi, :] @ r).astype(np.float32))
                frame_ids.append(fi)
                copy_ids.append(ci)

        consume(first, 0)
        for fi in range(1, nframes):
            consume(cn.read_frame(fh), fi)
        if fh.read(1):
            raise ValueError("unexpected trailing bytes after CG dataset")

    expected = len(raw_indices) * ncopies
    if len(descriptors) != expected:
        raise RuntimeError(f"selected CG sample count mismatch: got {len(descriptors)}, expected {expected}")
    return {
        "descriptors": np.asarray(descriptors, dtype=np.float32),
        "rotations": np.asarray(rotations, dtype=np.float64),
        "forces": np.asarray(forces, dtype=np.float32),
        "torques": np.asarray(torques, dtype=np.float32),
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "copy_ids": np.asarray(copy_ids, dtype=np.int16),
        "period": int(period),
        "copies": int(ncopies),
        "labels": labels,
        "current_rigid_mask": current_rigid_mask,
        "sites_per_copy": int(ref_xyz.shape[0]),
        "dataset_frames": int(nframes),
    }


def load_da_dt_orientations(
    raw_topology: Path,
    raw_trr: Path,
    raw_indices: Sequence[int],
    period: int,
    copies: int,
    labels: Sequence[str],
    rotations: np.ndarray,
) -> Tuple[np.ndarray, List[int], List[str]]:
    u = mda.Universe(str(raw_topology), str(raw_trr))
    dna_res = [r for r in u.residues if str(r.resname) in fs.DNA_RESNAMES]
    if len(dna_res) != period * copies:
        raise RuntimeError(
            f"raw atomistic DNA residue count {len(dna_res)} != period*copies {period*copies}"
        )
    raw_labels = [str(r.resname) for r in dna_res[:period]]
    if list(raw_labels) != list(labels):
        raise RuntimeError(f"raw atomistic residue order {raw_labels} != CG order {list(labels)}")
    for ci in range(copies):
        block_labels = [str(r.resname) for r in dna_res[ci * period:(ci + 1) * period]]
        if block_labels != raw_labels:
            raise RuntimeError(f"copy {ci}: atomistic residue order differs from copy 0")

    orient_local = [i for i, lab in enumerate(labels) if lab in ANCHORS]
    if not orient_local:
        raise RuntimeError("No DA/DT residues found in TEL22 copy")
    orient_labels = [str(labels[i]) for i in orient_local]

    # Validate anchor availability before traversing the trajectory.
    for local_i in orient_local:
        residue_orientation_frame(dna_res[local_i], np.asarray([1.0e9, 1.0e9, 1.0e9]))

    all_frames: List[np.ndarray] = []
    sample_i = 0
    for raw_i in raw_indices:
        ts = u.trajectory[int(raw_i)]
        box_nm = np.asarray(ts.dimensions[:3], dtype=np.float64) / 10.0
        for ci in range(copies):
            r_align = rotations[sample_i]
            per_copy = []
            base = ci * period
            for local_i in orient_local:
                frame = residue_orientation_frame(dna_res[base + local_i], box_nm)
                per_copy.append(frame @ r_align)
            all_frames.append(np.asarray(per_copy, dtype=np.float64))
            sample_i += 1

    arr = np.asarray(all_frames, dtype=np.float64)
    if arr.shape != (len(raw_indices) * copies, len(orient_local), 3, 3):
        raise RuntimeError(f"unexpected orientation array shape {arr.shape}")
    return arr, orient_local, orient_labels


def percentile_dict(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return cn.percentile_dict(x) if x.size else {"p05": math.nan, "p50": math.nan, "p95": math.nan, "mean": math.nan}


def orientation_pair_angles_deg(frames: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.empty((0,), dtype=np.float64)
    a = frames[pairs[:, 0]]
    b = frames[pairs[:, 1]]
    # Relative rotation trace for row-basis matrices.  Average RMS angle over DA/DT residues.
    rel = np.einsum("...ik,...jk->...ij", a, b)
    tr = np.trace(rel, axis1=-2, axis2=-1)
    cosang = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    ang = np.arccos(cosang)
    rms = np.sqrt(np.mean(np.square(ang), axis=1))
    return np.degrees(rms)


def type_target_metrics(
    pairs: np.ndarray,
    forces: np.ndarray,
    torques: np.ndarray,
    labels: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    if len(pairs) == 0:
        return {}
    i = pairs[:, 0]
    j = pairs[:, 1]
    labels_arr = np.asarray(labels)
    out: Dict[str, Dict[str, float]] = {}
    for label in sorted(set(labels)):
        mask = labels_arr == label
        df = forces[i][:, mask, :] - forces[j][:, mask, :]
        dt = torques[i][:, mask, :] - torques[j][:, mask, :]
        frms = cn.rms_components(forces[:, mask, :])
        trms = cn.rms_components(torques[:, mask, :])
        out[label] = {
            "positions_per_copy": int(mask.sum()),
            "force_target_component_rms": float(frms),
            "force_half_pair_difference_mse_fraction_of_target_mse": (
                float(np.mean(np.square(df, dtype=np.float64)) / (2.0 * frms * frms)) if frms > 0 else math.nan
            ),
            "torque_target_component_rms": float(trms),
            "torque_half_pair_difference_mse_fraction_of_target_mse": (
                float(np.mean(np.square(dt, dtype=np.float64)) / (2.0 * trms * trms)) if trms > 0 else math.nan
            ),
        }
    return out


def analyze_variant(
    name: str,
    descriptor: np.ndarray,
    base_descriptor: np.ndarray,
    orientation_frames: np.ndarray,
    forces: np.ndarray,
    torques: np.ndarray,
    labels: Sequence[str],
    copy_ids: np.ndarray,
    frame_ids: np.ndarray,
    same_copy_gap: int,
    seed: int,
):
    all_idx = np.arange(len(descriptor), dtype=np.int64)
    pair_sets = {
        "nearest_same_copy_gap": cn.nearest_same_copy_gap(descriptor, copy_ids, frame_ids, same_copy_gap),
        "nearest_different_copy": cn.nearest_different_copy(descriptor, copy_ids, all_idx),
    }
    force_rms = cn.rms_components(forces)
    torque_rms_all = cn.rms_components(torques)
    all_torque_mask = np.ones(len(labels), dtype=bool)
    reports = {}
    pair_arrays = {}
    for pi, (pair_name, pairs) in enumerate(pair_sets.items()):
        mode = "same_copy_gap" if pair_name == "nearest_same_copy_gap" else "different_copy"
        rng = np.random.default_rng(seed + 1009 * pi)
        random_pairs = cn.random_control_pairs(
            rng, len(pairs), all_idx, copy_ids, frame_ids, mode, same_copy_gap
        )
        near, near_arr = cn.pair_metrics(
            f"{name}_{pair_name}", pairs, descriptor, forces, torques, labels,
            all_torque_mask, force_rms, torque_rms_all,
        )
        rand, rand_arr = cn.pair_metrics(
            f"{name}_{pair_name}_random", random_pairs, descriptor, forces, torques, labels,
            all_torque_mask, force_rms, torque_rms_all,
        )
        item = {
            "nearest": near,
            "random_control": rand,
            "nearest_type_targets": type_target_metrics(pairs, forces, torques, labels),
            "random_type_targets": type_target_metrics(random_pairs, forces, torques, labels),
        }
        if len(pairs):
            dbase = base_descriptor[pairs[:, 0]] - base_descriptor[pairs[:, 1]]
            item["nearest_current_cg_rmsd_nm"] = percentile_dict(
                np.sqrt(np.mean(np.square(dbase, dtype=np.float64), axis=1))
            )
            item["nearest_da_dt_orientation_rms_angle_deg"] = percentile_dict(
                orientation_pair_angles_deg(orientation_frames, pairs)
            )
        if near.get("pairs", 0) and rand.get("pairs", 0):
            nf = near["force_half_pair_difference_mse_fraction_of_target_mse"]
            rf = rand["force_half_pair_difference_mse_fraction_of_target_mse"]
            nt = near["torque_half_pair_difference_mse_fraction_of_target_mse"]
            rt = rand["torque_half_pair_difference_mse_fraction_of_target_mse"]
            item["nearest_vs_random_force_half_mse_ratio"] = float(nf / rf) if rf > 0 else math.nan
            item["nearest_vs_random_torque_half_mse_ratio"] = float(nt / rt) if rt > 0 else math.nan
        reports[pair_name] = item
        pair_arrays[pair_name] = (pairs, random_pairs, near_arr, rand_arr)
    return reports, pair_arrays


def ratio_or_nan(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else math.nan


def main() -> None:
    ap = argparse.ArgumentParser(description="TEL22 DA/DT orientation conditional-noise ablation")
    ap.add_argument("--dataset", default="tel22_dataset.bin")
    ap.add_argument("--raw-topology", default="md.gro")
    ap.add_argument("--raw-trr", default="md.trr")
    ap.add_argument("--copy-dir", default="dna_self_full_reruns")
    ap.add_argument("--copy-manifest", default="dna_self_full_reruns/copy_groups.json")
    ap.add_argument("--same-copy-gap-frames", type=int, default=20)
    ap.add_argument("--orientation-scales-nm", nargs="+", type=float, default=[0.10, 0.20, 0.30])
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    if args.same_copy_gap_frames < 1:
        ap.error("--same-copy-gap-frames must be >= 1")
    scales = sorted(set(float(x) for x in args.orientation_scales_nm))
    if not scales or any((not np.isfinite(x) or x <= 0.0) for x in scales):
        ap.error("orientation scales must be finite and > 0")

    copy_dir = Path(args.copy_dir)
    manifest = json.loads(Path(args.copy_manifest).read_text())
    times, self_f, self_t, self_sig = load_self_targets(copy_dir, manifest)
    raw_indices = fs.raw_time_to_frame_indices(Path(args.raw_topology), Path(args.raw_trr), times)
    data = load_cg_samples(Path(args.dataset), raw_indices, self_f, self_t)
    if data["period"] != int(manifest["residues_per_copy"]) or data["copies"] != int(manifest["copies"]):
        raise RuntimeError("CG dataset repeated-copy topology does not match self-rerun manifest")
    if [x[0] for x in self_sig] != list(data["labels"]):
        raise RuntimeError("single-copy rerun residue order does not match CG TEL22 order")

    orientations, orient_local, orient_labels = load_da_dt_orientations(
        Path(args.raw_topology), Path(args.raw_trr), raw_indices,
        data["period"], data["copies"], data["labels"], data["rotations"]
    )
    orient_flat = orientations.reshape(len(orientations), -1).astype(np.float32)
    base = data["descriptors"]

    variants: Dict[str, np.ndarray] = {"cg_only": base}
    for scale in scales:
        tag = f"cg_plus_da_dt_orientation_{scale:.3f}nm".replace(".", "p")
        variants[tag] = np.concatenate([base, np.float32(scale) * orient_flat], axis=1)

    variant_reports = {}
    pair_cache = {}
    for name, descriptor in variants.items():
        rep, arr = analyze_variant(
            name, descriptor, base, orientations,
            data["forces"], data["torques"], data["labels"],
            data["copy_ids"], data["frame_ids"], args.same_copy_gap_frames, args.seed,
        )
        variant_reports[name] = rep
        pair_cache[name] = arr

    primary = "nearest_same_copy_gap"
    baseline = variant_reports["cg_only"][primary]
    comparisons = {}
    for name in variants:
        if name == "cg_only":
            continue
        cur = variant_reports[name][primary]
        bnear = baseline["nearest"]
        cnear = cur["nearest"]
        type_cmp = {}
        for lab in ("DA", "DT"):
            bt = baseline["nearest_type_targets"].get(lab, {})
            ct = cur["nearest_type_targets"].get(lab, {})
            type_cmp[lab] = {
                "force_half_mse_ratio_vs_cg_only": ratio_or_nan(
                    ct.get("force_half_pair_difference_mse_fraction_of_target_mse", math.nan),
                    bt.get("force_half_pair_difference_mse_fraction_of_target_mse", math.nan),
                ),
                "torque_half_mse_ratio_vs_cg_only": ratio_or_nan(
                    ct.get("torque_half_pair_difference_mse_fraction_of_target_mse", math.nan),
                    bt.get("torque_half_pair_difference_mse_fraction_of_target_mse", math.nan),
                ),
            }
        comparisons[name] = {
            "force_half_mse_ratio_vs_cg_only": ratio_or_nan(
                cnear.get("force_half_pair_difference_mse_fraction_of_target_mse", math.nan),
                bnear.get("force_half_pair_difference_mse_fraction_of_target_mse", math.nan),
            ),
            "torque_all_residues_half_mse_ratio_vs_cg_only": ratio_or_nan(
                cnear.get("torque_half_pair_difference_mse_fraction_of_target_mse", math.nan),
                bnear.get("torque_half_pair_difference_mse_fraction_of_target_mse", math.nan),
            ),
            "by_residue_type": type_cmp,
        }

    report = {
        "definition": {
            "purpose": "diagnostic test of whether atomistic DA/DT base orientation hidden by the current single-COM mapping explains TEL22 self-force conditional noise",
            "baseline_descriptor": "current retained CG copy geometry, centered and Kabsch-aligned exactly as in prior conditional-noise diagnostics",
            "augmented_descriptor": "baseline CG geometry plus a rigid DA/DT orientation frame extracted from atomistic base anchors and rotated by the same Kabsch transform",
            "orientation_only_guardrail": "anchor distances are discarded by normalization and Gram-Schmidt orthogonalization; the added feature contains orientation but not base bond lengths or other internal deformation",
            "virtual_marker_interpretation": "each 3x3 frame is equivalent to three orientation markers at COM + scale_nm * axis; multiple scales are reported because nearest-neighbor ranking depends on the relative geometry/orientation metric",
            "target": "single-copy GROMACS self generalized force and torque; no water, ions, or other TEL22 copies",
            "torque_metric": "for this mapping ablation, torque is evaluated for all residue types, including DA/DT, because an orientation-aware rigid mapping could represent their rotational generalized force",
        },
        "anchors": {k: list(v) for k, v in ANCHORS.items()},
        "inputs": {
            "frames": int(len(raw_indices)),
            "raw_dataset_frame_indices": [int(x) for x in raw_indices],
            "times_ps": [float(x) for x in times],
            "copies_per_frame": int(data["copies"]),
            "residues_per_copy": int(data["period"]),
            "current_cg_sites_per_copy": int(data["sites_per_copy"]),
            "oriented_residue_local_indices": [int(x) for x in orient_local],
            "oriented_residue_labels": orient_labels,
            "same_copy_min_gap_frames": int(args.same_copy_gap_frames),
            "orientation_scales_nm": scales,
            "seed": int(args.seed),
        },
        "target_scale": {
            "self_force_component_rms_kj_mol_nm": cn.rms_components(data["forces"]),
            "self_torque_all_residues_component_rms_kj_mol": cn.rms_components(data["torques"]),
        },
        "descriptor_variants": variant_reports,
        "primary_same_copy_comparison": comparisons,
        "interpretation": {
            "strong_orientation_signal": "If an augmented descriptor drives the same-copy nearest half-MSE ratios far below cg_only, DA/DT orientation lost by the current COM mapping is a major source of conditional noise.",
            "weak_orientation_signal": "If all scales remain close to cg_only, explicit DA/DT base orientation alone cannot explain most of the ~0.8 self-force floor; test additional retained DOFs/mapping changes next.",
            "do_not_overinterpret_scale": "Choose conclusions that are stable across a reasonable orientation-scale range; a result present at only one extreme scale can be a nearest-neighbor metric artifact.",
        },
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    rows = []
    for variant, by_pair in pair_cache.items():
        for pair_name, (near_pairs, random_pairs, near_arr, rand_arr) in by_pair.items():
            for control, pairs, arr in (("nearest", near_pairs, near_arr), ("random", random_pairs, rand_arr)):
                if len(pairs) == 0:
                    continue
                angles = orientation_pair_angles_deg(orientations, pairs)
                base_rms = np.sqrt(np.mean(np.square(base[pairs[:, 0]] - base[pairs[:, 1]], dtype=np.float64), axis=1))
                for k, (i, j) in enumerate(pairs):
                    rows.append({
                        "variant": variant,
                        "pair_set": pair_name,
                        "control": control,
                        "sample_i": int(i),
                        "sample_j": int(j),
                        "frame_i": int(data["frame_ids"][i]),
                        "frame_j": int(data["frame_ids"][j]),
                        "copy_i": int(data["copy_ids"][i]),
                        "copy_j": int(data["copy_ids"][j]),
                        "current_cg_rmsd_nm": float(base_rms[k]),
                        "da_dt_orientation_rms_angle_deg": float(angles[k]),
                        "force_pair_rms": float(arr["force_pair"][k]),
                        "torque_pair_rms": float(arr["torque_pair"][k]),
                    })
    fields = [
        "variant", "pair_set", "control", "sample_i", "sample_j", "frame_i", "frame_j",
        "copy_i", "copy_j", "current_cg_rmsd_nm", "da_dt_orientation_rms_angle_deg",
        "force_pair_rms", "torque_pair_rms",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("======================================================")
    print(" TEL22 DA/DT ORIENTATION CONDITIONAL-NOISE ABLATION")
    print("======================================================")
    print(f"frames={len(raw_indices)} copies/frame={data['copies']} samples={len(base)}")
    print(f"DA/DT oriented residues/copy={len(orient_local)} | anchors={ANCHORS}")
    print("Primary metric: self / nearest_same_copy_gap")
    for name in variants:
        item = variant_reports[name][primary]
        near = item["nearest"]
        print(
            f"  {name:38s} pairs={near.get('pairs', 0):5d} "
            f"Fhalf={near.get('force_half_pair_difference_mse_fraction_of_target_mse', math.nan):.4f} "
            f"Thalf(all)={near.get('torque_half_pair_difference_mse_fraction_of_target_mse', math.nan):.4f} "
            f"Fnear/random={item.get('nearest_vs_random_force_half_mse_ratio', math.nan):.4f} "
            f"orientP50={item.get('nearest_da_dt_orientation_rms_angle_deg', {}).get('p50', math.nan):.2f} deg"
        )
        if name != "cg_only":
            cmp = comparisons[name]
            print(
                f"    vs CG-only: Fhalf x{cmp['force_half_mse_ratio_vs_cg_only']:.4f} | "
                f"Tall x{cmp['torque_all_residues_half_mse_ratio_vs_cg_only']:.4f} | "
                f"DA F x{cmp['by_residue_type']['DA']['force_half_mse_ratio_vs_cg_only']:.4f} | "
                f"DT F x{cmp['by_residue_type']['DT']['force_half_mse_ratio_vs_cg_only']:.4f}"
            )
    print(f"JSON: {out_json}")
    print(f"CSV:  {out_csv}")


if __name__ == "__main__":
    main()
