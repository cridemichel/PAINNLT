#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import MDAnalysis as mda
import numpy as np

import analyze_dna_self_vs_intercopy as dsi
import analyze_force_source_decomposition as fs
import analyze_temporal_force_averaging as tfa
import build_dna_self_isolated_dataset as bsi

DNA_RESNAMES = set(fs.DNA_RESNAMES)
ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "P": 30.974,
    "S": 32.065,
    "K": 39.098,
}


def fallback_mass(atom_name: str) -> float:
    letters = "".join(c for c in str(atom_name) if c.isalpha()).upper()
    return float(ATOMIC_MASSES.get(letters[:1], 12.0)) if letters else 12.0


def topology_signature(u: mda.Universe) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    residues = [r for r in u.residues if str(r.resname) in DNA_RESNAMES]
    return tuple((str(r.resname), tuple(str(x) for x in r.atoms.names)) for r in residues)


def residue_metadata(topology: Path, expected_period: int, mapping_config: Path):
    u = mda.Universe(str(topology))
    residues = [r for r in u.residues if str(r.resname) in DNA_RESNAMES]
    if len(residues) != expected_period:
        raise RuntimeError(f"{topology}: DNA residue count {len(residues)} != {expected_period}")
    cfg = json.loads(mapping_config.read_text(encoding="utf-8"))
    mapping = cfg.get("mapping", {})
    if mapping.get("mapping_method", "COM") != "COM":
        raise RuntimeError("03r currently requires COM site mapping")
    mapping_by_resname = mapping.get("residues", {})

    natoms = len(u.atoms)
    masses = np.empty(natoms, dtype=np.float64)
    try:
        guessed = np.asarray(u.atoms.masses, dtype=np.float64)
    except Exception:
        guessed = np.full(natoms, np.nan, dtype=np.float64)
    for i, atom in enumerate(u.atoms):
        m = guessed[i] if i < len(guessed) else math.nan
        masses[i] = float(m) if np.isfinite(m) and m > 0 else fallback_mass(atom.name)

    infos = []
    current = np.zeros((expected_period, natoms), dtype=np.float64)
    com_coord = np.zeros_like(current)
    for ri, res in enumerate(residues):
        inds = np.asarray(res.atoms.indices, dtype=np.int64)
        rm = masses[inds]
        current[ri, inds] = 1.0
        com_coord[ri, inds] = rm / float(np.sum(rm))
        resmap = mapping_by_resname.get(str(res.resname))
        if not isinstance(resmap, dict) or not resmap:
            raise RuntimeError(f"mapping config lacks residue mapping for {res.resname}")
        site_items = list(resmap.items())
        is_single_com = len(site_items) == 1 and site_items[0][1] == ["*"]
        infos.append(
            {
                "local_index": ri,
                "resname": str(res.resname),
                "atom_indices": inds,
                "atom_names": [str(x) for x in res.atoms.names],
                "masses": rm.copy(),
                "single_site_exact_com": bool(is_single_com),
                "mapped_site_names": [str(k) for k, _ in site_items],
            }
        )
    if np.max(np.abs(current @ com_coord.T - np.eye(expected_period))) > 1.0e-12:
        raise RuntimeError("current residue-sum force map is not compatible with per-residue COM translations")
    return masses, current, com_coord, infos, topology_signature(u)


def unwrap_sequential(pos_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    pos_nm = np.asarray(pos_nm, dtype=np.float64)
    box_nm = np.asarray(box_nm, dtype=np.float64)
    if pos_nm.ndim != 2 or pos_nm.shape[1] != 3:
        raise ValueError(f"bad coordinate shape {pos_nm.shape}")
    out = np.empty_like(pos_nm)
    out[0] = pos_nm[0]
    for i in range(1, len(pos_nm)):
        d = pos_nm[i] - pos_nm[i - 1]
        d -= box_nm * np.round(d / box_nm)
        out[i] = out[i - 1] + d
    return out


def load_atomistic_frames(topology: Path, trr: Path, frame_positions: Sequence[int]):
    u = mda.Universe(str(topology), str(trr))
    frame_positions = np.asarray(frame_positions, dtype=np.int64)
    coords = np.empty((len(frame_positions), len(u.atoms), 3), dtype=np.float64)
    forces = np.empty_like(coords)
    times = np.empty(len(frame_positions), dtype=np.float64)
    for oi, fi in enumerate(frame_positions):
        ts = u.trajectory[int(fi)]
        if getattr(ts, "has_forces", None) is False:
            raise RuntimeError(f"trajectory lacks forces: {trr}")
        box_nm = np.asarray(ts.dimensions[:3], dtype=np.float64) / 10.0
        if np.any(~np.isfinite(box_nm)) or np.any(box_nm <= 0.0):
            raise RuntimeError(f"invalid periodic box in {trr}, frame {fi}")
        coords[oi] = unwrap_sequential(np.asarray(u.atoms.positions, dtype=np.float64) / 10.0, box_nm)
        # Same conversion used by preprocessing/build_cg_dataset.py for GROMACS TRR forces.
        forces[oi] = np.asarray(u.atoms.forces, dtype=np.float64) * 10.0
        times[oi] = float(ts.time)
    return coords, forces, times, topology_signature(u)


def deterministic_fit_samples(train_center_positions: np.ndarray, copies: int, max_samples: int):
    pairs = [(int(ti), ci) for ti in train_center_positions for ci in range(copies)]
    if max_samples <= 0 or max_samples >= len(pairs):
        return pairs
    sel = np.rint(np.linspace(0, len(pairs) - 1, max_samples)).astype(np.int64)
    if len(np.unique(sel)) != len(sel):
        raise RuntimeError("fit-sample linspace produced duplicate indices")
    return [pairs[int(i)] for i in sel]


def map_stats(row: np.ndarray, com_row: np.ndarray):
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    com_row = np.asarray(com_row, dtype=np.float64).reshape(-1)
    return {
        "compatibility_B_Ct_abs_error": float(abs(float(row @ com_row) - 1.0)),
        "coefficient_min": float(np.min(row)),
        "coefficient_max": float(np.max(row)),
        "coefficient_l2": float(np.linalg.norm(row)),
        "nonzero_coefficients": int(np.count_nonzero(np.abs(row) > 1.0e-10)),
    }


def read_pair_ids(path: Path, target_window_ps: float = 1.0):
    out = {"nearest": [], "random": []}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not math.isclose(float(row["window_ps"]), target_window_ps, abs_tol=1.0e-12, rel_tol=0.0):
                continue
            if row.get("pair_set") != "nearest_same_copy_gap":
                continue
            control = row.get("control")
            if control in out:
                out[control].append((int(row["sample_i"]), int(row["sample_j"])))
    result = {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}
    if any(len(v) == 0 for v in result.values()):
        raise RuntimeError(f"could not recover 1-ps nearest/random pair identities from {path}")
    return result


def half_mse_proxy(force_aligned: np.ndarray, pairs: np.ndarray, residue_indices: Sequence[int] | None = None):
    force_aligned = np.asarray(force_aligned, dtype=np.float64)
    if residue_indices is not None:
        force_aligned = force_aligned[:, np.asarray(residue_indices, dtype=np.int64), :]
    rms2 = float(np.mean(np.square(force_aligned, dtype=np.float64)))
    if rms2 <= 0.0:
        raise RuntimeError("non-positive target force variance")
    pairs = np.asarray(pairs, dtype=np.int64)
    diff = force_aligned[pairs[:, 0]] - force_aligned[pairs[:, 1]]
    half = 0.5 * float(np.mean(np.square(diff, dtype=np.float64)))
    return half / rms2, half, math.sqrt(rms2)


def aggforce_version() -> str:
    try:
        return importlib.metadata.version("aggforce")
    except importlib.metadata.PackageNotFoundError:
        return "unknown/source-install"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optimize TEL22 single-site DA/DT instantaneous self-force aggregation with aggforce"
    )
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--mapping-config", type=Path, required=True)
    ap.add_argument("--raw-topology", type=Path, required=True)
    ap.add_argument("--raw-trr", type=Path, required=True)
    ap.add_argument("--copy-dir", type=Path, required=True)
    ap.add_argument("--copy-manifest", type=Path, required=True)
    ap.add_argument("--temporal-report", type=Path, required=True)
    ap.add_argument("--temporal-pairs", type=Path, required=True)
    ap.add_argument("--validation-stride", type=int, default=5)
    ap.add_argument("--constraint-threshold", type=float, default=1.0e-3)
    ap.add_argument("--constraint-frames", type=int, default=100)
    ap.add_argument("--fit-max-samples", type=int, default=3000)
    ap.add_argument("--l2-regularization", type=float, default=1.0e3)
    ap.add_argument("--output-cache", type=Path, required=True)
    ap.add_argument("--output-maps", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    args = ap.parse_args()

    if args.validation_stride < 2:
        raise SystemExit("validation-stride must be >= 2")
    if args.constraint_threshold <= 0.0 or args.constraint_frames <= 0:
        raise SystemExit("constraint threshold/frames must be > 0")
    if args.fit_max_samples == 0:
        raise SystemExit("fit-max-samples must be positive or negative for all training samples")
    if args.l2_regularization < 0.0:
        raise SystemExit("l2-regularization must be >= 0")
    for p in (
        args.dataset,
        args.mapping_config,
        args.raw_topology,
        args.raw_trr,
        args.copy_manifest,
        args.temporal_report,
        args.temporal_pairs,
    ):
        if not p.exists():
            raise SystemExit(f"required file not found: {p}")

    try:
        from aggforce import (
            LinearMap,
            constraint_aware_uni_map,
            guess_pairwise_constraints,
            project_forces,
            qp_linear_map,
        )
    except Exception as exc:
        raise SystemExit(
            "aggforce import failed. Install the official package first, e.g.\n"
            "  python3 -m pip install 'git+https://github.com/noegroup/aggforce.git'\n"
            f"original import error: {exc}"
        )

    manifest = json.loads(args.copy_manifest.read_text(encoding="utf-8"))
    copies = int(manifest["copies"])
    period = int(manifest["residues_per_copy"])
    for ci in range(copies):
        for suffix in (".gro", "_rerun.trr"):
            p = args.copy_dir / f"copy_{ci:02d}{suffix}"
            if not p.exists():
                raise SystemExit(f"missing full single-copy rerun input: {p}")

    times, _f0, _t0, sig0 = dsi.load_targets(
        args.copy_dir / "copy_00.gro", args.copy_dir / "copy_00_rerun.trr"
    )
    times = np.asarray(times, dtype=np.float64)
    self_f, self_t, self_sig = dsi.load_self_targets(args.copy_dir, manifest, times)
    if sig0 != self_sig or len(self_sig) != period:
        raise RuntimeError("single-copy residue signature mismatch")
    self_f4 = np.asarray(self_f, dtype=np.float64).reshape(len(times), copies, period, 3)
    self_t4 = np.asarray(self_t, dtype=np.float64).reshape(len(times), copies, period, 3)

    raw_indices = np.asarray(
        fs.raw_time_to_frame_indices(args.raw_topology, args.raw_trr, times), dtype=np.int64
    )
    if len(np.unique(raw_indices)) != len(raw_indices):
        raise RuntimeError("raw time mapping produced duplicate dataset frame indices")

    temporal = json.loads(args.temporal_report.read_text(encoding="utf-8"))
    widths = [float(x) for x in temporal["inputs"]["windows_ps"]]
    edges = tfa.sample_cell_edges(times)
    centers = tfa.common_centers(times, edges, widths)
    if len(centers) != int(temporal["inputs"]["common_center_frames"]):
        raise RuntimeError("common-center count differs from 03p temporal diagnostic")
    if not math.isclose(
        float(times[centers[0]]),
        float(temporal["inputs"]["common_center_time_start_ps"]),
        abs_tol=1e-8,
        rel_tol=0.0,
    ):
        raise RuntimeError("common-center start time differs from 03p")
    if not math.isclose(
        float(times[centers[-1]]),
        float(temporal["inputs"]["common_center_time_end_ps"]),
        abs_tol=1e-8,
        rel_tol=0.0,
    ):
        raise RuntimeError("common-center end time differs from 03p")

    train_pos, val_pos, dataset_order = bsi.stratified_tail_order(len(centers), args.validation_stride)
    fit_pairs = deterministic_fit_samples(train_pos, copies, args.fit_max_samples)
    fit_by_copy: Dict[int, List[int]] = {ci: [] for ci in range(copies)}
    for center_pos, ci in fit_pairs:
        fit_by_copy[ci].append(int(centers[center_pos]))

    masses, current_matrix, com_matrix, residue_infos, sig_top = residue_metadata(
        args.copy_dir / "copy_00.gro", period, args.mapping_config
    )
    natoms = current_matrix.shape[1]
    eligible = [int(x["local_index"]) for x in residue_infos if x["single_site_exact_com"]]
    if not eligible:
        raise RuntimeError("no exact single-site COM residues found; 03r has nothing safe to optimize")
    noneligible = [i for i in range(period) if i not in set(eligible)]
    eligible_labels = [str(residue_infos[i]["resname"]) for i in eligible]

    # Assemble atomistic fit snapshots strictly from TRAIN center positions.  The same
    # fit sample list is used for every local DA/DT residue, so differences between
    # residue maps are not a data-split artifact.
    fit_coords_parts = []
    fit_forces_parts = []
    for ci in range(copies):
        positions = fit_by_copy[ci]
        if not positions:
            continue
        c, f, _t, sig = load_atomistic_frames(
            args.copy_dir / f"copy_{ci:02d}.gro",
            args.copy_dir / f"copy_{ci:02d}_rerun.trr",
            positions,
        )
        if sig != sig_top or c.shape[1] != natoms:
            raise RuntimeError(f"copy_{ci:02d}: atom topology/order differs from copy_00")
        fit_coords_parts.append(c)
        fit_forces_parts.append(f)
    fit_coords = np.concatenate(fit_coords_parts, axis=0)
    fit_forces = np.concatenate(fit_forces_parts, axis=0)
    if len(fit_coords) != len(fit_pairs):
        raise RuntimeError("fit sample assembly count mismatch")

    # Start from the existing exact total-force estimator for every residue.  Only rows
    # whose *actual model mapping* is one exact mass-weighted COM (DA/DT here) are
    # replaced.  Multi-site DG rows stay untouched because their rigid orientation map
    # is nonlinear; applying a 22-COM aggforce QP to them would optimize the wrong PMF.
    basic_matrix = current_matrix.copy()
    optim_matrix = current_matrix.copy()
    per_residue_maps = []
    for ri in eligible:
        info = residue_infos[ri]
        inds = np.asarray(info["atom_indices"], dtype=np.int64)
        local_coords = fit_coords[:, inds, :]
        local_forces = fit_forces[:, inds, :]
        local_com = np.asarray(info["masses"], dtype=np.float64)
        local_com /= float(np.sum(local_com))
        coord_map = LinearMap(local_com.reshape(1, -1))
        n_constraint = min(int(args.constraint_frames), len(local_coords))
        constraints = guess_pairwise_constraints(
            local_coords[:n_constraint], threshold=float(args.constraint_threshold)
        )
        print(
            f"[AGGFORCE] residue {ri:02d} {info['resname']}: atoms={len(inds)} "
            f"constraints={len(constraints)}"
        )
        basic_results = project_forces(
            coords=local_coords,
            forces=local_forces,
            coord_map=coord_map,
            constrained_inds=constraints,
            method=constraint_aware_uni_map,
        )
        optim_results = project_forces(
            coords=local_coords,
            forces=local_forces,
            coord_map=coord_map,
            constrained_inds=constraints,
            method=qp_linear_map,
            l2_regularization=float(args.l2_regularization),
        )
        brow = np.asarray(basic_results["tmap"].force_map.standard_matrix, dtype=np.float64).reshape(-1)
        orow = np.asarray(optim_results["tmap"].force_map.standard_matrix, dtype=np.float64).reshape(-1)
        if brow.shape != (len(inds),) or orow.shape != (len(inds),):
            raise RuntimeError(f"residue {ri}: unexpected aggforce map shape")
        for name, row in (("constraint_aware", brow), ("optimized", orow)):
            err = abs(float(row @ local_com) - 1.0)
            if err > 5.0e-5:
                raise RuntimeError(f"residue {ri} {name}: B C^T != 1 (error {err:g})")
        basic_matrix[ri, inds] = brow
        optim_matrix[ri, inds] = orow
        constraint_array = np.asarray(sorted(tuple(map(int, p)) for p in constraints), dtype=np.int32)
        if constraint_array.size == 0:
            constraint_array = np.empty((0, 2), dtype=np.int32)
        per_residue_maps.append(
            {
                "local_index": ri,
                "resname": str(info["resname"]),
                "atoms": len(inds),
                "atom_names": list(info["atom_names"]),
                "fit_samples": int(len(local_coords)),
                "constraint_detection_frames": int(n_constraint),
                "constraints_detected": int(len(constraints)),
                "constraint_pairs_local": constraint_array.tolist(),
                "current": map_stats(np.ones(len(inds)), local_com),
                "constraint_aware": map_stats(brow, local_com),
                "optimized": map_stats(orow, local_com),
            }
        )

    # Because each optimized row has support only inside its own DA/DT residue, it is
    # automatically orthogonal to every other model coordinate.  Combined with B C^T=1
    # for that residue's exact COM, this preserves the full mapping consistency for the
    # rows we change.  DG is left exactly at the original atom-sum target.
    for name, mat in (("constraint_aware", basic_matrix), ("optimized", optim_matrix)):
        if np.max(np.abs(mat[noneligible] - current_matrix[noneligible])) > 0.0:
            raise RuntimeError(f"{name}: non-single-site residue force rows changed unexpectedly")
        for ri in eligible:
            inds = np.asarray(residue_infos[ri]["atom_indices"], dtype=np.int64)
            outside = np.ones(natoms, dtype=bool)
            outside[inds] = False
            if np.max(np.abs(mat[ri, outside])) > 0.0:
                raise RuntimeError(f"{name}: residue {ri} has coefficients outside its own atoms")
            if abs(float(mat[ri] @ com_matrix[ri]) - 1.0) > 5.0e-5:
                raise RuntimeError(f"{name}: residue {ri} translational compatibility failed")

    mapped_current = np.empty_like(self_f4)
    mapped_basic = np.empty_like(self_f4)
    mapped_optim = np.empty_like(self_f4)
    all_positions = np.arange(len(times), dtype=np.int64)
    atomistic_current_guard = 0.0
    for ci in range(copies):
        _c, f, trr_times, sig = load_atomistic_frames(
            args.copy_dir / f"copy_{ci:02d}.gro",
            args.copy_dir / f"copy_{ci:02d}_rerun.trr",
            all_positions,
        )
        if sig != sig_top:
            raise RuntimeError(f"copy_{ci:02d}: topology/order changed")
        if len(trr_times) != len(times) or not np.allclose(trr_times, times, atol=1e-4, rtol=0.0):
            raise RuntimeError(f"copy_{ci:02d}: trajectory times differ")
        mapped_current[:, ci] = np.einsum("mn,tnd->tmd", current_matrix, f, optimize=True)
        mapped_basic[:, ci] = np.einsum("mn,tnd->tmd", basic_matrix, f, optimize=True)
        mapped_optim[:, ci] = np.einsum("mn,tnd->tmd", optim_matrix, f, optimize=True)
        atomistic_current_guard = max(
            atomistic_current_guard,
            float(np.max(np.abs(mapped_current[:, ci] - self_f4[:, ci]))),
        )
    denom = max(float(np.max(np.abs(self_f4))), 1.0)
    current_guard_rel = atomistic_current_guard / denom
    if current_guard_rel > 2.0e-6:
        raise RuntimeError(
            f"per-atom residue sum does not reproduce existing self target: rel max {current_guard_rel:g}"
        )

    full = dsi.geometry_with_targets(
        args.dataset,
        raw_indices,
        {
            "current": (
                mapped_current.reshape(len(times), copies * period, 3),
                self_t4.reshape(len(times), copies * period, 3),
            ),
            "constraint_aware": (
                mapped_basic.reshape(len(times), copies * period, 3),
                self_t4.reshape(len(times), copies * period, 3),
            ),
            "optimized": (
                mapped_optim.reshape(len(times), copies * period, 3),
                self_t4.reshape(len(times), copies * period, 3),
            ),
        },
    )
    pairs = read_pair_ids(args.temporal_pairs, 1.0)
    nsamples = len(centers) * copies
    for arr in pairs.values():
        if int(np.max(arr)) >= nsamples:
            raise RuntimeError("03p pair indices exceed common-center sample pool")

    by_type = {}
    for label in sorted(set(str(x["resname"]) for x in residue_infos)):
        by_type[label] = [int(x["local_index"]) for x in residue_infos if str(x["resname"]) == label]
    groups = {"all": list(range(period)), "optimized_single_site": eligible, **by_type}

    diagnostics = {}
    for name in ("current", "constraint_aware", "optimized"):
        aligned = np.asarray(full[name]["forces"], dtype=np.float64).reshape(len(times), copies, period, 3)
        aligned_center = aligned[centers].reshape(nsamples, period, 3)
        group_diag = {}
        for group, inds in groups.items():
            near_frac, near_abs, target_rms = half_mse_proxy(aligned_center, pairs["nearest"], inds)
            rand_frac, rand_abs, _ = half_mse_proxy(aligned_center, pairs["random"], inds)
            group_diag[group] = {
                "residue_local_indices": [int(x) for x in inds],
                "force_component_rms_kj_mol_nm": target_rms,
                "nearest_force_half_pair_difference_mse_fraction_of_target_mse": near_frac,
                "random_force_half_pair_difference_mse_fraction_of_target_mse": rand_frac,
                "nearest_vs_random_force_half_mse_ratio": near_frac / rand_frac,
                "nearest_absolute_half_pair_mse_kj2_mol2_nm2": near_abs,
                "random_absolute_half_pair_mse_kj2_mol2_nm2": rand_abs,
            }
        diagnostics[name] = {
            "force_component_rms_kj_mol_nm": group_diag["all"]["force_component_rms_kj_mol_nm"],
            "nearest_force_half_pair_difference_mse_fraction_of_target_mse": group_diag["all"]["nearest_force_half_pair_difference_mse_fraction_of_target_mse"],
            "random_force_half_pair_difference_mse_fraction_of_target_mse": group_diag["all"]["random_force_half_pair_difference_mse_fraction_of_target_mse"],
            "nearest_vs_random_force_half_mse_ratio": group_diag["all"]["nearest_vs_random_force_half_mse_ratio"],
            "nearest_absolute_half_pair_mse_kj2_mol2_nm2": group_diag["all"]["nearest_absolute_half_pair_mse_kj2_mol2_nm2"],
            "random_absolute_half_pair_mse_kj2_mol2_nm2": group_diag["all"]["random_absolute_half_pair_mse_kj2_mol2_nm2"],
            "groups": group_diag,
        }

    reference_proxy = float(
        temporal["windows"]["1ps"]["nearest_same_copy_gap"][
            "force_half_pair_difference_mse_fraction_of_target_mse"
        ]
    )
    if abs(
        diagnostics["current"]["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
        - reference_proxy
    ) > 2.0e-5:
        raise RuntimeError(
            "current-map nearest proxy does not reproduce 03p 1-ps baseline; pair ordering/rotation mismatch"
        )
    for name in ("constraint_aware", "optimized"):
        diagnostics[name]["normalized_noise_proxy_ratio_vs_current"] = (
            diagnostics[name]["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
            / diagnostics["current"]["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
        )
        diagnostics[name]["absolute_half_pair_mse_ratio_vs_current"] = (
            diagnostics[name]["nearest_absolute_half_pair_mse_kj2_mol2_nm2"]
            / diagnostics["current"]["nearest_absolute_half_pair_mse_kj2_mol2_nm2"]
        )
        for group in groups:
            d = diagnostics[name]["groups"][group]
            b = diagnostics["current"]["groups"][group]
            d["normalized_noise_proxy_ratio_vs_current"] = (
                d["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
                / b["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
            )
            d["absolute_half_pair_mse_ratio_vs_current"] = (
                d["nearest_absolute_half_pair_mse_kj2_mol2_nm2"]
                / b["nearest_absolute_half_pair_mse_kj2_mol2_nm2"]
            )

    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_cache,
        times_ps=times.astype(np.float64),
        raw_dataset_indices=raw_indices.astype(np.int32),
        center_rerun_positions=centers.astype(np.int32),
        center_raw_dataset_indices=raw_indices[centers].astype(np.int32),
        center_times_ps=times[centers].astype(np.float64),
        train_center_positions=train_pos.astype(np.int32),
        validation_center_positions=val_pos.astype(np.int32),
        dataset_center_order=dataset_order.astype(np.int32),
        force_current=mapped_current.astype(np.float32),
        force_constraint_aware=mapped_basic.astype(np.float32),
        force_optimized=mapped_optim.astype(np.float32),
        torque_current=self_t4.astype(np.float32),
    )
    args.output_maps.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_maps,
        per_residue_com_coordinate_map=com_matrix.astype(np.float64),
        current_residue_sum_force_map=current_matrix.astype(np.float64),
        constraint_aware_single_site_force_map=basic_matrix.astype(np.float64),
        optimized_single_site_force_map=optim_matrix.astype(np.float64),
        optimized_residue_local_indices=np.asarray(eligible, dtype=np.int32),
        atom_masses_amu=masses.astype(np.float64),
    )

    report = {
        "definition": {
            "purpose": "TEL22 self-force variance reduction with statistically optimized instantaneous force aggregation",
            "scope": "only residues whose actual retained mapping is one exact mass-weighted COM site are optimized (DA/DT); multi-site rigid DG remains at the original atom-sum force target",
            "why_partial": "aggforce linear-map consistency is exact for DA/DT COM coordinates. DG orientation/site reconstruction is a nonlinear rigid-body map, so optimizing it against a fictitious residue-COM map would target the wrong coarse variables",
            "current_force_map": "per-residue algebraic sum of atomistic instantaneous forces",
            "constraint_aware_map": "aggforce constraint_aware_uni_map on each eligible residue independently; DG unchanged",
            "optimized_force_map": "aggforce qp_linear_map on each eligible residue independently with fixed configuration-independent coefficients; DG unchanged",
            "thermodynamic_guardrail": "each changed force row has support only on atoms of its own exact single-site COM residue and satisfies B C^T = 1; all other model coordinates use disjoint atoms, so the changed row is orthogonal to them",
            "validation_guardrail": "force-map optimization is fit only on training-center configurations; validation target frames are excluded from optimization",
            "torque_guardrail": "torque mapping is unchanged; production torque loss already masks one-site DA/DT, so 03r can retain the same torque_weight=0.5 as the 03q baseline",
            "pair_guardrail": "conditional-noise proxies reuse the exact 1-ps nearest/random pair identities from 03p and the same Kabsch-aligned force convention",
        },
        "software": {
            "aggforce_version": aggforce_version(),
            "aggforce_repository": "https://github.com/noegroup/aggforce",
            "method": "qp_linear_map",
            "l2_regularization": float(args.l2_regularization),
        },
        "inputs": {
            "frames_per_copy": int(len(times)),
            "common_center_frames": int(len(centers)),
            "train_center_frames": int(len(train_pos)),
            "validation_center_frames": int(len(val_pos)),
            "copies": copies,
            "residues_per_copy": period,
            "atoms_per_copy": int(natoms),
            "fit_samples_available_train_only": int(len(train_pos) * copies),
            "fit_samples_used_per_optimized_residue": int(len(fit_pairs)),
            "fit_max_samples_requested": int(args.fit_max_samples),
            "validation_stride": int(args.validation_stride),
            "constraint_threshold": float(args.constraint_threshold),
            "common_center_time_start_ps": float(times[centers[0]]),
            "common_center_time_end_ps": float(times[centers[-1]]),
        },
        "mapping_scope": {
            "optimized_residue_local_indices": eligible,
            "optimized_residue_labels": eligible_labels,
            "unchanged_residue_local_indices": noneligible,
            "unchanged_residue_labels": [str(residue_infos[i]["resname"]) for i in noneligible],
            "per_residue_maps": per_residue_maps,
        },
        "mapping_guardrails": {
            "current_atom_sum_reproduces_existing_self_target_max_abs_kj_mol_nm": float(atomistic_current_guard),
            "current_atom_sum_reproduces_existing_self_target_relative_to_global_max": float(current_guard_rel),
            "noneligible_rows_exactly_unchanged": True,
            "optimized_rows_have_own_residue_support_only": True,
        },
        "force_noise_diagnostic": diagnostics,
        "comparison": {
            "current_03p_force_noise_proxy": reference_proxy,
            "constraint_aware_global_noise_proxy_ratio_vs_current": diagnostics["constraint_aware"]["normalized_noise_proxy_ratio_vs_current"],
            "optimized_global_noise_proxy_ratio_vs_current": diagnostics["optimized"]["normalized_noise_proxy_ratio_vs_current"],
            "constraint_aware_single_site_noise_proxy_ratio_vs_current": diagnostics["constraint_aware"]["groups"]["optimized_single_site"]["normalized_noise_proxy_ratio_vs_current"],
            "optimized_single_site_noise_proxy_ratio_vs_current": diagnostics["optimized"]["groups"]["optimized_single_site"]["normalized_noise_proxy_ratio_vs_current"],
            "optimized_global_absolute_half_pair_mse_ratio_vs_current": diagnostics["optimized"]["absolute_half_pair_mse_ratio_vs_current"],
            "optimized_single_site_absolute_half_pair_mse_ratio_vs_current": diagnostics["optimized"]["groups"]["optimized_single_site"]["absolute_half_pair_mse_ratio_vs_current"],
        },
        "outputs": {
            "target_cache": str(args.output_cache),
            "map_matrices": str(args.output_maps),
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    print("======================================================")
    print(" TEL22 AGGFORCE SINGLE-SITE SELF FORCE DIAGNOSTIC")
    print("======================================================")
    print(
        f"aggforce={report['software']['aggforce_version']} | atoms/copy={natoms} | "
        f"fit samples/residue={len(fit_pairs)} (training only)"
    )
    print(
        "optimized local residues: "
        + ", ".join(f"{i}:{residue_infos[i]['resname']}" for i in eligible)
    )
    print("variant             global_proxy xglobal   single_proxy xsingle")
    for name in ("current", "constraint_aware", "optimized"):
        d = diagnostics[name]
        s = d["groups"]["optimized_single_site"]
        xg = d["nearest_force_half_pair_difference_mse_fraction_of_target_mse"] / diagnostics["current"]["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
        xs = s["nearest_force_half_pair_difference_mse_fraction_of_target_mse"] / diagnostics["current"]["groups"]["optimized_single_site"]["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
        print(
            f"{name:18s} {d['nearest_force_half_pair_difference_mse_fraction_of_target_mse']:12.6f} "
            f"{xg:7.4f} {s['nearest_force_half_pair_difference_mse_fraction_of_target_mse']:12.6f} {xs:7.4f}"
        )
    print(f"cache:  {args.output_cache}")
    print(f"maps:   {args.output_maps}")
    print(f"report: {args.output_report}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import MDAnalysis as mda
import numpy as np

import analyze_dna_self_vs_intercopy as dsi
import analyze_force_source_decomposition as fs
import analyze_temporal_force_averaging as tfa
import build_dna_self_isolated_dataset as bsi

DNA_RESNAMES = set(fs.DNA_RESNAMES)
ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "P": 30.974,
    "S": 32.065,
    "K": 39.098,
}


def fallback_mass(atom_name: str) -> float:
    letters = "".join(c for c in str(atom_name) if c.isalpha()).upper()
    return float(ATOMIC_MASSES.get(letters[:1], 12.0)) if letters else 12.0


def topology_signature(u: mda.Universe) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    residues = [r for r in u.residues if str(r.resname) in DNA_RESNAMES]
    return tuple((str(r.resname), tuple(str(x) for x in r.atoms.names)) for r in residues)


def residue_metadata(topology: Path, expected_period: int, mapping_config: Path):
    u = mda.Universe(str(topology))
    residues = [r for r in u.residues if str(r.resname) in DNA_RESNAMES]
    if len(residues) != expected_period:
        raise RuntimeError(f"{topology}: DNA residue count {len(residues)} != {expected_period}")
    cfg = json.loads(mapping_config.read_text(encoding="utf-8"))
    mapping = cfg.get("mapping", {})
    if mapping.get("mapping_method", "COM") != "COM":
        raise RuntimeError("03r currently requires COM site mapping")
    mapping_by_resname = mapping.get("residues", {})

    natoms = len(u.atoms)
    masses = np.empty(natoms, dtype=np.float64)
    try:
        guessed = np.asarray(u.atoms.masses, dtype=np.float64)
    except Exception:
        guessed = np.full(natoms, np.nan, dtype=np.float64)
    for i, atom in enumerate(u.atoms):
        m = guessed[i] if i < len(guessed) else math.nan
        masses[i] = float(m) if np.isfinite(m) and m > 0 else fallback_mass(atom.name)

    infos = []
    current = np.zeros((expected_period, natoms), dtype=np.float64)
    com_coord = np.zeros_like(current)
    for ri, res in enumerate(residues):
        inds = np.asarray(res.atoms.indices, dtype=np.int64)
        rm = masses[inds]
        current[ri, inds] = 1.0
        com_coord[ri, inds] = rm / float(np.sum(rm))
        resmap = mapping_by_resname.get(str(res.resname))
        if not isinstance(resmap, dict) or not resmap:
            raise RuntimeError(f"mapping config lacks residue mapping for {res.resname}")
        site_items = list(resmap.items())
        is_single_com = len(site_items) == 1 and site_items[0][1] == ["*"]
        infos.append(
            {
                "local_index": ri,
                "resname": str(res.resname),
                "atom_indices": inds,
                "atom_names": [str(x) for x in res.atoms.names],
                "masses": rm.copy(),
                "single_site_exact_com": bool(is_single_com),
                "mapped_site_names": [str(k) for k, _ in site_items],
            }
        )
    if np.max(np.abs(current @ com_coord.T - np.eye(expected_period))) > 1.0e-12:
        raise RuntimeError("current residue-sum force map is not compatible with per-residue COM translations")
    return masses, current, com_coord, infos, topology_signature(u)


def unwrap_sequential(pos_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    pos_nm = np.asarray(pos_nm, dtype=np.float64)
    box_nm = np.asarray(box_nm, dtype=np.float64)
    if pos_nm.ndim != 2 or pos_nm.shape[1] != 3:
        raise ValueError(f"bad coordinate shape {pos_nm.shape}")
    out = np.empty_like(pos_nm)
    out[0] = pos_nm[0]
    for i in range(1, len(pos_nm)):
        d = pos_nm[i] - pos_nm[i - 1]
        d -= box_nm * np.round(d / box_nm)
        out[i] = out[i - 1] + d
    return out


def load_atomistic_frames(topology: Path, trr: Path, frame_positions: Sequence[int]):
    u = mda.Universe(str(topology), str(trr))
    frame_positions = np.asarray(frame_positions, dtype=np.int64)
    coords = np.empty((len(frame_positions), len(u.atoms), 3), dtype=np.float64)
    forces = np.empty_like(coords)
    times = np.empty(len(frame_positions), dtype=np.float64)
    for oi, fi in enumerate(frame_positions):
        ts = u.trajectory[int(fi)]
        if getattr(ts, "has_forces", None) is False:
            raise RuntimeError(f"trajectory lacks forces: {trr}")
        box_nm = np.asarray(ts.dimensions[:3], dtype=np.float64) / 10.0
        if np.any(~np.isfinite(box_nm)) or np.any(box_nm <= 0.0):
            raise RuntimeError(f"invalid periodic box in {trr}, frame {fi}")
        coords[oi] = unwrap_sequential(np.asarray(u.atoms.positions, dtype=np.float64) / 10.0, box_nm)
        # Same conversion used by preprocessing/build_cg_dataset.py for GROMACS TRR forces.
        forces[oi] = np.asarray(u.atoms.forces, dtype=np.float64) * 10.0
        times[oi] = float(ts.time)
    return coords, forces, times, topology_signature(u)


def deterministic_fit_samples(train_center_positions: np.ndarray, copies: int, max_samples: int):
    pairs = [(int(ti), ci) for ti in train_center_positions for ci in range(copies)]
    if max_samples <= 0 or max_samples >= len(pairs):
        return pairs
    sel = np.rint(np.linspace(0, len(pairs) - 1, max_samples)).astype(np.int64)
    if len(np.unique(sel)) != len(sel):
        raise RuntimeError("fit-sample linspace produced duplicate indices")
    return [pairs[int(i)] for i in sel]


def map_stats(row: np.ndarray, com_row: np.ndarray):
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    com_row = np.asarray(com_row, dtype=np.float64).reshape(-1)
    return {
        "compatibility_B_Ct_abs_error": float(abs(float(row @ com_row) - 1.0)),
        "coefficient_min": float(np.min(row)),
        "coefficient_max": float(np.max(row)),
        "coefficient_l2": float(np.linalg.norm(row)),
        "nonzero_coefficients": int(np.count_nonzero(np.abs(row) > 1.0e-10)),
    }


def read_pair_ids(path: Path, target_window_ps: float = 1.0):
    out = {"nearest": [], "random": []}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not math.isclose(float(row["window_ps"]), target_window_ps, abs_tol=1.0e-12, rel_tol=0.0):
                continue
            if row.get("pair_set") != "nearest_same_copy_gap":
                continue
            control = row.get("control")
            if control in out:
                out[control].append((int(row["sample_i"]), int(row["sample_j"])))
    result = {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}
    if any(len(v) == 0 for v in result.values()):
        raise RuntimeError(f"could not recover 1-ps nearest/random pair identities from {path}")
    return result


def half_mse_proxy(force_aligned: np.ndarray, pairs: np.ndarray, residue_indices: Sequence[int] | None = None):
    force_aligned = np.asarray(force_aligned, dtype=np.float64)
    if residue_indices is not None:
        force_aligned = force_aligned[:, np.asarray(residue_indices, dtype=np.int64), :]
    rms2 = float(np.mean(np.square(force_aligned, dtype=np.float64)))
    if rms2 <= 0.0:
        raise RuntimeError("non-positive target force variance")
    pairs = np.asarray(pairs, dtype=np.int64)
    diff = force_aligned[pairs[:, 0]] - force_aligned[pairs[:, 1]]
    half = 0.5 * float(np.mean(np.square(diff, dtype=np.float64)))
    return half / rms2, half, math.sqrt(rms2)


def aggforce_version() -> str:
    try:
        return importlib.metadata.version("aggforce")
    except importlib.metadata.PackageNotFoundError:
        return "unknown/source-install"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optimize TEL22 single-site DA/DT instantaneous self-force aggregation with aggforce"
    )
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--mapping-config", type=Path, required=True)
    ap.add_argument("--raw-topology", type=Path, required=True)
    ap.add_argument("--raw-trr", type=Path, required=True)
    ap.add_argument("--copy-dir", type=Path, required=True)
    ap.add_argument("--copy-manifest", type=Path, required=True)
    ap.add_argument("--temporal-report", type=Path, required=True)
    ap.add_argument("--temporal-pairs", type=Path, required=True)
    ap.add_argument("--validation-stride", type=int, default=5)
    ap.add_argument("--constraint-threshold", type=float, default=1.0e-3)
    ap.add_argument("--constraint-frames", type=int, default=100)
    ap.add_argument("--fit-max-samples", type=int, default=3000)
    ap.add_argument("--l2-regularization", type=float, default=1.0e3)
    ap.add_argument("--output-cache", type=Path, required=True)
    ap.add_argument("--output-maps", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    args = ap.parse_args()

    if args.validation_stride < 2:
        raise SystemExit("validation-stride must be >= 2")
    if args.constraint_threshold <= 0.0 or args.constraint_frames <= 0:
        raise SystemExit("constraint threshold/frames must be > 0")
    if args.fit_max_samples == 0:
        raise SystemExit("fit-max-samples must be positive or negative for all training samples")
    if args.l2_regularization < 0.0:
        raise SystemExit("l2-regularization must be >= 0")
    for p in (
        args.dataset,
        args.mapping_config,
        args.raw_topology,
        args.raw_trr,
        args.copy_manifest,
        args.temporal_report,
        args.temporal_pairs,
    ):
        if not p.exists():
            raise SystemExit(f"required file not found: {p}")

    try:
        from aggforce import (
            LinearMap,
            constraint_aware_uni_map,
            guess_pairwise_constraints,
            project_forces,
            qp_linear_map,
        )
    except Exception as exc:
        raise SystemExit(
            "aggforce import failed. Install the official package first, e.g.\n"
            "  python3 -m pip install 'git+https://github.com/noegroup/aggforce.git'\n"
            f"original import error: {exc}"
        )

    manifest = json.loads(args.copy_manifest.read_text(encoding="utf-8"))
    copies = int(manifest["copies"])
    period = int(manifest["residues_per_copy"])
    for ci in range(copies):
        for suffix in (".gro", "_rerun.trr"):
            p = args.copy_dir / f"copy_{ci:02d}{suffix}"
            if not p.exists():
                raise SystemExit(f"missing full single-copy rerun input: {p}")

    times, _f0, _t0, sig0 = dsi.load_targets(
        args.copy_dir / "copy_00.gro", args.copy_dir / "copy_00_rerun.trr"
    )
    times = np.asarray(times, dtype=np.float64)
    self_f, self_t, self_sig = dsi.load_self_targets(args.copy_dir, manifest, times)
    if sig0 != self_sig or len(self_sig) != period:
        raise RuntimeError("single-copy residue signature mismatch")
    self_f4 = np.asarray(self_f, dtype=np.float64).reshape(len(times), copies, period, 3)
    self_t4 = np.asarray(self_t, dtype=np.float64).reshape(len(times), copies, period, 3)

    raw_indices = np.asarray(
        fs.raw_time_to_frame_indices(args.raw_topology, args.raw_trr, times), dtype=np.int64
    )
    if len(np.unique(raw_indices)) != len(raw_indices):
        raise RuntimeError("raw time mapping produced duplicate dataset frame indices")

    temporal = json.loads(args.temporal_report.read_text(encoding="utf-8"))
    widths = [float(x) for x in temporal["inputs"]["windows_ps"]]
    edges = tfa.sample_cell_edges(times)
    centers = tfa.common_centers(times, edges, widths)
    if len(centers) != int(temporal["inputs"]["common_center_frames"]):
        raise RuntimeError("common-center count differs from 03p temporal diagnostic")
    if not math.isclose(
        float(times[centers[0]]),
        float(temporal["inputs"]["common_center_time_start_ps"]),
        abs_tol=1e-8,
        rel_tol=0.0,
    ):
        raise RuntimeError("common-center start time differs from 03p")
    if not math.isclose(
        float(times[centers[-1]]),
        float(temporal["inputs"]["common_center_time_end_ps"]),
        abs_tol=1e-8,
        rel_tol=0.0,
    ):
        raise RuntimeError("common-center end time differs from 03p")

    train_pos, val_pos, dataset_order = bsi.stratified_tail_order(len(centers), args.validation_stride)
    fit_pairs = deterministic_fit_samples(train_pos, copies, args.fit_max_samples)
    fit_by_copy: Dict[int, List[int]] = {ci: [] for ci in range(copies)}
    for center_pos, ci in fit_pairs:
        fit_by_copy[ci].append(int(centers[center_pos]))

    masses, current_matrix, com_matrix, residue_infos, sig_top = residue_metadata(
        args.copy_dir / "copy_00.gro", period, args.mapping_config
    )
    natoms = current_matrix.shape[1]
    eligible = [int(x["local_index"]) for x in residue_infos if x["single_site_exact_com"]]
    if not eligible:
        raise RuntimeError("no exact single-site COM residues found; 03r has nothing safe to optimize")
    noneligible = [i for i in range(period) if i not in set(eligible)]
    eligible_labels = [str(residue_infos[i]["resname"]) for i in eligible]

    # Assemble atomistic fit snapshots strictly from TRAIN center positions.  The same
    # fit sample list is used for every local DA/DT residue, so differences between
    # residue maps are not a data-split artifact.
    fit_coords_parts = []
    fit_forces_parts = []
    for ci in range(copies):
        positions = fit_by_copy[ci]
        if not positions:
            continue
        c, f, _t, sig = load_atomistic_frames(
            args.copy_dir / f"copy_{ci:02d}.gro",
            args.copy_dir / f"copy_{ci:02d}_rerun.trr",
            positions,
        )
        if sig != sig_top or c.shape[1] != natoms:
            raise RuntimeError(f"copy_{ci:02d}: atom topology/order differs from copy_00")
        fit_coords_parts.append(c)
        fit_forces_parts.append(f)
    fit_coords = np.concatenate(fit_coords_parts, axis=0)
    fit_forces = np.concatenate(fit_forces_parts, axis=0)
    if len(fit_coords) != len(fit_pairs):
        raise RuntimeError("fit sample assembly count mismatch")

    # Start from the existing exact total-force estimator for every residue.  Only rows
    # whose *actual model mapping* is one exact mass-weighted COM (DA/DT here) are
    # replaced.  Multi-site DG rows stay untouched because their rigid orientation map
    # is nonlinear; applying a 22-COM aggforce QP to them would optimize the wrong PMF.
    basic_matrix = current_matrix.copy()
    optim_matrix = current_matrix.copy()
    per_residue_maps = []
    for ri in eligible:
        info = residue_infos[ri]
        inds = np.asarray(info["atom_indices"], dtype=np.int64)
        local_coords = fit_coords[:, inds, :]
        local_forces = fit_forces[:, inds, :]
        local_com = np.asarray(info["masses"], dtype=np.float64)
        local_com /= float(np.sum(local_com))
        coord_map = LinearMap(local_com.reshape(1, -1))
        n_constraint = min(int(args.constraint_frames), len(local_coords))
        constraints = guess_pairwise_constraints(
            local_coords[:n_constraint], threshold=float(args.constraint_threshold)
        )
        print(
            f"[AGGFORCE] residue {ri:02d} {info['resname']}: atoms={len(inds)} "
            f"constraints={len(constraints)}"
        )
        basic_results = project_forces(
            coords=local_coords,
            forces=local_forces,
            coord_map=coord_map,
            constrained_inds=constraints,
            method=constraint_aware_uni_map,
        )
        optim_results = project_forces(
            coords=local_coords,
            forces=local_forces,
            coord_map=coord_map,
            constrained_inds=constraints,
            method=qp_linear_map,
            l2_regularization=float(args.l2_regularization),
        )
        brow = np.asarray(basic_results["tmap"].force_map.standard_matrix, dtype=np.float64).reshape(-1)
        orow = np.asarray(optim_results["tmap"].force_map.standard_matrix, dtype=np.float64).reshape(-1)
        if brow.shape != (len(inds),) or orow.shape != (len(inds),):
            raise RuntimeError(f"residue {ri}: unexpected aggforce map shape")
        for name, row in (("constraint_aware", brow), ("optimized", orow)):
            err = abs(float(row @ local_com) - 1.0)
            if err > 5.0e-5:
                raise RuntimeError(f"residue {ri} {name}: B C^T != 1 (error {err:g})")
        basic_matrix[ri, inds] = brow
        optim_matrix[ri, inds] = orow
        constraint_array = np.asarray(sorted(tuple(map(int, p)) for p in constraints), dtype=np.int32)
        if constraint_array.size == 0:
            constraint_array = np.empty((0, 2), dtype=np.int32)
        per_residue_maps.append(
            {
                "local_index": ri,
                "resname": str(info["resname"]),
                "atoms": len(inds),
                "atom_names": list(info["atom_names"]),
                "fit_samples": int(len(local_coords)),
                "constraint_detection_frames": int(n_constraint),
                "constraints_detected": int(len(constraints)),
                "constraint_pairs_local": constraint_array.tolist(),
                "current": map_stats(np.ones(len(inds)), local_com),
                "constraint_aware": map_stats(brow, local_com),
                "optimized": map_stats(orow, local_com),
            }
        )

    # Because each optimized row has support only inside its own DA/DT residue, it is
    # automatically orthogonal to every other model coordinate.  Combined with B C^T=1
    # for that residue's exact COM, this preserves the full mapping consistency for the
    # rows we change.  DG is left exactly at the original atom-sum target.
    for name, mat in (("constraint_aware", basic_matrix), ("optimized", optim_matrix)):
        if np.max(np.abs(mat[noneligible] - current_matrix[noneligible])) > 0.0:
            raise RuntimeError(f"{name}: non-single-site residue force rows changed unexpectedly")
        for ri in eligible:
            inds = np.asarray(residue_infos[ri]["atom_indices"], dtype=np.int64)
            outside = np.ones(natoms, dtype=bool)
            outside[inds] = False
            if np.max(np.abs(mat[ri, outside])) > 0.0:
                raise RuntimeError(f"{name}: residue {ri} has coefficients outside its own atoms")
            if abs(float(mat[ri] @ com_matrix[ri]) - 1.0) > 5.0e-5:
                raise RuntimeError(f"{name}: residue {ri} translational compatibility failed")

    mapped_current = np.empty_like(self_f4)
    mapped_basic = np.empty_like(self_f4)
    mapped_optim = np.empty_like(self_f4)
    all_positions = np.arange(len(times), dtype=np.int64)
    atomistic_current_guard = 0.0
    for ci in range(copies):
        _c, f, trr_times, sig = load_atomistic_frames(
            args.copy_dir / f"copy_{ci:02d}.gro",
            args.copy_dir / f"copy_{ci:02d}_rerun.trr",
            all_positions,
        )
        if sig != sig_top:
            raise RuntimeError(f"copy_{ci:02d}: topology/order changed")
        if len(trr_times) != len(times) or not np.allclose(trr_times, times, atol=1e-4, rtol=0.0):
            raise RuntimeError(f"copy_{ci:02d}: trajectory times differ")
        mapped_current[:, ci] = np.einsum("mn,tnd->tmd", current_matrix, f, optimize=True)
        mapped_basic[:, ci] = np.einsum("mn,tnd->tmd", basic_matrix, f, optimize=True)
        mapped_optim[:, ci] = np.einsum("mn,tnd->tmd", optim_matrix, f, optimize=True)
        atomistic_current_guard = max(
            atomistic_current_guard,
            float(np.max(np.abs(mapped_current[:, ci] - self_f4[:, ci]))),
        )
    denom = max(float(np.max(np.abs(self_f4))), 1.0)
    current_guard_rel = atomistic_current_guard / denom
    if current_guard_rel > 2.0e-6:
        raise RuntimeError(
            f"per-atom residue sum does not reproduce existing self target: rel max {current_guard_rel:g}"
        )

    full = dsi.geometry_with_targets(
        args.dataset,
        raw_indices,
        {
            "current": (
                mapped_current.reshape(len(times), copies * period, 3),
                self_t4.reshape(len(times), copies * period, 3),
            ),
            "constraint_aware": (
                mapped_basic.reshape(len(times), copies * period, 3),
                self_t4.reshape(len(times), copies * period, 3),
            ),
            "optimized": (
                mapped_optim.reshape(len(times), copies * period, 3),
                self_t4.reshape(len(times), copies * period, 3),
            ),
        },
    )
    pairs = read_pair_ids(args.temporal_pairs, 1.0)
    nsamples = len(centers) * copies
    for arr in pairs.values():
        if int(np.max(arr)) >= nsamples:
            raise RuntimeError("03p pair indices exceed common-center sample pool")

    by_type = {}
    for label in sorted(set(str(x["resname"]) for x in residue_infos)):
        by_type[label] = [int(x["local_index"]) for x in residue_infos if str(x["resname"]) == label]
    groups = {"all": list(range(period)), "optimized_single_site": eligible, **by_type}

    diagnostics = {}
    for name in ("current", "constraint_aware", "optimized"):
        aligned = np.asarray(full[name]["forces"], dtype=np.float64).reshape(len(times), copies, period, 3)
        aligned_center = aligned[centers].reshape(nsamples, period, 3)
        group_diag = {}
        for group, inds in groups.items():
            near_frac, near_abs, target_rms = half_mse_proxy(aligned_center, pairs["nearest"], inds)
            rand_frac, rand_abs, _ = half_mse_proxy(aligned_center, pairs["random"], inds)
            group_diag[group] = {
                "residue_local_indices": [int(x) for x in inds],
                "force_component_rms_kj_mol_nm": target_rms,
                "nearest_force_half_pair_difference_mse_fraction_of_target_mse": near_frac,
                "random_force_half_pair_difference_mse_fraction_of_target_mse": rand_frac,
                "nearest_vs_random_force_half_mse_ratio": near_frac / rand_frac,
                "nearest_absolute_half_pair_mse_kj2_mol2_nm2": near_abs,
                "random_absolute_half_pair_mse_kj2_mol2_nm2": rand_abs,
            }
        diagnostics[name] = {
            "force_component_rms_kj_mol_nm": group_diag["all"]["force_component_rms_kj_mol_nm"],
            "nearest_force_half_pair_difference_mse_fraction_of_target_mse": group_diag["all"]["nearest_force_half_pair_difference_mse_fraction_of_target_mse"],
            "random_force_half_pair_difference_mse_fraction_of_target_mse": group_diag["all"]["random_force_half_pair_difference_mse_fraction_of_target_mse"],
            "nearest_vs_random_force_half_mse_ratio": group_diag["all"]["nearest_vs_random_force_half_mse_ratio"],
            "nearest_absolute_half_pair_mse_kj2_mol2_nm2": group_diag["all"]["nearest_absolute_half_pair_mse_kj2_mol2_nm2"],
            "random_absolute_half_pair_mse_kj2_mol2_nm2": group_diag["all"]["random_absolute_half_pair_mse_kj2_mol2_nm2"],
            "groups": group_diag,
        }

    reference_proxy = float(
        temporal["windows"]["1ps"]["nearest_same_copy_gap"][
            "force_half_pair_difference_mse_fraction_of_target_mse"
        ]
    )
    if abs(
        diagnostics["current"]["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
        - reference_proxy
    ) > 2.0e-5:
        raise RuntimeError(
            "current-map nearest proxy does not reproduce 03p 1-ps baseline; pair ordering/rotation mismatch"
        )
    for name in ("constraint_aware", "optimized"):
        diagnostics[name]["normalized_noise_proxy_ratio_vs_current"] = (
            diagnostics[name]["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
            / diagnostics["current"]["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
        )
        diagnostics[name]["absolute_half_pair_mse_ratio_vs_current"] = (
            diagnostics[name]["nearest_absolute_half_pair_mse_kj2_mol2_nm2"]
            / diagnostics["current"]["nearest_absolute_half_pair_mse_kj2_mol2_nm2"]
        )
        for group in groups:
            d = diagnostics[name]["groups"][group]
            b = diagnostics["current"]["groups"][group]
            d["normalized_noise_proxy_ratio_vs_current"] = (
                d["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
                / b["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
            )
            d["absolute_half_pair_mse_ratio_vs_current"] = (
                d["nearest_absolute_half_pair_mse_kj2_mol2_nm2"]
                / b["nearest_absolute_half_pair_mse_kj2_mol2_nm2"]
            )

    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_cache,
        times_ps=times.astype(np.float64),
        raw_dataset_indices=raw_indices.astype(np.int32),
        center_rerun_positions=centers.astype(np.int32),
        center_raw_dataset_indices=raw_indices[centers].astype(np.int32),
        center_times_ps=times[centers].astype(np.float64),
        train_center_positions=train_pos.astype(np.int32),
        validation_center_positions=val_pos.astype(np.int32),
        dataset_center_order=dataset_order.astype(np.int32),
        force_current=mapped_current.astype(np.float32),
        force_constraint_aware=mapped_basic.astype(np.float32),
        force_optimized=mapped_optim.astype(np.float32),
        torque_current=self_t4.astype(np.float32),
    )
    args.output_maps.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_maps,
        per_residue_com_coordinate_map=com_matrix.astype(np.float64),
        current_residue_sum_force_map=current_matrix.astype(np.float64),
        constraint_aware_single_site_force_map=basic_matrix.astype(np.float64),
        optimized_single_site_force_map=optim_matrix.astype(np.float64),
        optimized_residue_local_indices=np.asarray(eligible, dtype=np.int32),
        atom_masses_amu=masses.astype(np.float64),
    )

    report = {
        "definition": {
            "purpose": "TEL22 self-force variance reduction with statistically optimized instantaneous force aggregation",
            "scope": "only residues whose actual retained mapping is one exact mass-weighted COM site are optimized (DA/DT); multi-site rigid DG remains at the original atom-sum force target",
            "why_partial": "aggforce linear-map consistency is exact for DA/DT COM coordinates. DG orientation/site reconstruction is a nonlinear rigid-body map, so optimizing it against a fictitious residue-COM map would target the wrong coarse variables",
            "current_force_map": "per-residue algebraic sum of atomistic instantaneous forces",
            "constraint_aware_map": "aggforce constraint_aware_uni_map on each eligible residue independently; DG unchanged",
            "optimized_force_map": "aggforce qp_linear_map on each eligible residue independently with fixed configuration-independent coefficients; DG unchanged",
            "thermodynamic_guardrail": "each changed force row has support only on atoms of its own exact single-site COM residue and satisfies B C^T = 1; all other model coordinates use disjoint atoms, so the changed row is orthogonal to them",
            "validation_guardrail": "force-map optimization is fit only on training-center configurations; validation target frames are excluded from optimization",
            "torque_guardrail": "torque mapping is unchanged; production torque loss already masks one-site DA/DT, so 03r can retain the same torque_weight=0.5 as the 03q baseline",
            "pair_guardrail": "conditional-noise proxies reuse the exact 1-ps nearest/random pair identities from 03p and the same Kabsch-aligned force convention",
        },
        "software": {
            "aggforce_version": aggforce_version(),
            "aggforce_repository": "https://github.com/noegroup/aggforce",
            "method": "qp_linear_map",
            "l2_regularization": float(args.l2_regularization),
        },
        "inputs": {
            "frames_per_copy": int(len(times)),
            "common_center_frames": int(len(centers)),
            "train_center_frames": int(len(train_pos)),
            "validation_center_frames": int(len(val_pos)),
            "copies": copies,
            "residues_per_copy": period,
            "atoms_per_copy": int(natoms),
            "fit_samples_available_train_only": int(len(train_pos) * copies),
            "fit_samples_used_per_optimized_residue": int(len(fit_pairs)),
            "fit_max_samples_requested": int(args.fit_max_samples),
            "validation_stride": int(args.validation_stride),
            "constraint_threshold": float(args.constraint_threshold),
            "common_center_time_start_ps": float(times[centers[0]]),
            "common_center_time_end_ps": float(times[centers[-1]]),
        },
        "mapping_scope": {
            "optimized_residue_local_indices": eligible,
            "optimized_residue_labels": eligible_labels,
            "unchanged_residue_local_indices": noneligible,
            "unchanged_residue_labels": [str(residue_infos[i]["resname"]) for i in noneligible],
            "per_residue_maps": per_residue_maps,
        },
        "mapping_guardrails": {
            "current_atom_sum_reproduces_existing_self_target_max_abs_kj_mol_nm": float(atomistic_current_guard),
            "current_atom_sum_reproduces_existing_self_target_relative_to_global_max": float(current_guard_rel),
            "noneligible_rows_exactly_unchanged": True,
            "optimized_rows_have_own_residue_support_only": True,
        },
        "force_noise_diagnostic": diagnostics,
        "comparison": {
            "current_03p_force_noise_proxy": reference_proxy,
            "constraint_aware_global_noise_proxy_ratio_vs_current": diagnostics["constraint_aware"]["normalized_noise_proxy_ratio_vs_current"],
            "optimized_global_noise_proxy_ratio_vs_current": diagnostics["optimized"]["normalized_noise_proxy_ratio_vs_current"],
            "constraint_aware_single_site_noise_proxy_ratio_vs_current": diagnostics["constraint_aware"]["groups"]["optimized_single_site"]["normalized_noise_proxy_ratio_vs_current"],
            "optimized_single_site_noise_proxy_ratio_vs_current": diagnostics["optimized"]["groups"]["optimized_single_site"]["normalized_noise_proxy_ratio_vs_current"],
            "optimized_global_absolute_half_pair_mse_ratio_vs_current": diagnostics["optimized"]["absolute_half_pair_mse_ratio_vs_current"],
            "optimized_single_site_absolute_half_pair_mse_ratio_vs_current": diagnostics["optimized"]["groups"]["optimized_single_site"]["absolute_half_pair_mse_ratio_vs_current"],
        },
        "outputs": {
            "target_cache": str(args.output_cache),
            "map_matrices": str(args.output_maps),
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    print("======================================================")
    print(" TEL22 AGGFORCE SINGLE-SITE SELF FORCE DIAGNOSTIC")
    print("======================================================")
    print(
        f"aggforce={report['software']['aggforce_version']} | atoms/copy={natoms} | "
        f"fit samples/residue={len(fit_pairs)} (training only)"
    )
    print(
        "optimized local residues: "
        + ", ".join(f"{i}:{residue_infos[i]['resname']}" for i in eligible)
    )
    print("variant             global_proxy xglobal   single_proxy xsingle")
    for name in ("current", "constraint_aware", "optimized"):
        d = diagnostics[name]
        s = d["groups"]["optimized_single_site"]
        xg = d["nearest_force_half_pair_difference_mse_fraction_of_target_mse"] / diagnostics["current"]["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
        xs = s["nearest_force_half_pair_difference_mse_fraction_of_target_mse"] / diagnostics["current"]["groups"]["optimized_single_site"]["nearest_force_half_pair_difference_mse_fraction_of_target_mse"]
        print(
            f"{name:18s} {d['nearest_force_half_pair_difference_mse_fraction_of_target_mse']:12.6f} "
            f"{xg:7.4f} {s['nearest_force_half_pair_difference_mse_fraction_of_target_mse']:12.6f} {xs:7.4f}"
        )
    print(f"cache:  {args.output_cache}")
    print(f"maps:   {args.output_maps}")
    print(f"report: {args.output_report}")


if __name__ == "__main__":
    main()
