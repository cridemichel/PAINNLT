#!/usr/bin/env python3
"""Audit TEL22 atomistic->CG generalized-force mapping and decompose AA force sources.

This tool has two modes.

1) --make-index writes deterministic GROMACS index groups for DNA, water, K, Cl,
   and the DNA+environment unions used by subset reruns.
2) --analyze consumes full and subset rerun TRRs and computes residue-level net
   forces/torques. The subset identity is

       F_full(DNA) = F_DNA-DNA + F_water-on-DNA + F_K-on-DNA + F_Cl-on-DNA

   with the source terms obtained by subtraction from subset reruns. A numerical
   closure check is mandatory before source fractions are interpreted.

The decomposition is diagnostic only. In particular, subtracting solvent/ion
forces from a force-matching target is NOT proposed here as a production target.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import MDAnalysis as mda
import numpy as np

I32 = struct.Struct("=i")
F32_3 = struct.Struct("=3f")
SITE = struct.Struct("=ifff")

DNA_RESNAMES = ("DA", "DG", "DT")
WATER_RESNAMES = ("SOL", "HOH", "WAT", "TIP3", "TIP3P")
K_RESNAMES = ("K", "K+", "POT")
CL_RESNAMES = ("CL", "CL-", "CLA")


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else math.nan


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / den) if den > 0.0 else math.nan


def percentiles(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {k: math.nan for k in ("p50", "p90", "p95", "p99", "max")}
    return {
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def select_resnames(u: mda.Universe, names: Sequence[str]) -> np.ndarray:
    wanted = set(names)
    idx = [int(a.index) for a in u.atoms if a.resname in wanted]
    return np.asarray(idx, dtype=np.int64)


def write_ndx_group(handle, name: str, indices0: np.ndarray) -> None:
    handle.write(f"[ {name} ]\n")
    one = np.asarray(indices0, dtype=np.int64) + 1
    for start in range(0, len(one), 15):
        handle.write(" ".join(str(int(v)) for v in one[start:start + 15]) + "\n")
    handle.write("\n")


def make_index(topology: Path, output: Path, manifest: Path) -> None:
    u = mda.Universe(str(topology))
    dna = select_resnames(u, DNA_RESNAMES)
    water = select_resnames(u, WATER_RESNAMES)
    potassium = select_resnames(u, K_RESNAMES)
    chloride = select_resnames(u, CL_RESNAMES)

    groups = {
        "FS_DNA_ONLY": dna,
        "FS_DNA_WATER": np.sort(np.concatenate((dna, water))),
        "FS_DNA_K": np.sort(np.concatenate((dna, potassium))),
        "FS_DNA_CL": np.sort(np.concatenate((dna, chloride))),
    }
    covered = np.unique(np.concatenate((dna, water, potassium, chloride)))
    missing = np.setdiff1d(np.arange(len(u.atoms), dtype=np.int64), covered)

    if len(dna) == 0:
        raise RuntimeError("No DA/DG/DT atoms found; cannot build TEL22 force-source groups")
    if len(water) == 0:
        raise RuntimeError(f"No water atoms found using resnames {WATER_RESNAMES}")
    if len(potassium) == 0:
        raise RuntimeError(f"No K atoms found using resnames {K_RESNAMES}")
    if len(chloride) == 0:
        raise RuntimeError(f"No Cl atoms found using resnames {CL_RESNAMES}")
    if len(missing) != 0:
        missing_names = sorted({u.atoms[int(i)].resname for i in missing[:1000]})
        raise RuntimeError(
            "Atom partition DNA/water/K/Cl does not cover System: "
            f"missing={len(missing)} atoms; example resnames={missing_names}. "
            "Refuse decomposition until groups are exhaustive."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for name, idx in groups.items():
            write_ndx_group(fh, name, idx)

    info = {
        "topology": str(topology),
        "atoms_total": int(len(u.atoms)),
        "atoms_dna": int(len(dna)),
        "atoms_water": int(len(water)),
        "atoms_k": int(len(potassium)),
        "atoms_cl": int(len(chloride)),
        "partition_missing_atoms": int(len(missing)),
        "groups": {k: int(len(v)) for k, v in groups.items()},
        "resnames_present": sorted({r.resname for r in u.residues}),
    }
    manifest.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


def mic(delta: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    out = np.asarray(delta, dtype=np.float64).copy()
    for k in range(3):
        L = float(box_nm[k])
        if L > 0:
            out[..., k] -= L * np.rint(out[..., k] / L)
    return out


def unwrap_residue(pos_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    pos_nm = np.asarray(pos_nm, dtype=np.float64)
    if len(pos_nm) <= 1:
        return pos_nm.copy()
    anchor = pos_nm[0]
    return anchor + mic(pos_nm - anchor, box_nm)


def dna_signature(u: mda.Universe) -> List[Tuple[str, int]]:
    return [(str(r.resname), int(len(r.atoms))) for r in u.residues if r.resname in DNA_RESNAMES]


def mapped_generalized_forces(topology: Path, trr: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, int]]]:
    u = mda.Universe(str(topology), str(trr))
    dna_res = [r for r in u.residues if r.resname in DNA_RESNAMES]
    if not dna_res:
        raise RuntimeError(f"No DNA residues in {topology}")
    signatures = [(str(r.resname), int(len(r.atoms))) for r in dna_res]
    times: List[float] = []
    all_f: List[np.ndarray] = []
    all_t: List[np.ndarray] = []

    for ts in u.trajectory:
        box_nm = np.asarray(ts.dimensions[:3], dtype=np.float64) / 10.0
        frame_f = []
        frame_t = []
        if getattr(ts, "has_forces", None) is False:
            raise RuntimeError(f"Trajectory lacks forces: {trr}")
        for res in dna_res:
            atoms = res.atoms
            p_nm = np.asarray(atoms.positions, dtype=np.float64) / 10.0
            p_nm = unwrap_residue(p_nm, box_nm)
            try:
                masses = np.asarray(atoms.masses, dtype=np.float64)
            except Exception as exc:
                raise RuntimeError(f"Masses unavailable for residue {res.resname}: {exc}")
            if not np.all(np.isfinite(masses)) or float(np.sum(masses)) <= 0:
                raise RuntimeError(f"Invalid masses for residue {res.resname}")
            center = np.sum(p_nm * masses[:, None], axis=0) / float(np.sum(masses))
            # MDAnalysis GROMACS force unit is kJ/(mol Angstrom); convert to kJ/(mol nm).
            f = np.asarray(atoms.forces, dtype=np.float64) * 10.0
            total_f = np.sum(f, axis=0)
            total_t = np.sum(np.cross(p_nm - center, f), axis=0)
            frame_f.append(total_f)
            frame_t.append(total_t)
        times.append(float(ts.time))
        all_f.append(np.asarray(frame_f, dtype=np.float64))
        all_t.append(np.asarray(frame_t, dtype=np.float64))

    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(all_f, dtype=np.float64),
        np.asarray(all_t, dtype=np.float64),
        signatures,
    )


def read_exact(handle, n: int) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise EOFError(f"unexpected EOF: wanted {n}, got {len(data)}")
    return data


def read_dataset_selected(dataset: Path, selected: Iterable[int]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    wanted = set(int(v) for v in selected)
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    with dataset.open("rb") as fh:
        # Dataset format begins with one global int32 num_frames, exactly as
        # written by build_cg_dataset.py and consumed by train_painn.cpp.
        nframes = I32.unpack(read_exact(fh, I32.size))[0]
        if nframes <= 0 or nframes > 10_000_000:
            raise ValueError(f"invalid dataset num_frames={nframes}")
        invalid = sorted(i for i in wanted if i < 0 or i >= nframes)
        if invalid:
            raise RuntimeError(
                f"Requested dataset frame indices outside [0,{nframes - 1}]: {invalid}"
            )

        for frame in range(nframes):
            nmol = I32.unpack(read_exact(fh, I32.size))[0]
            nsites_total = I32.unpack(read_exact(fh, I32.size))[0]
            if nmol <= 0 or nmol > 10_000_000:
                raise ValueError(f"invalid molecule count at frame {frame}: {nmol}")
            if nsites_total < nmol or nsites_total > 100_000_000:
                raise ValueError(f"invalid site count at frame {frame}: {nsites_total}")
            _box = read_exact(fh, F32_3.size)
            f_list = []
            t_list = []
            counted = 0
            for mol_idx in range(nmol):
                mol_id = I32.unpack(read_exact(fh, I32.size))[0]
                nsites = I32.unpack(read_exact(fh, I32.size))[0]
                if mol_id != mol_idx:
                    raise ValueError(
                        f"non-sequential molecule id at frame {frame}: "
                        f"expected {mol_idx}, got {mol_id}"
                    )
                if nsites <= 0 or counted + nsites > nsites_total:
                    raise ValueError(
                        f"invalid molecule site count at frame {frame}, molecule {mol_idx}: {nsites}"
                    )
                _center = read_exact(fh, F32_3.size)
                force = np.asarray(F32_3.unpack(read_exact(fh, F32_3.size)), dtype=np.float64)
                torque = np.asarray(F32_3.unpack(read_exact(fh, F32_3.size)), dtype=np.float64)
                for _s in range(nsites):
                    _ = SITE.unpack(read_exact(fh, SITE.size))
                counted += nsites
                f_list.append(force)
                t_list.append(torque)
            if counted != nsites_total:
                raise ValueError(
                    f"dataset site mismatch at frame {frame}: "
                    f"header={nsites_total}, parsed={counted}"
                )
            if frame in wanted:
                out[frame] = (np.asarray(f_list), np.asarray(t_list))

        if fh.read(1):
            raise ValueError("unexpected trailing bytes after declared dataset frames")

    missing = sorted(wanted.difference(out))
    if missing:
        raise RuntimeError(f"Requested dataset frames not found: {missing}")
    return out


def raw_time_to_frame_indices(topology: Path, raw_trr: Path, times: np.ndarray, tol_ps: float = 1e-4) -> List[int]:
    u = mda.Universe(str(topology), str(raw_trr))
    raw_times = np.asarray([float(ts.time) for ts in u.trajectory], dtype=np.float64)
    idx = []
    for t in times:
        j = int(np.argmin(np.abs(raw_times - t)))
        err = abs(float(raw_times[j] - t))
        if err > tol_ps:
            raise RuntimeError(f"Cannot map rerun time {t} ps to raw frame; nearest error={err} ps")
        idx.append(j)
    return idx


def component_report(x: np.ndarray, full: np.ndarray, residual: np.ndarray | None) -> Dict[str, object]:
    item: Dict[str, object] = {
        "component_rms": rms(x),
        "vector_norm_percentiles": percentiles(np.linalg.norm(x, axis=-1)),
        "over_full_rms": float(rms(x) / rms(full)) if rms(full) > 0 else math.nan,
        "cosine_vs_full": cosine(x, full),
    }
    if residual is not None:
        item.update({
            "over_residual_rms": float(rms(x) / rms(residual)) if rms(residual) > 0 else math.nan,
            "cosine_vs_residual": cosine(x, residual),
        })
    return item


def analyze(args: argparse.Namespace) -> None:
    sources = {
        "full": (Path(args.full_topology), Path(args.full_trr)),
        "dna": (Path(args.dna_topology), Path(args.dna_trr)),
        "dna_water": (Path(args.dna_water_topology), Path(args.dna_water_trr)),
        "dna_k": (Path(args.dna_k_topology), Path(args.dna_k_trr)),
        "dna_cl": (Path(args.dna_cl_topology), Path(args.dna_cl_trr)),
    }
    loaded = {}
    for key, (top, trr) in sources.items():
        loaded[key] = mapped_generalized_forces(top, trr)

    times, full_f, full_t, sig = loaded["full"]
    max_dt = 0.0
    for key in ("dna", "dna_water", "dna_k", "dna_cl"):
        t, _f, _tau, s = loaded[key]
        if s != sig:
            raise RuntimeError(f"DNA topology/signature mismatch in {key}")
        if len(t) != len(times):
            raise RuntimeError(f"Frame-count mismatch in {key}: {len(t)} vs {len(times)}")
        max_dt = max(max_dt, float(np.max(np.abs(t - times))))
    if max_dt > 1e-4:
        raise RuntimeError(f"Rerun time alignment failed: max |dt|={max_dt} ps")

    dna_f, dna_t = loaded["dna"][1], loaded["dna"][2]
    dw_f, dw_t = loaded["dna_water"][1], loaded["dna_water"][2]
    dk_f, dk_t = loaded["dna_k"][1], loaded["dna_k"][2]
    dc_f, dc_t = loaded["dna_cl"][1], loaded["dna_cl"][2]

    comp_f = {
        "dna_internal": dna_f,
        "water": dw_f - dna_f,
        "potassium": dk_f - dna_f,
        "chloride": dc_f - dna_f,
    }
    comp_t_all = {
        "dna_internal": dna_t,
        "water": dw_t - dna_t,
        "potassium": dk_t - dna_t,
        "chloride": dc_t - dna_t,
    }
    # Runtime/training rotational DOFs exist only for multi-site DG residues.
    rigid_mask = np.asarray([resname == "DG" for resname, _nat in sig], dtype=bool)
    if not np.any(rigid_mask):
        raise RuntimeError("No DG multi-site residues found for torque audit")
    comp_t = {name: arr[:, rigid_mask, :] for name, arr in comp_t_all.items()}
    full_t_rigid = full_t[:, rigid_mask, :]

    env_f = comp_f["water"] + comp_f["potassium"] + comp_f["chloride"]
    env_t = comp_t["water"] + comp_t["potassium"] + comp_t["chloride"]
    closure_f = comp_f["dna_internal"] + env_f
    closure_t = comp_t["dna_internal"] + env_t
    closure_rel_f = rms(closure_f - full_f) / max(rms(full_f), 1e-30)
    closure_rel_t = rms(closure_t - full_t_rigid) / max(rms(full_t_rigid), 1e-30)
    closure_pass = bool(closure_rel_f <= args.closure_tol and closure_rel_t <= args.closure_tol)

    raw_indices = raw_time_to_frame_indices(Path(args.full_topology), Path(args.raw_trr), times)
    ds = read_dataset_selected(Path(args.dataset), raw_indices)
    residual_f = np.stack([ds[i][0] for i in raw_indices], axis=0)
    residual_t_all = np.stack([ds[i][1] for i in raw_indices], axis=0)
    if residual_f.shape != full_f.shape:
        raise RuntimeError(f"Dataset/full mapped force shape mismatch: {residual_f.shape} vs {full_f.shape}")
    residual_t = residual_t_all[:, rigid_mask, :]

    implied_prior_f = full_f - residual_f
    implied_prior_t = full_t_rigid - residual_t

    # COM mapping compatibility for force aggregation b_i=1: sum_i b_i m_i/M = 1.
    u_full = mda.Universe(str(args.full_topology))
    compat_err = []
    for res in [r for r in u_full.residues if r.resname in DNA_RESNAMES]:
        m = np.asarray(res.atoms.masses, dtype=np.float64)
        c = m / np.sum(m)
        b = np.ones_like(c)
        compat_err.append(abs(float(np.dot(b, c)) - 1.0))

    report: Dict[str, object] = {
        "definition": {
            "coordinate_map": "per-residue center of mass (mass-weighted)",
            "force_map": "per-residue algebraic sum of atomistic forces",
            "torque_map": "sum_i (r_i-R_COM) x f_i",
            "source_decomposition": "subset rerun: DNA; DNA+water; DNA+K; DNA+Cl; source forces by subtraction from DNA-only",
            "warning": "source decomposition is diagnostic; do not subtract environment forces from production force-matching targets based on this report alone",
        },
        "counts": {
            "sampled_frames": int(len(times)),
            "dna_residues_per_frame": int(full_f.shape[1]),
            "force_components": int(full_f.size),
            "times_ps": [float(v) for v in times],
            "raw_dataset_frame_indices": [int(v) for v in raw_indices],
        },
        "mapping_compatibility": {
            "condition": "B C^T = I for translational COM coordinate map; B_i=1, C_i=m_i/M within each residue",
            "max_abs_error": float(max(compat_err) if compat_err else math.nan),
            "status": "PASS" if compat_err and max(compat_err) < 1e-12 else "FAIL",
        },
        "closure": {
            "tolerance": float(args.closure_tol),
            "force_relative_rms": float(closure_rel_f),
            "torque_relative_rms": float(closure_rel_t),
            "status": "PASS" if closure_pass else "FAIL",
        },
        "force": {
            "full_mapped_reference_rms": rms(full_f),
            "dataset_residual_target_rms": rms(residual_f),
            "implied_cg_prior_rms": rms(implied_prior_f),
            "components": {},
        },
        "torque": {
            "full_mapped_reference_rms": rms(full_t_rigid),
            "scope": "DG multi-site rigid bodies only",
            "dataset_residual_target_rms": rms(residual_t),
            "implied_cg_prior_rms": rms(implied_prior_t),
            "components": {},
        },
    }

    if closure_pass:
        for name, arr in {**comp_f, "environment_total": env_f}.items():
            report["force"]["components"][name] = component_report(arr, full_f, residual_f)
        for name, arr in {**comp_t, "environment_total": env_t}.items():
            report["torque"]["components"][name] = component_report(arr, full_t_rigid, residual_t)
        diagnostic_intrinsic_f = residual_f - env_f
        diagnostic_intrinsic_t = residual_t - env_t
        report["force"]["diagnostic_residual_minus_environment_rms"] = rms(diagnostic_intrinsic_f)
        report["force"]["diagnostic_residual_minus_environment_over_residual_rms"] = rms(diagnostic_intrinsic_f) / max(rms(residual_f), 1e-30)
        report["torque"]["diagnostic_residual_minus_environment_rms"] = rms(diagnostic_intrinsic_t)
        report["torque"]["diagnostic_residual_minus_environment_over_residual_rms"] = rms(diagnostic_intrinsic_t) / max(rms(residual_t), 1e-30)
    else:
        report["interpretation_blocked"] = (
            "Subset decomposition failed closure; source magnitudes are not reported because "
            "they would not be trustworthy."
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    print("======================================================")
    print(" TEL22 FORCE-MAPPING / FORCE-SOURCE AUDIT")
    print("======================================================")
    print(f"Mapped reference force RMS : {rms(full_f):.6g} kJ/(mol nm)")
    print(f"Dataset residual force RMS : {rms(residual_f):.6g} kJ/(mol nm)")
    print(f"COM force-map compatibility: {report['mapping_compatibility']['status']} (max err={report['mapping_compatibility']['max_abs_error']:.3g})")
    print(f"Subset closure force       : {closure_rel_f:.6g} relative RMS")
    print(f"Subset closure torque      : {closure_rel_t:.6g} relative RMS")
    print(f"Closure status             : {report['closure']['status']}")
    if closure_pass:
        print("\nForce source RMS [kJ/(mol nm)] and ratio to residual:")
        for name in ("dna_internal", "water", "potassium", "chloride", "environment_total"):
            r = report["force"]["components"][name]
            print(f"  {name:18s} {r['component_rms']:12.4f}   / residual={r['over_residual_rms']:.4f}   cos(res)={r['cosine_vs_residual']:.4f}")
        print("\nTorque source RMS [kJ/mol] and ratio to residual:")
        for name in ("dna_internal", "water", "potassium", "chloride", "environment_total"):
            r = report["torque"]["components"][name]
            print(f"  {name:18s} {r['component_rms']:12.4f}   / residual={r['over_residual_rms']:.4f}   cos(res)={r['cosine_vs_residual']:.4f}")
        print(f"\nDiagnostic residual-env force RMS ratio : {report['force']['diagnostic_residual_minus_environment_over_residual_rms']:.4f}")
        print(f"Diagnostic residual-env torque RMS ratio: {report['torque']['diagnostic_residual_minus_environment_over_residual_rms']:.4f}")
    print(f"\nReport: {out}")

    if not closure_pass:
        raise SystemExit(3)
    if report["mapping_compatibility"]["status"] != "PASS":
        raise SystemExit(4)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-index", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    p.add_argument("--topology")
    p.add_argument("--index-output")
    p.add_argument("--index-manifest")
    p.add_argument("--full-topology")
    p.add_argument("--raw-trr")
    p.add_argument("--full-trr")
    p.add_argument("--dna-topology")
    p.add_argument("--dna-trr")
    p.add_argument("--dna-water-topology")
    p.add_argument("--dna-water-trr")
    p.add_argument("--dna-k-topology")
    p.add_argument("--dna-k-trr")
    p.add_argument("--dna-cl-topology")
    p.add_argument("--dna-cl-trr")
    p.add_argument("--dataset")
    p.add_argument("--closure-tol", type=float, default=1e-3)
    p.add_argument("--output")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    if args.make_index:
        required = (args.topology, args.index_output, args.index_manifest)
        if any(v is None for v in required):
            p.error("--make-index requires --topology --index-output --index-manifest")
        make_index(Path(args.topology), Path(args.index_output), Path(args.index_manifest))
        return
    required_names = (
        "full_topology", "raw_trr", "full_trr", "dna_topology", "dna_trr",
        "dna_water_topology", "dna_water_trr", "dna_k_topology", "dna_k_trr",
        "dna_cl_topology", "dna_cl_trr", "dataset", "output",
    )
    missing = [name for name in required_names if getattr(args, name) is None]
    if missing:
        p.error("--analyze missing: " + ", ".join("--" + n.replace("_", "-") for n in missing))
    analyze(args)


if __name__ == "__main__":
    main()
