#!/usr/bin/env python3
"""Audit GROMACS TRR coordinate/force consistency for force matching.

This script is deliberately independent from the CG preprocessing.  It checks
whether the atomistic trajectory used as force-matching reference contains
synchronized positions/forces, whether `trjconv -pbc whole -force` preserved
forces, and (optionally) whether `mdrun -rerun` reproduces the stored forces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    token = value.strip().split()[0]
    try:
        return float(token)
    except (ValueError, IndexError):
        return None


def _as_int(value: str | None) -> int | None:
    f = _as_float(value)
    if f is None or not float(f).is_integer():
        return None
    return int(f)


def parse_mdp(path: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    if not path.exists():
        return params
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        params[key.strip().lower()] = value.strip()
    return params


def rel_rms(sum_delta_sq: float, sum_ref_sq: float) -> float:
    if sum_ref_sq <= 0.0:
        return 0.0 if sum_delta_sq <= 0.0 else math.inf
    return math.sqrt(sum_delta_sq / sum_ref_sq)


def load_universe(topology: Path, trajectory: Path) -> mda.Universe:
    try:
        return mda.Universe(str(topology), str(trajectory))
    except Exception as exc:  # pragma: no cover - message is the useful output
        raise RuntimeError(f"Unable to open {trajectory}: {exc}") from exc


def compare_raw_whole(
    topology: Path,
    raw_path: Path,
    whole_path: Path,
    csv_path: Path,
) -> dict[str, Any]:
    raw = load_universe(topology, raw_path)
    whole = load_universe(topology, whole_path)

    result: dict[str, Any] = {
        "raw_frames": len(raw.trajectory),
        "whole_frames": len(whole.trajectory),
        "raw_atoms": raw.atoms.n_atoms,
        "whole_atoms": whole.atoms.n_atoms,
    }

    if raw.atoms.n_atoms != whole.atoms.n_atoms:
        raise RuntimeError(
            f"Atom count differs: raw={raw.atoms.n_atoms}, whole={whole.atoms.n_atoms}"
        )
    if len(raw.trajectory) != len(whole.trajectory):
        raise RuntimeError(
            f"Frame count differs: raw={len(raw.trajectory)}, whole={len(whole.trajectory)}"
        )
    if len(raw.trajectory) == 0:
        raise RuntimeError("Trajectory contains zero frames")

    raw_missing_pos = 0
    raw_missing_force = 0
    whole_missing_pos = 0
    whole_missing_force = 0
    nonfinite_pos = 0
    nonfinite_force = 0
    max_time_delta = 0.0
    max_force_delta = 0.0
    sum_force_delta_sq = 0.0
    sum_force_ref_sq = 0.0
    force_components = 0
    times: list[float] = []
    rows: list[list[Any]] = []

    for idx in range(len(raw.trajectory)):
        ts_raw = raw.trajectory[idx]
        ts_whole = whole.trajectory[idx]

        has_raw_pos = bool(getattr(ts_raw, "has_positions", False))
        has_raw_force = bool(getattr(ts_raw, "has_forces", False))
        has_whole_pos = bool(getattr(ts_whole, "has_positions", False))
        has_whole_force = bool(getattr(ts_whole, "has_forces", False))
        raw_missing_pos += int(not has_raw_pos)
        raw_missing_force += int(not has_raw_force)
        whole_missing_pos += int(not has_whole_pos)
        whole_missing_force += int(not has_whole_force)

        time_raw = float(ts_raw.time)
        time_whole = float(ts_whole.time)
        times.append(time_raw)
        max_time_delta = max(max_time_delta, abs(time_raw - time_whole))

        frame_force_rms = math.nan
        frame_force_diff_rms = math.nan
        frame_force_diff_max = math.nan

        if has_raw_pos:
            pos = np.asarray(ts_raw.positions)
            nonfinite_pos += int(not np.isfinite(pos).all())
        if has_whole_pos:
            pos = np.asarray(ts_whole.positions)
            nonfinite_pos += int(not np.isfinite(pos).all())

        if has_raw_force:
            f_raw = np.asarray(ts_raw.forces, dtype=np.float64)
            nonfinite_force += int(not np.isfinite(f_raw).all())
            frame_force_rms = float(np.sqrt(np.mean(f_raw * f_raw)))
        if has_whole_force:
            f_whole = np.asarray(ts_whole.forces, dtype=np.float64)
            nonfinite_force += int(not np.isfinite(f_whole).all())

        if has_raw_force and has_whole_force:
            delta = f_whole - f_raw
            frame_force_diff_rms = float(np.sqrt(np.mean(delta * delta)))
            frame_force_diff_max = float(np.max(np.abs(delta)))
            max_force_delta = max(max_force_delta, frame_force_diff_max)
            sum_force_delta_sq += float(np.sum(delta * delta, dtype=np.float64))
            sum_force_ref_sq += float(np.sum(f_raw * f_raw, dtype=np.float64))
            force_components += int(f_raw.size)

        rows.append(
            [
                idx,
                time_raw,
                time_whole,
                abs(time_raw - time_whole),
                frame_force_rms,
                frame_force_diff_rms,
                frame_force_diff_max,
            ]
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "raw_time_ps",
                "whole_time_ps",
                "abs_time_delta_ps",
                "raw_force_component_rms_kj_mol_A",
                "raw_whole_force_component_rms_delta_kj_mol_A",
                "raw_whole_force_max_abs_delta_kj_mol_A",
            ]
        )
        writer.writerows(rows)

    time_diffs = np.diff(np.asarray(times, dtype=np.float64))
    positive_diffs = time_diffs[time_diffs > 0.0]
    median_frame_dt = (
        float(np.median(positive_diffs)) if positive_diffs.size else 0.0
    )
    max_frame_dt_deviation = (
        float(np.max(np.abs(positive_diffs - median_frame_dt)))
        if positive_diffs.size
        else 0.0
    )

    result.update(
        {
            "raw_missing_position_frames": raw_missing_pos,
            "raw_missing_force_frames": raw_missing_force,
            "whole_missing_position_frames": whole_missing_pos,
            "whole_missing_force_frames": whole_missing_force,
            "frames_with_nonfinite_positions": nonfinite_pos,
            "frames_with_nonfinite_forces": nonfinite_force,
            "first_time_ps": times[0],
            "last_time_ps": times[-1],
            "median_frame_dt_ps": median_frame_dt,
            "max_frame_dt_deviation_ps": max_frame_dt_deviation,
            "max_raw_whole_time_delta_ps": max_time_delta,
            "raw_whole_force_relative_rms": rel_rms(
                sum_force_delta_sq, sum_force_ref_sq
            ),
            "raw_whole_force_component_rms_delta_kj_mol_A": (
                math.sqrt(sum_force_delta_sq / force_components)
                if force_components
                else math.nan
            ),
            "raw_whole_force_max_abs_delta_kj_mol_A": max_force_delta,
        }
    )
    return result


def compare_rerun(
    topology: Path,
    stored_path: Path,
    rerun_path: Path,
    csv_path: Path,
) -> dict[str, Any]:
    stored = load_universe(topology, stored_path)
    rerun = load_universe(topology, rerun_path)

    result: dict[str, Any] = {
        "stored_subset_frames": len(stored.trajectory),
        "rerun_frames": len(rerun.trajectory),
        "stored_subset_atoms": stored.atoms.n_atoms,
        "rerun_atoms": rerun.atoms.n_atoms,
    }
    if stored.atoms.n_atoms != rerun.atoms.n_atoms:
        raise RuntimeError(
            f"Rerun atom count differs: stored={stored.atoms.n_atoms}, rerun={rerun.atoms.n_atoms}"
        )
    if len(stored.trajectory) != len(rerun.trajectory):
        raise RuntimeError(
            f"Rerun frame count differs: stored={len(stored.trajectory)}, rerun={len(rerun.trajectory)}"
        )
    if len(stored.trajectory) == 0:
        raise RuntimeError("Rerun subset contains zero frames")

    sum_delta_sq = 0.0
    sum_ref_sq = 0.0
    n_components = 0
    max_abs_delta = 0.0
    max_time_delta = 0.0
    rows: list[list[Any]] = []

    for idx in range(len(stored.trajectory)):
        ts_stored = stored.trajectory[idx]
        ts_rerun = rerun.trajectory[idx]
        if not bool(getattr(ts_stored, "has_positions", False)):
            raise RuntimeError(f"Stored subset frame {idx} has no positions")
        if not bool(getattr(ts_stored, "has_forces", False)):
            raise RuntimeError(f"Stored subset frame {idx} has no forces")
        if not bool(getattr(ts_rerun, "has_forces", False)):
            raise RuntimeError(f"Rerun frame {idx} has no forces")

        t0 = float(ts_stored.time)
        t1 = float(ts_rerun.time)
        max_time_delta = max(max_time_delta, abs(t0 - t1))

        f0 = np.asarray(ts_stored.forces, dtype=np.float64)
        f1 = np.asarray(ts_rerun.forces, dtype=np.float64)
        if not np.isfinite(f0).all() or not np.isfinite(f1).all():
            raise RuntimeError(f"Non-finite force in rerun comparison frame {idx}")

        delta = f1 - f0
        frame_ref_rms = float(np.sqrt(np.mean(f0 * f0)))
        frame_delta_rms = float(np.sqrt(np.mean(delta * delta)))
        frame_rel = frame_delta_rms / frame_ref_rms if frame_ref_rms > 0 else math.inf
        frame_max = float(np.max(np.abs(delta)))

        sum_delta_sq += float(np.sum(delta * delta, dtype=np.float64))
        sum_ref_sq += float(np.sum(f0 * f0, dtype=np.float64))
        n_components += int(f0.size)
        max_abs_delta = max(max_abs_delta, frame_max)

        rows.append([idx, t0, t1, frame_ref_rms, frame_delta_rms, frame_rel, frame_max])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "stored_time_ps",
                "rerun_time_ps",
                "stored_force_component_rms_kj_mol_A",
                "rerun_force_component_rms_delta_kj_mol_A",
                "rerun_force_relative_rms",
                "rerun_force_max_abs_delta_kj_mol_A",
            ]
        )
        writer.writerows(rows)

    result.update(
        {
            "max_time_delta_ps": max_time_delta,
            "force_relative_rms": rel_rms(sum_delta_sq, sum_ref_sq),
            "force_component_rms_delta_kj_mol_A": (
                math.sqrt(sum_delta_sq / n_components) if n_components else math.nan
            ),
            "force_max_abs_delta_kj_mol_A": max_abs_delta,
        }
    )
    return result


def choose_skip(n_frames: int, target_frames: int) -> int:
    if target_frames <= 1 or n_frames <= target_frames:
        return 1
    return max(1, int(math.ceil((n_frames - 1) / (target_frames - 1))))


def status_entry(ok: bool, detail: str, level_if_false: str = "FAIL") -> dict[str, str]:
    return {"status": "PASS" if ok else level_if_false, "detail": detail}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    topology = Path(args.topology)
    raw = Path(args.raw)
    whole = Path(args.whole)
    mdp_path = Path(args.mdp_from_tpr)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    mdp = parse_mdp(mdp_path)
    rw = compare_raw_whole(topology, raw, whole, outdir / "raw_vs_whole_frames.csv")

    nstxout = _as_int(mdp.get("nstxout"))
    nstfout = _as_int(mdp.get("nstfout"))
    dt = _as_float(mdp.get("dt"))
    nsteps = _as_int(mdp.get("nsteps"))
    expected_x_dt = dt * nstxout if dt is not None and nstxout and nstxout > 0 else None
    expected_f_dt = dt * nstfout if dt is not None and nstfout and nstfout > 0 else None

    checks: dict[str, dict[str, str]] = {}
    checks["raw_positions_present"] = status_entry(
        rw["raw_missing_position_frames"] == 0,
        f"missing={rw['raw_missing_position_frames']} / {rw['raw_frames']}",
    )
    checks["raw_forces_present"] = status_entry(
        rw["raw_missing_force_frames"] == 0,
        f"missing={rw['raw_missing_force_frames']} / {rw['raw_frames']}",
    )
    checks["whole_positions_present"] = status_entry(
        rw["whole_missing_position_frames"] == 0,
        f"missing={rw['whole_missing_position_frames']} / {rw['whole_frames']}",
    )
    checks["whole_forces_present"] = status_entry(
        rw["whole_missing_force_frames"] == 0,
        f"missing={rw['whole_missing_force_frames']} / {rw['whole_frames']}",
    )
    checks["finite_data"] = status_entry(
        rw["frames_with_nonfinite_positions"] == 0
        and rw["frames_with_nonfinite_forces"] == 0,
        "all trajectory positions/forces finite",
    )
    checks["raw_whole_time_alignment"] = status_entry(
        rw["max_raw_whole_time_delta_ps"] <= args.time_tol,
        f"max |dt|={rw['max_raw_whole_time_delta_ps']:.6g} ps",
    )
    checks["trjconv_force_preservation"] = status_entry(
        rw["raw_whole_force_relative_rms"] <= args.raw_whole_rel_tol,
        (
            f"relative RMS={rw['raw_whole_force_relative_rms']:.6g}; "
            f"max abs delta={rw['raw_whole_force_max_abs_delta_kj_mol_A']:.6g} kJ/(mol A)"
        ),
    )
    checks["tpr_nstfout"] = status_entry(
        nstfout is not None and nstfout > 0,
        f"nstfout={nstfout}",
    )
    checks["tpr_nstxout"] = status_entry(
        nstxout is not None and nstxout > 0,
        f"nstxout={nstxout}",
    )
    if nstxout is not None and nstfout is not None:
        checks["coordinate_force_output_stride"] = status_entry(
            nstxout == nstfout,
            f"nstxout={nstxout}, nstfout={nstfout}",
            level_if_false="WARN",
        )
    else:
        checks["coordinate_force_output_stride"] = {
            "status": "WARN",
            "detail": "could not parse nstxout/nstfout from TPR-derived MDP",
        }

    if expected_x_dt is not None and expected_f_dt is not None:
        observed = rw["median_frame_dt_ps"]
        expected = max(expected_x_dt, expected_f_dt) if nstxout != nstfout else expected_x_dt
        tol = max(args.time_tol, abs(expected) * 1.0e-5)
        checks["observed_frame_spacing"] = status_entry(
            abs(observed - expected) <= tol,
            (
                f"observed median={observed:.6g} ps; expected coordinate={expected_x_dt:.6g} ps; "
                f"expected force={expected_f_dt:.6g} ps"
            ),
            level_if_false="WARN",
        )

    rerun_skip = choose_skip(rw["raw_frames"], args.target_rerun_frames)
    report: dict[str, Any] = {
        "inputs": {
            "topology": str(topology),
            "raw_trr": str(raw),
            "whole_trr": str(whole),
            "mdp_from_tpr": str(mdp_path),
        },
        "tpr_parameters": {
            "integrator": mdp.get("integrator"),
            "dt_ps": dt,
            "nsteps": nsteps,
            "nstxout": nstxout,
            "nstfout": nstfout,
            "nstvout": _as_int(mdp.get("nstvout")),
            "nstxout_compressed": _as_int(mdp.get("nstxout-compressed")),
            "constraints": mdp.get("constraints"),
            "constraint_algorithm": mdp.get("constraint-algorithm"),
            "cutoff_scheme": mdp.get("cutoff-scheme"),
            "coulombtype": mdp.get("coulombtype"),
            "vdwtype": mdp.get("vdwtype"),
            "tcoupl": mdp.get("tcoupl"),
            "pcoupl": mdp.get("pcoupl"),
        },
        "raw_vs_whole": rw,
        "checks": checks,
        "rerun_selection": {
            "target_frames": args.target_rerun_frames,
            "trjconv_skip": rerun_skip,
            "expected_subset_frames_approx": int(math.ceil(rw["raw_frames"] / rerun_skip)),
        },
    }

    if args.stored_subset and args.rerun:
        rr = compare_rerun(
            topology,
            Path(args.stored_subset),
            Path(args.rerun),
            outdir / "stored_vs_rerun_frames.csv",
        )
        report["rerun"] = rr
        checks["rerun_time_alignment"] = status_entry(
            rr["max_time_delta_ps"] <= args.time_tol,
            f"max |dt|={rr['max_time_delta_ps']:.6g} ps",
        )
        checks["rerun_force_reproduction"] = status_entry(
            rr["force_relative_rms"] <= args.rerun_rel_tol,
            (
                f"relative RMS={rr['force_relative_rms']:.6g}; "
                f"RMS delta={rr['force_component_rms_delta_kj_mol_A']:.6g} kJ/(mol A); "
                f"max abs delta={rr['force_max_abs_delta_kj_mol_A']:.6g} kJ/(mol A); "
                f"tolerance={args.rerun_rel_tol:.3g}"
            ),
        )

    return report


def print_summary(report: dict[str, Any]) -> None:
    print("\n======================================================")
    print(" GROMACS FORCE AUDIT SUMMARY")
    print("======================================================")
    for name, entry in report["checks"].items():
        print(f"[{entry['status']:4s}] {name}: {entry['detail']}")
    rw = report["raw_vs_whole"]
    print(
        "[INFO] raw/whole frames="
        f"{rw['raw_frames']} | atoms={rw['raw_atoms']} | "
        f"median dt={rw['median_frame_dt_ps']:.6g} ps"
    )
    print(
        "[INFO] raw->whole force relative RMS="
        f"{rw['raw_whole_force_relative_rms']:.6g}"
    )
    if "rerun" in report:
        rr = report["rerun"]
        print(
            "[INFO] stored->rerun force relative RMS="
            f"{rr['force_relative_rms']:.6g} over {rr['rerun_frames']} frames"
        )
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for entry in report["checks"].values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    print(
        f"[INFO] checks: PASS={counts.get('PASS', 0)} "
        f"WARN={counts.get('WARN', 0)} FAIL={counts.get('FAIL', 0)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--whole", required=True)
    parser.add_argument("--mdp-from-tpr", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-rerun-frames", type=int, default=11)
    parser.add_argument("--stored-subset")
    parser.add_argument("--rerun")
    parser.add_argument("--time-tol", type=float, default=1.0e-5)
    parser.add_argument("--raw-whole-rel-tol", type=float, default=1.0e-7)
    parser.add_argument("--rerun-rel-tol", type=float, default=1.0e-3)
    parser.add_argument("--write-skip-file")
    args = parser.parse_args()

    if bool(args.stored_subset) != bool(args.rerun):
        parser.error("--stored-subset and --rerun must be provided together")
    if args.target_rerun_frames < 1:
        parser.error("--target-rerun-frames must be >= 1")

    try:
        report = build_report(args)
    except Exception as exc:
        print(f"[FAIL] GROMACS force audit could not complete: {exc}", file=sys.stderr)
        return 2

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = outdir / "gromacs_force_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.write_skip_file:
        Path(args.write_skip_file).write_text(
            str(report["rerun_selection"]["trjconv_skip"]) + "\n"
        )

    print_summary(report)
    print(f"[INFO] JSON report: {report_path}")

    return 1 if any(v["status"] == "FAIL" for v in report["checks"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
