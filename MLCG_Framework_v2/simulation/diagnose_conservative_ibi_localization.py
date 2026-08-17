#!/usr/bin/env python3
"""End-to-end localization suite for anomalous conservative-IBI sigma(E) scaling.

The suite is diagnostic-only.  It does not alter certification status or the
conservative spline kernel.  It combines:

1. U'' knot-jump inspection for every unique conservative table;
2. minimal bond/angle/rigid-angle NVE dynamics and reversibility;
3. configured no-IBI / bond-only / angle-only / full sigma(E) scans;
4. full-system translational/rotational finite-difference gradient checks;
5. knot-crossing vs per-step energy-error correlation;
6. kinetic/bonded/non-bonded energy decomposition;
7. link-cell vs all-pairs N-square state/energy A/B;
8. full-system time reversibility.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from conservative_ibi_energy_diagnostics import (  # noqa: E402
    analyze_energy_decomposition,
    analyze_knot_trace,
    compare_time_reversal,
    inspect_prior_smoothness,
    reverse_checkpoint_velocities,
    write_diagnostic_prior_variants,
)
from nve_analysis import analyze_energy_series, fit_metric_scaling, read_energy_csv  # noqa: E402
from nve_state_convergence import load_state_trajectory, state_error_at_frame  # noqa: E402


def resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    found = shutil.which(value)
    if found:
        return found
    raise FileNotFoundError(f"Executable not found: {value}")


def _run(command: list[str], *, log: Path, dry_run: bool) -> None:
    print("[CMD] " + " ".join(command))
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = log.read_text(errors="replace").splitlines()[-60:]
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(command)}\n"
            + "\n".join(tail)
        )


def _run_cg_base(args, priors: Path, dt: float, steps: int, energy: Path, *, neighbor: str, allow_mismatch: bool = False):
    cmd = [
        args.pypresso,
        str(ROOT / "simulation" / "run_cg_md.py"),
        "--model", str(args.model), "--disable_ml",
        "--config", str(args.config),
        "--priors", str(priors),
        "--rb_info", str(args.rb_info),
        "--dataset", str(args.dataset),
        "--checkpoint", str(args.checkpoint),
        "--dt", f"{dt:.17g}",
        "--steps", str(int(steps)),
        "--log_interval", "1",
        "--nve", "--device", args.device,
        "--ml_precision", args.ml_precision,
        "--neighbor_search", neighbor,
        "--energy_file", str(energy),
        "--no_vtf",
    ]
    if allow_mismatch:
        cmd.append("--allow_checkpoint_mismatch")
    return cmd


def _analyze_energy(path: Path, dt: float) -> dict[str, Any]:
    times, energy = read_energy_csv(path)
    metrics = analyze_energy_series(times, energy)
    metrics["dt_ps"] = float(dt)
    metrics["energy_file"] = str(path)
    return metrics


def _state_ab(path_a: Path, path_b: Path) -> dict[str, Any]:
    a = load_state_trajectory(path_a)
    b = load_state_trajectory(path_b)
    common = sorted(set(np.round(a["time_ps"], 14)).intersection(set(np.round(b["time_ps"], 14))))
    if len(common) < 2:
        raise ValueError("Neighbor-search A/B trajectories have no common sampled times")
    rows = []
    for t in common:
        ia = int(np.where(np.isclose(a["time_ps"], t, atol=1e-12, rtol=0.0))[0][0])
        ib = int(np.where(np.isclose(b["time_ps"], t, atol=1e-12, rtol=0.0))[0][0])
        row = state_error_at_frame(a, ia, b, ib)
        row["time_ps"] = float(t)
        rows.append(row)
    keys = ["position_rms_nm", "velocity_rms_nm_per_ps", "orientation_rms_rad", "omega_body_rms_per_ps"]
    return {
        "samples": int(len(rows)),
        "final": rows[-1],
        "max": {key: max(float(row[key]) for row in rows) for key in keys},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypresso", default="pypresso")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--priors", required=True, type=Path)
    parser.add_argument("--rb-info", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dts", nargs="+", type=float, required=True)
    parser.add_argument("--duration-ps", type=float, required=True)
    parser.add_argument("--micro-duration-ps", type=float, required=True)
    parser.add_argument("--trace-dt", type=float, required=True)
    parser.add_argument("--reversibility-dt", type=float, required=True)
    parser.add_argument("--reversibility-duration-ps", type=float, required=True)
    parser.add_argument("--neighbor-duration-ps", type=float, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ml-precision", choices=("float32", "float64"), required=True)
    parser.add_argument("--fd-max-bodies", type=int, required=True)
    parser.add_argument("--fine-max-dt", type=float, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pypresso = resolve_executable(args.pypresso)
    args.model = args.model.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.priors = args.priors.expanduser().resolve()
    args.rb_info = args.rb_info.expanduser().resolve()
    args.dataset = args.dataset.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in (args.model, Path(str(args.model) + ".manifest.json"), args.config, args.priors, args.rb_info, args.dataset, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.duration_ps <= 0 or args.micro_duration_ps <= 0 or min(args.dts) <= 0:
        raise ValueError("All durations and time steps must be positive")
    if args.trace_dt not in args.dts:
        raise ValueError("--trace-dt must be one of --dts so its full-system run can be reused")
    if args.output_dir.exists() and args.overwrite and not args.dry_run:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[CONSERVATIVE IBI ENERGY-LOCALIZATION PLAN]")
    print(f"priors       : {args.priors}")
    print(f"checkpoint   : {args.checkpoint}")
    print(f"full dt scan : {' '.join(f'{x:g}' for x in args.dts)} ps")
    print(f"duration     : {args.duration_ps:g} ps per full-system dt")
    print(f"micro        : {args.micro_duration_ps:g} ps per minimal-system dt")
    print(f"trace dt     : {args.trace_dt:g} ps")
    print("variants     : no_ibi / bonds_only / angles_only / full")
    print("neighbor A/B : link-cell / nsquare")
    print(f"output       : {args.output_dir}")
    print("[NOTE] Diagnostic-only: no certification artifact is modified.")

    plan = {
        "schema_version": 1,
        "kind": "conservative_ibi_energy_localization_plan",
        "priors": str(args.priors), "checkpoint": str(args.checkpoint),
        "dts_ps": [float(x) for x in args.dts], "duration_ps": float(args.duration_ps),
        "micro_duration_ps": float(args.micro_duration_ps), "trace_dt_ps": float(args.trace_dt),
        "variants": ["no_ibi", "bonds_only", "angles_only", "full"],
        "neighbor_modes": ["link-cell", "nsquare"],
    }
    (args.output_dir / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        for variant in plan["variants"]:
            for dt in args.dts:
                steps = int(round(args.duration_ps / dt))
                print(f"[PLAN] {variant:11s} dt={dt:g} ps steps={steps}")
        print("[DONE] Dry-run only; no dynamics launched.")
        return

    smooth = inspect_prior_smoothness(args.priors)
    (args.output_dir / "spline_smoothness_report.json").write_text(json.dumps(smooth, indent=2, sort_keys=True) + "\n")
    variants = write_diagnostic_prior_variants(args.priors, args.output_dir / "priors_variants")

    micro_report = args.output_dir / "minimal_dynamics_report.json"
    _run([
        args.pypresso, str(ROOT / "simulation" / "diagnose_conservative_spline_dynamics.py"),
        "--priors", str(args.priors), "--output", str(micro_report),
        "--duration-ps", f"{args.micro_duration_ps:.17g}",
        "--dts", *[f"{x:.17g}" for x in args.dts if x <= args.fine_max_dt],
        "--reversibility-dt", f"{args.reversibility_dt:.17g}",
        "--reversibility-duration-ps", f"{args.reversibility_duration_ps:.17g}",
    ], log=args.output_dir / "minimal_dynamics.log", dry_run=False)

    fd_energy = args.output_dir / "full_generalized_fd_energy.csv"
    fd_report = args.output_dir / "full_generalized_fd_report.json"
    fd_cmd = _run_cg_base(args, args.priors, args.trace_dt, 0, fd_energy, neighbor="link-cell")
    fd_cmd += ["--generalized_fd_report", str(fd_report), "--generalized_fd_max_bodies", str(args.fd_max_bodies)]
    _run(fd_cmd, log=args.output_dir / "full_generalized_fd.log", dry_run=False)

    scan_report: dict[str, Any] = {}
    full_trace_sample = args.output_dir / "full_trace_sample.npz"
    link_state = args.output_dir / "neighbor_linkcell_state.npz"
    for variant in ("no_ibi", "bonds_only", "angles_only", "full"):
        variant_dir = args.output_dir / "sigma_scan" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        priors = args.priors if variant == "full" else Path(variants[variant])
        runs = []
        for dt in args.dts:
            steps = max(4, int(round(args.duration_ps / dt)))
            energy = variant_dir / f"energy_dt_{dt:.9g}.csv"
            cmd = _run_cg_base(args, priors, dt, steps, energy, neighbor="link-cell", allow_mismatch=(variant != "full"))
            if variant == "full" and math.isclose(dt, args.trace_dt, rel_tol=0.0, abs_tol=1e-15):
                cmd += ["--sample_npz", str(full_trace_sample), "--state_sample_npz", str(link_state)]
            _run(cmd, log=variant_dir / f"run_dt_{dt:.9g}.log", dry_run=False)
            runs.append(_analyze_energy(energy, dt))
        scan_report[variant] = {"runs": runs, "sigma_fit": fit_metric_scaling(runs, "sigma_E", label="sigma_E")}

    knot = analyze_knot_trace(priors_path=args.priors, sample_npz=full_trace_sample,
                              energy_csv=Path(scan_report["full"]["runs"][[r["dt_ps"] for r in scan_report["full"]["runs"]].index(args.trace_dt)]["energy_file"]))
    decomposition = analyze_energy_decomposition(Path(scan_report["full"]["runs"][[r["dt_ps"] for r in scan_report["full"]["runs"]].index(args.trace_dt)]["energy_file"]))

    # Neighbor-search A/B over a short common physical interval.
    neighbor_steps = max(4, int(round(args.neighbor_duration_ps / args.trace_dt)))
    ns_energy = args.output_dir / "neighbor_nsquare_energy.csv"
    ns_state = args.output_dir / "neighbor_nsquare_state.npz"
    cmd = _run_cg_base(args, args.priors, args.trace_dt, neighbor_steps, ns_energy, neighbor="nsquare")
    cmd += ["--state_sample_npz", str(ns_state)]
    _run(cmd, log=args.output_dir / "neighbor_nsquare.log", dry_run=False)
    # The link-cell state from the full scan is longer; compare only common times.
    neighbor_ab = _state_ab(link_state, ns_state)
    link_energy_file = Path(scan_report["full"]["runs"][[r["dt_ps"] for r in scan_report["full"]["runs"]].index(args.trace_dt)]["energy_file"])
    t_link, e_link = read_energy_csv(link_energy_file)
    t_ns, e_ns = read_energy_csv(ns_energy)
    n = min(len(t_link), len(t_ns))
    neighbor_ab["energy_max_abs_difference"] = float(np.max(np.abs(e_link[:n] - e_ns[:n])))
    neighbor_ab["energy_sigma_linkcell_short"] = float(np.std(e_link[:n]))
    neighbor_ab["energy_sigma_nsquare"] = float(np.std(e_ns[:n]))

    # Full-system time reversibility.
    rev_dir = args.output_dir / "time_reversal"
    rev_dir.mkdir(parents=True, exist_ok=True)
    rev_steps = max(2, int(round(args.reversibility_duration_ps / args.reversibility_dt)))
    forward = rev_dir / "forward.npz"
    forward_energy = rev_dir / "forward_energy.csv"
    cmd = _run_cg_base(args, args.priors, args.reversibility_dt, rev_steps, forward_energy, neighbor="link-cell")
    cmd += ["--out_checkpoint", str(forward)]
    _run(cmd, log=rev_dir / "forward.log", dry_run=False)
    reversed_chk = rev_dir / "forward_velocity_reversed.npz"
    reverse_checkpoint_velocities(forward, reversed_chk)
    returned = rev_dir / "returned.npz"
    returned_energy = rev_dir / "returned_energy.csv"
    # The reversed checkpoint carries the same physical-input hashes, but its
    # source-checkpoint metadata is diagnostic. Runtime input validation remains strict.
    cmd = _run_cg_base(args, args.priors, args.reversibility_dt, rev_steps, returned_energy, neighbor="link-cell")
    # Replace the original checkpoint argument with the reversed one.
    idx = cmd.index("--checkpoint") + 1
    cmd[idx] = str(reversed_chk)
    cmd += ["--out_checkpoint", str(returned)]
    _run(cmd, log=rev_dir / "returned.log", dry_run=False)
    reversal = compare_time_reversal(args.checkpoint, returned)

    report = {
        "schema_version": 1,
        "kind": "conservative_ibi_energy_scaling_localization",
        "diagnostic_only": True,
        "smoothness": smooth,
        "minimal_dynamics_report": str(micro_report),
        "full_generalized_fd_report": str(fd_report),
        "sigma_scans": scan_report,
        "knot_crossing_correlation": knot,
        "energy_decomposition": decomposition,
        "neighbor_search_ab": neighbor_ab,
        "time_reversal": reversal,
    }
    target = args.output_dir / "localization_report.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[CONSERVATIVE IBI ENERGY-LOCALIZATION SUMMARY]")
    for variant, data in scan_report.items():
        fit = data["sigma_fit"]
        print(f"sigma_E {variant:11s}: p={fit['exponent_p']:.6f} R2={fit['loglog_r2']:.6f}")
    print(
        "knot crossings: "
        f"ratio=<|dE|>cross/<|dE|>no={knot['crossing_to_no_crossing_abs_delta_E_ratio']:.6g} "
        f"corr(count,|dE|)={knot['pearson_crossing_count_vs_abs_delta_E']:.6g} "
        f"corr(U2jump,|dE|)={knot['pearson_weighted_u2_jump_vs_abs_delta_E']:.6g}"
    )
    print(
        "neighbor A/B final: "
        f"dr={neighbor_ab['final']['position_rms_nm']:.3e} "
        f"dv={neighbor_ab['final']['velocity_rms_nm_per_ps']:.3e} "
        f"dtheta={neighbor_ab['final']['orientation_rms_rad']:.3e}"
    )
    print(
        "time reversal: "
        f"dr={reversal['position_rms_nm']:.3e} dv={reversal['velocity_rms_nm_per_ps']:.3e} "
        f"dtheta={reversal['orientation_rms_rad']:.3e} domega={reversal['omega_body_rms_per_ps']:.3e}"
    )
    print(f"[DONE] report: {target}")


if __name__ == "__main__":
    main()
