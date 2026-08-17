#!/usr/bin/env python3
"""Replica + long-structure validation for the selected smoothed IBI-angle prior.

This is a diagnostic validation gate only.  It never promotes or overwrites the
selected production priors.  One short-NVT/NVE replica may be reused from the
step-32 sweep; independent replicas and a matched long-NVT structural A/B are
then added before a single pass/fail summary is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "simulation", ROOT / "ibi"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ibi_angle_candidate_validation import (  # noqa: E402
    fit_sigma_range,
    nvt_energy_summary,
    structural_report,
)

SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _same_grid(values: list[float], expected: list[float], *, atol: float = 1e-15) -> bool:
    a = np.asarray(values, dtype=float)
    b = np.asarray(expected, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=atol))


def fixed_effects_sigma_slope(replica_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit log(sigma)=alpha_replica+p*log(dt), eliminating replica intercepts."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    names: list[str] = []
    for row in replica_rows:
        sr = row["sigma_range"]
        if not sr.get("available"):
            raise ValueError(f"Replica {row.get('name')} has unavailable sigma range")
        dt = np.asarray(sr["dt_ps"], dtype=float)
        sigma = np.asarray(sr["sigma_E"], dtype=float)
        if len(dt) < 3 or np.any(dt <= 0) or np.any(sigma <= 0):
            raise ValueError("Each replica needs at least three positive dt/sigma points")
        xs.append(np.log(dt))
        ys.append(np.log(sigma))
        names.append(str(row.get("name", f"replica{len(names)}")))

    xdm = [x - np.mean(x) for x in xs]
    ydm = [y - np.mean(y) for y in ys]
    denom = float(sum(np.dot(x, x) for x in xdm))
    if denom <= np.finfo(float).eps:
        raise ValueError("Degenerate dt grid for fixed-effects fit")
    p = float(sum(np.dot(x, y) for x, y in zip(xdm, ydm)) / denom)

    sse = 0.0
    sst = 0.0
    intercepts: dict[str, float] = {}
    for name, x, y in zip(names, xs, ys):
        alpha = float(np.mean(y) - p * np.mean(x))
        intercepts[name] = alpha
        pred = alpha + p * x
        sse += float(np.sum((y - pred) ** 2))
        centered = y - np.mean(y)
        sst += float(np.sum(centered * centered))
    r2 = 1.0 if sst <= np.finfo(float).eps else 1.0 - sse / sst
    return {
        "model": "log(sigma_E) = alpha_replica + p * log(dt)",
        "exponent_p": p,
        "within_replica_r2": float(r2),
        "replica_log_intercepts": intercepts,
        "n_replicas": len(replica_rows),
        "n_points": int(sum(len(x) for x in xs)),
    }


def replica_gate(
    replica_rows: list[Mapping[str, Any]],
    *,
    common_p_min: float,
    common_p_max: float,
    common_r2_min: float,
    full_clean_dt: float,
    min_clean_dt: float,
    min_full_clean_replicas: int,
    median_c2_spread_max: float,
    max_relative_block_drift: float,
) -> dict[str, Any]:
    common = fixed_effects_sigma_slope(replica_rows)
    clean = [float(r["sigma_range"]["max_clean_dt_factor_1p5"]) for r in replica_rows]
    spreads = [float(r["sigma_range"]["c2_spread_max_over_min"]) for r in replica_rows]
    max_drifts = [float(r.get("max_relative_block_drift", math.inf)) for r in replica_rows]
    n_full = sum(d >= full_clean_dt - 1e-15 for d in clean)
    checks = {
        "common_p": common_p_min <= float(common["exponent_p"]) <= common_p_max,
        "common_r2": float(common["within_replica_r2"]) >= common_r2_min,
        "enough_full_clean_replicas": n_full >= min_full_clean_replicas,
        "all_replicas_min_clean_dt": min(clean) >= min_clean_dt - 1e-15,
        "median_c2_spread": float(np.median(spreads)) <= median_c2_spread_max,
        "max_relative_block_drift": max(max_drifts) <= max_relative_block_drift,
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "common_fit": common,
        "clean_dt_factor_1p5_ps": clean,
        "n_full_clean_replicas": int(n_full),
        "c2_spreads": spreads,
        "median_c2_spread": float(np.median(spreads)),
        "max_relative_block_drifts": max_drifts,
        "thresholds": {
            "common_p_min": common_p_min,
            "common_p_max": common_p_max,
            "common_r2_min": common_r2_min,
            "full_clean_dt_ps": full_clean_dt,
            "min_clean_dt_ps": min_clean_dt,
            "min_full_clean_replicas": min_full_clean_replicas,
            "median_c2_spread_max": median_c2_spread_max,
            "max_relative_block_drift": max_relative_block_drift,
        },
    }


def structural_gate(
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    weighted_angle_delta_max: float,
    weighted_bond_delta_max: float,
    max_group_angle_delta_max: float,
    kinetic_relative_delta_max: float,
    curvature_reduction_min: float,
) -> dict[str, Any]:
    cur_s = current["structural"]
    can_s = candidate["structural"]
    cur_a = float(cur_s["angles"]["summary"]["weighted_mean_l1"])
    can_a = float(can_s["angles"]["summary"]["weighted_mean_l1"])
    cur_b = float(cur_s["bonds"]["summary"]["weighted_mean_l1"])
    can_b = float(can_s["bonds"]["summary"]["weighted_mean_l1"])
    group_deltas = {
        name: float(can_s["angles"]["groups"][name]["l1_runtime_vs_target"])
        - float(cur_s["angles"]["groups"][name]["l1_runtime_vs_target"])
        for name in cur_s["angles"]["groups"]
    }
    cur_k = float(current["nvt_kinetic_energy"]["mean_E_kin_second_half"])
    can_k = float(candidate["nvt_kinetic_energy"]["mean_E_kin_second_half"])
    krel = abs(can_k - cur_k) / max(abs(cur_k), np.finfo(float).eps)
    cur_u2 = float(cur_s["angle_curvature_runtime"]["p99_abs"])
    can_u2 = float(can_s["angle_curvature_runtime"]["p99_abs"])
    curvature_reduction = cur_u2 / can_u2 if can_u2 > 0 else math.inf
    checks = {
        "weighted_angle_l1": (can_a - cur_a) <= weighted_angle_delta_max,
        "weighted_bond_l1": (can_b - cur_b) <= weighted_bond_delta_max,
        "max_group_angle_l1": max(group_deltas.values()) <= max_group_angle_delta_max,
        "kinetic_energy_match": krel <= kinetic_relative_delta_max,
        "curvature_reduction": curvature_reduction >= curvature_reduction_min,
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "current_angle_weighted_l1": cur_a,
        "candidate_angle_weighted_l1": can_a,
        "delta_angle_weighted_l1": can_a - cur_a,
        "current_bond_weighted_l1": cur_b,
        "candidate_bond_weighted_l1": can_b,
        "delta_bond_weighted_l1": can_b - cur_b,
        "angle_group_l1_deltas": group_deltas,
        "max_angle_group_l1_delta": max(group_deltas.values()),
        "kinetic_relative_delta": krel,
        "current_angle_p99_curvature": cur_u2,
        "candidate_angle_p99_curvature": can_u2,
        "angle_p99_curvature_reduction": curvature_reduction,
        "thresholds": {
            "weighted_angle_delta_max": weighted_angle_delta_max,
            "weighted_bond_delta_max": weighted_bond_delta_max,
            "max_group_angle_delta_max": max_group_angle_delta_max,
            "kinetic_relative_delta_max": kinetic_relative_delta_max,
            "curvature_reduction_min": curvature_reduction_min,
        },
    }


def _max_relative_block_drift(variant: Mapping[str, Any]) -> float:
    vals = [
        abs(float(r["relative_block_mean_drift"]))
        for r in variant["nve_runs"]
        if r.get("status", "ok") == "ok" and "relative_block_mean_drift" in r
    ]
    return max(vals) if vals else math.inf


def _run_subvalidation(args: argparse.Namespace, *, seed: int, output_dir: Path) -> dict[str, Any]:
    cmd = [
        args.python_bin,
        str(ROOT / "simulation" / "ibi_angle_candidate_validation.py"),
        "--pypresso", args.pypresso,
        "--model", str(args.model),
        "--config", str(args.config),
        "--dataset", str(args.dataset),
        "--rb-info", str(args.rb_info),
        "--source-checkpoint", str(args.source_checkpoint),
        "--ibi-config", str(args.ibi_config),
        "--variant", f"current={args.candidate_priors}",
        "--dts", *[f"{dt:.17g}" for dt in args.dts],
        "--duration-ps", f"{args.nve_duration_ps:.17g}",
        "--branch-dt", f"{args.short_branch_dt:.17g}",
        "--branch-duration-ps", f"{args.short_branch_duration_ps:.17g}",
        "--kT", f"{args.kT:.17g}",
        "--thermostat-seed", str(seed),
        "--device", args.device,
        "--ml-precision", args.ml_precision,
        "--neighbor-search", args.neighbor_search,
        "--output-dir", str(output_dir),
    ]
    if args.resume:
        cmd.append("--resume")
    elif args.overwrite:
        cmd.append("--overwrite")
    print(f"[RUN] independent replica seed={seed}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Replica seed {seed} failed with exit code {proc.returncode}")
    report = json.loads((output_dir / "angle_candidate_validation_report.json").read_text())
    return report["variants"]["current"]


def _run_long_nvt(
    args: argparse.Namespace,
    *,
    name: str,
    priors: Path,
    output_dir: Path,
) -> dict[str, Any]:
    steps = int(round(args.long_branch_duration_ps / args.long_branch_dt))
    actual = steps * args.long_branch_dt
    if not math.isclose(actual, args.long_branch_duration_ps, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Long NVT duration must be an integer number of timesteps")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "nvt_checkpoint.npz"
    sample = output_dir / "nvt_structured_sample.npz"
    energy = output_dir / "nvt_energy.csv"
    complete = args.resume and checkpoint.is_file() and sample.is_file() and energy.is_file()
    if complete:
        print(f"[REUSE] long NVT {name}", flush=True)
    else:
        if output_dir.exists() and args.overwrite:
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.pypresso, str(ROOT / "simulation" / "run_cg_md.py"),
            "--model", str(args.model), "--disable_ml", "--config", str(args.config),
            "--priors", str(priors), "--rb_info", str(args.rb_info), "--dataset", str(args.dataset),
            "--checkpoint", str(args.source_checkpoint), "--allow_checkpoint_mismatch",
            "--dt", f"{args.long_branch_dt:.17g}", "--steps", str(steps), "--log_interval", "1",
            "--device", args.device, "--ml_precision", args.ml_precision,
            "--neighbor_search", args.neighbor_search, "--energy_file", str(energy), "--no_vtf",
            "--kT", f"{args.kT:.17g}", "--thermostat_seed", str(args.long_thermostat_seed),
            "--out_checkpoint", str(checkpoint), "--sample_npz", str(sample),
            "--sample_start_step", str(steps // 2),
        ]
        print(f"[RUN] long NVT {name}: {actual:g} ps ({steps} steps)", flush=True)
        log = output_dir / "nvt.log"
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").splitlines()[-30:])
            raise RuntimeError(f"Long NVT {name} failed ({proc.returncode})\n{tail}")
    struct = structural_report(args.dataset, priors, args.ibi_config, sample)
    kin = nvt_energy_summary(energy)
    return {
        "priors": str(priors),
        "checkpoint": str(checkpoint),
        "sample": str(sample),
        "nvt_energy": str(energy),
        "nvt_kinetic_energy": kin,
        "structural": struct,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--pypresso", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--rb-info", type=Path, required=True)
    p.add_argument("--source-checkpoint", type=Path, required=True)
    p.add_argument("--ibi-config", type=Path, required=True)
    p.add_argument("--current-priors", type=Path, required=True)
    p.add_argument("--candidate-priors", type=Path, required=True)
    p.add_argument("--step32-report", type=Path, required=True)
    p.add_argument("--step32-candidate-name", required=True)
    p.add_argument("--expected-sigma-rad", type=float, required=True)
    p.add_argument("--dts", type=float, nargs="+", required=True)
    p.add_argument("--nve-duration-ps", type=float, required=True)
    p.add_argument("--short-branch-dt", type=float, required=True)
    p.add_argument("--short-branch-duration-ps", type=float, required=True)
    p.add_argument("--new-replica-seeds", type=int, nargs="+", required=True)
    p.add_argument("--long-branch-dt", type=float, required=True)
    p.add_argument("--long-branch-duration-ps", type=float, required=True)
    p.add_argument("--long-thermostat-seed", type=int, required=True)
    p.add_argument("--kT", type=float, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--ml-precision", required=True)
    p.add_argument("--neighbor-search", choices=("verlet", "link-cell"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--common-p-min", type=float, required=True)
    p.add_argument("--common-p-max", type=float, required=True)
    p.add_argument("--common-r2-min", type=float, required=True)
    p.add_argument("--full-clean-dt", type=float, required=True)
    p.add_argument("--min-clean-dt", type=float, required=True)
    p.add_argument("--min-full-clean-replicas", type=int, required=True)
    p.add_argument("--median-c2-spread-max", type=float, required=True)
    p.add_argument("--max-relative-block-drift", type=float, required=True)
    p.add_argument("--weighted-angle-delta-max", type=float, required=True)
    p.add_argument("--weighted-bond-delta-max", type=float, required=True)
    p.add_argument("--max-group-angle-delta-max", type=float, required=True)
    p.add_argument("--kinetic-relative-delta-max", type=float, required=True)
    p.add_argument("--curvature-reduction-min", type=float, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    args.dts = sorted({float(x) for x in args.dts})
    if len(args.dts) < 3 or min(args.dts) <= 0:
        raise ValueError("Need at least three positive dt values")
    if len(args.new_replica_seeds) < 1 or len(set(args.new_replica_seeds)) != len(args.new_replica_seeds):
        raise ValueError("Need unique new replica seeds")
    for attr in (
        "model", "config", "dataset", "rb_info", "source_checkpoint", "ibi_config",
        "current_priors", "candidate_priors", "step32_report",
    ):
        path = getattr(args, attr).expanduser().resolve()
        setattr(args, attr, path)
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir = args.output_dir.expanduser().resolve()

    step32 = json.loads(args.step32_report.read_text())
    selected = step32.get("candidates", {}).get(args.step32_candidate_name)
    if selected is None:
        raise KeyError(f"Step-32 report has no candidate {args.step32_candidate_name!r}")
    if not math.isclose(float(selected.get("sigma_rad", math.nan)), args.expected_sigma_rad, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Step-32 candidate sigma does not match expected sigma")
    report_priors = Path(str(selected.get("priors", ""))).expanduser().resolve()
    if not report_priors.is_file() or sha256_file(report_priors) != sha256_file(args.candidate_priors):
        raise ValueError("Candidate priors do not match the selected step-32 artifact")
    candidate_payload = json.loads(args.candidate_priors.read_text())
    meta = candidate_payload.get("regularization_candidate", {})
    if not math.isclose(float(meta.get("body_sigma_rad", math.nan)), args.expected_sigma_rad, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Candidate metadata body_sigma_rad mismatch")

    nve_steps = sum(int(round(args.nve_duration_ps / dt)) for dt in args.dts)
    short_steps = int(round(args.short_branch_duration_ps / args.short_branch_dt))
    long_steps = int(round(args.long_branch_duration_ps / args.long_branch_dt))
    new_steps = len(args.new_replica_seeds) * (short_steps + nve_steps) + 2 * long_steps
    print("[IBI ANGLE FINAL-CANDIDATE VALIDATION PLAN]")
    print(f"candidate      : {args.step32_candidate_name} sigma={args.expected_sigma_rad:g} rad")
    print("replicas       : step32 reused + " + " + ".join(str(s) for s in args.new_replica_seeds))
    print("dt scan        : " + " ".join(f"{x:g}" for x in args.dts) + " ps")
    print(f"NVE duration   : {args.nve_duration_ps:g} ps per dt")
    print(f"short NVT      : {args.short_branch_duration_ps:g} ps at dt={args.short_branch_dt:g} ps")
    print(f"long structural: current + candidate, {args.long_branch_duration_ps:g} ps each at dt={args.long_branch_dt:g} ps")
    print(f"new integration: about {new_steps} steps (step32 replica reused at zero cost)")
    print(f"output         : {args.output_dir}")
    print("[NOTE] Validation-only. No priors are promoted or overwritten.")
    if args.dry_run:
        return

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    elif args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; use --resume or --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    replicas: list[dict[str, Any]] = []
    reused_sr = selected["sigma_range_common_grid"]
    if not _same_grid(list(reused_sr["dt_ps"]), args.dts):
        raise ValueError("Step-32 reused replica does not use the requested dt grid")
    reused = {
        "name": "replica00_step32",
        "source": "reused_step32",
        "thermostat_seed": int(step32.get("thermostat_seed", -1)),
        "sigma_range": reused_sr,
        "max_relative_block_drift": _max_relative_block_drift({"nve_runs": selected["nve_runs_common_grid"]}),
        "nve_runs": selected["nve_runs_common_grid"],
        "checkpoint": selected.get("checkpoint"),
    }
    replicas.append(reused)
    print(
        f"[REUSE] replica00 step32 p={reused_sr['fit']['exponent_p']:.4f} "
        f"R2={reused_sr['fit']['loglog_r2']:.4f} clean1.5x={reused_sr['max_clean_dt_factor_1p5']:.3g} ps"
    )

    for i, seed in enumerate(args.new_replica_seeds, start=1):
        rdir = args.output_dir / "replicas" / f"replica{i:02d}_seed{seed}"
        variant = _run_subvalidation(args, seed=seed, output_dir=rdir)
        sr = variant["sigma_range"]
        row = {
            "name": f"replica{i:02d}_seed{seed}",
            "source": "new_step33",
            "thermostat_seed": int(seed),
            "sigma_range": sr,
            "max_relative_block_drift": _max_relative_block_drift(variant),
            "nve_runs": variant["nve_runs"],
            "checkpoint": variant.get("checkpoint"),
            "nvt_kinetic_energy": variant.get("nvt_kinetic_energy"),
        }
        replicas.append(row)
        print(
            f"[REPLICA] {row['name']} p={sr['fit']['exponent_p']:.4f} "
            f"R2={sr['fit']['loglog_r2']:.4f} C2spread={sr['c2_spread_max_over_min']:.3f} "
            f"clean1.5x={sr['max_clean_dt_factor_1p5']:.3g} ps"
        )

    rg = replica_gate(
        replicas,
        common_p_min=args.common_p_min,
        common_p_max=args.common_p_max,
        common_r2_min=args.common_r2_min,
        full_clean_dt=args.full_clean_dt,
        min_clean_dt=args.min_clean_dt,
        min_full_clean_replicas=args.min_full_clean_replicas,
        median_c2_spread_max=args.median_c2_spread_max,
        max_relative_block_drift=args.max_relative_block_drift,
    )

    long_current = _run_long_nvt(args, name="current", priors=args.current_priors, output_dir=args.output_dir / "long_structure" / "current")
    long_candidate = _run_long_nvt(args, name="candidate", priors=args.candidate_priors, output_dir=args.output_dir / "long_structure" / "candidate")
    sg = structural_gate(
        long_current,
        long_candidate,
        weighted_angle_delta_max=args.weighted_angle_delta_max,
        weighted_bond_delta_max=args.weighted_bond_delta_max,
        max_group_angle_delta_max=args.max_group_angle_delta_max,
        kinetic_relative_delta_max=args.kinetic_relative_delta_max,
        curvature_reduction_min=args.curvature_reduction_min,
    )

    overall_pass = bool(rg["pass"] and sg["pass"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ibi_angle_final_candidate_validation",
        "diagnostic_only": True,
        "validated": overall_pass,
        "candidate_name": args.step32_candidate_name,
        "candidate_sigma_rad": float(args.expected_sigma_rad),
        "candidate_priors": str(args.candidate_priors),
        "candidate_priors_sha256": sha256_file(args.candidate_priors),
        "current_priors": str(args.current_priors),
        "step32_report": str(args.step32_report),
        "dts_ps": args.dts,
        "nve_duration_ps": float(args.nve_duration_ps),
        "short_branch_dt_ps": float(args.short_branch_dt),
        "short_branch_duration_ps": float(args.short_branch_duration_ps),
        "long_branch_dt_ps": float(args.long_branch_dt),
        "long_branch_duration_ps": float(args.long_branch_duration_ps),
        "kT": float(args.kT),
        "estimated_new_integration_steps": int(new_steps),
        "replicas": replicas,
        "replica_gate": rg,
        "long_structure": {"current": long_current, "candidate": long_candidate, "gate": sg},
        "pass": overall_pass,
        "note": "Passing this step validates the candidate for promotion consideration; it does not modify production priors or replace final Hamiltonian certification.",
    }
    out = args.output_dir / "angle_final_candidate_validation_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[FINAL CANDIDATE VALIDATION]")
    print(
        f"[NVE] common p={rg['common_fit']['exponent_p']:.6f} "
        f"R2within={rg['common_fit']['within_replica_r2']:.6f} "
        f"full-clean={rg['n_full_clean_replicas']}/{len(replicas)} "
        f"medianC2spread={rg['median_c2_spread']:.3f} pass={rg['pass']}"
    )
    print(
        f"[STRUCT] dAngleL1={sg['delta_angle_weighted_l1']:+.6f} "
        f"dBondL1={sg['delta_bond_weighted_l1']:+.6f} "
        f"maxGroupAngleDelta={sg['max_angle_group_l1_delta']:+.6f} "
        f"kineticRelDelta={sg['kinetic_relative_delta']:.3e} "
        f"P99U2red={sg['angle_p99_curvature_reduction']:.3f}x pass={sg['pass']}"
    )
    print(f"[FINAL] pass={overall_pass}")
    print(f"[DONE] report: {out}")
    print("[NOTE] No automatic promotion was performed.")


if __name__ == "__main__":
    main()
