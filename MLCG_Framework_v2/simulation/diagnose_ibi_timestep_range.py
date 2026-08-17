#!/usr/bin/env python3
"""Matched reference-prior vs conservative-IBI coarse-timestep localization.

Runs a short NVT branch for four matched bonded Hamiltonians, then reuses the
standard NVE diagnostic on the configured timestep grid.  In parallel it
measures U'' only in the coordinate ranges actually visited by each branch.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from nve_analysis import analyze_energy_series, read_energy_csv  # noqa: E402
from ibi_timestep_range_diagnostics import (  # noqa: E402
    sigma_range_diagnostics,
    stiffness_ratios,
    visited_curvature_report,
    write_matched_prior_variants,
)


def resolve_executable(value: str) -> str:
    p = Path(value).expanduser()
    if p.exists(): return str(p.resolve())
    found = shutil.which(value)
    if found: return found
    raise FileNotFoundError(value)


def run(cmd: list[str], log: Path, dry: bool = False) -> None:
    print("[CMD] " + " ".join(cmd))
    if dry: return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as h:
        proc = subprocess.run(cmd, stdout=h, stderr=subprocess.STDOUT, text=True)
    if proc.returncode:
        tail = log.read_text(errors="replace").splitlines()[-80:]
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n" + "\n".join(tail))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pypresso", default="pypresso")
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--old-priors", required=True, type=Path)
    p.add_argument("--ibi-priors", required=True, type=Path)
    p.add_argument("--rb-info", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--source-checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dts", nargs="+", type=float, required=True)
    p.add_argument("--duration-ps", type=float, required=True)
    p.add_argument("--branch-dt", type=float, required=True)
    p.add_argument("--branch-duration-ps", type=float, required=True)
    p.add_argument("--branch-kT", type=float, required=True)
    p.add_argument("--seed-base", type=int, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--ml-precision", choices=("float32", "float64"), required=True)
    p.add_argument("--neighbor-search", choices=("verlet", "link-cell"), required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--resume", action="store_true",
        help="Reuse complete NVT/NVE artifacts already present and run only missing/incomplete work",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args(); args.pypresso = resolve_executable(args.pypresso)
    for name in ("model", "config", "old_priors", "ibi_priors", "rb_info", "dataset", "source_checkpoint"):
        value = getattr(args, name).expanduser().resolve(); setattr(args, name, value)
        if not value.is_file(): raise FileNotFoundError(value)
    manifest = Path(str(args.model) + ".manifest.json")
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if len(args.dts) < 3 or min(args.dts) <= 0 or args.duration_ps <= 0 or args.branch_dt <= 0 or args.branch_duration_ps <= 0:
        raise ValueError("Positive durations and at least three positive dts are required")
    branch_steps = int(round(args.branch_duration_ps / args.branch_dt))
    if branch_steps < 2: raise ValueError("NVT branch must contain at least two steps")
    sample_start = branch_steps // 2
    if args.output_dir.exists() and args.overwrite and not args.dry_run:
        shutil.rmtree(args.output_dir)
    elif (
        args.output_dir.exists() and any(args.output_dir.iterdir())
        and not args.resume and not args.dry_run
    ):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. "
            "Use --resume to reuse completed work or --overwrite to start over."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = write_matched_prior_variants(args.old_priors, args.ibi_priors, args.output_dir / "priors_variants")

    print("[IBI TIMESTEP-RANGE LOCALIZATION PLAN]")
    print(f"old priors   : {args.old_priors}")
    print(f"IBI priors   : {args.ibi_priors}")
    print(f"source chk   : {args.source_checkpoint}")
    print(f"variants     : {' / '.join(variants)}")
    print(f"dt scan      : {' '.join(f'{x:g}' for x in args.dts)} ps")
    print(f"NVE duration : {args.duration_ps:g} ps per dt")
    print(f"NVT branch   : {args.branch_duration_ps:g} ps at dt={args.branch_dt:g} ps; curvature sampled on final half")
    nve_steps = sum(int(round(args.duration_ps / dt)) for dt in args.dts)
    total = len(variants) * (branch_steps + nve_steps)
    print(f"integration  : about {total} total steps")
    print(f"output       : {args.output_dir}")
    print("[NOTE] Diagnostic-only. No certification artifact is modified.")

    plan = {
        "schema_version": 1, "kind": "ibi_timestep_range_localization_plan",
        "old_priors": str(args.old_priors), "ibi_priors": str(args.ibi_priors),
        "source_checkpoint": str(args.source_checkpoint), "variants": list(variants),
        "dts_ps": [float(x) for x in args.dts], "duration_ps": args.duration_ps,
        "branch_dt_ps": args.branch_dt, "branch_duration_ps": args.branch_duration_ps,
        "branch_sample_start_step": sample_start, "estimated_total_steps": total,
        "resume": bool(args.resume),
    }
    (args.output_dir / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        for name in variants:
            print(f"[PLAN] {name}: NVT {branch_steps} steps + NVE {nve_steps} steps")
        return

    variant_reports: dict[str, Any] = {}
    curvature_reports: dict[str, Any] = {}
    for i, (name, priors_path) in enumerate(variants.items()):
        vdir = args.output_dir / name; vdir.mkdir(parents=True, exist_ok=True)
        checkpoint = vdir / "nvt_checkpoint.npz"
        sample = vdir / "nvt_structured_sample.npz"
        nvt_energy = vdir / "nvt_energy.csv"
        nvt_steps = branch_steps
        nvt_cmd = [
            args.pypresso, str(ROOT / "simulation" / "run_cg_md.py"),
            "--model", str(args.model), "--disable_ml", "--config", str(args.config),
            "--priors", str(priors_path), "--rb_info", str(args.rb_info), "--dataset", str(args.dataset),
            "--checkpoint", str(args.source_checkpoint), "--allow_checkpoint_mismatch",
            "--dt", f"{args.branch_dt:.17g}", "--steps", str(nvt_steps), "--log_interval", "1",
            "--device", args.device, "--ml_precision", args.ml_precision,
            "--neighbor_search", args.neighbor_search, "--energy_file", str(nvt_energy), "--no_vtf",
            "--kT", f"{args.branch_kT:.17g}", "--thermostat_seed", str(args.seed_base + i),
            "--out_checkpoint", str(checkpoint), "--sample_npz", str(sample),
            "--sample_start_step", str(sample_start),
        ]
        nvt_reused = bool(args.resume and checkpoint.is_file() and sample.is_file() and nvt_energy.is_file())
        if nvt_reused:
            print(f"[REUSE] {name}: existing NVT checkpoint/sample")
        else:
            run(nvt_cmd, vdir / "nvt.log")
        curvature = visited_curvature_report(priors_path, sample)
        (vdir / "visited_curvature_report.json").write_text(json.dumps(curvature, indent=2, sort_keys=True) + "\n")
        curvature_reports[name] = curvature

        nve_dir = vdir / "nve_coarse_scan"
        if not nvt_reused and nve_dir.exists():
            shutil.rmtree(nve_dir)
        nve_dir.mkdir(parents=True, exist_ok=True)
        run_metrics: list[dict[str, Any]] = []
        for dt in sorted(args.dts, reverse=True):
            steps = int(round(args.duration_ps / dt))
            actual_duration = steps * dt
            run_dir = nve_dir / f"dt_{dt:.8g}".replace(".", "p")
            energy_csv = run_dir / "energy.csv"
            run_log = run_dir / "run.log"
            reuse_trace = False
            if args.resume and energy_csv.is_file():
                try:
                    times, energies = read_energy_csv(energy_csv)
                    tol = max(1.0e-12, 1.0e-9 * max(1.0, actual_duration))
                    reuse_trace = (
                        len(times) == steps + 1
                        and abs(float(times[-1] - times[0]) - actual_duration) <= tol
                    )
                except Exception:
                    reuse_trace = False
            if reuse_trace:
                print(f"[REUSE] {name}: dt={dt:g} ps ({steps} steps)")
            else:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
                nve_cmd = [
                    args.pypresso, str(ROOT / "simulation" / "run_cg_md.py"),
                    "--model", str(args.model), "--disable_ml", "--config", str(args.config),
                    "--priors", str(priors_path), "--rb_info", str(args.rb_info),
                    "--dataset", str(args.dataset), "--checkpoint", str(checkpoint),
                    "--dt", f"{dt:.17g}", "--steps", str(steps), "--log_interval", "1",
                    "--device", args.device, "--ml_precision", args.ml_precision,
                    "--neighbor_search", args.neighbor_search, "--energy_file", str(energy_csv),
                    "--no_vtf", "--nve",
                ]
                run(nve_cmd, run_log)
                times, energies = read_energy_csv(energy_csv)
            metrics = analyze_energy_series(times, energies)
            metrics.update({
                "dt_ps": float(dt), "steps": int(steps), "actual_duration_ps": float(actual_duration),
                "energy_csv": str(energy_csv), "run_log": str(run_log),
            })
            run_metrics.append(metrics)
            print(
                f"[NVE] {name:15s} dt={dt:g} sigma_E={metrics['sigma_E']:.6g} "
                f"rel_block_drift={metrics['relative_block_mean_drift']:.3e}"
            )

        report_path = nve_dir / "coarse_scan_report.json"
        report = {
            "schema_version": 1, "kind": "ibi_timestep_range_coarse_nve_scan",
            "diagnostic_only": True, "variant": name, "runs": run_metrics,
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        sigma = sigma_range_diagnostics(run_metrics)
        variant_reports[name] = {
            "priors": str(priors_path), "checkpoint": str(checkpoint),
            "nve_report": str(report_path), "sigma_range": sigma,
            "max_relative_block_drift": max(abs(float(r["relative_block_mean_drift"])) for r in run_metrics),
        }
        print(
            f"[RESULT] {name:15s} p={sigma['fit']['exponent_p']:.6f} "
            f"R2={sigma['fit']['loglog_r2']:.6f} C2spread={sigma['c2_spread_max_over_min']:.3f}"
        )

    ratios = stiffness_ratios(curvature_reports)
    final = {
        "schema_version": 1, "kind": "ibi_timestep_range_localization", "diagnostic_only": True,
        "variants": variant_reports, "visited_curvature": curvature_reports,
        "stiffness_ratios_vs_reference": ratios,
        "interpretation_keys": {
            "reference_pass_bonds_fail": "IBI bond stiffness/representation narrows the clean timestep range",
            "reference_pass_angles_fail": "IBI angle stiffness/representation narrows the clean timestep range",
            "isolated_pass_full_fail": "bond-angle coupling narrows the clean timestep range",
            "large_curvature_ratio": "sqrt(curvature ratio) is a frequency-scale proxy for the same generalized coordinate",
        },
    }
    out = args.output_dir / "timestep_range_localization_report.json"
    out.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print("[STIFFNESS P99 RATIOS VS REFERENCE]")
    for name, row in ratios.items():
        print(
            f"{name:15s} bond={row['bond']['p99_abs_curvature_ratio_vs_reference']:.3g} "
            f"(sqrt={row['bond']['sqrt_ratio_frequency_proxy']:.3g})  "
            f"angle={row['angle']['p99_abs_curvature_ratio_vs_reference']:.3g} "
            f"(sqrt={row['angle']['sqrt_ratio_frequency_proxy']:.3g})"
        )
    print(f"[DONE] report: {out}")


if __name__ == "__main__":
    main()
