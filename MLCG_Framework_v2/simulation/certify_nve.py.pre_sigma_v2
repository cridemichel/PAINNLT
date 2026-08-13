#!/usr/bin/env python3
"""Run a reproducible multi-dt NVE scan and certify energy conservation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from framework_utils import nonconservative_prior_entries
from nve_analysis import analyze_energy_series, certify_metrics, read_energy_csv


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    found = shutil.which(value)
    if found:
        return found
    raise FileNotFoundError(f"Executable not found: {value}")


def existing_file(value: str | None, name: str, *, required: bool = True) -> Path | None:
    if value is None:
        if required:
            raise ValueError(f"--{name} is required")
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def tail(path: Path, n: int = 30) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-n:])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Certify a conservative CG Hamiltonian by running the same checkpoint at "
            "multiple Velocity-Verlet time steps and fitting RMS energy error ~ dt^p."
        )
    )
    parser.add_argument("--pypresso", default="pypresso", help="ESPResSo pypresso executable")
    parser.add_argument(
        "--runner",
        default=str(Path(__file__).with_name("run_cg_md.py")),
        help="Path to generic run_cg_md.py",
    )
    parser.add_argument("--model", default=None, help="PaiNN model; omit for a classical-only test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--priors", required=True)
    parser.add_argument("--rb-info", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dts",
        nargs="+",
        type=float,
        default=[0.002, 0.001, 0.0005, 0.00025],
        help="Time steps in ps (default: 0.002 0.001 0.0005 0.00025)",
    )
    parser.add_argument("--duration-ps", type=float, default=5.0, help="Physical duration of each NVE run")
    parser.add_argument("--log-interval-ps", type=float, default=0.01, help="Energy sampling interval")
    parser.add_argument("--device", default="cpu", help="PaiNN device; CPU is the certification reference")
    parser.add_argument("--allow-nonreference-device", action="store_true")
    parser.add_argument("--output-dir", default="nve_certification")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--slope-min", type=float, default=1.7)
    parser.add_argument("--slope-max", type=float, default=2.3)
    parser.add_argument("--min-r2", type=float, default=0.97)
    parser.add_argument(
        "--max-relative-drift",
        type=float,
        default=1.0e-4,
        help="Maximum |mean(final 20%% E)-mean(initial 20%% E)| / characteristic |E|",
    )
    parser.add_argument("--allow-missing-model-manifest", action="store_true")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print run plan without launching ESPResSo")
    args = parser.parse_args()

    if args.duration_ps <= 0.0 or args.log_interval_ps <= 0.0:
        raise ValueError("--duration-ps and --log-interval-ps must be positive")
    if len(args.dts) < 3 or any(dt <= 0.0 for dt in args.dts):
        raise ValueError("Provide at least three positive --dts")
    if len(set(args.dts)) != len(args.dts):
        raise ValueError("--dts must be unique")
    if args.device.lower() != "cpu" and not args.allow_nonreference_device:
        raise RuntimeError(
            "NVE certification defaults to CPU because it is the reference numerical path. "
            "Pass --allow-nonreference-device only for an explicit comparison."
        )

    pypresso = resolve_executable(args.pypresso)
    runner = existing_file(args.runner, "runner")
    model = existing_file(args.model, "model", required=False)
    config = existing_file(args.config, "config")
    priors = existing_file(args.priors, "priors")
    rb_info = existing_file(args.rb_info, "rb-info")
    dataset = existing_file(args.dataset, "dataset")
    checkpoint = existing_file(args.checkpoint, "checkpoint")
    assert runner and config and priors and rb_info and dataset and checkpoint

    prior_data = json.loads(priors.read_text(encoding="utf-8"))
    unsafe = nonconservative_prior_entries(prior_data)
    if unsafe:
        raise RuntimeError(
            "NVE certification requires a conservative Hamiltonian. Explicitly tabulated "
            "priors are forbidden: " + ", ".join(unsafe)
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite or a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs: dict[str, Path] = {
        "runner": runner,
        "config": config,
        "priors": priors,
        "rb_info": rb_info,
        "dataset": dataset,
        "checkpoint": checkpoint,
    }
    if model is not None:
        inputs["model"] = model
        manifest = Path(str(model) + ".manifest.json")
        if manifest.is_file():
            inputs["model_manifest"] = manifest

    input_hashes = {name: sha256_file(path) for name, path in inputs.items()}
    run_metrics: list[dict[str, Any]] = []
    run_plan: list[dict[str, Any]] = []

    for dt in sorted(args.dts, reverse=True):
        steps = int(round(args.duration_ps / dt))
        if steps < 2:
            raise ValueError(f"duration {args.duration_ps} ps is too short for dt={dt} ps")
        actual_duration = steps * dt
        log_steps = max(1, int(round(args.log_interval_ps / dt)))
        run_dir = output_dir / f"dt_{dt:.8g}".replace(".", "p")
        energy_csv = run_dir / "energy.csv"
        log_file = run_dir / "run.log"
        command = [
            pypresso,
            str(runner),
            "--config", str(config),
            "--priors", str(priors),
            "--rb_info", str(rb_info),
            "--dataset", str(dataset),
            "--checkpoint", str(checkpoint),
            "--dt", f"{dt:.17g}",
            "--steps", str(steps),
            "--log_interval", str(log_steps),
            "--device", args.device,
            "--nve",
            "--no_vtf",
            "--energy_file", "energy.csv",
        ]
        if model is not None:
            command[2:2] = ["--model", str(model)]
        if args.allow_missing_model_manifest:
            command.append("--allow_missing_model_manifest")
        if args.allow_legacy_checkpoint:
            command.append("--allow_legacy_checkpoint")

        plan_item = {
            "dt_ps": dt,
            "steps": steps,
            "requested_duration_ps": args.duration_ps,
            "actual_duration_ps": actual_duration,
            "log_interval_steps": log_steps,
            "run_dir": str(run_dir),
            "command": command,
        }
        run_plan.append(plan_item)
        print(
            f"[PLAN] dt={dt:g} ps steps={steps} duration={actual_duration:g} ps "
            f"log_every={log_steps} steps"
        )
        if args.dry_run:
            continue

        if run_dir.exists() and args.overwrite:
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=run_dir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=os.environ.copy(),
            )
        if completed.returncode != 0:
            print(f"[ERROR] NVE run failed for dt={dt:g} ps; tail of {log_file}:", file=sys.stderr)
            print(tail(log_file), file=sys.stderr)
            return 3
        if not energy_csv.is_file():
            raise RuntimeError(f"Expected energy log was not produced: {energy_csv}")

        times, energies = read_energy_csv(energy_csv)
        metrics = analyze_energy_series(times, energies)
        metrics.update({
            "dt_ps": dt,
            "steps": steps,
            "log_interval_steps": log_steps,
            "energy_csv": str(energy_csv),
            "run_log": str(log_file),
        })
        run_metrics.append(metrics)
        print(
            f"[RESULT] dt={dt:g} rms_dE={metrics['rms_delta_E']:.6g} "
            f"rel_block_drift={metrics['relative_block_mean_drift']:.3e}"
        )

    plan_path = output_dir / "run_plan.json"
    plan_path.write_text(json.dumps({"inputs_sha256": input_hashes, "runs": run_plan}, indent=2) + "\n")
    if args.dry_run:
        print(f"[DONE] Dry-run plan: {plan_path}")
        return 0

    certification = certify_metrics(
        run_metrics,
        slope_min=args.slope_min,
        slope_max=args.slope_max,
        min_r2=args.min_r2,
        max_relative_drift=args.max_relative_drift,
    )
    report = {
        "definition": {
            "quantity": "RMS of E_total(t)-E_total(0) over fixed physical duration",
            "scaling_model": "RMS_dE = C * dt^p",
            "drift_metric": "abs(mean(last 20% E)-mean(first 20% E)) / characteristic_abs_energy",
            "linear_fit": "reported as an additional diagnostic, not used as the pass/fail drift metric",
            "same_initial_state": "all runs load the identical checkpoint file",
            "integrator": "explicit ESPResSo Velocity Verlet",
            "thermostat": "off (--nve)",
            "force_cap": 0.0,
            "reference_device": "cpu",
        },
        "inputs_sha256": input_hashes,
        "device": args.device,
        "runs": run_metrics,
        "certification": certification,
    }
    report_path = output_dir / "nve_certification_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    csv_path = output_dir / "nve_certification_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dt_ps", "steps", "duration_ps", "samples", "rms_delta_E",
                "relative_rms_delta_E", "peak_to_peak_E", "linear_drift_kjmol_per_ps",
                "relative_linear_drift_span", "relative_block_mean_drift", "drift_span_over_rms_delta",
            ],
        )
        writer.writeheader()
        for item in sorted(run_metrics, key=lambda row: row["dt_ps"], reverse=True):
            writer.writerow({
                "dt_ps": item["dt_ps"],
                "steps": item["steps"],
                "duration_ps": item["duration_ps"],
                "samples": item["samples"],
                "rms_delta_E": item["rms_delta_E"],
                "relative_rms_delta_E": item["relative_rms_delta_E"],
                "peak_to_peak_E": item["peak_to_peak_E"],
                "linear_drift_kjmol_per_ps": item["linear_drift_kjmol_per_ps"],
                "relative_linear_drift_span": item["relative_linear_drift_span"],
                "relative_block_mean_drift": item["relative_block_mean_drift"],
                "drift_span_over_rms_delta": item["drift_span_over_rms_delta"],
            })

    scaling = certification["scaling"]
    status = "PASS" if certification["pass"] else "FAIL"
    print(
        f"[{status}] NVE certification: p={scaling['exponent_p']:.6f}, "
        f"R2={scaling['loglog_r2']:.6f}, drift_pass={certification['drift_pass']}"
    )
    print(f"       report: {report_path}")
    print(f"       table:  {csv_path}")
    return 0 if certification["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
