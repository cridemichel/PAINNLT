#!/usr/bin/env python3
"""Run and analyse a reproducible NVE timestep-scaling experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NVE timestep scaling with drift and block-bootstrap diagnostics."
    )
    parser.add_argument("--pypresso", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--physical_time", type=float, default=5.0, help="Physical time per run in ps")
    parser.add_argument(
        "--dts", type=float, nargs="+",
        default=[0.004, 0.002, 0.001, 0.0005, 0.00025, 0.000125],
    )
    parser.add_argument("--samples", type=int, default=800, help="Approximate logged samples per timestep")
    parser.add_argument("--discard_fraction", type=float, default=0.1)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Moving-block bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit_min_dt", type=float, default=None)
    parser.add_argument("--fit_max_dt", type=float, default=None)
    parser.add_argument("--strict", action="store_true", help="Return non-zero unless fitted slope is in [1.8, 2.2]")
    return parser.parse_args()


def read_energy_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 4:
        raise RuntimeError(f"Too few energy samples in {path}: {len(rows)}")
    steps = np.asarray([int(row["Step"]) for row in rows], dtype=np.int64)
    energies = np.asarray([float(row["E_tot"]) for row in rows], dtype=float)
    if not np.isfinite(energies).all():
        raise RuntimeError(f"Non-finite total energy in {path}")
    if np.any(np.diff(steps) <= 0):
        raise RuntimeError(f"Non-increasing Step column in {path}")
    return steps, energies


def estimate_block_length(values: np.ndarray) -> int:
    centered = values - np.mean(values)
    variance = float(np.dot(centered, centered) / len(centered))
    if variance <= 0.0:
        return 1
    max_lag = min(len(values) // 4, 2000)
    tau_int = 0.5
    for lag in range(1, max_lag + 1):
        corr = float(np.dot(centered[:-lag], centered[lag:]) / ((len(centered) - lag) * variance))
        if corr <= 0.0:
            break
        tau_int += corr
    return max(1, min(len(values) // 4, int(math.ceil(2.0 * tau_int))))


def moving_block_bootstrap_std(values: np.ndarray, block: int, replicates: int, rng: np.random.Generator) -> np.ndarray:
    if replicates <= 0:
        return np.asarray([], dtype=float)
    n = len(values)
    blocks_needed = int(math.ceil(n / block))
    result = np.empty(replicates, dtype=float)
    offsets = np.arange(block)
    for i in range(replicates):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = (starts[:, None] + offsets[None, :]) % n
        sample = values[indices.ravel()[:n]]
        result[i] = np.std(sample, ddof=1)
    return result


def analyse_series(steps: np.ndarray, energies: np.ndarray, dt: float, discard_fraction: float, bootstrap: int, rng: np.random.Generator):
    discard = int(math.floor(len(energies) * discard_fraction))
    if len(energies) - discard < 20:
        raise RuntimeError("Too few samples remain after discarding the transient")
    steps = steps[discard:]
    energies = energies[discard:]
    times = steps.astype(float) * dt

    drift_slope, intercept = np.polyfit(times, energies, 1)
    detrended = energies - (drift_slope * times + intercept)
    raw_std = float(np.std(energies, ddof=1))
    detrended_std = float(np.std(detrended, ddof=1))
    span = float(np.ptp(energies))
    block = estimate_block_length(detrended)
    boot = moving_block_bootstrap_std(detrended, block, bootstrap, rng)
    ci_low, ci_high = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (math.nan, math.nan))
    relative_drift = float(drift_slope / max(abs(float(np.mean(energies))), 1e-30))
    return {
        "samples": len(energies),
        "raw_std": raw_std,
        "detrended_std": detrended_std,
        "std_ci_low": float(ci_low),
        "std_ci_high": float(ci_high),
        "range": span,
        "drift_kj_mol_ps": float(drift_slope),
        "relative_drift_per_ps": relative_drift,
        "block_length_samples": block,
        "bootstrap_stds": boot,
    }


def fit_slope(dts: np.ndarray, stds: np.ndarray) -> tuple[float, float]:
    coeff = np.polyfit(np.log(dts), np.log(stds), 1)
    predicted = np.polyval(coeff, np.log(dts))
    residual = np.sum((np.log(stds) - predicted) ** 2)
    total = np.sum((np.log(stds) - np.mean(np.log(stds))) ** 2)
    r2 = 1.0 - residual / total if total > 0 else 1.0
    return float(coeff[0]), float(r2)


def main():
    args = parse_args()
    if args.physical_time <= 0 or args.samples < 20:
        raise ValueError("--physical_time must be positive and --samples must be at least 20")
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100 for a meaningful confidence interval")
    if len(set(args.dts)) != len(args.dts):
        raise ValueError("--dts must not contain duplicate values")
    if not 0.0 <= args.discard_fraction < 0.8:
        raise ValueError("--discard_fraction must be in [0, 0.8)")

    root = Path(__file__).resolve().parents[2]
    tutorial = Path(__file__).resolve().parent
    pypresso = args.pypresso.expanduser().resolve()
    required = {
        "pypresso": pypresso,
        "dataset": tutorial / "my_ethanol_dataset.bin",
        "model": tutorial / "my_ethanol_model.pt",
        "manifest": tutorial / "my_ethanol_model.pt.manifest.json",
        "config": tutorial / "fast_training_config.json",
        "priors": tutorial / "cg_priors.json",
        "rb_info": tutorial / "rigid_bodies_info.json",
        "checkpoint": tutorial / "equilibrated_ethanol.npz",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    with required["config"].open() as handle:
        config = json.load(handle)
    print(f"[INFO] Model cutoff: {config['cutoff']} nm")

    rng = np.random.default_rng(args.seed)
    results = []
    for dt in args.dts:
        if dt <= 0:
            raise ValueError("All timesteps must be positive")
        steps = max(1, int(round(args.physical_time / dt)))
        actual_time = steps * dt
        log_interval = max(1, int(math.floor(steps / args.samples)))
        energy_csv = tutorial / "energy.csv"
        energy_csv.unlink(missing_ok=True)
        (tutorial / "cg_trajectory.vtf").unlink(missing_ok=True)

        command = [
            str(pypresso), str(root / "simulation" / "run_cg_md.py"),
            "--model", str(required["model"]),
            "--config", str(required["config"]),
            "--priors", str(required["priors"]),
            "--rb_info", str(required["rb_info"]),
            "--dataset", str(required["dataset"]),
            "--checkpoint", str(required["checkpoint"]),
            "--dt", str(dt), "--steps", str(steps),
            "--log_interval", str(log_interval), "--device", args.device, "--nve",
        ]
        print("[RUN]", " ".join(command))
        completed = subprocess.run(command, cwd=tutorial, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"NVE run failed for dt={dt} with code {completed.returncode}")
        if not energy_csv.is_file():
            raise RuntimeError(f"energy.csv was not produced for dt={dt}")

        archive = tutorial / f"energy_dt_{dt:.9g}.csv"
        shutil.copy2(energy_csv, archive)
        sample_steps, energies = read_energy_series(energy_csv)
        metrics = analyse_series(sample_steps, energies, dt, args.discard_fraction, args.bootstrap, rng)
        metrics.update({"dt": float(dt), "steps": steps, "physical_time_ps": actual_time, "energy_file": archive.name})
        results.append(metrics)

    results.sort(key=lambda row: row["dt"], reverse=True)
    fit_rows = [
        row for row in results
        if (args.fit_min_dt is None or row["dt"] >= args.fit_min_dt)
        and (args.fit_max_dt is None or row["dt"] <= args.fit_max_dt)
        and row["detrended_std"] > 0
    ]
    if len(fit_rows) < 3:
        raise RuntimeError("At least three timestep points are required for a scaling fit")

    fit_dts = np.asarray([row["dt"] for row in fit_rows])
    fit_stds = np.asarray([row["detrended_std"] for row in fit_rows])
    slope, r2 = fit_slope(fit_dts, fit_stds)

    slope_boot = np.empty(args.bootstrap, dtype=float)
    for b in range(args.bootstrap):
        sampled = []
        for row in fit_rows:
            boot = row["bootstrap_stds"]
            sampled.append(max(float(boot[b]), np.finfo(float).tiny) if len(boot) else row["detrended_std"])
        slope_boot[b], _ = fit_slope(fit_dts, np.asarray(sampled))
    slope_ci = np.percentile(slope_boot, [2.5, 97.5]) if len(slope_boot) else [math.nan, math.nan]

    print(f"{'dt (ps)':>12} {'steps':>10} {'std detr.':>16} {'drift/ps':>16} {'ratio':>10} {'block':>8}")
    for index, row in enumerate(results):
        ratio = "-"
        if index + 1 < len(results):
            next_row = results[index + 1]
            ratio = f"{row['detrended_std'] / next_row['detrended_std']:.4f}"
        print(
            f"{row['dt']:12.6g} {row['steps']:10d} {row['detrended_std']:16.8e} "
            f"{row['drift_kj_mol_ps']:16.8e} {ratio:>10} {row['block_length_samples']:8d}"
        )

    output_csv = tutorial / "energy_scaling.csv"
    fieldnames = [
        "dt_ps", "steps", "physical_time_ps", "samples", "raw_std_total_energy",
        "detrended_std_total_energy", "std_ci_low", "std_ci_high", "range_total_energy",
        "drift_kj_mol_ps", "relative_drift_per_ps", "block_length_samples", "energy_file",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "dt_ps": row["dt"], "steps": row["steps"], "physical_time_ps": row["physical_time_ps"],
                "samples": row["samples"], "raw_std_total_energy": row["raw_std"],
                "detrended_std_total_energy": row["detrended_std"], "std_ci_low": row["std_ci_low"],
                "std_ci_high": row["std_ci_high"], "range_total_energy": row["range"],
                "drift_kj_mol_ps": row["drift_kj_mol_ps"],
                "relative_drift_per_ps": row["relative_drift_per_ps"],
                "block_length_samples": row["block_length_samples"], "energy_file": row["energy_file"],
            })

    fit_summary = {
        "slope": slope, "slope_ci_95": [float(slope_ci[0]), float(slope_ci[1])], "r_squared": r2,
        "fit_dts_ps": fit_dts.tolist(), "physical_time_ps": args.physical_time,
        "discard_fraction": args.discard_fraction, "bootstrap_replicates": args.bootstrap,
    }
    (tutorial / "energy_scaling_fit.json").write_text(json.dumps(fit_summary, indent=2) + "\n")

    dts = np.asarray([row["dt"] for row in results])
    stds = np.asarray([row["detrended_std"] for row in results])
    low = np.asarray([row["std_ci_low"] for row in results])
    high = np.asarray([row["std_ci_high"] for row in results])
    plt.figure(figsize=(8, 6))
    lower_error = np.maximum(0.0, stds - low)
    upper_error = np.maximum(0.0, high - stds)
    plt.errorbar(dts, stds, yerr=np.vstack([lower_error, upper_error]), fmt="o-", capsize=3, label="Detrended std")
    reference = fit_stds[-1] * (dts / fit_dts[-1]) ** 2
    plt.loglog(dts, reference, "--", label=r"Reference $O(\Delta t^2)$")
    plt.xlabel("Timestep (ps)")
    plt.ylabel("Std(total energy, detrended) (kJ/mol)")
    plt.title(f"NVE scaling: slope={slope:.3f}, 95% CI [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(tutorial / "scaling_plot.png", dpi=300)

    print(f"[RESULT] fitted slope: {slope:.6f}; 95% CI [{slope_ci[0]:.6f}, {slope_ci[1]:.6f}]; R^2={r2:.6f}")
    print(f"[RESULT] wrote {output_csv.name}, energy_scaling_fit.json and scaling_plot.png")
    passed = 1.8 <= slope <= 2.2
    if not passed:
        print("[WARNING] Fitted slope is outside the certification interval [1.8, 2.2].")
    if args.strict and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
