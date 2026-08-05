#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a reproducible NVE timestep-scaling test with the CG pipeline outputs."
    )
    parser.add_argument("--pypresso", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--physical_time", type=float, default=0.2, help="Physical time in ps")
    parser.add_argument(
        "--dts",
        type=float,
        nargs="+",
        default=[0.004, 0.002, 0.001, 0.0005, 0.00025, 0.000125],
    )
    return parser.parse_args()


def read_total_energies(path: Path) -> np.ndarray:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No energy samples were written to {path}")
    values = np.asarray([float(row["E_tot"]) for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite total energy in {path}")
    return values


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    tutorial = Path(__file__).resolve().parent
    pypresso = args.pypresso.expanduser().resolve()

    required = {
        "pypresso": pypresso,
        "dataset": tutorial / "my_ethanol_dataset.bin",
        "model": tutorial / "my_ethanol_model.pt",
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

    results = []
    for dt in args.dts:
        if dt <= 0:
            raise ValueError("All timesteps must be positive")
        steps = max(1, int(round(args.physical_time / dt)))
        log_interval = max(1, steps // 400)
        energy_csv = tutorial / "energy.csv"
        energy_csv.unlink(missing_ok=True)
        (tutorial / "cg_trajectory.vtf").unlink(missing_ok=True)

        command = [
            str(pypresso),
            str(root / "simulation" / "run_cg_md.py"),
            "--model", str(required["model"]),
            "--config", str(required["config"]),
            "--priors", str(required["priors"]),
            "--rb_info", str(required["rb_info"]),
            "--dataset", str(required["dataset"]),
            "--checkpoint", str(required["checkpoint"]),
            "--dt", str(dt),
            "--steps", str(steps),
            "--log_interval", str(log_interval),
            "--device", args.device,
            "--nve",
        ]
        print("[RUN]", " ".join(command))
        completed = subprocess.run(command, cwd=tutorial, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"NVE run failed for dt={dt} with code {completed.returncode}")
        if not energy_csv.is_file():
            raise RuntimeError(f"energy.csv was not produced for dt={dt}")

        energies = read_total_energies(energy_csv)
        std = float(np.std(energies))
        span = float(np.ptp(energies))
        results.append((dt, steps, std, span))

    print(f"{'dt (ps)':>12} {'steps':>10} {'std(E)':>16} {'range(E)':>16} {'ratio':>12}")
    previous = None
    for dt, steps, std, span in results:
        ratio = "-" if previous is None else f"{std / previous:.6f}"
        print(f"{dt:12.6g} {steps:10d} {std:16.8e} {span:16.8e} {ratio:>12}")
        previous = std

    dts = np.asarray([row[0] for row in results], dtype=float)
    stds = np.asarray([row[2] for row in results], dtype=float)
    valid = (stds > 0) & np.isfinite(stds)
    if valid.sum() < 2:
        raise RuntimeError("Not enough finite non-zero points for a log-log fit")
    slope = float(np.polyfit(np.log(dts[valid]), np.log(stds[valid]), 1)[0])

    output_csv = tutorial / "energy_scaling.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dt_ps", "steps", "std_total_energy", "range_total_energy"])
        writer.writerows(results)

    plt.figure(figsize=(8, 6))
    plt.loglog(dts, stds, "o-", label="Measured")
    reference = stds[-1] * (dts / dts[-1]) ** 2
    plt.loglog(dts, reference, "--", label=r"Reference $O(\Delta t^2)$")
    plt.xlabel("Timestep (ps)")
    plt.ylabel("Std(total energy) (kJ/mol)")
    plt.title(f"NVE energy-error scaling; fitted slope={slope:.3f}")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(tutorial / "scaling_plot.png", dpi=300)

    print(f"[RESULT] fitted log-log slope: {slope:.6f}")
    print(f"[RESULT] wrote {output_csv.name} and scaling_plot.png")
    if not (1.5 <= slope <= 2.5):
        print("[WARNING] The measured slope is not close to the expected value 2.")


if __name__ == "__main__":
    main()
