#!/usr/bin/env python3
"""Certify TorchMD Velocity-Verlet on a TEL22-sized conservative test system.

This benchmark is intentionally independent of the TEL22 Hamiltonian.  It tests
TorchMD's NVE integration/precision path on a deterministic 820-particle smooth
Hamiltonian and measures sigma_E ~ dt^p plus secular drift.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

from analysis import analyze_energy_series, certify_metrics


DEFAULT_DTS_PS = (0.001, 0.0015, 0.002, 0.003, 0.004, 0.005)
BOLTZMANN_KCAL_MOL_K = 0.001987191


@dataclass
class MinimalSystem:
    """Only the state fields used by torchmd.integrator.Integrator."""

    pos: Any
    vel: Any
    forces: Any
    masses: Any
    box: Any


class HarmonicReferenceForces:
    """Analytic conservative force field with a spectrum of oscillator frequencies."""

    def __init__(self, equilibrium, stiffness):
        self.equilibrium = equilibrium
        self.stiffness = stiffness

    def compute(self, pos, box, forces, *args, **kwargs):
        del box, args, kwargs
        delta = pos - self.equilibrium
        forces.copy_(-self.stiffness * delta)
        energy = 0.5 * (self.stiffness * delta * delta).sum(dim=(1, 2))
        return energy.detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TorchMD NVE certification on a deterministic TEL22-sized harmonic reference system."
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, mps, or auto")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--particles", type=int, default=820)
    parser.add_argument("--duration-ps", type=float, default=1.98)
    parser.add_argument("--dts", nargs="+", type=float, default=list(DEFAULT_DTS_PS))
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--mass-g-mol", type=float, default=72.0)
    parser.add_argument("--k-min", type=float, default=20.0, help="minimum harmonic k in kcal/mol/A^2")
    parser.add_argument("--k-max", type=float, default=80.0, help="maximum harmonic k in kcal/mol/A^2")
    parser.add_argument("--displacement-sigma-a", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=220526)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--slope-min", type=float, default=1.7)
    parser.add_argument("--slope-max", type=float, default=2.3)
    parser.add_argument("--min-r2", type=float, default=0.97)
    parser.add_argument("--max-relative-drift", type=float, default=1.0e-4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(torch, requested: str):
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device.type == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
    return device


def version_or_unknown(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def validate_args(args: argparse.Namespace) -> None:
    if args.particles < 1:
        raise ValueError("--particles must be positive")
    if args.duration_ps <= 0.0:
        raise ValueError("--duration-ps must be positive")
    if len(args.dts) < 3 or any(dt <= 0.0 for dt in args.dts):
        raise ValueError("provide at least three positive --dts values")
    if len(set(args.dts)) != len(args.dts):
        raise ValueError("--dts values must be unique")
    if any(args.duration_ps / dt < 20 for dt in args.dts):
        raise ValueError("each trajectory must contain at least 20 integration steps")
    if args.temperature_k <= 0.0 or args.mass_g_mol <= 0.0:
        raise ValueError("temperature and mass must be positive")
    if not (0.0 < args.k_min <= args.k_max):
        raise ValueError("require 0 < k-min <= k-max")
    if args.displacement_sigma_a <= 0.0:
        raise ValueError("--displacement-sigma-a must be positive")


def build_initial_state(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    n = args.particles
    side = int(math.ceil(n ** (1.0 / 3.0)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side), np.arange(side), indexing="ij"), axis=-1)
    equilibrium = 4.0 * grid.reshape(-1, 3)[:n].astype(np.float64)
    displacement = rng.normal(0.0, args.displacement_sigma_a, size=(n, 3))
    positions = equilibrium + displacement

    masses = np.full((n, 1), args.mass_g_mol, dtype=np.float64)
    velocity_sigma = math.sqrt(BOLTZMANN_KCAL_MOL_K * args.temperature_k / args.mass_g_mol)
    velocities = rng.normal(0.0, velocity_sigma, size=(n, 3))
    velocities -= velocities.mean(axis=0, keepdims=True)

    phase = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    stiffness = args.k_min + (args.k_max - args.k_min) * (0.5 + 0.5 * np.sin(phase))
    stiffness = stiffness[:, None]
    return {
        "equilibrium": equilibrium,
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "stiffness": stiffness,
    }


def make_system(torch, initial: dict[str, np.ndarray], dtype, device):
    def tensor(name: str):
        return torch.as_tensor(initial[name], dtype=dtype, device=device)

    pos = tensor("positions").unsqueeze(0).clone()
    vel = tensor("velocities").unsqueeze(0).clone()
    masses = tensor("masses").clone()
    forces = torch.zeros_like(pos)
    box = torch.zeros((1, 3, 3), dtype=dtype, device=device)
    equilibrium = tensor("equilibrium").unsqueeze(0)
    stiffness = tensor("stiffness").unsqueeze(0)
    return MinimalSystem(pos=pos, vel=vel, forces=forces, masses=masses, box=box), equilibrium, stiffness


def extract_scalar(value: Any) -> float:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"expected scalar-like value, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def run_one_dt(torch, Integrator, args, initial, dtype, device, dt_ps: float, out_dir: Path) -> dict[str, Any]:
    exact_steps = args.duration_ps / dt_ps
    steps = int(round(exact_steps))
    if not math.isclose(steps * dt_ps, args.duration_ps, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"duration {args.duration_ps:g} ps is not commensurate with dt={dt_ps:g} ps")

    system, equilibrium, stiffness = make_system(torch, initial, dtype, device)
    forcefield = HarmonicReferenceForces(equilibrium, stiffness)
    forcefield.compute(system.pos, system.box, system.forces)
    integrator = Integrator(system, forcefield, dt_ps * 1000.0, str(device), gamma=None, T=None)

    from torchmd.integrator import kinetic_energy

    e_pot0 = extract_scalar(forcefield.compute(system.pos, system.box, system.forces))
    e_kin0 = extract_scalar(kinetic_energy(system.masses, system.vel).detach().cpu().numpy())

    times = [0.0]
    epot = [e_pot0]
    ekin = [e_kin0]
    etot = [e_pot0 + e_kin0]
    temperatures = [2.0 * e_kin0 / (3.0 * args.particles * BOLTZMANN_KCAL_MOL_K)]

    start = time.perf_counter()
    for step in range(1, steps + 1):
        e_kin, e_pot, temp = integrator.step(1)
        e_kin_value = extract_scalar(e_kin)
        e_pot_value = extract_scalar(e_pot)
        times.append(step * dt_ps)
        epot.append(e_pot_value)
        ekin.append(e_kin_value)
        etot.append(e_pot_value + e_kin_value)
        temperatures.append(extract_scalar(temp))
    wall_s = time.perf_counter() - start

    if not np.isfinite(np.asarray(etot)).all():
        raise RuntimeError(f"non-finite total energy at dt={dt_ps:g} ps")

    metrics = analyze_energy_series(times, etot)
    metrics.update({
        "dt_ps": float(dt_ps),
        "dt_fs": float(dt_ps * 1000.0),
        "steps": int(steps),
        "wall_seconds": float(wall_s),
        "ms_per_step": float(1000.0 * wall_s / steps),
        "C2_sigma_over_dt2": float(metrics["sigma_E"] / (dt_ps * dt_ps)),
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "energy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time_ps", "E_pot_kcal_mol", "E_kin_kcal_mol", "E_tot_kcal_mol", "Temperature_K"])
        writer.writerows(zip(times, epot, ekin, etot, temperatures))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def write_summary_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fields = [
        "dt_ps", "dt_fs", "steps", "sigma_E", "C2_sigma_over_dt2",
        "relative_block_mean_drift", "max_abs_delta_E", "ms_per_step",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(runs, key=lambda item: item["dt_ps"]):
            writer.writerow({key: row[key] for key in fields})


def main() -> int:
    args = parse_args()
    validate_args(args)

    if args.dry_run:
        print("[DRY-RUN] isolated TorchMD TEL22-sized NVE certification")
        print(f"[DRY-RUN] particles={args.particles} duration={args.duration_ps:g} ps")
        print("[DRY-RUN] dts_ps=" + ",".join(f"{x:g}" for x in args.dts))
        print(f"[DRY-RUN] device={args.device} precision={args.precision}")
        print("[DRY-RUN] Hamiltonian: independent 3D harmonic modes; no TEL22/IBI inputs are modified or read")
        return 0

    try:
        import torch
        import torchmd  # noqa: F401
        from torchmd.integrator import Integrator
    except Exception as exc:
        print(f"[ERROR] TorchMD environment unavailable: {exc}", file=sys.stderr)
        print("[ERROR] Run this script inside the environment where torchmd is installed.", file=sys.stderr)
        return 2

    device = resolve_device(torch, args.device)
    dtype = torch.float32 if args.precision == "float32" else torch.float64
    if device.type == "mps" and dtype == torch.float64:
        raise RuntimeError("MPS does not provide the float64 path required by this certification")

    out_root = Path(args.output_dir).expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {out_root}; pass --overwrite")
    out_root.mkdir(parents=True, exist_ok=True)

    initial = build_initial_state(args)
    np.savez_compressed(out_root / "initial_state.npz", **initial)

    provenance = {
        "kind": "torchmd_tel22_sized_nve_certification",
        "scope": "TorchMD integrator/precision certification; NOT TEL22 Hamiltonian parity",
        "particles": args.particles,
        "device": str(device),
        "precision": args.precision,
        "duration_ps": args.duration_ps,
        "dts_ps": list(args.dts),
        "temperature_K": args.temperature_k,
        "mass_g_mol": args.mass_g_mol,
        "k_range_kcal_mol_A2": [args.k_min, args.k_max],
        "seed": args.seed,
        "torch_version": str(torch.__version__),
        "torchmd_version": version_or_unknown("torchmd"),
        "python_version": platform.python_version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (out_root / "run_plan.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[TORCHMD TEL22-SIZED NVE CERTIFICATION]")
    print(f"device / precision : {device} / {args.precision}")
    print(f"particles          : {args.particles}")
    print(f"duration           : {args.duration_ps:g} ps per dt")
    print("dt grid            : " + " ".join(f"{dt:g}" for dt in args.dts) + " ps")
    print("Hamiltonian        : deterministic harmonic reference (not TEL22 parity)")

    runs: list[dict[str, Any]] = []
    for dt_ps in args.dts:
        label = str(dt_ps).replace(".", "p")
        print(f"[RUN] dt={dt_ps:g} ps ({dt_ps * 1000:g} fs)")
        metrics = run_one_dt(torch, Integrator, args, initial, dtype, device, dt_ps, out_root / f"dt_{label}")
        runs.append(metrics)
        print(
            f"      sigma_E={metrics['sigma_E']:.8g} kcal/mol  "
            f"C2={metrics['C2_sigma_over_dt2']:.8g}  "
            f"drift={metrics['relative_block_mean_drift']:.3e}  "
            f"{metrics['ms_per_step']:.3f} ms/step"
        )

    certification = certify_metrics(
        runs,
        slope_min=args.slope_min,
        slope_max=args.slope_max,
        min_r2=args.min_r2,
        max_relative_drift=args.max_relative_drift,
    )
    report = {
        **provenance,
        "energy_units": "kcal/mol",
        "length_units": "angstrom",
        "time_input_units": "fs (dt grid specified in ps and converted by x1000)",
        "runs": sorted(runs, key=lambda item: item["dt_ps"]),
        "certification": certification,
    }
    (out_root / "nve_certification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary_csv(out_root / "nve_certification_runs.csv", runs)

    scaling = certification["scaling"]
    print("[RESULT]")
    print(f"  p              = {scaling['exponent_p']:.6f}")
    print(f"  log-log R2     = {scaling['loglog_r2']:.6f}")
    print(f"  C2 spread      = {certification['c2_spread_max_over_min']:.3f}")
    print(f"  max drift      = {max(r['relative_block_mean_drift'] for r in runs):.3e}")
    print(f"  certification  = {'PASS' if certification['pass'] else 'FAIL'}")
    print(f"  report         = {out_root / 'nve_certification_report.json'}")
    return 0 if certification["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
