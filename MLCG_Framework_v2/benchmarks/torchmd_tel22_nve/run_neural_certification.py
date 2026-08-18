#!/usr/bin/env python3
"""TorchMD NVE certification with a frozen conservative neural potential and autograd forces."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

from analysis import analyze_energy_series, certify_metrics
from neural_potential import (
    AutogradNeuralForces,
    NeuralPotentialConfig,
    build_neural_model,
    model_fingerprint,
)
from run_certification import (
    BOLTZMANN_KCAL_MOL_K,
    DEFAULT_DTS_PS,
    build_initial_state,
    extract_scalar,
    make_system,
    resolve_device,
    validate_args,
    version_or_unknown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TorchMD NVE certification with a frozen conservative MLP and autograd forces."
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, mps, or auto")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--particles", type=int, default=820)
    parser.add_argument("--duration-ps", type=float, default=1.98)
    parser.add_argument("--dts", nargs="+", type=float, default=list(DEFAULT_DTS_PS))
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--mass-g-mol", type=float, default=72.0)
    parser.add_argument("--k-min", type=float, default=20.0)
    parser.add_argument("--k-max", type=float, default=80.0)
    parser.add_argument("--displacement-sigma-a", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=220526, help="initial-state seed")
    parser.add_argument("--network-seed", type=int, default=220527)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--modulation-amplitude", type=float, default=0.05)
    parser.add_argument("--network-length-scale-a", type=float, default=0.50)
    parser.add_argument("--output-dir", default="results/neural_cpu_float64")
    parser.add_argument("--slope-min", type=float, default=1.7)
    parser.add_argument("--slope-max", type=float, default=2.3)
    parser.add_argument("--min-r2", type=float, default=0.97)
    parser.add_argument("--max-relative-drift", type=float, default=1.0e-4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_neural_args(args: argparse.Namespace) -> NeuralPotentialConfig:
    validate_args(args)
    return NeuralPotentialConfig(
        hidden_channels=args.hidden_channels,
        hidden_layers=args.hidden_layers,
        modulation_amplitude=args.modulation_amplitude,
        length_scale_a=args.network_length_scale_a,
        seed=args.network_seed,
    )


def run_one_dt(torch, Integrator, args, initial, config, dtype, device, dt_ps: float, out_dir: Path):
    exact_steps = args.duration_ps / dt_ps
    steps = int(round(exact_steps))
    if not math.isclose(steps * dt_ps, args.duration_ps, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"duration {args.duration_ps:g} ps is not commensurate with dt={dt_ps:g} ps")

    system, equilibrium, stiffness = make_system(torch, initial, dtype, device)
    model = build_neural_model(torch, config, dtype=dtype, device=device)
    forcefield = AutogradNeuralForces(torch, model, equilibrium, stiffness)
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
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
        "force_compute_calls": int(forcefield.compute_calls),
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "energy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time_ps", "E_pot_kcal_mol", "E_kin_kcal_mol", "E_tot_kcal_mol", "Temperature_K"])
        writer.writerows(zip(times, epot, ekin, etot, temperatures))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics, model_fingerprint(model), sum(p.numel() for p in model.parameters())


def write_summary_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fields = [
        "dt_ps", "dt_fs", "steps", "sigma_E", "C2_sigma_over_dt2",
        "relative_block_mean_drift", "max_abs_delta_E", "ms_per_step", "force_compute_calls",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(runs, key=lambda item: item["dt_ps"]):
            writer.writerow({key: row[key] for key in fields})


def main() -> int:
    args = parse_args()
    config = validate_neural_args(args)

    if args.dry_run:
        print("[DRY-RUN] TorchMD neural-potential NVE certification")
        print(f"[DRY-RUN] particles={args.particles} duration={args.duration_ps:g} ps")
        print("[DRY-RUN] dts_ps=" + ",".join(f"{x:g}" for x in args.dts))
        print(f"[DRY-RUN] device={args.device} precision={args.precision}")
        print(
            f"[DRY-RUN] MLP={config.hidden_layers}x{config.hidden_channels} SiLU; "
            f"modulation={config.modulation_amplitude:g}; frozen parameters; F=-dE/dr via autograd"
        )
        return 0

    try:
        import torch
        import torchmd  # noqa: F401
        from torchmd.integrator import Integrator
    except Exception as exc:
        print(f"[ERROR] TorchMD environment unavailable: {exc}", file=sys.stderr)
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

    print("[TORCHMD NEURAL-POTENTIAL NVE CERTIFICATION]")
    print(f"device / precision : {device} / {args.precision}")
    print(f"particles          : {args.particles}")
    print(f"duration           : {args.duration_ps:g} ps per dt")
    print("dt grid            : " + " ".join(f"{dt:g}" for dt in args.dts) + " ps")
    print(
        f"neural potential   : frozen {config.hidden_layers}x{config.hidden_channels} SiLU MLP; "
        "autograd coordinate forces"
    )

    runs: list[dict[str, Any]] = []
    fingerprints = set()
    parameter_counts = set()
    for dt_ps in args.dts:
        label = str(dt_ps).replace(".", "p")
        print(f"[RUN] dt={dt_ps:g} ps ({dt_ps * 1000:g} fs)")
        metrics, fingerprint, parameter_count = run_one_dt(
            torch, Integrator, args, initial, config, dtype, device, dt_ps, out_root / f"dt_{label}"
        )
        runs.append(metrics)
        fingerprints.add(fingerprint)
        parameter_counts.add(parameter_count)
        print(
            f"      sigma_E={metrics['sigma_E']:.8g} kcal/mol  "
            f"C2={metrics['C2_sigma_over_dt2']:.8g}  "
            f"drift={metrics['relative_block_mean_drift']:.3e}  "
            f"{metrics['ms_per_step']:.3f} ms/step"
        )
    if len(fingerprints) != 1 or len(parameter_counts) != 1:
        raise RuntimeError("neural model provenance changed across timestep runs")

    certification = certify_metrics(
        runs,
        slope_min=args.slope_min,
        slope_max=args.slope_max,
        min_r2=args.min_r2,
        max_relative_drift=args.max_relative_drift,
    )
    provenance = {
        "kind": "torchmd_tel22_sized_neural_nve_certification",
        "scope": "TorchMD frozen-neural-potential/autograd NVE certification; NOT TEL22 Hamiltonian parity",
        "particles": args.particles,
        "device": str(device),
        "precision": args.precision,
        "duration_ps": args.duration_ps,
        "dts_ps": list(args.dts),
        "temperature_K": args.temperature_k,
        "mass_g_mol": args.mass_g_mol,
        "k_range_kcal_mol_A2": [args.k_min, args.k_max],
        "initial_state_seed": args.seed,
        "neural_potential": {
            **config.as_dict(),
            "activation": "SiLU",
            "input": "per-particle squared displacement from equilibrium",
            "output": "positive bounded modulation of harmonic energy",
            "parameters_frozen": True,
            "force_definition": "F = -dE/dr via torch.autograd.grad",
            "parameter_count": next(iter(parameter_counts)),
            "parameter_sha256_canonical_float64": next(iter(fingerprints)),
        },
        "torch_version": str(torch.__version__),
        "torchmd_version": version_or_unknown("torchmd"),
        "python_version": platform.python_version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (out_root / "run_plan.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        **provenance,
        "energy_units": "kcal/mol",
        "length_units": "angstrom",
        "time_input_units": "fs (dt grid specified in ps and converted by x1000)",
        "runs": sorted(runs, key=lambda item: item["dt_ps"]),
        "certification": certification,
    }
    report_path = out_root / "neural_nve_certification_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_csv(out_root / "neural_nve_certification_runs.csv", runs)

    scaling = certification["scaling"]
    print("[RESULT]")
    print(f"  p              = {scaling['exponent_p']:.6f}")
    print(f"  log-log R2     = {scaling['loglog_r2']:.6f}")
    print(f"  C2 spread      = {certification['c2_spread_max_over_min']:.3f}")
    print(f"  max drift      = {max(r['relative_block_mean_drift'] for r in runs):.3e}")
    print(f"  certification  = {'PASS' if certification['pass'] else 'FAIL'}")
    print(f"  report         = {report_path}")
    return 0 if certification["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
