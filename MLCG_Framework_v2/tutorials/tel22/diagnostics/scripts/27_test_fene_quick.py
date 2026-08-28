#!/usr/bin/env python3
"""Fast, isolated analytic/ESPResSo/NVE closure for a single FENE bond."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import numpy as np

import espressomd
import espressomd.interactions


def fene_energy(r: float, k: float, r0: float, rmax: float) -> float:
    x = r - r0
    return -0.5 * k * rmax**2 * math.log1p(-(x / rmax) ** 2)


def fene_force_on_right(r: float, k: float, r0: float, rmax: float) -> float:
    """Cartesian x force on the right particle for a bond aligned with +x."""
    x = r - r0
    return -k * x / (1.0 - (x / rmax) ** 2)


def relative_error(observed: float, expected: float, floor: float = 1.0e-12) -> float:
    return abs(observed - expected) / max(abs(expected), floor)


def get_energy(system: espressomd.System, component: str) -> float:
    energies = system.analysis.energy()
    if component in energies:
        return float(energies[component])
    if component == "bonded":
        values = [
            float(value)
            for key, value in energies.items()
            if isinstance(key, tuple) and key and key[0] == "bonded"
        ]
        if values:
            return float(sum(values))
    raise KeyError(f"energy component {component!r} absent; keys={list(energies)}")


def configure_system(k: float, r0: float, rmax: float):
    system = espressomd.System(box_l=[10.0, 10.0, 10.0])
    system.time_step = 0.001
    system.cell_system.skin = 0.2
    system.thermostat.turn_off()
    system.integrator.set_vv()

    center = np.array([5.0, 5.0, 5.0])
    p0 = system.part.add(id=0, pos=center - np.array([0.5 * r0, 0.0, 0.0]))
    p1 = system.part.add(id=1, pos=center + np.array([0.5 * r0, 0.0, 0.0]))
    bond = espressomd.interactions.FeneBond(k=k, d_r_max=rmax, r_0=r0)
    system.bonded_inter.add(bond)
    p0.add_bond((bond, p1))
    return system, p0, p1, center


def set_state(system, p0, p1, center: np.ndarray, r: float) -> None:
    offset = np.array([0.5 * r, 0.0, 0.0])
    p0.pos = center - offset
    p1.pos = center + offset
    p0.v = [0.0, 0.0, 0.0]
    p1.v = [0.0, 0.0, 0.0]
    system.time = 0.0
    system.integrator.run(0, recalc_forces=True)


def snapshot(system, p0, p1, center: np.ndarray, r: float) -> dict:
    set_state(system, p0, p1, center, r)
    return {
        "r": r,
        "energy": get_energy(system, "bonded"),
        "force_left": np.asarray(p0.f, dtype=float).tolist(),
        "force_right": np.asarray(p1.f, dtype=float).tolist(),
    }


def analytic_runtime_closure(system, p0, p1, center, k, r0, rmax) -> dict:
    fractions = [-0.80, -0.60, -0.30, -0.10, 0.0, 0.10, 0.30, 0.60, 0.80, 0.95]
    samples = []
    max_energy_abs_error = 0.0
    max_force_abs_error = 0.0
    max_force_relative_error = 0.0
    max_action_reaction_error = 0.0
    max_transverse_force = 0.0

    for fraction in fractions:
        r = r0 + fraction * rmax
        row = snapshot(system, p0, p1, center, r)
        expected_energy = fene_energy(r, k, r0, rmax)
        expected_force = fene_force_on_right(r, k, r0, rmax)
        force_left = np.asarray(row.pop("force_left"))
        force_right = np.asarray(row.pop("force_right"))
        energy_abs_error = abs(row["energy"] - expected_energy)
        force_abs_error = abs(force_right[0] - expected_force)
        force_relative_error = relative_error(force_right[0], expected_force)
        action_reaction_error = float(np.linalg.norm(force_left + force_right))
        transverse_force = float(np.linalg.norm(force_right[1:]))
        max_energy_abs_error = max(max_energy_abs_error, energy_abs_error)
        max_force_abs_error = max(max_force_abs_error, force_abs_error)
        max_force_relative_error = max(max_force_relative_error, force_relative_error)
        max_action_reaction_error = max(max_action_reaction_error, action_reaction_error)
        max_transverse_force = max(max_transverse_force, transverse_force)
        samples.append(
            {
                **row,
                "extension_fraction": fraction,
                "expected_energy": expected_energy,
                "force_right_x": float(force_right[0]),
                "expected_force_right_x": expected_force,
                "energy_abs_error": energy_abs_error,
                "force_abs_error": force_abs_error,
                "force_relative_error": force_relative_error,
                "action_reaction_error": action_reaction_error,
                "transverse_force": transverse_force,
            }
        )

    passed = (
        max_energy_abs_error <= 1.0e-10
        and max_force_abs_error <= 1.0e-9
        and max_action_reaction_error <= 1.0e-10
        and max_transverse_force <= 1.0e-10
    )
    return {
        "pass": passed,
        "max_energy_abs_error": max_energy_abs_error,
        "max_force_abs_error": max_force_abs_error,
        "max_force_relative_error_nonzero_samples": max_force_relative_error,
        "max_action_reaction_error": max_action_reaction_error,
        "max_transverse_force": max_transverse_force,
        "largest_tested_abs_extension_fraction": max(abs(x) for x in fractions),
        "samples": samples,
    }


def finite_difference_closure(system, p0, p1, center, k, r0, rmax) -> dict:
    h = 1.0e-6
    fractions = [-0.70, -0.25, 0.20, 0.65]
    rows = []
    max_abs_error = 0.0
    max_relative_error = 0.0
    for fraction in fractions:
        r = r0 + fraction * rmax
        energy_plus = snapshot(system, p0, p1, center, r + h)["energy"]
        energy_minus = snapshot(system, p0, p1, center, r - h)["energy"]
        force_fd = -(energy_plus - energy_minus) / (2.0 * h)
        force_runtime = snapshot(system, p0, p1, center, r)["force_right"][0]
        abs_error = abs(force_runtime - force_fd)
        rel_error = relative_error(force_runtime, force_fd)
        max_abs_error = max(max_abs_error, abs_error)
        max_relative_error = max(max_relative_error, rel_error)
        rows.append(
            {
                "r": r,
                "extension_fraction": fraction,
                "force_runtime": force_runtime,
                "force_from_energy_finite_difference": force_fd,
                "abs_error": abs_error,
                "relative_error": rel_error,
            }
        )
    return {
        "pass": max_relative_error <= 2.0e-7,
        "finite_difference_h": h,
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error,
        "samples": rows,
    }


def nve_scaling_closure(system, p0, p1, center, k, r0, rmax, duration) -> dict:
    dts = [0.005, 0.0075, 0.010, 0.015, 0.020, 0.025]
    initial_extension = 0.30 * rmax
    runs = []

    for dt in dts:
        system.time_step = dt
        set_state(system, p0, p1, center, r0 + initial_extension)
        energies = [get_energy(system, "total")]
        min_r = math.inf
        max_r = -math.inf
        steps = int(round(duration / dt))
        for _ in range(steps):
            system.integrator.run(1)
            energies.append(get_energy(system, "total"))
            r = float(np.linalg.norm(np.asarray(p1.pos) - np.asarray(p0.pos)))
            min_r = min(min_r, r)
            max_r = max(max_r, r)
        energy_array = np.asarray(energies, dtype=float)
        sigma = float(np.std(energy_array))
        quarter = max(2, len(energy_array) // 4)
        drift = abs(float(np.mean(energy_array[-quarter:]) - np.mean(energy_array[:quarter])))
        drift /= max(abs(float(np.mean(energy_array))), 1.0e-30)
        runs.append(
            {
                "dt": dt,
                "steps": steps,
                "duration": steps * dt,
                "sigma_E": sigma,
                "C2_sigma_over_dt2": sigma / dt**2,
                "relative_block_mean_drift": drift,
                "min_r": min_r,
                "max_r": max_r,
                "max_abs_extension_fraction": max(abs(min_r - r0), abs(max_r - r0)) / rmax,
            }
        )

    log_dt = np.log(np.asarray([row["dt"] for row in runs]))
    log_sigma = np.log(np.asarray([row["sigma_E"] for row in runs]))
    slope, intercept = np.polyfit(log_dt, log_sigma, 1)
    fitted = intercept + slope * log_dt
    residual = float(np.sum((log_sigma - fitted) ** 2))
    total = float(np.sum((log_sigma - np.mean(log_sigma)) ** 2))
    r2 = 1.0 - residual / total
    max_drift = max(row["relative_block_mean_drift"] for row in runs)
    max_extension_fraction = max(row["max_abs_extension_fraction"] for row in runs)
    passed = 1.80 <= slope <= 2.20 and r2 >= 0.995 and max_extension_fraction < 0.90
    return {
        "pass": passed,
        "exponent_p": float(slope),
        "loglog_r2": r2,
        "max_relative_block_mean_drift": max_drift,
        "max_abs_extension_fraction": max_extension_fraction,
        "acceptance": {"p_min": 1.80, "p_max": 2.20, "min_r2": 0.995},
        "runs": runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", type=float, default=30.0)
    parser.add_argument("--r0", type=float, default=1.2)
    parser.add_argument("--rmax", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=4.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    if args.k <= 0.0 or args.r0 <= args.rmax or args.rmax <= 0.0 or args.duration <= 0.0:
        raise ValueError("require k>0, r0>rmax>0, and duration>0")
    system, p0, p1, center = configure_system(args.k, args.r0, args.rmax)
    analytic = analytic_runtime_closure(system, p0, p1, center, args.k, args.r0, args.rmax)
    finite_difference = finite_difference_closure(
        system, p0, p1, center, args.k, args.r0, args.rmax
    )
    nve = nve_scaling_closure(
        system, p0, p1, center, args.k, args.r0, args.rmax, args.duration
    )
    overall = analytic["pass"] and finite_difference["pass"] and nve["pass"]
    return {
        "schema_version": 1,
        "kind": "fene_espresso_quick_closure",
        "scope": "isolated two-particle FENE; no PaiNN, nonbonded interactions, thermostat, or production files",
        "parameters": {
            "k": args.k,
            "r0": args.r0,
            "d_r_max": args.rmax,
            "nve_duration": args.duration,
        },
        "formula": {
            "energy": "-0.5*k*d_r_max^2*log(1-((r-r0)/d_r_max)^2)",
            "force_on_right": "-k*(r-r0)/(1-((r-r0)/d_r_max)^2)",
            "domain": "abs(r-r0) < d_r_max",
        },
        "espresso_version": getattr(espressomd, "__version__", "unknown"),
        "analytic_runtime_closure": analytic,
        "finite_difference_closure": finite_difference,
        "nve_scaling_closure": nve,
        "overall_pass": overall,
        "limitation": "This fast test validates ESPResSo FENE mechanics and integration; it does not exercise a project-specific residual-dataset FENE mapping.",
    }


def main() -> int:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args)
    except Exception as exc:  # preserve a machine-readable failure report
        report = {
            "schema_version": 1,
            "kind": "fene_espresso_quick_closure",
            "overall_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("[FENE QUICK CLOSURE]")
    if "analytic_runtime_closure" in report:
        print(f"analytic/runtime : {'PASS' if report['analytic_runtime_closure']['pass'] else 'FAIL'}")
        print(f"finite difference: {'PASS' if report['finite_difference_closure']['pass'] else 'FAIL'}")
        print(
            "NVE scaling      : "
            f"{'PASS' if report['nve_scaling_closure']['pass'] else 'FAIL'} "
            f"(p={report['nve_scaling_closure']['exponent_p']:.6f}, "
            f"R2={report['nve_scaling_closure']['loglog_r2']:.6f})"
        )
    else:
        print(f"ERROR            : {report['error']}")
    print(f"overall          : {'PASS' if report['overall_pass'] else 'FAIL'}")
    print(f"report           : {args.output}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
