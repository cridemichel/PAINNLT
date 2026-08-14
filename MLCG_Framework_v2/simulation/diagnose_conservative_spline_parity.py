#!/usr/bin/env python3
"""ESPResSo runtime/preprocessing parity for conservative bond/angle splines.

This diagnostic must be run with pypresso after installing the MLCG conservative
spline bonded interactions.  It probes every unique conservative spline table
referenced by the selected priors and compares both Cartesian force and bonded
energy against the preprocessing implementation of the same Hermite kernel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import espressomd
import espressomd.interactions
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "simulation"))

from conservative_spline import (  # noqa: E402
    conservative_angle_forces,
    conservative_distance_forces,
    conservative_spline_value,
    load_conservative_spline,
)
from conservative_spline_runtime import create_conservative_spline_interaction  # noqa: E402


def system_singleton():
    system = espressomd.System(box_l=[30.0, 30.0, 30.0])
    system.time_step = 0.001
    system.cell_system.skin = 0.2
    system.thermostat.turn_off()
    system.integrator.set_vv()
    return system


def reset(system, positions):
    system.part.clear()
    system.bonded_inter.clear()
    system.time = 0.0
    return [system.part.add(pos=p, type=0, mass=1.0) for p in positions]


def unique_entries(priors_path: Path):
    data = json.loads(priors_path.read_text())
    seen = set()
    for json_key, kind in (("bonds", "bond"), ("angles", "angle")):
        for idx, entry in enumerate(data.get(json_key, [])):
            if str(entry.get("type", "")).lower() != "conservative_spline":
                continue
            key = (kind, str(entry.get("file", "")))
            if key in seen:
                continue
            seen.add(key)
            yield json_key, idx, kind, entry


def probe_bond(system, priors_path: Path, entry, table):
    max_df = 0.0
    max_de = 0.0
    for frac in (0.173, 0.371, 0.713):
        q = table.minimum + frac * (table.maximum - table.minimum)
        center = np.asarray([10.0, 10.0, 10.0])
        pos = np.asarray([center, center + np.asarray([q, 0.0, 0.0])])
        particles = reset(system, pos)
        ia = create_conservative_spline_interaction(
            espressomd.interactions, entry, kind="bond", priors_path=priors_path
        )
        system.bonded_inter.add(ia)
        particles[0].add_bond((ia, particles[1].id))
        system.integrator.run(0, recalc_forces=True)
        actual_f = np.asarray([particle.f for particle in particles])
        expected_f = np.vstack(
            conservative_distance_forces(pos[0], pos[1], np.asarray(system.box_l), table)
        )
        expected_e = conservative_spline_value(table, float(q))[0]
        actual_e = float(system.analysis.energy()["bonded"])
        max_df = max(max_df, float(np.max(np.abs(actual_f - expected_f))))
        max_de = max(max_de, abs(actual_e - expected_e))
    return max_df, max_de


def probe_angle(system, priors_path: Path, entry, table):
    max_df = 0.0
    max_de = 0.0
    # Avoid the singular 0/pi endpoints by construction.
    for frac in (0.173, 0.371, 0.713):
        theta = table.minimum + frac * (table.maximum - table.minimum)
        center = np.asarray([10.0, 10.0, 10.0])
        pos = np.asarray([
            center + np.asarray([1.0, 0.0, 0.0]),
            center,
            center + np.asarray([np.cos(theta), np.sin(theta), 0.0]),
        ])
        particles = reset(system, pos)
        ia = create_conservative_spline_interaction(
            espressomd.interactions, entry, kind="angle", priors_path=priors_path
        )
        system.bonded_inter.add(ia)
        particles[1].add_bond((ia, particles[0].id, particles[2].id))
        system.integrator.run(0, recalc_forces=True)
        actual_f = np.asarray([particle.f for particle in particles])
        expected_f = np.vstack(
            conservative_angle_forces(*pos, np.asarray(system.box_l), table)
        )
        expected_e = conservative_spline_value(table, float(theta))[0]
        actual_e = float(system.analysis.energy()["bonded"])
        max_df = max(max_df, float(np.max(np.abs(actual_f - expected_f))))
        max_de = max(max_de, abs(actual_e - expected_e))
    return max_df, max_de


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priors", required=True, type=Path)
    parser.add_argument("--force-atol", type=float, default=1.0e-9)
    parser.add_argument("--energy-atol", type=float, default=1.0e-10)
    args = parser.parse_args()

    priors_path = args.priors.expanduser().resolve()
    if not priors_path.is_file():
        raise FileNotFoundError(priors_path)
    for name in ("ConservativeSplineDistance", "ConservativeSplineAngle"):
        if not hasattr(espressomd.interactions, name):
            raise RuntimeError(
                f"Missing espressomd.interactions.{name}; install/rebuild the conservative spline plugin first"
            )

    entries = list(unique_entries(priors_path))
    if not entries:
        raise ValueError(f"No conservative bond/angle spline entries found in {priors_path}")

    system = system_singleton()
    results = []
    worst_f = 0.0
    worst_e = 0.0
    for json_key, idx, kind, entry in entries:
        table = load_conservative_spline(entry, kind=kind, priors_path=priors_path)
        if kind == "bond":
            df, de = probe_bond(system, priors_path, entry, table)
        else:
            df, de = probe_angle(system, priors_path, entry, table)
        results.append((kind, str(entry["file"]), df, de))
        worst_f = max(worst_f, df)
        worst_e = max(worst_e, de)

    print("[CONSERVATIVE SPLINE RUNTIME/PREPROCESSING PARITY]")
    print(f"priors: {priors_path}")
    for kind, filename, df, de in results:
        print(f"{kind:5s} {filename}: max |dF|={df:.3e} |dE|={de:.3e}")
    print(f"worst: max |dF|={worst_f:.3e} |dE|={worst_e:.3e}")
    if (
        not np.isfinite(worst_f + worst_e)
        or worst_f > args.force_atol
        or worst_e > args.energy_atol
    ):
        raise RuntimeError(
            "Conservative spline runtime parity failed: "
            f"max|dF|={worst_f:.6g}, max|dE|={worst_e:.6g}"
        )
    print("[PASS] ESPResSo and preprocessing evaluate the same conservative Hermite energy/force kernel on every unique converted table.")


if __name__ == "__main__":
    main()
