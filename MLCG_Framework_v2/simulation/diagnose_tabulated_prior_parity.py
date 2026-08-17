#!/usr/bin/env python3
"""Runtime parity check for v2 tabulated bonded prior subtraction.

Run with the ESPResSo ``pypresso`` executable.  The diagnostic configures
synthetic TabulatedDistance/Angle/Dihedral interactions in ESPResSo, evaluates
forces with ``run(0, recalc_forces=True)``, and compares them with the pure
NumPy kernels used by ``preprocessing/build_cg_dataset.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import espressomd
import espressomd.interactions
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))

from prior_kernels import (  # noqa: E402
    TabulatedPrior,
    tabulated_angle_forces,
    tabulated_dihedral_forces,
    tabulated_distance_forces,
)


def make_prior(x, energy, force, kind):
    return TabulatedPrior(
        x=np.asarray(x, dtype=float),
        energy=np.asarray(energy, dtype=float),
        force=np.asarray(force, dtype=float),
        minimum=float(x[0]),
        maximum=float(x[-1]),
        kind=kind,
        path=Path(f"synthetic_{kind}.dat"),
    )


def create_system():
    """Create the singleton ESPResSo System used by all parity probes."""
    system = espressomd.System(box_l=[30.0, 30.0, 30.0])
    system.time_step = 0.001
    system.cell_system.skin = 0.2
    system.thermostat.turn_off()
    system.integrator.set_vv()
    return system


def reset_system(system, positions):
    """Reset particles/bonds while respecting ESPResSo's one-System rule."""
    # Particles own bond records, so remove them before clearing bonded_inter.
    # ESPResSo permits only one System instance per Python process; reusing the
    # same object also keeps the three probes independent without relying on
    # garbage collection of previous System wrappers.
    system.part.clear()
    system.bonded_inter.clear()
    system.time = 0.0
    return [system.part.add(pos=pos, type=0, mass=1.0) for pos in positions]


def evaluate_bond(system):
    k, r0 = 7.0, 1.2
    x = np.linspace(0.5, 3.0, 2001)
    energy = 0.5 * k * (x - r0) ** 2
    force = -k * (x - r0)
    table = make_prior(x, energy, force, "bond")
    positions = np.asarray([[1.0, 2.0, 3.0], [2.7, 2.2, 3.0]])
    p = reset_system(system, positions)
    interaction = espressomd.interactions.TabulatedDistance(
        min=float(x[0]), max=float(x[-1]), energy=energy, force=force
    )
    system.bonded_inter.add(interaction)
    p[0].add_bond((interaction, p[1].id))
    system.integrator.run(0, recalc_forces=True)
    actual = np.asarray([part.f for part in p], dtype=float)
    expected = np.vstack(tabulated_distance_forces(*positions, np.asarray(system.box_l), table))
    return float(np.max(np.abs(actual - expected)))


def evaluate_angle(system):
    k, theta0 = 5.0, 1.3
    x = np.linspace(0.0, np.pi, 2001)
    energy = 0.5 * k * (x - theta0) ** 2
    gradient = k * (x - theta0)
    table = make_prior(x, energy, gradient, "angle")
    positions = np.asarray([
        [0.2, 0.4, 0.1],
        [1.0, 1.0, 1.0],
        [2.2, 1.4, 0.7],
    ])
    p = reset_system(system, positions)
    interaction = espressomd.interactions.TabulatedAngle(
        min=0.0, max=float(np.pi), energy=energy, force=gradient
    )
    system.bonded_inter.add(interaction)
    p[1].add_bond((interaction, p[0].id, p[2].id))
    system.integrator.run(0, recalc_forces=True)
    actual = np.asarray([part.f for part in p], dtype=float)
    expected = np.vstack(tabulated_angle_forces(*positions, np.asarray(system.box_l), table))
    return float(np.max(np.abs(actual - expected)))


def evaluate_dihedral(system):
    k = 3.7
    x = np.linspace(0.0, 2.0 * np.pi, 4001)
    energy = k * (1.0 - np.cos(x))
    # ESPResSo TabulatedDihedral stores a torsional geometry factor.  For this
    # analytic cosine profile the corresponding factor is the constant -k.
    force_factor = np.full_like(x, -k)
    table = make_prior(x, energy, force_factor, "dihedral")
    positions = np.asarray([
        [0.2, 0.4, 0.1],
        [1.1, 0.9, 0.6],
        [2.0, 1.5, 1.2],
        [2.8, 2.2, 0.5],
    ])
    p = reset_system(system, positions)
    interaction = espressomd.interactions.TabulatedDihedral(
        min=0.0, max=float(2.0 * np.pi), energy=energy, force=force_factor
    )
    system.bonded_inter.add(interaction)
    p[1].add_bond((interaction, p[0].id, p[2].id, p[3].id))
    system.integrator.run(0, recalc_forces=True)
    actual = np.asarray([part.f for part in p], dtype=float)
    expected = np.vstack(tabulated_dihedral_forces(*positions, np.asarray(system.box_l), table))
    return float(np.max(np.abs(actual - expected)))


def main():
    system = create_system()
    errors = {
        "bond": evaluate_bond(system),
        "angle": evaluate_angle(system),
        "dihedral": evaluate_dihedral(system),
    }
    print("[TABULATED PRIOR RUNTIME/PREPROCESSING PARITY]")
    for name, error in errors.items():
        print(f"{name:8s} max |dF| = {error:.3e}")
    worst = max(errors.values())
    if not np.isfinite(worst) or worst > 1.0e-9:
        raise RuntimeError(f"Tabulated prior parity failed: max |dF|={worst:.6g}")
    print("[PASS] ESPResSo tabulated bonded forces match preprocessing subtraction kernels.")


if __name__ == "__main__":
    main()
