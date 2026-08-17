#!/usr/bin/env python3
"""Synthetic ESPResSo smoke test for MLCG conservative bonded splines.

Run with the rebuilt ``pypresso``.  The test is independent of any IBI run: it
creates small synthetic distance, angle and periodic dihedral spline tables, loads them through the
same runtime loader used by production, and checks directly in ESPResSo that
particle forces are the negative Cartesian finite-difference gradient of the
reported bonded energy.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import espressomd
import espressomd.interactions
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from conservative_spline_runtime import create_conservative_spline_interaction  # noqa: E402


def make_system():
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
    return [system.part.add(pos=pos, type=0, mass=1.0) for pos in positions]


def evaluate(system, *, kind, positions, entry, priors_path):
    particles = reset(system, positions)
    interaction = create_conservative_spline_interaction(
        espressomd.interactions,
        entry,
        kind=kind,
        priors_path=priors_path,
    )
    system.bonded_inter.add(interaction)
    if kind == "bond":
        particles[0].add_bond((interaction, particles[1].id))
    elif kind == "angle":
        particles[1].add_bond((interaction, particles[0].id, particles[2].id))
    elif kind == "dihedral":
        particles[1].add_bond((interaction, particles[0].id, particles[2].id, particles[3].id))
    else:
        raise ValueError(kind)
    system.integrator.run(0, recalc_forces=True)
    forces = np.asarray([np.asarray(p.f, dtype=float) for p in particles])
    energy = float(system.analysis.energy()["bonded"])
    return forces, energy


def finite_difference_forces(system, *, kind, positions, entry, priors_path, eps):
    positions = np.asarray(positions, dtype=float)
    result = np.zeros_like(positions)
    for i in range(positions.shape[0]):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[i, axis] += eps
            minus[i, axis] -= eps
            _fp, ep = evaluate(
                system,
                kind=kind,
                positions=plus,
                entry=entry,
                priors_path=priors_path,
            )
            _fm, em = evaluate(
                system,
                kind=kind,
                positions=minus,
                entry=entry,
                priors_path=priors_path,
            )
            result[i, axis] = -(ep - em) / (2.0 * eps)
    return result


def write_table(path: Path, x, energy, derivative):
    np.savetxt(
        path,
        np.column_stack((x, energy, derivative)),
        fmt="%.17g",
        header="q U dU_dq synthetic conservative runtime smoke",
    )


def probe(system, *, kind, positions, entry, priors_path, eps):
    actual_f, energy = evaluate(
        system,
        kind=kind,
        positions=positions,
        entry=entry,
        priors_path=priors_path,
    )
    fd_f = finite_difference_forces(
        system,
        kind=kind,
        positions=positions,
        entry=entry,
        priors_path=priors_path,
        eps=eps,
    )
    max_error = float(np.max(np.abs(actual_f - fd_f)))
    net_force = float(np.max(np.abs(np.sum(actual_f, axis=0))))
    if not np.isfinite(energy) or not np.all(np.isfinite(actual_f)):
        raise RuntimeError(f"{kind} conservative spline produced NaN/Inf")
    return max_error, net_force, energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd-eps", type=float, default=1.0e-6)
    parser.add_argument("--force-atol", type=float, default=2.0e-6)
    parser.add_argument("--net-force-atol", type=float, default=1.0e-12)
    args = parser.parse_args()

    for name in ("ConservativeSplineDistance", "ConservativeSplineAngle", "ConservativeSplineDihedral"):
        if not hasattr(espressomd.interactions, name):
            raise RuntimeError(
                f"Missing espressomd.interactions.{name}; install/rebuild the conservative spline plugin first"
            )

    system = make_system()
    with tempfile.TemporaryDirectory(prefix="mlcg_conservative_smoke_") as tmpdir:
        root = Path(tmpdir)
        priors_path = root / "cg_priors.json"
        priors_path.write_text(json.dumps({"bonds": [], "angles": [], "dihedrals": []}))

        xb = np.linspace(0.4, 2.0, 33)
        ub = 3.0 * (xb - 1.1) ** 2 + 0.15 * (xb - 1.1) ** 4
        dub = 6.0 * (xb - 1.1) + 0.6 * (xb - 1.1) ** 3
        write_table(root / "bond.dat", xb, ub, dub)
        bond_entry = {
            "type": "conservative_spline",
            "spline_schema": "pchip_hermite_v1",
            "file": "bond.dat",
            "min": float(xb[0]),
            "max": float(xb[-1]),
        }
        center = np.asarray([10.0, 10.0, 10.0])
        bond_positions = np.asarray([center, center + np.asarray([1.23, 0.17, -0.09])])
        bond_error, bond_net, bond_energy = probe(
            system,
            kind="bond",
            positions=bond_positions,
            entry=bond_entry,
            priors_path=priors_path,
            eps=args.fd_eps,
        )

        xa = np.linspace(0.0, np.pi, 65)
        ua = 1.7 * (xa - 1.35) ** 2 + 0.08 * (xa - 1.35) ** 4
        dua = 3.4 * (xa - 1.35) + 0.32 * (xa - 1.35) ** 3
        write_table(root / "angle.dat", xa, ua, dua)
        angle_entry = {
            "type": "conservative_spline",
            "spline_schema": "pchip_hermite_v1",
            "file": "angle.dat",
            "min": 0.0,
            "max": float(np.pi),
        }
        theta = 1.17
        angle_positions = np.asarray([
            center + np.asarray([1.13, 0.08, 0.04]),
            center,
            center + 0.91 * np.asarray([np.cos(theta), np.sin(theta), 0.11]),
        ])
        angle_error, angle_net, angle_energy = probe(
            system,
            kind="angle",
            positions=angle_positions,
            entry=angle_entry,
            priors_path=priors_path,
            eps=args.fd_eps,
        )

        xd = np.linspace(0.0, 2.0 * np.pi, 129)
        kd = 2.3
        ud = kd * (1.0 - np.cos(xd))
        dud = kd * np.sin(xd)
        write_table(root / "dihedral.dat", xd, ud, dud)
        dihedral_entry = {
            "type": "conservative_spline",
            "spline_schema": "pchip_hermite_v1",
            "file": "dihedral.dat",
            "min": 0.0,
            "max": float(2.0 * np.pi),
        }
        dihedral_positions = center + np.asarray([
            [0.2, 0.4, 0.1],
            [1.1, 0.9, 0.6],
            [2.0, 1.5, 1.2],
            [2.8, 2.2, 0.5],
        ])
        dihedral_error, dihedral_net, dihedral_energy = probe(
            system,
            kind="dihedral",
            positions=dihedral_positions,
            entry=dihedral_entry,
            priors_path=priors_path,
            eps=args.fd_eps,
        )

    print("[CONSERVATIVE SPLINE SYNTHETIC RUNTIME SMOKE]")
    print(
        f"bond : E={bond_energy:.12g} max |F + grad(U)|={bond_error:.3e} "
        f"max |sum(F)|={bond_net:.3e}"
    )
    print(
        f"angle: E={angle_energy:.12g} max |F + grad(U)|={angle_error:.3e} "
        f"max |sum(F)|={angle_net:.3e}"
    )
    print(
        f"dihed: E={dihedral_energy:.12g} max |F + grad(U)|={dihedral_error:.3e} "
        f"max |sum(F)|={dihedral_net:.3e}"
    )
    worst_force = max(bond_error, angle_error, dihedral_error)
    worst_net = max(bond_net, angle_net, dihedral_net)
    if worst_force > args.force_atol or worst_net > args.net_force_atol:
        raise RuntimeError(
            "Conservative spline runtime smoke failed: "
            f"max|F+gradU|={worst_force:.6g}, max|sumF|={worst_net:.6g}"
        )
    print(
        "[PASS] ESPResSo conservative distance/angle/dihedral splines return forces equal to "
        "the negative Cartesian gradient of their bonded energy."
    )


if __name__ == "__main__":
    main()
