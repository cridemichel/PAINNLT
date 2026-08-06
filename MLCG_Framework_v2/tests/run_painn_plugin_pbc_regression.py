#!/usr/bin/env python3
"""Runtime regression for PaiNN periodic ghosts and the zero-edge energy path.

Run this file with the patched ESPResSo ``pypresso`` executable, not with the
system Python interpreter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import espressomd
import espressomd.painn
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rtol", type=float, default=2.0e-5)
    parser.add_argument("--atol", type=float, default=2.0e-6)
    parser.add_argument("--fd_rtol", type=float, default=1.0e-2)
    parser.add_argument("--fd_atol", type=float, default=1.0e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.config.open() as handle:
        config = json.load(handle)

    required = ["num_species", "hidden_channels", "n_layers", "num_rbf", "cutoff", "toxvaerd_alpha"]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing model configuration keys: {missing}")
    if int(config["num_species"]) < 1:
        raise ValueError("num_species must be at least one")

    cutoff = float(config["cutoff"])
    skin = min(0.2, max(0.02, 0.1 * cutoff))
    box_length = max(4.0, 4.0 * (cutoff + skin))
    delta = min(0.2, max(0.02, 0.25 * cutoff))

    system = espressomd.System(box_l=[box_length] * 3)
    system.time_step = 1.0e-4
    system.cell_system.skin = skin
    system.thermostat.turn_off()
    system.force_cap = 0.0

    p1 = system.part.add(
        pos=[0.5 * delta, 0.5 * box_length, 0.5 * box_length],
        type=0,
        mass=1.0,
        mol_id=0,
    )
    p2 = system.part.add(
        pos=[box_length - 0.5 * delta, 0.5 * box_length, 0.5 * box_length],
        type=min(1, int(config["num_species"]) - 1),
        mass=1.0,
        mol_id=1,
    )

    # A zero-strength short-range interaction tells ESPResSo which cutoff must
    # be represented in its cell/Verlet neighbour traversal.
    for type_i in range(int(config["num_species"])):
        for type_j in range(type_i, int(config["num_species"])):
            system.non_bonded_inter[type_i, type_j].soft_sphere.set_params(
                a=0.0, n=1, cutoff=cutoff, offset=0.0
            )

    espressomd.painn.activate_painn_potential(
        str(args.model.resolve()),
        int(config["num_species"]),
        int(config["hidden_channels"]),
        int(config["n_layers"]),
        int(config["num_rbf"]),
        cutoff,
        float(config["toxvaerd_alpha"]),
        args.device,
    )

    def evaluate(pos1, pos2):
        p1.pos = pos1
        p2.pos = pos2
        system.integrator.run(0, recalc_forces=True)
        energy = float(espressomd.painn.get_painn_energy())
        forces = np.asarray([p1.f, p2.f], dtype=float)
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise RuntimeError("Non-finite PaiNN energy or force")
        return energy, forces

    center = 0.5 * box_length
    pbc_energy, pbc_forces = evaluate(
        [0.5 * delta, center, center],
        [box_length - 0.5 * delta, center, center],
    )
    interior_energy, interior_forces = evaluate(
        [center, center, center],
        [center - delta, center, center],
    )

    if not np.isclose(pbc_energy, interior_energy, rtol=args.rtol, atol=args.atol):
        raise AssertionError(
            f"PBC energy mismatch: boundary={pbc_energy:.16e}, interior={interior_energy:.16e}"
        )
    if not np.allclose(pbc_forces, interior_forces, rtol=args.rtol, atol=args.atol):
        raise AssertionError(
            "PBC force mismatch:\n"
            f"boundary={pbc_forces}\ninterior={interior_forces}"
        )

    # Direct Hamiltonian check: the force reported by ESPResSo must be the
    # negative central finite difference of the same PaiNN energy scalar.
    fd_step = max(1.0e-4, 1.0e-3 * delta)
    plus_energy, _ = evaluate(
        [center + fd_step, center, center],
        [center - delta, center, center],
    )
    minus_energy, _ = evaluate(
        [center - fd_step, center, center],
        [center - delta, center, center],
    )
    finite_difference_force = -(plus_energy - minus_energy) / (2.0 * fd_step)
    plugin_force = float(interior_forces[0, 0])
    if not np.isclose(
        plugin_force, finite_difference_force, rtol=args.fd_rtol, atol=args.fd_atol
    ):
        raise AssertionError(
            "PaiNN force is not the negative derivative of the reported energy: "
            f"plugin={plugin_force:.16e}, finite_difference={finite_difference_force:.16e}"
        )

    far_energy_1, far_forces_1 = evaluate(
        [0.25 * box_length, center, center],
        [0.75 * box_length, center, center],
    )
    far_energy_2, far_forces_2 = evaluate(
        [0.20 * box_length, 0.40 * box_length, center],
        [0.70 * box_length, 0.40 * box_length, center],
    )
    if not np.isclose(far_energy_1, far_energy_2, rtol=args.rtol, atol=args.atol):
        raise AssertionError(
            f"Zero-edge baseline energy changed with position: {far_energy_1} vs {far_energy_2}"
        )
    if not np.isclose(far_energy_1, 0.0, rtol=0.0, atol=args.atol):
        raise AssertionError(
            f"Isolated-species gauge did not zero the no-edge energy: {far_energy_1}"
        )
    force_limit = max(args.atol, 1.0e-6)
    if np.max(np.abs(far_forces_1)) > force_limit or np.max(np.abs(far_forces_2)) > force_limit:
        raise AssertionError(
            f"Zero-edge forces are not zero: {far_forces_1}, {far_forces_2}"
        )

    print("[PASS] PaiNN PBC energy/force translation invariance")
    print("[PASS] PaiNN force equals the finite-difference energy gradient")
    print("[PASS] PaiNN isolated-species gauge and zero-edge forces")
    print(f"[INFO] interacting energy: {pbc_energy:.16e}")
    print(f"[INFO] isolated-sites energy: {far_energy_1:.16e}")


if __name__ == "__main__":
    main()
