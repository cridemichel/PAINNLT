#!/usr/bin/env python3
"""Runtime smoke test for the analytic ESPResSo MorseBond extension.

Run this script with the *rebuilt* ESPResSo Python launcher (normally
``pypresso``).  It checks that the Python binding is present and that bonded
energy and force agree with the analytic Morse expression on a two-particle
system.
"""

from __future__ import annotations

import math

import espressomd
from espressomd import interactions


D = 8.5
a = 3.2
r0 = 0.37
r = 0.51
r_cut = 2.0


def morse_energy(distance: float) -> float:
    exp_term = math.exp(-a * (distance - r0))
    return D * (1.0 - exp_term) ** 2


def dU_dr(distance: float) -> float:
    exp_term = math.exp(-a * (distance - r0))
    return 2.0 * a * D * (1.0 - exp_term) * exp_term


def main() -> None:
    cls = getattr(interactions, "MorseBond", None)
    if cls is None:
        raise RuntimeError(
            "espressomd.interactions.MorseBond is unavailable. Install the "
            "MLCG extension into the ESPResSo source tree and rebuild ESPResSo."
        )

    system = espressomd.System(box_l=[10.0, 10.0, 10.0])
    system.time_step = 1.0e-3
    system.cell_system.skin = 0.1

    p0 = system.part.add(pos=[4.0, 5.0, 5.0])
    p1 = system.part.add(pos=[4.0 + r, 5.0, 5.0])

    bond = cls(D=D, a=a, r_0=r0, r_cut=r_cut)
    system.bonded_inter.add(bond)
    p0.add_bond((bond, p1))

    system.integrator.run(0, recalc_forces=True)

    observed_energy = float(system.analysis.energy()["bonded"])
    expected_energy = morse_energy(r)

    # p0 lies to the left of p1.  For r > r0 the attractive force on p0 is
    # +x with magnitude dU/dr; p1 receives the opposite force.
    expected_fx_p0 = dU_dr(r)
    observed_fx_p0 = float(p0.f[0])
    observed_fx_p1 = float(p1.f[0])

    energy_tol = 1.0e-10 * max(1.0, abs(expected_energy))
    force_tol = 1.0e-10 * max(1.0, abs(expected_fx_p0))

    if abs(observed_energy - expected_energy) > energy_tol:
        raise RuntimeError(
            "MorseBond energy mismatch: "
            f"observed={observed_energy:.16e}, expected={expected_energy:.16e}"
        )
    if abs(observed_fx_p0 - expected_fx_p0) > force_tol:
        raise RuntimeError(
            "MorseBond force mismatch on particle 0: "
            f"observed={observed_fx_p0:.16e}, expected={expected_fx_p0:.16e}"
        )
    if abs(observed_fx_p1 + expected_fx_p0) > force_tol:
        raise RuntimeError(
            "MorseBond Newton-pair mismatch on particle 1: "
            f"observed={observed_fx_p1:.16e}, expected={-expected_fx_p0:.16e}"
        )

    print("[PASS] Runtime analytic MorseBond smoke test")
    print(f"       U(r)={observed_energy:.12g}")
    print(f"       F0x={observed_fx_p0:.12g}  F1x={observed_fx_p1:.12g}")


if __name__ == "__main__":
    main()
