"""Factories for ESPResSo interactions used by the generic MLCG runtime."""

from __future__ import annotations

from typing import Any


DEFAULT_MORSE_R_CUT_NM = 15.0


def make_analytic_morse_bond(interactions: Any, prior: dict[str, Any]):
    """Construct the conservative MLCG Morse bonded interaction.

    No tabulated fallback is allowed: independent interpolation of energy and
    force would invalidate strict NVE energy-conservation tests.
    """
    cls = getattr(interactions, "MorseBond", None)
    if cls is None:
        raise RuntimeError(
            "This simulation contains a Morse bonded prior, but the current "
            "ESPResSo build has no espressomd.interactions.MorseBond. Re-run "
            "simulation/espresso_plugin/copy_plugin_files.sh and rebuild ESPResSo."
        )

    D = float(prior["D"])
    a = float(prior["a"])
    r0 = float(prior["r0"])
    r_cut = float(prior.get("r_cut", DEFAULT_MORSE_R_CUT_NM))
    if D < 0.0:
        raise ValueError("Morse D must be >= 0")
    if a <= 0.0:
        raise ValueError("Morse a must be > 0")
    if r0 < 0.0:
        raise ValueError("Morse r0 must be >= 0")
    if r_cut <= 0.0:
        raise ValueError("Morse r_cut must be > 0")
    return cls(D=D, a=a, r_0=r0, r_cut=r_cut)
