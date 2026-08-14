#!/usr/bin/env python3
"""Runtime diagnostic for site-addressable pair-specific Morse force/torque transfer.

The test is intentionally independent of TEL22 inputs.  It constructs two rigid
bodies, attaches one physical CG virtual site to each body, and asks the normal
MLCG pair-specific Morse machinery to create technical markers on those sites.
Only the marker-marker switched Morse interaction is active.  A force evaluation
is then compared with the analytic Morse energy, translational forces, and the
lab-frame torques expected on the two real COM particles.

Run this script with ESPResSo's ``pypresso``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from espresso_interactions import (
    configure_pair_specific_morse,
    create_pair_specific_morse_markers,
    prepare_pair_specific_morse,
    switched_morse_energy_radial_force,
)
from framework_utils import configure_neighbor_search


DEFAULT_PARAMS = {
    "D": 6.0,
    "a": 1.2,
    "r0": 2.0,
    "r_switch": 3.0,
    "r_cut": 4.0,
}


def _vec(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _sub(a: Iterable[float], b: Iterable[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _add(a: Iterable[float], b: Iterable[float]) -> list[float]:
    return [float(x) + float(y) for x, y in zip(a, b)]


def _scale(value: float, vector: Iterable[float]) -> list[float]:
    return [float(value) * float(x) for x in vector]


def _norm(vector: Iterable[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vector))


def _cross(a: Iterable[float], b: Iterable[float]) -> list[float]:
    ax, ay, az = _vec(a)
    bx, by, bz = _vec(b)
    return [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]


def _minimum_image(delta: Iterable[float], box_l: Iterable[float]) -> list[float]:
    result = []
    for value, length in zip(delta, box_l):
        value = float(value)
        length = float(length)
        result.append(value - math.floor(value / length + 0.5) * length)
    return result


def max_abs_error(actual: Iterable[float], expected: Iterable[float]) -> float:
    return max((abs(float(a) - float(e)) for a, e in zip(actual, expected)), default=0.0)


def expected_site_pair_wrench(
    *,
    com_i: Iterable[float],
    site_i: Iterable[float],
    com_j: Iterable[float],
    site_j: Iterable[float],
    box_l: Iterable[float],
    params: dict[str, float],
) -> dict[str, Any]:
    """Return analytic pair energy, forces, and COM torques for two site endpoints.

    ``switched_morse_energy_radial_force`` uses the signed radial convention
    ``F_r=-dU/dr``.  For a unit vector from endpoint i to endpoint j, the force
    on i is ``-F_r * e_ij`` and the force on j is its negative.
    """
    box = _vec(box_l)
    delta = _minimum_image(_sub(site_j, site_i), box)
    distance = _norm(delta)
    if distance <= 0.0:
        raise ValueError("Diagnostic endpoint distance must be positive")
    unit = _scale(1.0 / distance, delta)
    energy, radial_force = switched_morse_energy_radial_force(distance, **params)
    force_i = _scale(-radial_force, unit)
    force_j = _scale(-1.0, force_i)
    lever_i = _minimum_image(_sub(site_i, com_i), box)
    lever_j = _minimum_image(_sub(site_j, com_j), box)
    torque_i = _cross(lever_i, force_i)
    torque_j = _cross(lever_j, force_j)
    return {
        "distance": distance,
        "energy": energy,
        "radial_force": radial_force,
        "force_i": force_i,
        "force_j": force_j,
        "lever_i": lever_i,
        "lever_j": lever_j,
        "torque_i": torque_i,
        "torque_j": torque_j,
    }


def _add_physical_virtual_site(system: Any, *, parent: Any, pos: list[float], ptype: int, mol_id: int):
    site = system.part.add(
        pos=pos,
        type=int(ptype),
        mass=1.0e-5,
        rinertia=[1.0e-5, 1.0e-5, 1.0e-5],
        mol_id=int(mol_id),
    )
    site.virtual = True
    site.vs_auto_relate_to(parent.id)
    site.gamma = 0.0
    site.gamma_rot = 0.0
    return site


def create_runtime_system(params: dict[str, float]):
    import espressomd

    box_l = [10.0, 10.0, 10.0]
    system = espressomd.System(box_l=box_l)
    system.time_step = 1.0e-3
    system.cell_system.skin = 0.2
    system.force_cap = 0.0
    system.integrator.set_vv()
    system.thermostat.turn_off()

    # The physical sites are deliberately off-COM and not collinear with the
    # endpoint-endpoint force, so a correct back-transfer produces non-zero torque.
    com_i_pos = [2.5, 5.0, 5.0]
    com_j_pos = [5.0, 5.0, 5.0]
    site_i_pos = [2.5, 5.4, 5.0]
    site_j_pos = [5.0, 4.7, 5.0]

    num_species = 2
    com_type = num_species + 1
    com_i = system.part.add(
        pos=com_i_pos,
        type=com_type,
        mass=10.0,
        rinertia=[2.0, 2.0, 2.0],
        rotation=[True, True, True],
        mol_id=0,
    )
    com_j = system.part.add(
        pos=com_j_pos,
        type=com_type,
        mass=10.0,
        rinertia=[2.0, 2.0, 2.0],
        rotation=[True, True, True],
        mol_id=1,
    )
    site_i = _add_physical_virtual_site(
        system, parent=com_i, pos=site_i_pos, ptype=0, mol_id=0
    )
    site_j = _add_physical_virtual_site(
        system, parent=com_j, pos=site_j_pos, ptype=1, mol_id=1
    )

    priors = {
        "bonds": [
            {
                "type": "morse",
                "mol_i": 0,
                "site_i": 0,
                "mol_j": 1,
                "site_j": 0,
                **params,
            }
        ]
    }
    marker_types, contacts = prepare_pair_specific_morse(priors, num_species)
    marker_parts = create_pair_specific_morse_markers(
        system,
        marker_types,
        {0: com_i.id, 1: com_j.id},
        {(0, 0): site_i.id, (1, 0): site_j.id},
    )

    # Mirror the production architecture: physical sites are on the regular
    # side, whereas COMs and technical explicit-contact markers are N-square.
    configure_neighbor_search(
        system,
        "link-cell",
        n_square_types={com_type, *marker_types.values()},
        cutoff_regular=1.0,
    )
    configure_pair_specific_morse(system, contacts, marker_types)

    return system, {
        "com_i": com_i,
        "com_j": com_j,
        "site_i": site_i,
        "site_j": site_j,
        "marker_i": system.part.by_id(marker_parts[(0, 0)]),
        "marker_j": system.part.by_id(marker_parts[(1, 0)]),
        "marker_types": marker_types,
        "physical_site_types_before": [int(site_i.type), int(site_j.type)],
    }


def run_probe(params: dict[str, float]) -> dict[str, Any]:
    system, p = create_runtime_system(params)
    system.integrator.run(0, recalc_forces=True)

    actual = {
        "energy": float(system.analysis.energy()["non_bonded"]),
        "force_i": _vec(p["com_i"].f),
        "force_j": _vec(p["com_j"].f),
        "torque_i": _vec(p["com_i"].torque_lab),
        "torque_j": _vec(p["com_j"].torque_lab),
        "site_i_pos": _vec(p["site_i"].pos),
        "site_j_pos": _vec(p["site_j"].pos),
        "marker_i_pos": _vec(p["marker_i"].pos),
        "marker_j_pos": _vec(p["marker_j"].pos),
        "physical_site_types_after": [int(p["site_i"].type), int(p["site_j"].type)],
        "marker_runtime_types": [int(p["marker_i"].type), int(p["marker_j"].type)],
    }
    expected = expected_site_pair_wrench(
        com_i=p["com_i"].pos,
        site_i=p["marker_i"].pos,
        com_j=p["com_j"].pos,
        site_j=p["marker_j"].pos,
        box_l=system.box_l,
        params=params,
    )

    errors = {
        "energy_abs": abs(actual["energy"] - expected["energy"]),
        "force_i_max_abs": max_abs_error(actual["force_i"], expected["force_i"]),
        "force_j_max_abs": max_abs_error(actual["force_j"], expected["force_j"]),
        "torque_i_max_abs": max_abs_error(actual["torque_i"], expected["torque_i"]),
        "torque_j_max_abs": max_abs_error(actual["torque_j"], expected["torque_j"]),
        "marker_i_site_max_abs": max_abs_error(actual["marker_i_pos"], actual["site_i_pos"]),
        "marker_j_site_max_abs": max_abs_error(actual["marker_j_pos"], actual["site_j_pos"]),
        "net_force_max_abs": max_abs_error(_add(actual["force_i"], actual["force_j"]), [0.0, 0.0, 0.0]),
    }
    invariants = {
        "physical_site_types_unchanged": actual["physical_site_types_after"]
        == p["physical_site_types_before"],
        "marker_types_are_technical": min(actual["marker_runtime_types"]) >= 4,
        "analytic_torque_is_nonzero": _norm(expected["torque_i"]) > 1.0e-6
        and _norm(expected["torque_j"]) > 1.0e-6,
    }
    return {
        "diagnostic": "pair-specific Morse site-to-site rigid-body force/torque back-transfer",
        "parameters": params,
        "actual": actual,
        "expected": expected,
        "errors": errors,
        "invariants": invariants,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("morse_site_torque_report.json"),
    )
    parser.add_argument("--atol", type=float, default=1.0e-9)
    parser.add_argument("--assert-expected", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.atol <= 0.0:
        raise ValueError("--atol must be positive")
    report = run_probe(dict(DEFAULT_PARAMS))
    errors = report["errors"]
    invariants = report["invariants"]

    numeric_ok = all(float(value) <= args.atol for value in errors.values())
    invariant_ok = all(bool(value) for value in invariants.values())
    report["tolerance"] = args.atol
    report["pass"] = bool(numeric_ok and invariant_ok)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    exp = report["expected"]
    act = report["actual"]
    print("[PAIR-SPECIFIC MORSE SITE/TORQUE DIAGNOSTIC]")
    print(
        f"distance={exp['distance']:.12g}  energy actual={act['energy']:.12g} "
        f"expected={exp['energy']:.12g}  |dU|={errors['energy_abs']:.3e}"
    )
    print(
        "body 0: "
        f"|dF|max={errors['force_i_max_abs']:.3e}  "
        f"|dTau|max={errors['torque_i_max_abs']:.3e}  "
        f"expected_tau={exp['torque_i']}"
    )
    print(
        "body 1: "
        f"|dF|max={errors['force_j_max_abs']:.3e}  "
        f"|dTau|max={errors['torque_j_max_abs']:.3e}  "
        f"expected_tau={exp['torque_j']}"
    )
    print(
        "marker/site coincidence: "
        f"body0={errors['marker_i_site_max_abs']:.3e}  "
        f"body1={errors['marker_j_site_max_abs']:.3e}"
    )
    print(
        "physical site types unchanged: "
        f"{invariants['physical_site_types_unchanged']}  "
        f"marker types={act['marker_runtime_types']}"
    )
    print(f"net force max abs={errors['net_force_max_abs']:.3e}")
    print(f"report: {args.output}")

    if args.assert_expected:
        failed = [name for name, value in errors.items() if float(value) > args.atol]
        failed.extend(name for name, value in invariants.items() if not bool(value))
        if failed:
            raise RuntimeError(
                "Pair-specific Morse site/torque diagnostic failed: " + ", ".join(failed)
            )
        print(
            "[PASS] Site-addressable pair-specific Morse transfers the analytic "
            "force and torque to both rigid bodies."
        )


if __name__ == "__main__":
    main()
