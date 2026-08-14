#!/usr/bin/env python3
"""Black-box diagnostic for breakable Morse bonds in an ESPResSo build.

The MLCG analytic ``MorseBond`` returns an empty ``std::optional`` when
``r >= r_cut``.  ESPResSo uses empty optionals to signal a broken pair bond.
This script probes the *runtime* consequences without changing the force law:

* evaluate bonded force and energy below, at, and above ``r_cut``;
* record whether ESPResSo raises an exception;
* record whether the bond is still present in ``Particle.bonds`` afterwards;
* drive a two-particle system across ``r_cut`` and record where integration
  stops.

Run with the rebuilt ESPResSo ``pypresso`` executable, not plain CPython.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_MORSE_R_CUT_NM = 15.0


def morse_energy(distance: float, D: float, a: float, r0: float) -> float:
    exp_term = math.exp(-a * (distance - r0))
    return D * (1.0 - exp_term) ** 2


def morse_radial_force(distance: float, D: float, a: float, r0: float) -> float:
    """Return radial force coefficient F_r = -dU/dr."""
    exp_term = math.exp(-a * (distance - r0))
    return -2.0 * a * D * (1.0 - exp_term) * exp_term


def load_morse_prior(path: Path, morse_index: int) -> dict[str, float]:
    data = json.loads(path.read_text())
    morse = [b for b in data.get("bonds", []) if b.get("type") == "morse"]
    if not morse:
        raise RuntimeError(f"No Morse bonds found in {path}")
    if morse_index < 0 or morse_index >= len(morse):
        raise IndexError(
            f"Morse index {morse_index} out of range; file contains {len(morse)} Morse bonds"
        )
    prior = morse[morse_index]
    return {
        "D": float(prior["D"]),
        "a": float(prior["a"]),
        "r0": float(prior["r0"]),
        "r_cut": float(prior.get("r_cut", DEFAULT_MORSE_R_CUT_NM)),
    }


def exception_record(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def bonds_snapshot(particle: Any) -> dict[str, Any]:
    try:
        bonds = tuple(particle.bonds)
        return {
            "available": True,
            "count": len(bonds),
            "repr": [repr(bond) for bond in bonds],
        }
    except Exception as exc:  # diagnostic path: preserve the observation
        return {
            "available": False,
            "error": exception_record(exc),
        }


def configure_cells(system: Any) -> None:
    """Use deterministic link-cell traversal if supported by this ESPResSo."""
    setter = getattr(system.cell_system, "set_regular_decomposition", None)
    if setter is not None:
        try:
            setter(use_verlet_lists=False)
            return
        except TypeError:
            pass


def create_system(params: dict[str, float], max_distance: float):
    import espressomd
    from espressomd import interactions

    cls = getattr(interactions, "MorseBond", None)
    if cls is None:
        raise RuntimeError(
            "espressomd.interactions.MorseBond is unavailable. Install the MLCG "
            "analytic Morse extension and rebuild ESPResSo."
        )

    r_cut = params["r_cut"]
    # Keep every probe well inside half the box so minimum-image wrapping cannot
    # change the requested pair distance.  ESPResSo permits only one System per
    # process, therefore this box is created once and reused for all probes.
    box_l = max(10.0, 4.0 * max(r_cut, max_distance) + 4.0)
    system = espressomd.System(box_l=[box_l, box_l, box_l])
    system.time_step = 1.0e-4
    system.cell_system.skin = 0.1
    configure_cells(system)
    return system, cls


def reset_pair(system: Any, morse_cls: Any, params: dict[str, float], distance: float):
    """Reset the singleton ESPResSo System to one isolated Morse-bonded pair."""
    # Remove particles first because their bond records reference entries in
    # bonded_inter.  Clearing both containers makes each probe independent while
    # respecting ESPResSo's one-System-per-process restriction.
    system.part.clear()
    system.bonded_inter.clear()
    system.time = 0.0
    system.time_step = 1.0e-4

    box_l = float(system.box_l[0])
    center = 0.5 * box_l
    p0 = system.part.add(pos=[center - 0.5 * distance, center, center])
    p1 = system.part.add(pos=[center + 0.5 * distance, center, center])

    bond = morse_cls(
        D=params["D"], a=params["a"], r_0=params["r0"], r_cut=params["r_cut"]
    )
    system.bonded_inter.add(bond)
    p0.add_bond((bond, p1))
    return p0, p1, bond


def probe_force(
    system: Any, morse_cls: Any, params: dict[str, float], distance: float
) -> dict[str, Any]:
    p0, p1, _ = reset_pair(system, morse_cls, params, distance)
    result: dict[str, Any] = {
        "distance": distance,
        "bond_before": bonds_snapshot(p0),
    }
    try:
        system.integrator.run(0, recalc_forces=True)
        result.update(
            {
                "success": True,
                "p0_force": [float(x) for x in p0.f],
                "p1_force": [float(x) for x in p1.f],
            }
        )
    except Exception as exc:
        result.update({"success": False, "exception": exception_record(exc)})
    result["bond_after"] = bonds_snapshot(p0)
    return result


def probe_energy(
    system: Any, morse_cls: Any, params: dict[str, float], distance: float
) -> dict[str, Any]:
    p0, _p1, _ = reset_pair(system, morse_cls, params, distance)
    result: dict[str, Any] = {
        "distance": distance,
        "bond_before": bonds_snapshot(p0),
    }
    try:
        energies = system.analysis.energy()
        result.update(
            {
                "success": True,
                "bonded_energy": float(energies["bonded"]),
            }
        )
    except Exception as exc:
        result.update({"success": False, "exception": exception_record(exc)})
    result["bond_after"] = bonds_snapshot(p0)
    return result


def current_distance(p0: Any, p1: Any) -> float:
    dx = [float(p1.pos[i]) - float(p0.pos[i]) for i in range(3)]
    return math.sqrt(sum(x * x for x in dx))


def probe_dynamic_crossing(
    system: Any, morse_cls: Any, params: dict[str, float], *,
    margin: float, time_step: float, max_steps: int
) -> dict[str, Any]:
    r_cut = params["r_cut"]
    start = r_cut - margin
    if start <= 0.0:
        raise ValueError("dynamic margin must be smaller than r_cut")

    p0, p1, _ = reset_pair(system, morse_cls, params, start)
    system.time_step = time_step

    # Relative displacement per nominal step is 2*margin.  Thus the first
    # propagated step should cross r_cut even if the attractive Morse force
    # slightly reduces the separation rate.
    relative_speed = 2.0 * margin / time_step
    p0.v = [-0.5 * relative_speed, 0.0, 0.0]
    p1.v = [+0.5 * relative_speed, 0.0, 0.0]

    result: dict[str, Any] = {
        "start_distance": start,
        "r_cut": r_cut,
        "margin": margin,
        "time_step": time_step,
        "relative_speed": relative_speed,
        "max_steps": max_steps,
        "bond_before": bonds_snapshot(p0),
        "steps_completed": 0,
        "raised": False,
    }

    try:
        # Establish a valid initial force while still below cutoff.
        system.integrator.run(0, recalc_forces=True)
        for step in range(1, max_steps + 1):
            system.integrator.run(1)
            result["steps_completed"] = step
            result["last_success_distance"] = current_distance(p0, p1)
    except Exception as exc:
        result["raised"] = True
        result["exception"] = exception_record(exc)
        result["distance_at_exception"] = current_distance(p0, p1)
        result["system_time_at_exception"] = float(system.time)

    result["bond_after"] = bonds_snapshot(p0)
    return result


def bond_count_unchanged(probe: dict[str, Any]) -> bool | None:
    before = probe.get("bond_before", {})
    after = probe.get("bond_after", {})
    if not before.get("available") or not after.get("available"):
        return None
    return before.get("count") == after.get("count")


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    probes = report["static_probes"]
    below = probes["below_cutoff"]
    at = probes["at_cutoff"]
    above = probes["above_cutoff"]
    dynamic = report["dynamic_crossing"]

    return {
        "below_cutoff_force_succeeds": bool(below["force"].get("success")),
        "below_cutoff_energy_succeeds": bool(below["energy"].get("success")),
        "at_cutoff_force_reports_broken": not bool(at["force"].get("success")),
        "at_cutoff_energy_reports_broken": not bool(at["energy"].get("success")),
        "above_cutoff_force_reports_broken": not bool(above["force"].get("success")),
        "above_cutoff_energy_reports_broken": not bool(above["energy"].get("success")),
        "bond_topology_unchanged_after_at_cutoff_force_error": bond_count_unchanged(at["force"]),
        "bond_topology_unchanged_after_above_cutoff_force_error": bond_count_unchanged(above["force"]),
        "dynamic_crossing_stops_integrator": bool(dynamic.get("raised")),
        "bond_topology_unchanged_after_dynamic_error": bond_count_unchanged(dynamic),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priors", type=Path, help="JSON priors file containing Morse bonds")
    parser.add_argument(
        "--morse-index", type=int, default=0,
        help="zero-based index among Morse bonds in --priors (default: 0)",
    )
    parser.add_argument("--D", type=float)
    parser.add_argument("--a", type=float)
    parser.add_argument("--r0", type=float)
    parser.add_argument("--r-cut", dest="r_cut", type=float)
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="distance offset used for below/above-cutoff probes",
    )
    parser.add_argument("--dynamic-dt", type=float, default=1.0e-4)
    parser.add_argument("--dynamic-steps", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("morse_breakage_report.json"))
    parser.add_argument(
        "--assert-expected", action="store_true",
        help="exit non-zero unless runtime behavior matches ESPResSo breakable-bond semantics",
    )
    return parser.parse_args()


def resolve_params(args: argparse.Namespace) -> dict[str, float]:
    if args.priors is not None:
        params = load_morse_prior(args.priors, args.morse_index)
    else:
        missing = [name for name in ("D", "a", "r0", "r_cut") if getattr(args, name) is None]
        if missing:
            raise ValueError(
                "Provide --priors or all explicit parameters --D --a --r0 --r-cut; "
                f"missing: {', '.join(missing)}"
            )
        params = {name: float(getattr(args, name)) for name in ("D", "a", "r0", "r_cut")}

    if params["D"] < 0.0 or params["a"] <= 0.0 or params["r0"] < 0.0 or params["r_cut"] <= 0.0:
        raise ValueError(f"Invalid Morse parameters: {params}")
    return params


def main() -> None:
    args = parse_args()
    params = resolve_params(args)
    r_cut = params["r_cut"]
    epsilon = args.epsilon
    if epsilon is None:
        epsilon = max(1.0e-6, 1.0e-6 * r_cut)
    if epsilon <= 0.0 or epsilon >= r_cut:
        raise ValueError("epsilon must satisfy 0 < epsilon < r_cut")

    distances = {
        "equilibrium": params["r0"],
        "below_cutoff": r_cut - epsilon,
        "at_cutoff": r_cut,
        "above_cutoff": r_cut + epsilon,
    }

    margin = max(10.0 * epsilon, 1.0e-5 * r_cut)
    max_distance = max(max(distances.values()), r_cut + 3.0 * margin)
    system, morse_cls = create_system(params, max_distance)

    static_probes: dict[str, Any] = {}
    for label, distance in distances.items():
        static_probes[label] = {
            "distance": distance,
            "analytic_untruncated_energy": morse_energy(distance, params["D"], params["a"], params["r0"]),
            "analytic_untruncated_radial_force": morse_radial_force(distance, params["D"], params["a"], params["r0"]),
            "force": probe_force(system, morse_cls, params, distance),
            "energy": probe_energy(system, morse_cls, params, distance),
        }

    dynamic = probe_dynamic_crossing(
        system, morse_cls, params,
        margin=margin,
        time_step=args.dynamic_dt,
        max_steps=args.dynamic_steps,
    )

    report: dict[str, Any] = {
        "diagnostic": "ESPResSo breakable MorseBond cutoff behavior",
        "morse_parameters": params,
        "epsilon": epsilon,
        "cutoff_condition_in_mlcg_kernel": "r >= r_cut returns empty std::optional",
        "static_probes": static_probes,
        "dynamic_crossing": dynamic,
    }
    report["summary"] = summarize(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[MORSE BREAKAGE DIAGNOSTIC]")
    print(
        f"params: D={params['D']:.12g} a={params['a']:.12g} "
        f"r0={params['r0']:.12g} r_cut={params['r_cut']:.12g}"
    )
    print("label           distance       force_eval   energy_eval   bond_after_force")
    for label in ("equilibrium", "below_cutoff", "at_cutoff", "above_cutoff"):
        item = static_probes[label]
        fp = item["force"]
        ep = item["energy"]
        unchanged = bond_count_unchanged(fp)
        print(
            f"{label:15s} {item['distance']:12.6g} "
            f"{'OK' if fp.get('success') else 'ERROR':>12s} "
            f"{'OK' if ep.get('success') else 'ERROR':>13s} "
            f"{str(unchanged):>18s}"
        )
        if not fp.get("success"):
            err = fp.get("exception", {})
            print(f"  force exception: {err.get('type')}: {err.get('message')}")
        if not ep.get("success"):
            err = ep.get("exception", {})
            print(f"  energy exception: {err.get('type')}: {err.get('message')}")

    print(
        "dynamic crossing: "
        f"{'STOPPED WITH ERROR' if dynamic.get('raised') else 'NO ERROR'}; "
        f"bond topology unchanged={bond_count_unchanged(dynamic)}"
    )
    if dynamic.get("raised"):
        print(
            f"  distance_at_exception={dynamic.get('distance_at_exception'):.12g} "
            f"time={dynamic.get('system_time_at_exception'):.12g}"
        )
        err = dynamic.get("exception", {})
        print(f"  exception: {err.get('type')}: {err.get('message')}")
    print(f"report: {args.output}")

    if args.assert_expected:
        summary = report["summary"]
        required_true = [
            "below_cutoff_force_succeeds",
            "below_cutoff_energy_succeeds",
            "at_cutoff_force_reports_broken",
            "above_cutoff_force_reports_broken",
            "dynamic_crossing_stops_integrator",
            "bond_topology_unchanged_after_at_cutoff_force_error",
            "bond_topology_unchanged_after_above_cutoff_force_error",
            "bond_topology_unchanged_after_dynamic_error",
        ]
        failed = [name for name in required_true if summary.get(name) is not True]
        if failed:
            raise RuntimeError(
                "Observed Morse breakage behavior differs from expected ESPResSo semantics: "
                + ", ".join(failed)
            )
        print("[PASS] Observed runtime behavior matches expected breakable-bond semantics.")


if __name__ == "__main__":
    main()
