#!/usr/bin/env python3
"""Black-box diagnostic for the reversible switched non-bonded Morse prior.

Run with ESPResSo's pypresso after installing/rebuilding the MLCG switched
non-bonded Morse extension.  The test is isolated from PaiNN, WCA, bonded
priors, and rigid bodies: two particles of distinct types interact only via one
Morse type pair.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from espresso_interactions import (
    DEFAULT_MORSE_R_CUT_NM,
    DEFAULT_MORSE_SWITCH_FRACTION,
    switched_morse_energy_radial_force,
)


def load_morse_prior(path: Path, morse_index: int) -> dict[str, float]:
    data = json.loads(path.read_text())
    morse = [b for b in data.get("bonds", []) if str(b.get("type", "")).lower() == "morse"]
    if not morse:
        raise RuntimeError(f"No Morse priors found in {path}")
    if not (0 <= morse_index < len(morse)):
        raise IndexError(
            f"Morse index {morse_index} out of range; file contains {len(morse)} Morse priors"
        )
    prior = morse[morse_index]
    D = float(prior["D"])
    a = float(prior["a"])
    r0 = float(prior["r0"])
    r_cut = float(prior.get("r_cut", DEFAULT_MORSE_R_CUT_NM))
    r_switch = float(
        prior.get("r_switch", r0 + DEFAULT_MORSE_SWITCH_FRACTION * (r_cut - r0))
    )
    if not (D >= 0.0 and a > 0.0 and r0 >= 0.0 and r0 < r_switch < r_cut):
        raise ValueError(
            f"Invalid switched Morse parameters: D={D}, a={a}, r0={r0}, "
            f"r_switch={r_switch}, r_cut={r_cut}"
        )
    return {"D": D, "a": a, "r0": r0, "r_switch": r_switch, "r_cut": r_cut}


def exception_record(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def create_system(params: dict[str, float], max_distance: float):
    import espressomd

    box_l = max(10.0, 4.0 * max(max_distance, params["r_cut"]) + 4.0)
    system = espressomd.System(box_l=[box_l, box_l, box_l])
    system.time_step = 1.0e-4
    system.cell_system.skin = 0.1
    system.cell_system.set_n_square(use_verlet_lists=False)

    center = 0.5 * box_l
    p0 = system.part.add(pos=[center, center, center], type=0, mass=1.0)
    p1 = system.part.add(pos=[center + params["r0"], center, center], type=1, mass=1.0)
    if not espressomd.has_features("MORSE"):
        raise RuntimeError(
            "This ESPResSo binary was compiled without the MORSE feature, so the "
            "non-bonded '.morse' handle does not exist. Run "
            "simulation/espresso_plugin/copy_plugin_files.sh, re-run CMake, and rebuild "
            "ESPResSo before this diagnostic."
        )

    pair_handle = system.non_bonded_inter[0, 1]
    if not hasattr(pair_handle, "morse"):
        raise RuntimeError(
            "ESPResSo reports MORSE as compiled, but NonBondedInteractionHandle has no "
            "'.morse' attribute. Re-run CMake and rebuild the Python interface."
        )
    try:
        pair_handle.morse.set_params(
            eps=params["D"],
            alpha=params["a"],
            rmin=params["r0"],
            cutoff=params["r_cut"],
            switch_start=params["r_switch"],
        )
    except Exception as exc:
        raise RuntimeError(
            "The MORSE feature is compiled, but the MLCG 'switch_start' extension is "
            "not available in this binary. Run simulation/espresso_plugin/copy_plugin_files.sh, "
            "re-run CMake, and rebuild ESPResSo."
        ) from exc
    return system, p0, p1


def set_distance(system: Any, p0: Any, p1: Any, distance: float) -> None:
    center = 0.5 * float(system.box_l[0])
    p0.pos = [center - 0.5 * distance, center, center]
    p1.pos = [center + 0.5 * distance, center, center]
    p0.v = [0.0, 0.0, 0.0]
    p1.v = [0.0, 0.0, 0.0]
    system.time = 0.0


def current_distance(p0: Any, p1: Any) -> float:
    dx = [float(p1.pos[i]) - float(p0.pos[i]) for i in range(3)]
    return math.sqrt(sum(value * value for value in dx))


def radial_force_from_p0(p0: Any) -> float:
    # p1 is always to +x of p0 in the static probes.  The force on p0 points
    # with the signed radial convention used by the pair kernel.
    return -float(p0.f[0])


def probe_static(system: Any, p0: Any, p1: Any, params: dict[str, float], distance: float):
    set_distance(system, p0, p1, distance)
    result: dict[str, Any] = {"distance": distance}
    analytic_energy, analytic_force = switched_morse_energy_radial_force(
        distance,
        D=params["D"],
        a=params["a"],
        r0=params["r0"],
        r_switch=params["r_switch"],
        r_cut=params["r_cut"],
    )
    result["analytic_energy"] = analytic_energy
    result["analytic_radial_force"] = analytic_force
    try:
        system.integrator.run(0, recalc_forces=True)
        energies = system.analysis.energy()
        result.update(
            {
                "success": True,
                "p0_force": [float(x) for x in p0.f],
                "p1_force": [float(x) for x in p1.f],
                "radial_force": radial_force_from_p0(p0),
                "non_bonded_energy": float(energies["non_bonded"]),
            }
        )
    except Exception as exc:
        result.update({"success": False, "exception": exception_record(exc)})
    return result


def probe_dynamic_roundtrip(
    system: Any,
    p0: Any,
    p1: Any,
    params: dict[str, float],
    *,
    margin: float,
    time_step: float,
) -> dict[str, Any]:
    start = params["r_cut"] - margin
    set_distance(system, p0, p1, start)
    system.time_step = time_step

    # Make the ballistic displacement dominate the tiny switched-tail force.
    relative_speed = 2.5 * margin / time_step
    p0.v = [-0.5 * relative_speed, 0.0, 0.0]
    p1.v = [+0.5 * relative_speed, 0.0, 0.0]

    result: dict[str, Any] = {
        "start_distance": start,
        "relative_speed": relative_speed,
        "time_step": time_step,
        "raised": False,
    }
    try:
        system.integrator.run(0, recalc_forces=True)
        result["initial_force_norm"] = math.sqrt(sum(float(x) ** 2 for x in p0.f))
        system.integrator.run(2)
        result["outside_distance"] = current_distance(p0, p1)
        system.integrator.run(0, recalc_forces=True)
        result["outside_force_norm"] = math.sqrt(sum(float(x) ** 2 for x in p0.f))
        result["outside_energy"] = float(system.analysis.energy()["non_bonded"])

        # Reverse the velocities. The pair must be able to re-enter the active
        # region because no topology was deleted on the outward crossing.
        p0.v = [+0.5 * relative_speed, 0.0, 0.0]
        p1.v = [-0.5 * relative_speed, 0.0, 0.0]
        system.integrator.run(3)
        result["reentry_distance"] = current_distance(p0, p1)
        system.integrator.run(0, recalc_forces=True)
        result["reentry_force_norm"] = math.sqrt(sum(float(x) ** 2 for x in p0.f))
        result["reentry_energy"] = float(system.analysis.energy()["non_bonded"])
    except Exception as exc:
        result["raised"] = True
        result["exception"] = exception_record(exc)
        result["distance_at_exception"] = current_distance(p0, p1)
        result["time_at_exception"] = float(system.time)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--morse-index", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--dynamic-dt", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path, default=Path("morse_reversibility_report.json"))
    parser.add_argument("--assert-expected", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_morse_prior(args.priors, args.morse_index)
    epsilon = args.epsilon
    if epsilon is None:
        epsilon = max(1.0e-6, 1.0e-6 * params["r_cut"])
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    midpoint = 0.5 * (params["r_switch"] + params["r_cut"])
    distances = {
        "equilibrium": params["r0"],
        "switch_start": params["r_switch"],
        "switch_midpoint": midpoint,
        "below_cutoff": params["r_cut"] - epsilon,
        "at_cutoff": params["r_cut"],
        "above_cutoff": params["r_cut"] + epsilon,
    }
    margin = max(20.0 * epsilon, 2.0e-5 * params["r_cut"])
    system, p0, p1 = create_system(params, max(distances.values()) + 10.0 * margin)

    probes = {
        label: probe_static(system, p0, p1, params, distance)
        for label, distance in distances.items()
    }
    dynamic = probe_dynamic_roundtrip(
        system, p0, p1, params, margin=margin, time_step=args.dynamic_dt
    )

    max_energy_error = max(
        abs(item["non_bonded_energy"] - item["analytic_energy"])
        for item in probes.values()
        if item.get("success")
    )
    max_force_error = max(
        abs(item["radial_force"] - item["analytic_radial_force"])
        for item in probes.values()
        if item.get("success")
    )
    summary = {
        "all_static_probes_succeed": all(item.get("success") for item in probes.values()),
        "energy_zero_at_cutoff": abs(probes["at_cutoff"].get("non_bonded_energy", math.inf)) < 1.0e-12,
        "force_zero_at_cutoff": abs(probes["at_cutoff"].get("radial_force", math.inf)) < 1.0e-12,
        "energy_zero_above_cutoff": abs(probes["above_cutoff"].get("non_bonded_energy", math.inf)) < 1.0e-12,
        "force_zero_above_cutoff": abs(probes["above_cutoff"].get("radial_force", math.inf)) < 1.0e-12,
        "dynamic_crossing_succeeds": not dynamic.get("raised", True),
        "outside_force_zero": abs(dynamic.get("outside_force_norm", math.inf)) < 1.0e-12,
        "reentry_force_restored": dynamic.get("reentry_force_norm", 0.0) > 1.0e-10,
        "max_energy_abs_error": max_energy_error,
        "max_radial_force_abs_error": max_force_error,
    }
    report = {
        "diagnostic": "MLCG reversible switched non-bonded Morse",
        "morse_parameters": params,
        "epsilon": epsilon,
        "static_probes": probes,
        "dynamic_roundtrip": dynamic,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[MORSE REVERSIBILITY DIAGNOSTIC]")
    print(
        f"params: D={params['D']:.12g} a={params['a']:.12g} r0={params['r0']:.12g} "
        f"r_switch={params['r_switch']:.12g} r_cut={params['r_cut']:.12g}"
    )
    print("label              distance      status        energy          F_radial")
    for label, item in probes.items():
        if item.get("success"):
            print(
                f"{label:18s} {item['distance']:11.6g} {'OK':>10s} "
                f"{item['non_bonded_energy']:14.7g} {item['radial_force']:14.7g}"
            )
        else:
            err = item.get("exception", {})
            print(f"{label:18s} {item['distance']:11.6g} {'ERROR':>10s} {err.get('message')}")
    print(
        "dynamic roundtrip: "
        + ("OK" if not dynamic.get("raised") else "ERROR")
        + f"; outside_r={dynamic.get('outside_distance')} reentry_r={dynamic.get('reentry_distance')}"
    )
    print(
        f"analytic agreement: max |dU|={max_energy_error:.3e}, "
        f"max |dF|={max_force_error:.3e}"
    )
    print(f"report: {args.output}")

    if args.assert_expected:
        required = [
            "all_static_probes_succeed",
            "energy_zero_at_cutoff",
            "force_zero_at_cutoff",
            "energy_zero_above_cutoff",
            "force_zero_above_cutoff",
            "dynamic_crossing_succeeds",
            "outside_force_zero",
            "reentry_force_restored",
        ]
        failed = [name for name in required if summary.get(name) is not True]
        if max_energy_error > 1.0e-9:
            failed.append("analytic_energy_agreement")
        if max_force_error > 1.0e-9:
            failed.append("analytic_force_agreement")
        if failed:
            raise RuntimeError("Reversible Morse diagnostic failed: " + ", ".join(failed))
        print("[PASS] Morse crosses the cutoff and reforms without topology events.")


if __name__ == "__main__":
    main()
