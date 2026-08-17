#!/usr/bin/env python3
"""Minimal ESPResSo dynamics diagnostics for conservative IBI splines.

This script localizes energy-scaling failures with deliberately small systems:

* a two-particle distance spline inside one Hermite cell and across a knot;
* a three-point angle spline inside one cell and across a knot;
* the same angle carried by rigid-body virtual sites, including torque finite differences;
* forward/reverse time-reversibility checks.

The selected spline tables come from the real conservative IBI priors.  Run with
ESPResSo's ``pypresso`` after installing the conservative spline plugin.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import espressomd
import espressomd.interactions
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "simulation"))

from conservative_spline import (  # noqa: E402
    ConservativeSplinePrior,
    conservative_spline_value,
    load_conservative_spline,
    save_conservative_spline,
)
from conservative_spline_runtime import create_conservative_spline_interaction  # noqa: E402
from conservative_ibi_energy_diagnostics import curvature_jumps, unique_conservative_entries  # noqa: E402
from nve_analysis import analyze_energy_series, fit_metric_scaling  # noqa: E402


MASS = 5000.0
INERTIA = 500.0
BOX = np.asarray([30.0, 30.0, 30.0], dtype=float)
CENTER = np.asarray([15.0, 15.0, 15.0], dtype=float)


_SYSTEM_SINGLETON = None


def _system(dt: float):
    """Return the one ESPResSo System instance, reset for a fresh micro-run.

    ESPResSo permits only one ``System`` instance in a Python process.  The
    localization suite executes many independent micro-trajectories in one
    pypresso process, so creating a new System for every dt/case is invalid.
    Reuse the singleton and clear all state that these micro-tests install.
    """
    global _SYSTEM_SINGLETON

    if _SYSTEM_SINGLETON is None:
        _SYSTEM_SINGLETON = espressomd.System(box_l=BOX.tolist())
    system = _SYSTEM_SINGLETON

    # Clear the previous micro-system before installing the next independent
    # case.  Particle removal must precede bonded-interaction removal because
    # particles may still reference the interaction objects through bonds.
    system.thermostat.turn_off()
    system.part.clear()
    system.bonded_inter.clear()
    system.time = 0.0

    system.box_l = BOX.tolist()
    system.time_step = float(dt)
    system.cell_system.skin = 0.2
    system.cell_system.set_n_square(use_verlet_lists=False)
    system.force_cap = 0.0
    system.integrator.set_vv()
    return system


def _energy(system) -> float:
    return float(system.analysis.energy()["total"])


def _fit(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return fit_metric_scaling(runs, "sigma_E", label="sigma_E")


def _choose_entry(priors_path: Path, kind: str):
    priors = json.loads(priors_path.read_text())
    candidates = []
    for _key, _idx, entry_kind, entry in unique_conservative_entries(priors):
        if entry_kind != kind:
            continue
        table = load_conservative_spline(entry, kind=kind, priors_path=priors_path)
        smooth = curvature_jumps(table)
        energies = np.asarray(table.energy, dtype=float)
        e_cut = float(np.quantile(energies, 0.50))
        eligible = [
            row for row in smooth["knots"]
            if table.energy[row["knot_index"]] <= e_cut
            and 2 <= row["knot_index"] <= len(table.x) - 3
        ]
        if not eligible:
            eligible = [row for row in smooth["knots"] if 2 <= row["knot_index"] <= len(table.x) - 3]
        if not eligible:
            continue
        knot = max(eligible, key=lambda row: row["abs_jump"])
        candidates.append((float(knot["abs_jump"]), entry, table, knot))
    if not candidates:
        raise ValueError(f"No usable conservative {kind} spline found in {priors_path}")
    _score, entry, table, knot = max(candidates, key=lambda item: item[0])
    return entry, table, knot


def _bond_local_entry(table: ConservativeSplinePrior, knot_index: int, root: Path, *, crossing: bool):
    if crossing:
        indices = np.asarray([knot_index - 1, knot_index, knot_index + 1], dtype=int)
    else:
        # Use the exact left Hermite segment as a one-cell table.  This removes
        # all internal knots while preserving the polynomial coefficients on
        # that segment exactly.
        indices = np.asarray([knot_index - 1, knot_index], dtype=int)
    x = table.x[indices]
    u = table.energy[indices]
    du = table.derivative[indices]
    path = root / ("bond_crossing.dat" if crossing else "bond_one_cell.dat")
    save_conservative_spline(path, x, u, du)
    return {
        "type": "conservative_spline",
        "spline_schema": "pchip_hermite_v1",
        "file": path.name,
        "min": float(x[0]),
        "max": float(x[-1]),
    }, x


def _bond_builder(entry, priors_path: Path, *, q0: float, relative_speed: float):
    def build(dt: float):
        system = _system(dt)
        ia = create_conservative_spline_interaction(
            espressomd.interactions, entry, kind="bond", priors_path=priors_path
        )
        system.bonded_inter.add(ia)
        p0 = system.part.add(pos=CENTER - [q0 / 2.0, 0.0, 0.0], type=0, mass=MASS)
        p1 = system.part.add(pos=CENTER + [q0 / 2.0, 0.0, 0.0], type=0, mass=MASS)
        p0.v = [-0.5 * relative_speed, 0.0, 0.0]
        p1.v = [0.5 * relative_speed, 0.0, 0.0]
        p0.add_bond((ia, p1.id))
        return system, [p0, p1], lambda: float(system.distance(p0, p1))
    return build


def _angle_positions(theta: float) -> np.ndarray:
    return np.asarray([
        CENTER + [1.0, 0.0, 0.0],
        CENTER,
        CENTER + [math.cos(theta), math.sin(theta), 0.0],
    ], dtype=float)


def _point_angle_builder(entry, priors_path: Path, *, theta0: float, angular_speed: float):
    def build(dt: float):
        system = _system(dt)
        ia = create_conservative_spline_interaction(
            espressomd.interactions, entry, kind="angle", priors_path=priors_path
        )
        system.bonded_inter.add(ia)
        pos = _angle_positions(theta0)
        particles = [system.part.add(pos=p, type=0, mass=MASS) for p in pos]
        # Tangential velocity of the third endpoint changes theta to first order.
        tangent = np.asarray([-math.sin(theta0), math.cos(theta0), 0.0])
        particles[2].v = angular_speed * tangent
        particles[1].add_bond((ia, particles[0].id, particles[2].id))
        def coordinate():
            a = np.asarray(particles[0].pos) - np.asarray(particles[1].pos)
            b = np.asarray(particles[2].pos) - np.asarray(particles[1].pos)
            return float(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))
        return system, particles, coordinate
    return build


def _add_virtual_site(system, parent, position, ptype):
    site = system.part.add(
        pos=position, type=int(ptype), mass=1.0e-5,
        rinertia=[1.0e-5, 1.0e-5, 1.0e-5], mol_id=int(parent.mol_id),
    )
    site.virtual = True
    site.vs_auto_relate_to(parent.id)
    site.gamma = 0.0
    site.gamma_rot = 0.0
    return site


def _rigid_angle_builder(entry, priors_path: Path, *, theta0: float, angular_speed: float):
    def build(dt: float):
        system = _system(dt)
        ia = create_conservative_spline_interaction(
            espressomd.interactions, entry, kind="angle", priors_path=priors_path
        )
        system.bonded_inter.add(ia)
        site_pos = _angle_positions(theta0)
        offsets = np.asarray([[0.0, 0.27, 0.11], [0.13, -0.21, 0.09], [-0.18, 0.24, -0.08]])
        bodies = []
        sites = []
        for i in range(3):
            com_pos = site_pos[i] - offsets[i]
            body = system.part.add(
                pos=com_pos, type=10 + i, mass=MASS,
                rinertia=[INERTIA, 1.1 * INERTIA, 0.9 * INERTIA],
                rotation=[True, True, True], mol_id=i,
            )
            bodies.append(body)
            sites.append(_add_virtual_site(system, body, site_pos[i], i))
        tangent = np.asarray([-math.sin(theta0), math.cos(theta0), 0.0])
        bodies[2].v = angular_speed * tangent
        sites[1].add_bond((ia, sites[0].id, sites[2].id))
        def coordinate():
            a = np.asarray(sites[0].pos) - np.asarray(sites[1].pos)
            b = np.asarray(sites[2].pos) - np.asarray(sites[1].pos)
            return float(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))
        return system, bodies, coordinate
    return build


def _run_trace(builder: Callable[[float], tuple[Any, list[Any], Callable[[], float]]], dt: float, duration: float):
    system, real_particles, coordinate = builder(dt)
    nsteps = max(4, int(round(duration / dt)))
    system.integrator.run(0, recalc_forces=True)
    times = [0.0]
    energies = [_energy(system)]
    coords = [coordinate()]
    for step in range(1, nsteps + 1):
        system.integrator.run(1)
        times.append(step * dt)
        energies.append(_energy(system))
        coords.append(coordinate())
    metrics = analyze_energy_series(times, energies)
    metrics.update({"dt_ps": float(dt), "steps": int(nsteps), "q_min": float(np.min(coords)), "q_max": float(np.max(coords))})
    return metrics, np.asarray(coords, dtype=float)


def _reversibility(builder, dt: float, duration: float) -> dict[str, float]:
    system, particles, _coordinate = builder(dt)
    nsteps = max(2, int(round(duration / dt)))
    system.integrator.run(0, recalc_forces=True)
    initial = {
        int(p.id): {
            "pos": np.asarray(p.pos, dtype=float).copy(),
            "v": np.asarray(p.v, dtype=float).copy(),
            "quat": np.asarray(p.quat, dtype=float).copy(),
            "omega": np.asarray(getattr(p, "omega_body", [0.0, 0.0, 0.0]), dtype=float).copy(),
        }
        for p in particles
    }
    system.integrator.run(nsteps)
    for p in particles:
        p.v = -np.asarray(p.v, dtype=float)
        if any(bool(v) for v in p.rotation):
            p.omega_body = -np.asarray(p.omega_body, dtype=float)
    system.integrator.run(nsteps)
    pos_err = []
    vel_err = []
    orient_err = []
    omega_err = []
    for p in particles:
        ref = initial[int(p.id)]
        d = np.asarray(p.pos, dtype=float) - ref["pos"]
        d -= BOX * np.round(d / BOX)
        pos_err.extend(d.tolist())
        vel_err.extend((np.asarray(p.v, dtype=float) + ref["v"]).tolist())
        qa = ref["quat"] / np.linalg.norm(ref["quat"])
        qb = np.asarray(p.quat, dtype=float).copy()
        qb /= np.linalg.norm(qb)
        orient_err.append(2.0 * math.acos(float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))))
        omega_err.extend((np.asarray(p.omega_body, dtype=float) + ref["omega"]).tolist())
    return {
        "dt_ps": float(dt), "duration_each_way_ps": float(nsteps * dt),
        "position_rms_nm": float(np.sqrt(np.mean(np.square(pos_err)))),
        "velocity_rms_nm_per_ps": float(np.sqrt(np.mean(np.square(vel_err)))),
        "orientation_rms_rad": float(np.sqrt(np.mean(np.square(orient_err)))),
        "omega_body_rms_per_ps": float(np.sqrt(np.mean(np.square(omega_err)))),
    }


def _rigid_torque_fd(builder, dt: float, eps: float) -> dict[str, Any]:
    system, bodies, _coordinate = builder(dt)
    system.integrator.run(0, recalc_forces=True)
    # Pick the body with the largest torque norm so the signal is not dominated by roundoff.
    body = max(bodies, key=lambda p: float(np.linalg.norm(p.torque_lab)))
    base_quat = np.asarray(body.quat, dtype=float).copy()
    actual = np.asarray(body.torque_lab, dtype=float).copy()
    rows = []
    for axis_index, axis in enumerate(np.eye(3)):
        body.quat = base_quat
        body.rotate(axis, float(eps))
        system.integrator.run(0, recalc_forces=True)
        plus = _energy(system)
        body.quat = base_quat
        body.rotate(axis, -float(eps))
        system.integrator.run(0, recalc_forces=True)
        minus = _energy(system)
        body.quat = base_quat
        system.integrator.run(0, recalc_forces=True)
        fd = -(plus - minus) / (2.0 * eps)
        rows.append({
            "axis": int(axis_index), "actual_torque_lab": float(actual[axis_index]),
            "fd_torque": float(fd), "abs_error": float(abs(fd - actual[axis_index])),
        })
    return {
        "particle_id": int(body.id),
        "eps_rad": float(eps),
        "torque_norm": float(np.linalg.norm(actual)),
        "max_abs_error": max(float(row["abs_error"]) for row in rows),
        "axes": rows,
    }


def _run_case(name: str, builder, dts: list[float], duration: float, table: ConservativeSplinePrior):
    runs = []
    coordinate_traces = []
    h = float(table.x[1] - table.x[0])
    for dt in dts:
        metrics, coords = _run_trace(builder, dt, duration)
        cells = np.floor((coords - table.minimum) / h).astype(int)
        crossings = int(np.sum(np.abs(np.diff(cells))))
        metrics["cell_crossings"] = crossings
        runs.append(metrics)
        coordinate_traces.append({"dt_ps": float(dt), "cell_crossings": crossings})
    fit = _fit(runs)
    return {"name": name, "runs": runs, "fit": fit, "crossings": coordinate_traces}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration-ps", type=float, default=0.096)
    parser.add_argument("--dts", nargs="+", type=float, default=[0.001, 0.0005, 0.00025, 0.000125])
    parser.add_argument("--reversibility-dt", type=float, default=0.0005)
    parser.add_argument("--reversibility-duration-ps", type=float, default=0.024)
    parser.add_argument("--torque-fd-eps", type=float, default=1.0e-6)
    args = parser.parse_args()

    priors_path = args.priors.expanduser().resolve()
    if args.duration_ps <= 0.0 or min(args.dts) <= 0.0:
        raise ValueError("durations and time steps must be positive")
    for cls in ("ConservativeSplineDistance", "ConservativeSplineAngle"):
        if not hasattr(espressomd.interactions, cls):
            raise RuntimeError(f"Missing espressomd.interactions.{cls}; rebuild the conservative spline plugin first")

    bond_entry, bond_table, bond_knot = _choose_entry(priors_path, "bond")
    angle_entry, angle_table, angle_knot = _choose_entry(priors_path, "angle")
    bond_k = int(bond_knot["knot_index"])
    angle_k = int(angle_knot["knot_index"])
    hb = float(bond_table.x[1] - bond_table.x[0])
    ha = float(angle_table.x[1] - angle_table.x[0])

    with tempfile.TemporaryDirectory(prefix="mlcg_spline_dynamics_") as tmpdir:
        tmp = Path(tmpdir)
        dummy_priors = tmp / "cg_priors.json"
        dummy_priors.write_text("{}\n")
        bond_inside_entry, inside_nodes = _bond_local_entry(bond_table, bond_k, tmp, crossing=False)
        bond_cross_entry, cross_nodes = _bond_local_entry(bond_table, bond_k, tmp, crossing=True)

        # Small imposed coordinate speeds; large masses keep the microdynamics
        # local while the crossing case is deliberately aimed through one knot.
        bond_inside_q = float(np.mean(inside_nodes))
        bond_inside_speed = 0.05 * hb / args.duration_ps
        bond_cross_q = float(bond_table.x[bond_k] - 0.20 * hb)
        bond_cross_speed = 0.55 * hb / args.duration_ps
        bond_inside_builder = _bond_builder(bond_inside_entry, dummy_priors, q0=bond_inside_q, relative_speed=bond_inside_speed)
        bond_cross_builder = _bond_builder(bond_cross_entry, dummy_priors, q0=bond_cross_q, relative_speed=bond_cross_speed)

        angle_inside_q = float(angle_table.x[angle_k] - 0.55 * ha)
        angle_cross_q = float(angle_table.x[angle_k] - 0.20 * ha)
        angle_inside_speed = 0.05 * ha / args.duration_ps
        angle_cross_speed = 0.55 * ha / args.duration_ps
        point_inside_builder = _point_angle_builder(angle_entry, priors_path, theta0=angle_inside_q, angular_speed=angle_inside_speed)
        point_cross_builder = _point_angle_builder(angle_entry, priors_path, theta0=angle_cross_q, angular_speed=angle_cross_speed)
        rigid_inside_builder = _rigid_angle_builder(angle_entry, priors_path, theta0=angle_inside_q, angular_speed=angle_inside_speed)
        rigid_cross_builder = _rigid_angle_builder(angle_entry, priors_path, theta0=angle_cross_q, angular_speed=angle_cross_speed)

        cases = {
            "bond_inside_single_hermite_cell": _run_case("bond_inside_single_hermite_cell", bond_inside_builder, args.dts, args.duration_ps, bond_table),
            "bond_crossing_real_knot": _run_case("bond_crossing_real_knot", bond_cross_builder, args.dts, args.duration_ps, bond_table),
            "angle_point_inside_cell": _run_case("angle_point_inside_cell", point_inside_builder, args.dts, args.duration_ps, angle_table),
            "angle_point_crossing_knot": _run_case("angle_point_crossing_knot", point_cross_builder, args.dts, args.duration_ps, angle_table),
            "angle_rigid_inside_cell": _run_case("angle_rigid_inside_cell", rigid_inside_builder, args.dts, args.duration_ps, angle_table),
            "angle_rigid_crossing_knot": _run_case("angle_rigid_crossing_knot", rigid_cross_builder, args.dts, args.duration_ps, angle_table),
        }
        reversibility = {
            "bond_inside": _reversibility(bond_inside_builder, args.reversibility_dt, args.reversibility_duration_ps),
            "bond_crossing": _reversibility(bond_cross_builder, args.reversibility_dt, args.reversibility_duration_ps),
            "angle_point_crossing": _reversibility(point_cross_builder, args.reversibility_dt, args.reversibility_duration_ps),
            "angle_rigid_crossing": _reversibility(rigid_cross_builder, args.reversibility_dt, args.reversibility_duration_ps),
        }
        torque_fd = _rigid_torque_fd(rigid_cross_builder, args.reversibility_dt, args.torque_fd_eps)

    report = {
        "schema_version": 1,
        "kind": "conservative_spline_minimal_dynamics_localization",
        "priors": str(priors_path),
        "duration_ps": float(args.duration_ps),
        "dts_ps": [float(x) for x in args.dts],
        "selected_bond": {
            "name": str(bond_entry.get("name", bond_entry.get("file"))),
            "file": str(bond_entry.get("file")),
            "knot_index": bond_k,
            "knot_q": float(bond_table.x[bond_k]),
            "u2_jump": float(bond_knot["jump"]),
        },
        "selected_angle": {
            "name": str(angle_entry.get("name", angle_entry.get("file"))),
            "file": str(angle_entry.get("file")),
            "knot_index": angle_k,
            "knot_q": float(angle_table.x[angle_k]),
            "u2_jump": float(angle_knot["jump"]),
        },
        "cases": cases,
        "reversibility": reversibility,
        "rigid_angle_torque_fd": torque_fd,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[CONSERVATIVE SPLINE MINIMAL DYNAMICS]")
    for key, case in cases.items():
        fit = case["fit"]
        crossings = [row["cell_crossings"] for row in case["crossings"]]
        print(f"{key}: sigma_E p={fit['exponent_p']:.6f} R2={fit['loglog_r2']:.6f} crossings={crossings}")
    print(
        "rigid-angle torque FD: "
        f"max|dTau|={torque_fd['max_abs_error']:.3e} torque_norm={torque_fd['torque_norm']:.3e}"
    )
    for key, value in reversibility.items():
        print(
            f"reversibility {key}: dr={value['position_rms_nm']:.3e} "
            f"dv={value['velocity_rms_nm_per_ps']:.3e} dtheta={value['orientation_rms_rad']:.3e} "
            f"domega={value['omega_body_rms_per_ps']:.3e}"
        )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
