#!/usr/bin/env python3
"""Diagnose legacy TabulatedDihedral energy/force consistency versus conservative splines.

This is a test-only runtime diagnostic for a configured step-37 workflow.  It compares the
same periodic torsional priors in two representations:

* ESPResSo ``TabulatedDihedral``: energy and force-factor columns are linearly
  interpolated independently.  The force-factor convention satisfies
  ``dU/dphi = -factor*sin(phi)`` only if the table is conservative.
* ``ConservativeSplineDihedral``: a single periodic Hermite energy spline is
  fundamental and Cartesian forces are computed from its analytical dU/dphi.

The script probes both the pre-update (iteration 000) and post-update
(iteration 001) step-35 torsional tables.  It performs no MD integration and
never modifies any prior artifact.
"""
from __future__ import annotations

import argparse
import hashlib
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
    conservative_dihedral_forces,
    conservative_spline_value,
    load_conservative_spline,
)
from conservative_spline_runtime import create_conservative_spline_interaction  # noqa: E402
from prior_kernels import (  # noqa: E402
    espresso_dihedral_geometry,
    load_tabulated_prior,
    tabulated_dihedral_forces,
    tabulated_value,
)

SCHEMA_VERSION = 1
KIND = "tabulated_dihedral_conservativity_diagnostic"
TWOPI = 2.0 * np.pi


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def group_entries(priors_path: Path, *, expected_type: str) -> dict[str, dict]:
    data = load_json(priors_path)
    groups: dict[str, dict] = {}
    files: dict[str, Path] = {}
    for idx, entry in enumerate(data.get("dihedrals", [])):
        if str(entry.get("type", "")).lower() != expected_type:
            continue
        name = str(entry.get("name", ""))
        require(bool(name), f"dihedrals[{idx}] in {priors_path} is missing name")
        require("file" in entry, f"dihedrals[{idx}] {name!r} in {priors_path} is missing file")
        table_path = Path(str(entry["file"])).expanduser()
        if not table_path.is_absolute():
            table_path = priors_path.resolve().parent / table_path
        table_path = table_path.resolve()
        if not table_path.is_file():
            raise FileNotFoundError(table_path)
        if name in groups:
            require(files[name] == table_path, f"Group {name!r} references multiple tables in {priors_path}")
            continue
        groups[name] = dict(entry)
        files[name] = table_path
    require(bool(groups), f"No {expected_type} dihedral groups found in {priors_path}")
    return groups


def system_singleton():
    system = espressomd.System(box_l=[30.0, 30.0, 30.0])
    system.time_step = 0.001
    system.cell_system.skin = 0.2
    system.thermostat.turn_off()
    system.integrator.set_vv()
    return system


def reset(system, positions: np.ndarray):
    system.part.clear()
    system.bonded_inter.clear()
    system.time = 0.0
    return [system.part.add(pos=p, type=0, mass=1.0) for p in positions]


def positions_for_phi(phi: float) -> np.ndarray:
    """Construct a well-conditioned four-particle geometry with ESPResSo phi."""
    center = np.asarray([14.0, 14.0, 14.0], dtype=float)
    # With the ESPResSo convention in prior_kernels this geometry has
    # n12=(0,0,1), cos(phi)=n12.n23 and the sign from n12.v34.
    offsets = np.asarray([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, np.cos(phi), np.sin(phi)],
    ], dtype=float)
    pos = center + offsets
    geom = espresso_dihedral_geometry(*pos, np.asarray([30.0, 30.0, 30.0]))
    if geom is None:
        raise RuntimeError(f"Internal probe geometry is undefined for phi={phi}")
    wrapped = float(phi % TWOPI)
    delta = abs(((geom[0] - wrapped + np.pi) % TWOPI) - np.pi)
    if delta > 1.0e-10:
        raise RuntimeError(f"Internal probe geometry phi mismatch: requested={wrapped}, got={geom[0]}")
    return pos


def add_dihedral(system, particles, interaction) -> None:
    system.bonded_inter.add(interaction)
    # ESPResSo attaches a dihedral to the second particle, with p1,p3,p4 as partners.
    particles[1].add_bond((interaction, particles[0].id, particles[2].id, particles[3].id))


def legacy_interaction(entry: dict, priors_path: Path):
    table = load_tabulated_prior(entry, kind="dihedral", priors_path=priors_path)
    return espressomd.interactions.TabulatedDihedral(
        min=table.minimum,
        max=table.maximum,
        energy=table.energy.tolist(),
        force=table.force.tolist(),
    )


def evaluate(system, positions: np.ndarray, *, representation: str, entry: dict, priors_path: Path):
    particles = reset(system, positions)
    if representation == "legacy":
        interaction = legacy_interaction(entry, priors_path)
    elif representation == "conservative":
        interaction = create_conservative_spline_interaction(
            espressomd.interactions, entry, kind="dihedral", priors_path=priors_path
        )
    else:
        raise ValueError(representation)
    add_dihedral(system, particles, interaction)
    system.integrator.run(0, recalc_forces=True)
    energy = float(system.analysis.energy()["bonded"])
    forces = np.asarray([p.f for p in particles], dtype=float)
    return energy, forces


def finite_difference_gradient(
    system,
    positions: np.ndarray,
    *,
    representation: str,
    entry: dict,
    priors_path: Path,
    eps: float,
) -> np.ndarray:
    grad = np.zeros_like(positions, dtype=float)
    for i in range(4):
        for a in range(3):
            plus = np.asarray(positions, dtype=float).copy()
            minus = np.asarray(positions, dtype=float).copy()
            plus[i, a] += eps
            minus[i, a] -= eps
            ep, _ = evaluate(system, plus, representation=representation, entry=entry, priors_path=priors_path)
            em, _ = evaluate(system, minus, representation=representation, entry=entry, priors_path=priors_path)
            grad[i, a] = (ep - em) / (2.0 * eps)
    return grad


def linear_energy_derivative(table, phi: float) -> float:
    """Exact dU/dphi of ESPResSo's piecewise-linear tabulated energy away from knots."""
    q = float(np.clip(phi, table.minimum, table.maximum))
    h = (table.maximum - table.minimum) / (len(table.energy) - 1)
    scaled = (q - table.minimum) / h
    lo = min(int(np.floor(scaled)), len(table.energy) - 2)
    if q >= table.maximum:
        lo = len(table.energy) - 2
    return float((table.energy[lo + 1] - table.energy[lo]) / h)


def select_probe_phis(table, n: int, sin_min: float) -> np.ndarray:
    mids = 0.5 * (table.x[:-1] + table.x[1:])
    valid = mids[np.abs(np.sin(mids)) >= sin_min]
    require(valid.size >= n, f"Not enough non-singular dihedral intervals for {n} probes")
    indices = np.linspace(0, valid.size - 1, n, dtype=int)
    return np.asarray(valid[indices], dtype=float)


def rms(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def summarize_pair(
    system,
    *,
    label: str,
    legacy_priors: Path,
    conservative_priors: Path,
    probes: int,
    sin_min: float,
    fd_eps: float,
) -> dict:
    legacy_groups = group_entries(legacy_priors, expected_type="tabulated")
    cons_groups = group_entries(conservative_priors, expected_type="conservative_spline")
    require(sorted(legacy_groups) == sorted(cons_groups), (
        f"Dihedral group mismatch for {label}: legacy={sorted(legacy_groups)}, "
        f"conservative={sorted(cons_groups)}"
    ))

    group_reports = {}
    all_scalar_residual = []
    all_legacy_fd = []
    all_cons_fd = []
    all_representation_df = []
    all_representation_de = []
    all_legacy_parity = []

    for name in sorted(legacy_groups):
        le = legacy_groups[name]
        ce = cons_groups[name]
        tab = load_tabulated_prior(le, kind="dihedral", priors_path=legacy_priors)
        spline = load_conservative_spline(ce, kind="dihedral", priors_path=conservative_priors)
        require(np.allclose(tab.x, spline.x, rtol=0.0, atol=1.0e-12), f"Grid mismatch for {label}/{name}")
        require(np.allclose(tab.energy, spline.energy, rtol=0.0, atol=1.0e-10), (
            f"Nodal energy mismatch for {label}/{name}; this diagnostic requires the conservative projection "
            "to preserve legacy energy nodes"
        ))

        phis = select_probe_phis(tab, probes, sin_min)
        scalar_residuals = []
        runtime_legacy_fd = []
        runtime_cons_fd = []
        runtime_df = []
        runtime_de = []
        runtime_legacy_parity = []
        scalar_rows = []

        for phi in phis:
            factor = tabulated_value(tab, float(phi), column="force")
            implied_du = -factor * np.sin(phi)
            runtime_du = linear_energy_derivative(tab, float(phi))
            scalar_residuals.append(implied_du - runtime_du)

            pos = positions_for_phi(float(phi))
            legacy_e, legacy_f = evaluate(
                system, pos, representation="legacy", entry=le, priors_path=legacy_priors
            )
            cons_e, cons_f = evaluate(
                system, pos, representation="conservative", entry=ce, priors_path=conservative_priors
            )
            expected_legacy = np.vstack(
                tabulated_dihedral_forces(*pos, np.asarray(system.box_l), tab)
            )
            runtime_legacy_parity.append(float(np.max(np.abs(legacy_f - expected_legacy))))

            legacy_grad = finite_difference_gradient(
                system, pos, representation="legacy", entry=le, priors_path=legacy_priors, eps=fd_eps
            )
            cons_grad = finite_difference_gradient(
                system, pos, representation="conservative", entry=ce, priors_path=conservative_priors, eps=fd_eps
            )
            legacy_fd_err = float(np.max(np.abs(legacy_f + legacy_grad)))
            cons_fd_err = float(np.max(np.abs(cons_f + cons_grad)))
            runtime_legacy_fd.append(legacy_fd_err)
            runtime_cons_fd.append(cons_fd_err)
            runtime_df.append(float(np.max(np.abs(legacy_f - cons_f))))
            runtime_de.append(abs(legacy_e - cons_e))
            _uc, duc = conservative_spline_value(spline, float(phi))
            scalar_rows.append({
                "phi": float(phi),
                "legacy_force_factor": float(factor),
                "legacy_implied_dU_dphi": float(implied_du),
                "legacy_runtime_linear_dU_dphi": float(runtime_du),
                "conservative_dU_dphi": float(duc),
                "legacy_energy": float(legacy_e),
                "conservative_energy": float(cons_e),
                "legacy_force_minus_conservative_max_abs": runtime_df[-1],
                "legacy_force_plus_gradE_max_abs": legacy_fd_err,
                "conservative_force_plus_gradE_max_abs": cons_fd_err,
            })

        scale = max(rms([linear_energy_derivative(tab, float(phi)) for phi in phis]), 1.0e-12)
        rec = {
            "legacy_table": str(tab.path),
            "legacy_table_sha256": sha256_file(tab.path),
            "conservative_table": str(spline.path),
            "conservative_table_sha256": sha256_file(spline.path),
            "probes": int(len(phis)),
            "scalar_force_factor_vs_linear_energy_derivative": {
                "rms_abs_dU_dphi_residual": rms(scalar_residuals),
                "p99_abs_dU_dphi_residual": float(np.percentile(np.abs(scalar_residuals), 99.0)),
                "max_abs_dU_dphi_residual": float(np.max(np.abs(scalar_residuals))),
                "rms_relative_to_runtime_energy_slope": float(rms(scalar_residuals) / scale),
            },
            "legacy_runtime_preprocessing_parity_max_abs": float(np.max(runtime_legacy_parity)),
            "legacy_runtime_force_plus_gradE": {
                "rms_max_component_error": rms(runtime_legacy_fd),
                "max_component_error": float(np.max(runtime_legacy_fd)),
            },
            "conservative_runtime_force_plus_gradE": {
                "rms_max_component_error": rms(runtime_cons_fd),
                "max_component_error": float(np.max(runtime_cons_fd)),
            },
            "legacy_vs_conservative_runtime": {
                "rms_max_force_component_difference": rms(runtime_df),
                "max_force_component_difference": float(np.max(runtime_df)),
                "rms_energy_difference": rms(runtime_de),
                "max_energy_difference": float(np.max(runtime_de)),
            },
            "samples": scalar_rows,
        }
        group_reports[name] = rec
        all_scalar_residual.extend(scalar_residuals)
        all_legacy_fd.extend(runtime_legacy_fd)
        all_cons_fd.extend(runtime_cons_fd)
        all_representation_df.extend(runtime_df)
        all_representation_de.extend(runtime_de)
        all_legacy_parity.extend(runtime_legacy_parity)

    legacy_fd_rms = rms(all_legacy_fd)
    cons_fd_rms = rms(all_cons_fd)
    ratio = legacy_fd_rms / max(cons_fd_rms, 1.0e-16)
    return {
        "label": label,
        "legacy_priors": str(legacy_priors),
        "legacy_priors_sha256": sha256_file(legacy_priors),
        "conservative_priors": str(conservative_priors),
        "conservative_priors_sha256": sha256_file(conservative_priors),
        "groups": group_reports,
        "aggregate": {
            "groups": len(group_reports),
            "scalar_dU_residual_rms": rms(all_scalar_residual),
            "scalar_dU_residual_p99": float(np.percentile(np.abs(all_scalar_residual), 99.0)),
            "legacy_runtime_force_plus_gradE_rms": legacy_fd_rms,
            "legacy_runtime_force_plus_gradE_max": float(np.max(all_legacy_fd)),
            "conservative_runtime_force_plus_gradE_rms": cons_fd_rms,
            "conservative_runtime_force_plus_gradE_max": float(np.max(all_cons_fd)),
            "legacy_to_conservative_nonconservativity_ratio": float(ratio),
            "legacy_vs_conservative_force_rms": rms(all_representation_df),
            "legacy_vs_conservative_force_max": float(np.max(all_representation_df)),
            "legacy_vs_conservative_energy_rms": rms(all_representation_de),
            "legacy_runtime_preprocessing_parity_max": float(np.max(all_legacy_parity)),
        },
    }


def diagnostic_hint(pairs: list[dict], *, legacy_residual_min: float, ratio_min: float, conservative_residual_max: float) -> str:
    # Heuristic only. A large legacy F+gradE residual combined with a small
    # conservative one directly localizes the representation jump to independent
    # legacy energy/force-factor interpolation rather than the conservative kernel.
    ratios = [float(p["aggregate"]["legacy_to_conservative_nonconservativity_ratio"]) for p in pairs]
    legacy = [float(p["aggregate"]["legacy_runtime_force_plus_gradE_rms"]) for p in pairs]
    cons = [float(p["aggregate"]["conservative_runtime_force_plus_gradE_rms"]) for p in pairs]
    if max(legacy) > legacy_residual_min and min(ratios) > ratio_min and max(cons) < conservative_residual_max:
        return "legacy_tabulated_energy_force_inconsistency_dominant"
    if max(legacy) <= legacy_residual_min and max(cons) <= conservative_residual_max:
        return "both_representations_locally_conservative_check_interpolation_difference"
    return "mixed_or_requires_deeper_runtime_analysis"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration0-priors", required=True, type=Path)
    parser.add_argument("--iteration0-conservative", required=True, type=Path)
    parser.add_argument("--iteration1-priors", required=True, type=Path)
    parser.add_argument("--iteration1-conservative", required=True, type=Path)
    parser.add_argument("--probes-per-group", type=int, required=True)
    parser.add_argument("--sin-min", type=float, required=True)
    parser.add_argument("--fd-eps", type=float, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--legacy-residual-min", type=float, required=True)
    parser.add_argument("--ratio-min", type=float, required=True)
    parser.add_argument("--conservative-residual-max", type=float, required=True)
    parser.add_argument("--model-config-provenance", type=Path, default=None)
    args = parser.parse_args()

    if args.probes_per_group < 3:
        raise ValueError("--probes-per-group must be at least 3")
    if not (0.0 < args.sin_min < 1.0):
        raise ValueError("--sin-min must be in (0,1)")
    if args.fd_eps <= 0.0:
        raise ValueError("--fd-eps must be positive")
    for name in ("ConservativeSplineDihedral", "TabulatedDihedral"):
        if not hasattr(espressomd.interactions, name):
            raise RuntimeError(f"ESPResSo is missing espressomd.interactions.{name}")

    paths = [
        args.iteration0_priors,
        args.iteration0_conservative,
        args.iteration1_priors,
        args.iteration1_conservative,
    ]
    paths = [p.expanduser().resolve() for p in paths]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    system = system_singleton()
    pair0 = summarize_pair(
        system,
        label="iteration_000_pre_update",
        legacy_priors=paths[0],
        conservative_priors=paths[1],
        probes=args.probes_per_group,
        sin_min=args.sin_min,
        fd_eps=args.fd_eps,
    )
    pair1 = summarize_pair(
        system,
        label="iteration_001_post_update",
        legacy_priors=paths[2],
        conservative_priors=paths[3],
        probes=args.probes_per_group,
        sin_min=args.sin_min,
        fd_eps=args.fd_eps,
    )
    pairs = [pair0, pair1]
    hint = diagnostic_hint(pairs, legacy_residual_min=args.legacy_residual_min, ratio_min=args.ratio_min, conservative_residual_max=args.conservative_residual_max)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "test_only": True,
        "production_modified": False,
        "force_factor_convention": "dU_dphi = -legacy_force_factor * sin(phi)",
        "diagnostic_hint_policy": {"legacy_residual_min": args.legacy_residual_min, "ratio_min": args.ratio_min, "conservative_residual_max": args.conservative_residual_max},
        "model_config_provenance": str(args.model_config_provenance.resolve()) if args.model_config_provenance else None,
        "probe_policy": {
            "probes_per_group": int(args.probes_per_group),
            "minimum_abs_sin_phi": float(args.sin_min),
            "cartesian_fd_eps": float(args.fd_eps),
            "phi_points_are_midpoints_of_legacy_energy_intervals": True,
        },
        "pairs": pairs,
        "diagnostic_hint": hint,
        "hint_is_gating": False,
    }
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[TABULATED DIHEDRAL CONSERVATIVITY DIAGNOSTIC]")
    print("[CONVENTION] dU/dphi = -force_factor*sin(phi)")
    for pair in pairs:
        a = pair["aggregate"]
        print(
            f"[{pair['label']}] scalar_dU_resid_rms={a['scalar_dU_residual_rms']:.6g} "
            f"legacy|F+gradE|rms={a['legacy_runtime_force_plus_gradE_rms']:.6g} "
            f"cons|F+gradE|rms={a['conservative_runtime_force_plus_gradE_rms']:.6g} "
            f"legacy/cons={a['legacy_to_conservative_nonconservativity_ratio']:.3g} "
            f"legacy-vs-cons dF_rms={a['legacy_vs_conservative_force_rms']:.6g}"
        )
        print(
            f"[{pair['label']}] legacy preprocessing/runtime parity max="
            f"{a['legacy_runtime_preprocessing_parity_max']:.3e}"
        )
    print(f"[HINT] {hint} (diagnostic only, non-gating)")
    print("[FINAL] promotion_ready=False")
    print(f"[DONE] report: {out}")


if __name__ == "__main__":
    main()
