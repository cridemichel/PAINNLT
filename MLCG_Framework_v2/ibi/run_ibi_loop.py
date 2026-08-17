#!/usr/bin/env python3
"""Run bonded Iterative Boltzmann Inversion for MLCG Framework v2.

The workflow is deliberately pre-training/prior-only:

For a fresh run the driver derives initial DBI tables from the mapped reference
trajectory.  For a continuation run it instead loads an already evaluated
self-contained tabulated prior set and preserves those exact tables as the
starting potential.  It then runs classical NVT CG sampling with all explicit
priors (no PaiNN model), updates only entries carrying ``ibi_mode: ibi``, and
writes self-contained iteration/final prior sets.

Entries carrying ``ibi_mode: dbi`` are sampled but never updated.  This driver
handles bonded distances, angles and dihedrals only; it is not an RDF/nonbonded
IBI implementation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conservative_spline import SCHEMA, save_conservative_spline  # noqa: E402
from build_dbi_priors import (  # noqa: E402
    build_initial_dbi_priors,
    load_continuation_priors,
    safe_group_name,
)
from geometry_io import pool_requested, read_sampled_distributions  # noqa: E402
from ibi_core import (  # noqa: E402
    histogram_density,
    normalize_density,
    save_tabulated_potential,
    update_ibi_potential,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_logged(command, *, cwd: Path, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("[COMMAND] " + " ".join(str(item) for item in command) + "\n\n")
        log.flush()
        result = subprocess.run(
            [str(item) for item in command],
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        lines = log_path.read_text(errors="replace").splitlines()
        tail = "\n".join(lines[-60:])
        raise RuntimeError(
            f"ESPResSo sampling failed with exit code {result.returncode}. "
            f"Log: {log_path}\n--- log tail ---\n{tail}"
        )


def _distribution_l1(sim_density, target_density, grid):
    sim = normalize_density(sim_density, grid)
    target = normalize_density(target_density, grid)
    return float(np.trapezoid(np.abs(sim - target), grid))


def _resolve_table_path(filename, reference_priors: Path) -> Path:
    path = Path(str(filename)).expanduser()
    if path.is_absolute():
        return path
    return reference_priors.resolve().parent / path


def _write_iteration_priors(template, groups, iteration_dir: Path, *, source_priors_path: Path):
    iteration_dir.mkdir(parents=True, exist_ok=True)
    priors = copy.deepcopy(template)

    group_indices = {
        json_key: {idx for state in category.values() for idx in state["indices"]}
        for json_key, category in groups.items()
    }
    # Make fixed file-backed interactions self-contained.  This includes the
    # promoted production bond/angle conservative splines used as the frozen
    # background during dihedral-only IBI.
    for json_key in ("bonds", "angles", "dihedrals"):
        for idx, entry in enumerate(priors.get(json_key, [])):
            if idx in group_indices.get(json_key, set()):
                continue
            entry_type = str(entry.get("type", "")).lower()
            if entry_type not in {"tabulated", "conservative_spline"}:
                continue
            if "file" not in entry:
                raise ValueError(f"Fixed {entry_type} {json_key}[{idx}] is missing 'file'")
            source = _resolve_table_path(entry["file"], source_priors_path).resolve()
            if not source.is_file():
                raise FileNotFoundError(
                    f"Missing fixed {entry_type} table for {json_key}[{idx}]: {source}"
                )
            suffix = source.suffix or ".dat"
            destination = iteration_dir / f"fixed_{json_key}_{idx}{suffix}"
            shutil.copy2(source, destination)
            entry["file"] = destination.name

    for json_key, category in groups.items():
        for name, state in category.items():
            use_conservative_dihedral = (
                state["kind"] == "dihedral"
                and str(state.get("representation", "tabulated")) == "conservative_spline"
            )
            if use_conservative_dihedral:
                filename = f"dihedral_conservative_{safe_group_name(name)}.dat"
                table_path = iteration_dir / filename
                save_conservative_spline(
                    table_path, state["grid"], state["energy"], state["force"]
                )
            else:
                filename = f"{state['kind']}_tabulated_{safe_group_name(name)}.dat"
                table_path = iteration_dir / filename
                save_tabulated_potential(
                    table_path, state["grid"], state["energy"], state["force"]
                )
            for idx in state["indices"]:
                entry = priors[json_key][idx]
                entry["type"] = "conservative_spline" if use_conservative_dihedral else "tabulated"
                entry["ibi_mode"] = state["mode"]
                entry["file"] = filename
                entry["min"] = float(state["grid"][0])
                entry["max"] = float(state["grid"][-1])
                if use_conservative_dihedral:
                    entry["spline_schema"] = SCHEMA
                    entry["ibi_runtime_representation"] = "conservative_spline"
                else:
                    entry.pop("spline_schema", None)
                    entry.pop("ibi_runtime_representation", None)
    priors_path = iteration_dir / "cg_priors.json"
    priors_path.write_text(json.dumps(priors, indent=2) + "\n")
    return priors, priors_path

def _write_final_priors(template, groups, outdir: Path, *, source_priors_path: Path):
    final_dir = outdir / "final"
    final_priors, final_internal = _write_iteration_priors(
        template, groups, final_dir, source_priors_path=source_priors_path
    )

    # Root-level convenience JSON keeps every tabulated path valid by pointing
    # into the self-contained final/ directory, including non-IBI fixed tables.
    root_priors = copy.deepcopy(final_priors)
    for json_key in ("bonds", "angles", "dihedrals"):
        for entry in root_priors.get(json_key, []):
            if str(entry.get("type", "")).lower() in {"tabulated", "conservative_spline"}:
                entry["file"] = str(Path("final") / entry["file"])
    root_path = outdir / "cg_priors_final.json"
    root_path.write_text(json.dumps(root_priors, indent=2) + "\n")
    return root_path, final_internal


def _assert_conservative_dihedral_loop(priors, groups, priors_path: Path) -> None:
    """Fail closed if an active torsional IBI group can reach legacy runtime."""
    active = 0
    for name, state in groups.get("dihedrals", {}).items():
        if state["mode"] != "ibi":
            continue
        active += 1
        if str(state.get("representation", "")) != "conservative_spline":
            raise RuntimeError(
                f"Conservative dihedral IBI loop requested but state {name!r} is "
                f"{state.get('representation')!r}"
            )
        for idx in state["indices"]:
            entry = priors.get("dihedrals", [])[idx]
            if str(entry.get("type", "")).lower() != "conservative_spline":
                raise RuntimeError(
                    f"Conservative dihedral IBI loop refuses {priors_path}: "
                    f"dihedrals[{idx}] is type={entry.get('type')!r}"
                )
            if str(entry.get("ibi_runtime_representation", "")) != "conservative_spline":
                raise RuntimeError(
                    f"Conservative dihedral IBI loop is missing the explicit runtime marker "
                    f"for dihedrals[{idx}] in {priors_path}"
                )
    if active == 0:
        raise RuntimeError("Conservative dihedral IBI loop requested but no active IBI dihedral groups exist")


def _simulation_parameters(settings):
    try:
        sim = settings["simulation"]
        dt = float(sim["dt"])
        production_steps = int(sim["steps"])
        sample_interval = int(sim["log_interval"])
        burn_in_steps = int(sim["burn_in_steps"])
    except KeyError as exc:
        raise ValueError(f"Explicit IBI settings are missing required simulation key: {exc.args[0]}") from exc
    if dt <= 0.0:
        raise ValueError("IBI simulation dt must be positive")
    if production_steps <= 0:
        raise ValueError("IBI simulation steps must be positive")
    if burn_in_steps < 0:
        raise ValueError("IBI burn-in steps must be non-negative")
    if sample_interval <= 0:
        raise ValueError("IBI sampling interval must be positive")
    if burn_in_steps % sample_interval != 0:
        raise ValueError("IBI burn-in steps must be a multiple of the sampling interval")
    if production_steps % sample_interval != 0:
        raise ValueError("IBI production steps must be a multiple of the sampling interval")
    return dt, burn_in_steps, production_steps, sample_interval


def run_ibi(
    *,
    dataset,
    seed_priors=None,
    resume_priors=None,
    config,
    rb_info,
    outdir,
    iterations,
    pypresso,
    ibi_config=None,
    neighbor_search="verlet",
    overwrite=False,
    velocity_seed=314159,
    thermostat_seed=42,
    iteration_offset=0,
    conservative_dihedrals_in_loop=False,
):
    dataset = Path(dataset).resolve()
    seed_priors = Path(seed_priors).resolve() if seed_priors is not None else None
    resume_priors = Path(resume_priors).resolve() if resume_priors is not None else None
    config = Path(config).resolve()
    rb_info = Path(rb_info).resolve()
    outdir = Path(outdir).resolve()
    pypresso = Path(pypresso).resolve()
    ibi_config_path = Path(ibi_config).resolve() if ibi_config is not None else None
    if iterations < 0:
        raise ValueError("--iterations must be non-negative")
    if iteration_offset < 0:
        raise ValueError("--iteration-offset must be non-negative")
    if (seed_priors is None) == (resume_priors is None):
        raise ValueError("Specify exactly one of seed_priors or resume_priors")
    if resume_priors is None and iteration_offset != 0:
        raise ValueError("--iteration-offset is only valid with --resume-priors")

    required_paths = [
        (dataset, "dataset"),
        (config, "config"),
        (rb_info, "rigid-body info"),
        (pypresso, "pypresso"),
    ]
    required_paths.append(
        (seed_priors, "seed priors") if seed_priors is not None
        else (resume_priors, "resume priors")
    )
    for path, label in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if ibi_config_path is not None and not ibi_config_path.exists():
        raise FileNotFoundError(f"Missing IBI settings: {ibi_config_path}")

    input_hashes = {
        "dataset": _sha256_file(dataset),
        "config": _sha256_file(config),
        "rb_info": _sha256_file(rb_info),
        "pypresso": _sha256_file(pypresso),
    }
    if seed_priors is not None:
        input_hashes["seed_priors"] = _sha256_file(seed_priors)
    else:
        input_hashes["resume_priors"] = _sha256_file(resume_priors)
    if ibi_config_path is not None:
        input_hashes["ibi_config"] = _sha256_file(ibi_config_path)

    if outdir.exists() and any(outdir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"IBI output directory is not empty: {outdir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if resume_priors is None:
        iteration0 = outdir / "iteration_000"
        initial = build_initial_dbi_priors(
            dataset,
            seed_priors,
            iteration0,
            output_priors=iteration0 / "cg_priors.json",
            ibi_config=ibi_config_path,
            conservative_dihedrals=conservative_dihedrals_in_loop,
        )
        settings = initial["settings"]
        groups = initial["groups"]
        current_priors = initial["priors"]
        current_priors_path = initial["output_priors"]
        run_mode = "initial_dbi"
    else:
        initial = load_continuation_priors(
            dataset,
            resume_priors,
            ibi_config=ibi_config_path,
            allow_conservative_spline=conservative_dihedrals_in_loop,
            conservative_dihedral_update=conservative_dihedrals_in_loop,
        )
        settings = initial["settings"]
        groups = initial["groups"]
        current_priors = initial["priors"]
        current_priors, current_priors_path = _write_iteration_priors(
            current_priors,
            groups,
            outdir / "resume_start",
            source_priors_path=resume_priors,
        )
        run_mode = "continuation"
        print(
            f"[INFO] Continuing IBI from evaluated priors {resume_priors}; "
            f"sampling iteration offset={iteration_offset}"
        )

    ibi_group_count = sum(
        state["mode"] == "ibi"
        for category in groups.values()
        for state in category.values()
    )
    dbi_group_count = sum(
        state["mode"] == "dbi"
        for category in groups.values()
        for state in category.values()
    )
    state_label = "Continuation state" if resume_priors is not None else "Initial bonded inversion"
    print(
        f"[INFO] {state_label}: {ibi_group_count} iterative IBI groups, "
        f"{dbi_group_count} DBI-only groups"
    )
    if conservative_dihedrals_in_loop:
        _assert_conservative_dihedral_loop(current_priors, groups, current_priors_path)
        print("[PASS] Active dihedral IBI groups use ConservativeSplineDihedral before the first sampling step.")

    dt, burn_in_steps, production_steps, sample_interval = _simulation_parameters(settings)
    total_steps = burn_in_steps + production_steps
    kT = float(settings["kT"])
    run_script = ROOT / "simulation" / "run_cg_md.py"

    all_metrics = []
    for continuation_step in range(1, iterations + 1):
        if ibi_group_count == 0:
            print("[INFO] No type=ibi groups are active; stopping after initial DBI generation.")
            break

        iteration = iteration_offset + continuation_step
        if resume_priors is None:
            iteration_label = f"{iteration}/{iterations}"
        else:
            iteration_label = (
                f"{iteration} (continuation {continuation_step}/{iterations}, "
                f"offset={iteration_offset})"
            )
        print(
            f"\n[INFO] === IBI iteration {iteration_label}: "
            f"burn-in={burn_in_steps}, production={production_steps}, dt={dt} ps ==="
        )
        sample_dir = outdir / "sampling" / f"iteration_{iteration:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / "trajectory.npz"
        log_path = sample_dir / "run.log"

        if conservative_dihedrals_in_loop:
            _assert_conservative_dihedral_loop(current_priors, groups, current_priors_path)

        command = [
            pypresso,
            run_script,
            "--config", config,
            "--priors", current_priors_path,
            "--rb_info", rb_info,
            "--dataset", dataset,
            "--dt", str(dt),
            "--steps", str(total_steps),
            "--log_interval", str(sample_interval),
            "--sample_start_step", str(burn_in_steps),
            "--sample_npz", sample_path,
            "--kT", str(kT),
            "--init_kT", str(kT),
            "--velocity_seed", str(int(velocity_seed) + iteration - 1),
            "--thermostat_seed", str(int(thermostat_seed) + iteration - 1),
            "--neighbor_search", neighbor_search,
            "--no_log",
        ]
        _run_logged(command, cwd=sample_dir, log_path=log_path)

        sampled = read_sampled_distributions(sample_path, current_priors)
        sampled_by_key = {
            "bonds": sampled[0],
            "angles": sampled[1],
            "dihedrals": sampled[2],
        }
        iteration_metrics = {
            "iteration": iteration,
            "continuation_step": continuation_step,
            "source_priors": str(current_priors_path),
            "sample": str(sample_path),
            "groups": {},
        }

        for json_key, category in groups.items():
            sim_groups = pool_requested(current_priors, sampled_by_key[json_key], json_key)
            for name, state in category.items():
                sim_group = sim_groups.get(name)
                if sim_group is None:
                    raise RuntimeError(f"Sampling did not produce requested {json_key} group {name!r}")
                sim_values = np.asarray(sim_group["values"], dtype=float)
                if sim_values.size == 0:
                    raise RuntimeError(f"No simulated samples for {json_key} group {name!r}")
                if state["kind"] == "dihedral":
                    sim_values = np.mod(sim_values, 2.0 * np.pi)

                if state["kind"] == "bond":
                    safety_fraction = float(settings["bond"]["runtime_safety_fraction"])
                    max_sample = float(np.max(sim_values))
                    if max_sample >= safety_fraction * float(state["grid"][-1]):
                        raise RuntimeError(
                            f"Bond group {name!r} sampled r={max_sample:.6g} nm, too close to "
                            f"the TabulatedDistance break limit {state['grid'][-1]:.6g} nm"
                        )

                sim_counts, sim_density, sim_hist_x = histogram_density(sim_values, state["bins"])
                if not np.allclose(sim_hist_x, state["hist_x"], rtol=0.0, atol=1.0e-12):
                    raise RuntimeError(f"Histogram grid changed for {json_key} group {name!r}")
                l1_before = _distribution_l1(
                    sim_density, state["target_density"], state["hist_x"]
                )

                group_metrics = {
                    "kind": state["kind"],
                    "mode": state["mode"],
                    "runtime_representation": str(state.get("representation", "tabulated")),
                    "samples": int(sim_values.size),
                    "distribution_l1": l1_before,
                    "sample_min": float(np.min(sim_values)),
                    "sample_max": float(np.max(sim_values)),
                }

                if state["mode"] == "ibi":
                    _x, next_energy, next_force = update_ibi_potential(
                        state["energy"],
                        sim_density,
                        state["target_density"],
                        state["hist_x"],
                        state["grid"],
                        state["target_counts"],
                        sim_counts,
                        periodic=state["kind"] == "dihedral",
                        target_type=state["kind"],
                        settings=settings,
                        previous_force=state["force"],
                        conservative_dihedral=(
                            state["kind"] == "dihedral"
                            and str(state.get("representation", "")) == "conservative_spline"
                        ),
                    )
                    state["energy"] = next_energy
                    state["force"] = next_force
                    print(
                        f"[IBI] {state['kind']} {name}: N={sim_values.size}, "
                        f"L1(Psim,Ptarget)={l1_before:.6g}"
                    )
                else:
                    print(
                        f"[DBI] {state['kind']} {name}: fixed table, N={sim_values.size}, "
                        f"L1(Psim,Ptarget)={l1_before:.6g}"
                    )
                iteration_metrics["groups"][f"{json_key}:{name}"] = group_metrics

        next_dir = outdir / f"iteration_{iteration:03d}"
        current_priors, current_priors_path = _write_iteration_priors(
            current_priors, groups, next_dir, source_priors_path=current_priors_path
        )
        if conservative_dihedrals_in_loop:
            _assert_conservative_dihedral_loop(current_priors, groups, current_priors_path)
        metrics_path = sample_dir / "metrics.json"
        metrics_path.write_text(json.dumps(iteration_metrics, indent=2) + "\n")
        all_metrics.append(iteration_metrics)
        print(f"[INFO] Updated priors: {current_priors_path}")

    final_path, final_internal = _write_final_priors(
        current_priors, groups, outdir, source_priors_path=current_priors_path
    )
    summary = {
        "schema_version": 2,
        "run_mode": run_mode,
        "dataset": str(dataset),
        "seed_priors": str(seed_priors) if seed_priors is not None else None,
        "resume_priors": str(resume_priors) if resume_priors is not None else None,
        "iteration_offset": int(iteration_offset),
        "config": str(config),
        "rb_info": str(rb_info),
        "iterations_requested": int(iterations),
        "iterations_completed": len(all_metrics),
        "ibi_groups": int(ibi_group_count),
        "dbi_groups": int(dbi_group_count),
        "kT": kT,
        "dt_ps": dt,
        "burn_in_steps": burn_in_steps,
        "production_steps": production_steps,
        "sample_interval": sample_interval,
        "neighbor_search": neighbor_search,
        "conservative_dihedrals_in_loop": bool(conservative_dihedrals_in_loop),
        "dihedral_runtime_representation": (
            "conservative_spline" if conservative_dihedrals_in_loop else "legacy_tabulated"
        ),
        "velocity_seed": int(velocity_seed),
        "thermostat_seed": int(thermostat_seed),
        "settings": settings,
        "inputs_sha256": input_hashes,
        "final_priors": str(final_path),
        "final_internal_priors": str(final_internal),
        "metrics": all_metrics,
    }
    summary_path = outdir / "ibi_report.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[SUCCESS] Bonded IBI/DBI workflow completed.")
    print(f"[SUCCESS] Final priors: {final_path}")
    print(f"[SUCCESS] Report: {summary_path}")
    print(
        "[NEXT] Evaluate/select a sampled prior set before rebuilding the force-matching "
        "dataset. The final post-update priors written by this run have not yet been sampled."
    )
    return summary


def main():
    default_pypresso = ROOT / "espresso" / "build" / "pypresso"
    parser = argparse.ArgumentParser(description="Run bonded IBI/DBI for MLCG Framework v2")
    parser.add_argument("--dataset", required=True, help="Mapped CG binary dataset used as target geometry")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--priors", help="Fresh seed priors with type=ibi/dbi entries")
    source.add_argument(
        "--resume-priors",
        help="Previously evaluated priors carrying ibi_mode=ibi/dbi; representation must match the selected IBI runtime mode",
    )
    parser.add_argument("--iteration-offset", type=int, default=0, help="Sampling-iteration offset for a continuation run")
    parser.add_argument("--config", required=True, help="CG/PaiNN config used by run_cg_md.py")
    parser.add_argument("--rb_info", required=True, help="rigid_bodies_info.json")
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--ibi-config", "--ibi_config", dest="ibi_config", required=True, help="Authoritative external IBI settings JSON")
    parser.add_argument("--pypresso", default=str(default_pypresso))
    parser.add_argument("--neighbor_search", choices=("verlet", "link-cell"), required=True)
    parser.add_argument("--velocity_seed", type=int, required=True)
    parser.add_argument("--thermostat_seed", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--conservative-dihedrals-in-loop",
        action="store_true",
        help=(
            "Generate/update active IBI dihedrals directly as ConservativeSplineDihedral and "
            "fail if a legacy TabulatedDihedral enters sampling. Bond/angle behavior is unchanged."
        ),
    )
    args = parser.parse_args()

    run_ibi(
        dataset=args.dataset,
        seed_priors=args.priors,
        resume_priors=args.resume_priors,
        config=args.config,
        rb_info=args.rb_info,
        outdir=args.outdir,
        iterations=args.iterations,
        pypresso=args.pypresso,
        ibi_config=args.ibi_config,
        neighbor_search=args.neighbor_search,
        overwrite=args.overwrite,
        velocity_seed=args.velocity_seed,
        thermostat_seed=args.thermostat_seed,
        iteration_offset=args.iteration_offset,
        conservative_dihedrals_in_loop=args.conservative_dihedrals_in_loop,
    )


if __name__ == "__main__":
    main()
