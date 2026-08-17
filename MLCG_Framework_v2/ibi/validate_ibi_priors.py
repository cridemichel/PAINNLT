#!/usr/bin/env python3
"""Read-only validation sampling for an evaluated bonded IBI prior set.

The supplied priors are never updated.  A fresh NVT trajectory is sampled with
independent random seeds, the bonded target distributions are reconstructed
from the mapped reference dataset, and L1 distances are reported using the same
histogram definitions as the IBI loop.  SHA256 hashes of the prior JSON and all
referenced tabulated tables are checked before and after sampling to enforce
read-only semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dbi_priors import load_continuation_priors  # noqa: E402
from geometry_io import pool_requested, read_sampled_distributions  # noqa: E402
from ibi_core import histogram_density  # noqa: E402
from run_ibi_loop import (  # noqa: E402
    _distribution_l1,
    _run_logged,
    _simulation_parameters,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _referenced_artifact_hashes(priors_path: Path) -> dict[str, str]:
    priors_path = priors_path.resolve()
    priors = json.loads(priors_path.read_text())
    hashes = {str(priors_path): _sha256_file(priors_path)}
    for key in ("bonds", "angles", "dihedrals"):
        for idx, entry in enumerate(priors.get(key, [])):
            if str(entry.get("type", "")).lower() != "tabulated":
                continue
            if "file" not in entry:
                raise ValueError(f"Tabulated {key}[{idx}] is missing 'file'")
            table = Path(str(entry["file"])).expanduser()
            if not table.is_absolute():
                table = priors_path.parent / table
            table = table.resolve()
            if not table.is_file():
                raise FileNotFoundError(f"Missing referenced table for {key}[{idx}]: {table}")
            hashes[str(table)] = _sha256_file(table)
    return dict(sorted(hashes.items()))


def _reference_metrics(summary_path: Path | None):
    if summary_path is None:
        return None
    summary = json.loads(summary_path.read_text())
    best_iteration = int(summary["best_sampling_iteration"])
    best_row = next(
        (row for row in summary.get("iterations", []) if int(row["sampling_iteration"]) == best_iteration),
        None,
    )
    if best_row is None:
        raise ValueError(
            f"Reference summary does not contain its best sampling iteration {best_iteration}: {summary_path}"
        )
    return {
        "summary": str(summary_path.resolve()),
        "best_sampling_iteration": best_iteration,
        "mean_l1": float(best_row["mean_l1"]),
        "max_l1": float(best_row["max_l1"]),
        "mean_l1_by_kind": {
            str(k): float(v) for k, v in best_row.get("mean_l1_by_kind", {}).items()
        },
    }


def validate_priors(
    *,
    dataset,
    priors,
    config,
    rb_info,
    pypresso,
    outdir,
    ibi_config=None,
    neighbor_search="link-cell",
    velocity_seed=271828,
    thermostat_seed=161803,
    reference_summary=None,
    overwrite=False,
):
    dataset = Path(dataset).resolve()
    priors = Path(priors).resolve()
    config = Path(config).resolve()
    rb_info = Path(rb_info).resolve()
    pypresso = Path(pypresso).resolve()
    outdir = Path(outdir).resolve()
    ibi_config = Path(ibi_config).resolve() if ibi_config is not None else None
    reference_summary = Path(reference_summary).resolve() if reference_summary is not None else None

    for path, label in (
        (dataset, "dataset"),
        (priors, "priors"),
        (config, "config"),
        (rb_info, "rigid-body info"),
        (pypresso, "pypresso"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if ibi_config is not None and not ibi_config.is_file():
        raise FileNotFoundError(f"Missing IBI config: {ibi_config}")
    if reference_summary is not None and not reference_summary.is_file():
        raise FileNotFoundError(f"Missing reference convergence summary: {reference_summary}")

    # Validation output must not live inside the immutable source-prior directory.
    try:
        outdir.relative_to(priors.parent)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"Validation outdir must not be inside the source-prior directory {priors.parent}: {outdir}"
        )

    if outdir.exists():
        if not overwrite and any(outdir.iterdir()):
            raise FileExistsError(f"Validation output directory is not empty: {outdir}")
        if overwrite:
            shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    immutable_hashes_before = _referenced_artifact_hashes(priors)
    loaded = load_continuation_priors(dataset, priors, ibi_config=ibi_config)
    settings = loaded["settings"]
    groups = loaded["groups"]
    current_priors = loaded["priors"]

    dt, burn_in_steps, production_steps, sample_interval = _simulation_parameters(settings)
    total_steps = burn_in_steps + production_steps
    kT = float(settings["kT"])
    sample_path = outdir / "trajectory.npz"
    log_path = outdir / "run.log"
    run_script = ROOT / "simulation" / "run_cg_md.py"

    print(
        f"[INFO] Read-only validation: burn-in={burn_in_steps}, production={production_steps}, "
        f"dt={dt} ps, velocity_seed={velocity_seed}, thermostat_seed={thermostat_seed}"
    )
    command = [
        pypresso,
        run_script,
        "--config", config,
        "--priors", priors,
        "--rb_info", rb_info,
        "--dataset", dataset,
        "--dt", str(dt),
        "--steps", str(total_steps),
        "--log_interval", str(sample_interval),
        "--sample_start_step", str(burn_in_steps),
        "--sample_npz", sample_path,
        "--kT", str(kT),
        "--init_kT", str(kT),
        "--velocity_seed", str(int(velocity_seed)),
        "--thermostat_seed", str(int(thermostat_seed)),
        "--neighbor_search", neighbor_search,
        "--no_log",
    ]
    _run_logged(command, cwd=outdir, log_path=log_path)

    sampled = read_sampled_distributions(sample_path, current_priors)
    sampled_by_key = {"bonds": sampled[0], "angles": sampled[1], "dihedrals": sampled[2]}
    group_metrics = {}
    ibi_values = []
    by_kind: dict[str, list[float]] = {}

    for json_key, category in groups.items():
        sim_groups = pool_requested(current_priors, sampled_by_key[json_key], json_key)
        for name, state in category.items():
            sim_group = sim_groups.get(name)
            if sim_group is None:
                raise RuntimeError(f"Validation sampling did not produce {json_key} group {name!r}")
            sim_values = np.asarray(sim_group["values"], dtype=float)
            if sim_values.size == 0:
                raise RuntimeError(f"Validation produced no samples for {json_key} group {name!r}")
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

            _counts, sim_density, sim_hist_x = histogram_density(sim_values, state["bins"])
            if not np.allclose(sim_hist_x, state["hist_x"], rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"Histogram grid changed for validation {json_key} group {name!r}")
            l1 = _distribution_l1(sim_density, state["target_density"], state["hist_x"])
            metric = {
                "kind": state["kind"],
                "mode": state["mode"],
                "samples": int(sim_values.size),
                "distribution_l1": l1,
                "sample_min": float(np.min(sim_values)),
                "sample_max": float(np.max(sim_values)),
            }
            group_metrics[f"{json_key}:{name}"] = metric
            print(f"[VALIDATE] {state['kind']} {name}: N={sim_values.size}, L1(Psim,Ptarget)={l1:.6g}")
            if state["mode"] == "ibi":
                ibi_values.append(l1)
                by_kind.setdefault(state["kind"], []).append(l1)

    if not ibi_values:
        raise ValueError("Validation priors contain no ibi_mode=ibi groups")

    immutable_hashes_after = _referenced_artifact_hashes(priors)
    if immutable_hashes_after != immutable_hashes_before:
        changed = sorted(
            set(immutable_hashes_before) | set(immutable_hashes_after)
        )
        changed = [
            path for path in changed
            if immutable_hashes_before.get(path) != immutable_hashes_after.get(path)
        ]
        raise RuntimeError(f"Read-only validation modified source prior artifacts: {changed}")

    mean_l1 = float(np.mean(ibi_values))
    max_l1 = float(np.max(ibi_values))
    mean_by_kind = {kind: float(np.mean(values)) for kind, values in sorted(by_kind.items())}
    reference = _reference_metrics(reference_summary)
    comparison = None
    if reference is not None:
        delta = mean_l1 - reference["mean_l1"]
        comparison = {
            "mean_l1_delta": delta,
            "mean_l1_ratio": mean_l1 / reference["mean_l1"] if reference["mean_l1"] != 0.0 else None,
            "relative_mean_l1_change": delta / reference["mean_l1"] if reference["mean_l1"] != 0.0 else None,
        }

    report = {
        "schema_version": 1,
        "mode": "read_only_validation",
        "dataset": str(dataset),
        "priors": str(priors),
        "config": str(config),
        "rb_info": str(rb_info),
        "ibi_config": str(ibi_config) if ibi_config is not None else None,
        "pypresso": str(pypresso),
        "neighbor_search": neighbor_search,
        "velocity_seed": int(velocity_seed),
        "thermostat_seed": int(thermostat_seed),
        "simulation": {
            "dt_ps": dt,
            "burn_in_steps": burn_in_steps,
            "production_steps": production_steps,
            "sample_interval_steps": sample_interval,
        },
        "sample": str(sample_path),
        "groups": group_metrics,
        "mean_l1": mean_l1,
        "max_l1": max_l1,
        "mean_l1_by_kind": mean_by_kind,
        "reference_best": reference,
        "comparison_to_reference_best": comparison,
        "source_artifact_sha256": immutable_hashes_after,
        "source_priors_unchanged": True,
    }
    report_path = outdir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("[IBI READ-ONLY VALIDATION SUMMARY]")
    kinds = " ".join(f"{kind}={value:.6f}" for kind, value in mean_by_kind.items())
    print(f"mean={mean_l1:.6f} max={max_l1:.6f} {kinds}")
    if reference is not None:
        print(
            f"reference_best={reference['mean_l1']:.6f} "
            f"delta={comparison['mean_l1_delta']:+.6f} "
            f"ratio={comparison['mean_l1_ratio']:.6f}"
        )
    print("[PASS] Source priors and referenced tables are byte-identical after validation sampling.")
    print(f"[DONE] Validation report: {report_path}")
    return report


def main():
    default_pypresso = ROOT / "espresso" / "build" / "pypresso"
    parser = argparse.ArgumentParser(description="Read-only validation of evaluated bonded IBI priors")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--priors", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rb_info", required=True)
    parser.add_argument("--pypresso", default=str(default_pypresso))
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--ibi-config", default=None)
    parser.add_argument("--neighbor_search", choices=("verlet", "link-cell"), default="link-cell")
    parser.add_argument("--velocity_seed", type=int, default=271828)
    parser.add_argument("--thermostat_seed", type=int, default=161803)
    parser.add_argument("--reference-summary", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    validate_priors(
        dataset=args.dataset,
        priors=args.priors,
        config=args.config,
        rb_info=args.rb_info,
        pypresso=args.pypresso,
        outdir=args.outdir,
        ibi_config=args.ibi_config,
        neighbor_search=args.neighbor_search,
        velocity_seed=args.velocity_seed,
        thermostat_seed=args.thermostat_seed,
        reference_summary=args.reference_summary,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
