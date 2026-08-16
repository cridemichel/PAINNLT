#!/usr/bin/env python3
"""Replica/window diagnostics for sigma(E) timestep scaling.

This diagnostic separates three effects that a single NVE scaling scan mixes:
1. timestep dependence;
2. finite observation-window dependence;
3. initial-state dependence.

Each replica is prepared by a short, independently seeded IBI-only Langevin NVT
branch from the same provenance-bound conservative checkpoint.  For each replica
and timestep, one NVE trajectory is run to the longest requested duration.  The
shorter-duration sigma(E) values are computed from exact prefixes of that same
energy trace, so changing the duration never changes the underlying trajectory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from framework_utils import input_hashes, sha256_file  # noqa: E402
from nve_analysis import analyze_energy_series, fit_metric_scaling, read_energy_csv  # noqa: E402

HAMILTONIAN_MODE = "conservative_classical_model_provenance_ml_disabled"


def resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    found = shutil.which(value)
    if found:
        return found
    raise FileNotFoundError(f"Executable not found: {value}")


def _run(command: list[str], *, log: Path, dry_run: bool) -> None:
    print("[CMD] " + " ".join(command))
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(command)}\n"
            + "\n".join(tail)
        )


def _require_commensurate(dts: Iterable[float], durations: Iterable[float]) -> None:
    for dt in dts:
        for duration in durations:
            steps = int(round(duration / dt))
            if steps < 2:
                raise ValueError(f"Duration {duration:g} ps is too short for dt={dt:g} ps")
            actual = steps * dt
            tol = max(1e-12, 1e-10 * max(1.0, abs(duration)))
            if not math.isclose(actual, duration, rel_tol=0.0, abs_tol=tol):
                raise ValueError(
                    "Every requested duration must be an integer number of steps for every dt; "
                    f"duration={duration:g} ps, dt={dt:g} ps gives {duration / dt:.12g} steps"
                )


def prefix_metrics(
    times_ps: Iterable[float],
    energies: Iterable[float],
    *,
    dt_ps: float,
    duration_ps: float,
) -> dict[str, Any]:
    """Analyze the exact [0,duration] prefix of an every-step energy trace."""
    t = np.asarray(list(times_ps), dtype=float)
    e = np.asarray(list(energies), dtype=float)
    if t.ndim != 1 or e.ndim != 1 or t.size != e.size:
        raise ValueError("Energy time/value arrays must be one-dimensional and equally sized")
    expected_steps = int(round(duration_ps / dt_ps))
    count = expected_steps + 1
    if t.size < count:
        raise ValueError(
            f"Energy trace has {t.size} samples but {count} are required for "
            f"duration={duration_ps:g} ps at dt={dt_ps:g} ps"
        )
    tp = t[:count]
    ep = e[:count]
    target = float(t[0] + duration_ps)
    tol = max(5e-12, 1e-8 * dt_ps)
    if not math.isclose(float(tp[-1]), target, rel_tol=0.0, abs_tol=tol):
        raise ValueError(
            f"Energy prefix endpoint mismatch: got {tp[-1]:.17g} ps, expected {target:.17g} ps"
        )
    metrics = analyze_energy_series(tp, ep)
    metrics["target_duration_ps"] = float(duration_ps)
    metrics["dt_ps"] = float(dt_ps)
    return metrics


def aggregate_sigma(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate sigma(E) across replicas at fixed duration and dt."""
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        grouped[float(row["dt_ps"])].append(float(row["sigma_E"]))
    out: list[dict[str, Any]] = []
    for dt in sorted(grouped):
        values = np.asarray(grouped[dt], dtype=float)
        if np.any(values <= 0.0) or not np.isfinite(values).all():
            raise ValueError("Replica sigma(E) values must be finite and positive")
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        out.append({
            "dt_ps": float(dt),
            "n_replicas": int(values.size),
            "sigma_mean": mean,
            "sigma_median": float(np.median(values)),
            "sigma_geometric_mean": float(np.exp(np.mean(np.log(values)))),
            "sigma_std_across_replicas": std,
            "sigma_sem_across_replicas": float(std / math.sqrt(values.size)) if values.size else math.nan,
            "sigma_cv_across_replicas": float(std / mean) if mean > 0.0 else math.nan,
            "sigma_min": float(np.min(values)),
            "sigma_max": float(np.max(values)),
        })
    return out


def fixed_effects_loglog_fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit log sigma = replica intercept + p log(dt).

    Replica-specific intercepts absorb state-dependent prefactors.  The slope is
    estimated from within-replica variation only.
    """
    by_replica: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_replica[int(row["replica_index"])].append(row)
    if len(by_replica) < 2:
        raise ValueError("Fixed-effects fit requires at least two replicas")
    x_within: list[float] = []
    y_within: list[float] = []
    intercepts: dict[str, float] = {}
    for replica, rep_rows in sorted(by_replica.items()):
        if len(rep_rows) < 3:
            raise ValueError("Each replica needs at least three timesteps for a scaling fit")
        x = np.log(np.asarray([float(r["dt_ps"]) for r in rep_rows], dtype=float))
        y = np.log(np.asarray([float(r["sigma_E"]) for r in rep_rows], dtype=float))
        if not np.isfinite(y).all():
            raise ValueError("Non-finite log sigma(E) in fixed-effects fit")
        x_within.extend((x - np.mean(x)).tolist())
        y_within.extend((y - np.mean(y)).tolist())
    xw = np.asarray(x_within, dtype=float)
    yw = np.asarray(y_within, dtype=float)
    denom = float(np.dot(xw, xw))
    if denom <= np.finfo(float).tiny:
        raise ValueError("Degenerate timestep grid in fixed-effects fit")
    slope = float(np.dot(xw, yw) / denom)
    pred = slope * xw
    ss_res = float(np.sum((yw - pred) ** 2))
    ss_tot = float(np.sum(yw * yw))
    r2_within = 1.0 if ss_tot <= np.finfo(float).eps else 1.0 - ss_res / ss_tot
    for replica, rep_rows in sorted(by_replica.items()):
        x = np.log(np.asarray([float(r["dt_ps"]) for r in rep_rows], dtype=float))
        y = np.log(np.asarray([float(r["sigma_E"]) for r in rep_rows], dtype=float))
        intercepts[str(replica)] = float(np.mean(y) - slope * np.mean(x))
    return {
        "model": "log(sigma_E[r,dt]) = alpha_replica[r] + p*log(dt)",
        "exponent_p": slope,
        "within_loglog_r2": float(r2_within),
        "n_replicas": int(len(by_replica)),
        "n_observations": int(len(rows)),
        "replica_log_intercepts": intercepts,
    }


def bootstrap_fixed_effects_slope(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        return {"samples": 0, "seed": int(seed), "p025": None, "p50": None, "p975": None}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["replica_index"])].append(dict(row))
    replica_ids = sorted(grouped)
    if len(replica_ids) < 2:
        raise ValueError("Bootstrap requires at least two replicas")
    rng = np.random.default_rng(seed)
    slopes = np.empty(samples, dtype=float)
    for i in range(samples):
        drawn = rng.choice(replica_ids, size=len(replica_ids), replace=True)
        boot_rows: list[dict[str, Any]] = []
        for synthetic_replica, source_replica in enumerate(drawn):
            for original in grouped[int(source_replica)]:
                row = dict(original)
                row["replica_index"] = int(synthetic_replica)
                boot_rows.append(row)
        slopes[i] = fixed_effects_loglog_fit(boot_rows)["exponent_p"]
    q = np.quantile(slopes, [0.025, 0.5, 0.975])
    return {
        "samples": int(samples),
        "seed": int(seed),
        "p025": float(q[0]),
        "p50": float(q[1]),
        "p975": float(q[2]),
        "slope_std": float(np.std(slopes, ddof=1)) if samples > 1 else 0.0,
    }


def _fit_aggregate(aggregate: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return fit_metric_scaling(aggregate, key, label=key)


def _replica_fits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["replica_index"])].append(row)
    fits = []
    for replica, rep_rows in sorted(grouped.items()):
        fit = fit_metric_scaling(rep_rows, "sigma_E", label="sigma_E")
        fit["replica_index"] = int(replica)
        fits.append(fit)
    return fits


def summarize_duration(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    aggregate = aggregate_sigma(rows)
    replica_fits = _replica_fits(rows)
    slopes = np.asarray([float(item["exponent_p"]) for item in replica_fits], dtype=float)
    r2 = np.asarray([float(item["loglog_r2"]) for item in replica_fits], dtype=float)
    fixed = fixed_effects_loglog_fit(rows)
    n_replicas = len(replica_fits)
    effective_bootstrap_samples = int(bootstrap_samples) if n_replicas >= 3 else 0
    fixed["bootstrap"] = bootstrap_fixed_effects_slope(
        rows, samples=effective_bootstrap_samples, seed=bootstrap_seed
    )
    fixed["bootstrap"]["requested_samples"] = int(bootstrap_samples)
    fixed["bootstrap"]["status"] = (
        "enabled" if effective_bootstrap_samples > 0 else "disabled_too_few_replicas"
    )
    return {
        "n_replicas": int(len(replica_fits)),
        "aggregate_by_dt": aggregate,
        "fit_mean_sigma": _fit_aggregate(aggregate, "sigma_mean"),
        "fit_median_sigma": _fit_aggregate(aggregate, "sigma_median"),
        "fit_geometric_mean_sigma": _fit_aggregate(aggregate, "sigma_geometric_mean"),
        "fixed_effects_fit": fixed,
        "per_replica_fits": replica_fits,
        "per_replica_slope_summary": {
            "median_p": float(np.median(slopes)),
            "min_p": float(np.min(slopes)),
            "max_p": float(np.max(slopes)),
            "median_r2": float(np.median(r2)),
            "fraction_second_order_like": float(np.mean((slopes >= 1.7) & (slopes <= 2.3) & (r2 >= 0.95))),
        },
    }


def _metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as chk:
        if "metadata_json" not in chk.files:
            raise ValueError(f"Checkpoint has no metadata_json: {path}")
        raw = np.asarray(chk["metadata_json"])
        if raw.shape != ():
            raise ValueError(f"Checkpoint metadata_json is not scalar: {path}")
        return json.loads(str(raw.item()))


def validate_replica_checkpoint(
    path: Path,
    *,
    base_checkpoint: Path,
    expected_hashes: dict[str, str | None],
    eq_dt: float,
    eq_kT: float,
    eq_steps: int,
    thermostat_seed: int,
) -> dict[str, Any]:
    metadata = _metadata(path)
    errors: list[str] = []
    if metadata.get("hamiltonian_mode") != HAMILTONIAN_MODE:
        errors.append(f"hamiltonian_mode={metadata.get('hamiltonian_mode')!r}")
    if metadata.get("sampling_ensemble") != "NVT_Langevin":
        errors.append(f"sampling_ensemble={metadata.get('sampling_ensemble')!r}")
    if metadata.get("source_checkpoint_sha256") != sha256_file(base_checkpoint):
        errors.append("source_checkpoint_sha256 mismatch")
    if metadata.get("input_hashes") != expected_hashes:
        errors.append("input_hashes mismatch")
    if int(metadata.get("completed_steps", -1)) != int(eq_steps):
        errors.append(f"completed_steps={metadata.get('completed_steps')!r}")
    if int(metadata.get("thermostat_seed", -1)) != int(thermostat_seed):
        errors.append(f"thermostat_seed={metadata.get('thermostat_seed')!r}")
    if not math.isclose(float(metadata.get("created_with_dt_ps", math.nan)), eq_dt, rel_tol=0.0, abs_tol=1e-15):
        errors.append(f"created_with_dt_ps={metadata.get('created_with_dt_ps')!r}")
    if not math.isclose(float(metadata.get("created_with_kT_kJ_mol", math.nan)), eq_kT, rel_tol=0.0, abs_tol=1e-12):
        errors.append(f"created_with_kT_kJ_mol={metadata.get('created_with_kT_kJ_mol')!r}")
    if metadata.get("ml_active") is not False or metadata.get("ml_disabled_by_flag") is not True:
        errors.append("ML-disabled checkpoint flags are inconsistent")
    if errors:
        raise ValueError(f"Replica checkpoint provenance mismatch for {path}: " + "; ".join(errors))
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "thermostat_seed": int(thermostat_seed),
        "equilibration_dt_ps": float(eq_dt),
        "equilibration_kT_kj_mol": float(eq_kT),
        "equilibration_steps": int(eq_steps),
        "equilibration_duration_ps": float(eq_steps * eq_dt),
        "source_checkpoint_sha256": metadata.get("source_checkpoint_sha256"),
    }


def collect_existing_complete_replicas(
    args: argparse.Namespace,
    *,
    expected_hashes: dict[str, str | None],
    eq_steps: int,
    max_duration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reuse only provenance-valid replicas with complete max-duration traces for every dt."""
    replicas: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for replica in range(args.replicas):
        rep_dir = args.output_dir / "replicas" / f"replica_{replica:02d}"
        seed = args.seed_base + replica
        checkpoint = rep_dir / "nvt_checkpoint.npz"
        try:
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
            rep_summary = validate_replica_checkpoint(
                checkpoint,
                base_checkpoint=args.base_checkpoint,
                expected_hashes=expected_hashes,
                eq_dt=args.replica_equilibration_dt,
                eq_kT=args.kT,
                eq_steps=eq_steps,
                thermostat_seed=seed,
            )

            traces: dict[float, tuple[np.ndarray, np.ndarray, Path]] = {}
            for dt in args.dts:
                energy = rep_dir / "nve" / f"energy_dt_{dt:.9g}.csv"
                if not energy.is_file():
                    raise FileNotFoundError(f"missing NVE energy trace for dt={dt:g}: {energy}")
                times, values = read_energy_csv(energy)
                # Validate the longest prefix before accepting the replica at all.
                prefix_metrics(times, values, dt_ps=dt, duration_ps=max_duration)
                traces[dt] = (np.asarray(times, dtype=float), np.asarray(values, dtype=float), energy)

            rep_summary["replica_index"] = int(replica)
            rep_summary["analysis_source"] = "existing_complete_replica"
            replicas.append(rep_summary)
            for dt in args.dts:
                times, values, energy = traces[dt]
                for duration in args.durations:
                    metrics = prefix_metrics(times, values, dt_ps=dt, duration_ps=duration)
                    metrics.update({
                        "replica_index": int(replica),
                        "replica_thermostat_seed": int(seed),
                        "replica_checkpoint": str(checkpoint),
                        "replica_checkpoint_sha256": rep_summary["checkpoint_sha256"],
                        "energy_file": str(energy),
                    })
                    observations.append(metrics)
            print(f"[USE] replica={replica:02d}: complete for all {len(args.dts)} dt values through {max_duration:g} ps")
        except (FileNotFoundError, ValueError) as exc:
            reason = str(exc)
            skipped.append({
                "replica_index": int(replica),
                "replica_dir": str(rep_dir),
                "reason": reason,
            })
            print(f"[SKIP] replica={replica:02d}: {reason}")

    if len(replicas) < 2:
        raise RuntimeError(
            "Existing-trace analysis requires at least two complete, provenance-valid replicas; "
            f"found {len(replicas)} of {args.replicas} requested"
        )
    return replicas, observations, skipped


def _base_run_args(args, checkpoint: Path, dt: float, steps: int, energy: Path) -> list[str]:
    return [
        args.pypresso,
        str(ROOT / "simulation" / "run_cg_md.py"),
        "--model", str(args.model), "--disable_ml",
        "--config", str(args.config),
        "--priors", str(args.priors),
        "--rb_info", str(args.rb_info),
        "--dataset", str(args.dataset),
        "--checkpoint", str(checkpoint),
        "--dt", f"{dt:.17g}",
        "--steps", str(int(steps)),
        "--log_interval", "1",
        "--device", args.device,
        "--ml_precision", args.ml_precision,
        "--neighbor_search", args.neighbor_search,
        "--energy_file", str(energy),
        "--no_vtf",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypresso", default="pypresso")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--rb-info", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--dts", type=float, nargs="+", required=True)
    parser.add_argument("--durations", type=float, nargs="+", required=True)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--replica-equilibration-dt", type=float, default=0.0005)
    parser.add_argument("--replica-equilibration-duration-ps", type=float, default=1.0)
    parser.add_argument("--kT", type=float, default=2.49)
    parser.add_argument("--seed-base", type=int, default=280000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ml-precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--neighbor-search", choices=("link-cell", "nsquare"), default="link-cell")
    parser.add_argument("--localization-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analyze-existing",
        action="store_true",
        help=(
            "Do not run NVT/NVE. Reuse only complete existing replica traces in output-dir; "
            "incomplete replicas are skipped and at least two complete replicas are required."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analyze_existing and args.overwrite:
        raise ValueError("--analyze-existing must not be combined with --overwrite")
    if args.analyze_existing and args.dry_run:
        raise ValueError("--analyze-existing performs analysis only and must not be combined with --dry-run")
    if not args.analyze_existing:
        args.pypresso = resolve_executable(args.pypresso)
    for name in ("model", "config", "priors", "rb_info", "dataset", "base_checkpoint"):
        value = Path(getattr(args, name)).expanduser().resolve()
        if not value.is_file():
            raise FileNotFoundError(value)
        setattr(args, name, value)
    manifest = Path(str(args.model) + ".manifest.json")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dts = sorted(set(float(x) for x in args.dts), reverse=True)
    args.durations = sorted(set(float(x) for x in args.durations))
    if args.replicas < 2:
        raise ValueError("At least two replicas are required")
    if len(args.dts) < 3:
        raise ValueError("At least three timesteps are required")
    if min(args.dts) <= 0.0 or min(args.durations) <= 0.0:
        raise ValueError("Timesteps and durations must be positive")
    if args.replica_equilibration_dt <= 0.0 or args.replica_equilibration_duration_ps <= 0.0:
        raise ValueError("Replica equilibration dt/duration must be positive")
    if args.kT <= 0.0:
        raise ValueError("kT must be positive")
    _require_commensurate(args.dts, args.durations)
    eq_steps = int(round(args.replica_equilibration_duration_ps / args.replica_equilibration_dt))
    if not math.isclose(eq_steps * args.replica_equilibration_dt, args.replica_equilibration_duration_ps,
                        rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Replica equilibration duration must be an integer number of equilibration steps")
    max_duration = max(args.durations)
    nve_steps_per_replica = sum(int(round(max_duration / dt)) for dt in args.dts)
    total_steps = args.replicas * (eq_steps + nve_steps_per_replica)

    if args.analyze_existing:
        if not args.output_dir.is_dir():
            raise FileNotFoundError(
                f"Existing-trace analysis requires an existing output directory: {args.output_dir}"
            )
    else:
        if args.output_dir.exists() and args.overwrite and not args.dry_run:
            shutil.rmtree(args.output_dir)
        if args.output_dir.exists() and not args.dry_run and not args.overwrite:
            raise FileExistsError(f"Output directory exists; use --overwrite: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    expected_hashes = input_hashes(
        dataset=args.dataset, config=args.config, priors=args.priors, rb_info=args.rb_info, model=args.model
    )
    plan = {
        "schema_version": 1,
        "kind": "conservative_ibi_sigma_energy_replica_window_plan",
        "diagnostic_only": True,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "dts_ps": args.dts,
        "durations_ps": args.durations,
        "max_duration_ps": float(max_duration),
        "replicas": int(args.replicas),
        "replica_equilibration": {
            "dt_ps": float(args.replica_equilibration_dt),
            "duration_ps": float(args.replica_equilibration_duration_ps),
            "steps": int(eq_steps),
            "kT_kj_mol": float(args.kT),
            "seed_base": int(args.seed_base),
        },
        "nve_steps_per_replica": int(nve_steps_per_replica),
        "total_integration_steps": int(total_steps),
        "analysis": "raw population sigma(E) on exact prefixes; replica fixed-effects log-log slope",
        "analysis_mode": "existing_complete_replicas" if args.analyze_existing else "generate_replicas_and_analyze",
    }
    plan_path = args.output_dir / ("analyze_existing_plan.json" if args.analyze_existing else "run_plan.json")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    print("[CONSERVATIVE IBI SIGMA(E) REPLICA/WINDOW PLAN]")
    print(f"base checkpoint : {args.base_checkpoint}")
    print(f"dts             : {' '.join(f'{x:g}' for x in args.dts)} ps")
    print(f"durations       : {' '.join(f'{x:g}' for x in args.durations)} ps (prefixes of one run)")
    print(f"replicas        : {args.replicas}")
    print(f"NVT branch      : dt={args.replica_equilibration_dt:g} ps, duration={args.replica_equilibration_duration_ps:g} ps, kT={args.kT:g}")
    print(f"NVE max/run     : {max_duration:g} ps")
    if args.analyze_existing:
        print("integration     : 0 new steps (reuse complete existing traces only)")
        print(f"requested reps  : {args.replicas}; incomplete replicas will be skipped")
    else:
        print(f"integration     : {total_steps} total steps ({nve_steps_per_replica} NVE + {eq_steps} NVT per replica)")
    print(f"output          : {args.output_dir}")
    print("[NOTE] sigma(E) is the raw population standard deviation; no detrending is applied.")
    print("[NOTE] Diagnostic-only. It does not modify steps 23-27 or certification artifacts.")
    if args.dry_run:
        for replica in range(args.replicas):
            seed = args.seed_base + replica
            print(f"[PLAN] replica={replica:02d} NVT seed={seed} steps={eq_steps}")
            for dt in args.dts:
                print(f"[PLAN] replica={replica:02d} NVE dt={dt:g} ps steps={int(round(max_duration/dt))}")
        print(f"[DONE] Dry-run plan: {plan_path}")
        return

    skipped_replicas: list[dict[str, Any]] = []
    if args.analyze_existing:
        replicas, observations, skipped_replicas = collect_existing_complete_replicas(
            args,
            expected_hashes=expected_hashes,
            eq_steps=eq_steps,
            max_duration=max_duration,
        )
    else:
        replicas = []
        observations = []
        for replica in range(args.replicas):
            rep_dir = args.output_dir / "replicas" / f"replica_{replica:02d}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            seed = args.seed_base + replica
            checkpoint = rep_dir / "nvt_checkpoint.npz"
            nvt_energy = rep_dir / "nvt_energy.csv"
            nvt_cmd = _base_run_args(
                args, args.base_checkpoint, args.replica_equilibration_dt, eq_steps, nvt_energy
            )
            nvt_cmd += ["--kT", f"{args.kT:.17g}", "--thermostat_seed", str(seed), "--out_checkpoint", str(checkpoint)]
            _run(nvt_cmd, log=rep_dir / "nvt.log", dry_run=False)
            rep_summary = validate_replica_checkpoint(
                checkpoint,
                base_checkpoint=args.base_checkpoint,
                expected_hashes=expected_hashes,
                eq_dt=args.replica_equilibration_dt,
                eq_kT=args.kT,
                eq_steps=eq_steps,
                thermostat_seed=seed,
            )
            rep_summary["replica_index"] = int(replica)
            rep_summary["analysis_source"] = "generated_in_this_run"
            replicas.append(rep_summary)

            for dt in args.dts:
                run_dir = rep_dir / "nve"
                run_dir.mkdir(parents=True, exist_ok=True)
                steps = int(round(max_duration / dt))
                energy = run_dir / f"energy_dt_{dt:.9g}.csv"
                cmd = _base_run_args(args, checkpoint, dt, steps, energy)
                cmd += ["--nve"]
                _run(cmd, log=run_dir / f"run_dt_{dt:.9g}.log", dry_run=False)
                times, values = read_energy_csv(energy)
                for duration in args.durations:
                    metrics = prefix_metrics(times, values, dt_ps=dt, duration_ps=duration)
                    metrics.update({
                        "replica_index": int(replica),
                        "replica_thermostat_seed": int(seed),
                        "replica_checkpoint": str(checkpoint),
                        "replica_checkpoint_sha256": rep_summary["checkpoint_sha256"],
                        "energy_file": str(energy),
                    })
                    observations.append(metrics)

    by_duration: dict[str, Any] = {}
    for duration_index, duration in enumerate(args.durations):
        rows = [row for row in observations if math.isclose(float(row["target_duration_ps"]), duration, rel_tol=0.0, abs_tol=1e-12)]
        by_duration[f"{duration:.12g}"] = summarize_duration(
            rows,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + duration_index,
        )
        by_duration[f"{duration:.12g}"]["duration_ps"] = float(duration)

    legacy_reference = None
    if args.localization_report is not None:
        loc = args.localization_report.expanduser().resolve()
        if loc.is_file():
            data = json.loads(loc.read_text(encoding="utf-8"))
            full_fit = data.get("sigma_scans", {}).get("full", {}).get("sigma_fit")
            legacy_reference = {
                "path": str(loc),
                "sha256": sha256_file(loc),
                "full_fine_sigma_fit": full_fit,
            }

    report = {
        "schema_version": 1,
        "kind": "conservative_ibi_sigma_energy_replica_window_diagnostic",
        "diagnostic_only": True,
        "hamiltonian_mode": HAMILTONIAN_MODE,
        "model_active": False,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "input_hashes": expected_hashes,
        "dts_ps": args.dts,
        "durations_ps": args.durations,
        "analysis_mode": "existing_complete_replicas" if args.analyze_existing else "generated_replicas",
        "replicas_requested": int(args.replicas),
        "replicas_used": int(len(replicas)),
        "replicas": replicas,
        "skipped_replicas": skipped_replicas,
        "analysis_notes": [
            "sigma_E is numpy population standard deviation (ddof=0) of E_tot; no detrending is applied.",
            "Each replica/dt trajectory is run once to the maximum duration; shorter windows are exact prefixes of that trajectory.",
            "Replica fixed-effects fits estimate a common log-log slope while allowing one log-prefactor per initial-state replica.",
            "Replica checkpoints are independently Langevin-branched from the same conservative IBI-only checkpoint using distinct thermostat seeds.",
            "When --analyze-existing is used, no dynamics are launched; only complete provenance-valid replicas are included.",
            "Replica bootstrap intervals are disabled when fewer than three replicas are available; fixed-effects and per-replica fits are still reported.",
        ],
        "by_duration": by_duration,
        "localization_reference": legacy_reference,
    }
    target = args.output_dir / "sigma_energy_replica_report.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    observations_csv = args.output_dir / "sigma_energy_replica_observations.csv"
    obs_fields = [
        "replica_index", "replica_thermostat_seed", "target_duration_ps", "dt_ps", "sigma_E",
        "rms_delta_E", "relative_block_mean_drift", "samples", "energy_file", "replica_checkpoint_sha256",
    ]
    with observations_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=obs_fields)
        writer.writeheader()
        for row in sorted(observations, key=lambda r: (r["target_duration_ps"], r["replica_index"], -r["dt_ps"])):
            writer.writerow({key: row.get(key) for key in obs_fields})

    aggregate_csv = args.output_dir / "sigma_energy_replica_aggregate.csv"
    agg_fields = [
        "duration_ps", "dt_ps", "n_replicas", "sigma_mean", "sigma_median", "sigma_geometric_mean",
        "sigma_std_across_replicas", "sigma_sem_across_replicas", "sigma_cv_across_replicas", "sigma_min", "sigma_max",
    ]
    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=agg_fields)
        writer.writeheader()
        for key in sorted(by_duration, key=float):
            duration_data = by_duration[key]
            for row in duration_data["aggregate_by_dt"]:
                writer.writerow({"duration_ps": duration_data["duration_ps"], **row})

    print("[SIGMA(E) REPLICA/WINDOW SUMMARY]")
    print(f"replicas used: {len(replicas)}/{args.replicas}; skipped: {len(skipped_replicas)}")
    for key in sorted(by_duration, key=float):
        data = by_duration[key]
        fixed = data["fixed_effects_fit"]
        boot = fixed["bootstrap"]
        mean_fit = data["fit_mean_sigma"]
        rep = data["per_replica_slope_summary"]
        if boot["status"] == "enabled":
            ci_text = f"95%CI=[{boot['p025']:.6f},{boot['p975']:.6f}]"
        else:
            ci_text = "bootstrap=disabled(n_replicas<3)"
        print(
            f"T={data['duration_ps']:g} ps: fixed-effects p={fixed['exponent_p']:.6f} "
            f"R2_within={fixed['within_loglog_r2']:.6f} "
            f"{ci_text} "
            f"mean-sigma p={mean_fit['exponent_p']:.6f} "
            f"replica median p={rep['median_p']:.6f} range=[{rep['min_p']:.6f},{rep['max_p']:.6f}]"
        )
    print(f"[DONE] report: {target}")
    print(f"       observations: {observations_csv}")
    print(f"       aggregate:    {aggregate_csv}")


if __name__ == "__main__":
    main()
