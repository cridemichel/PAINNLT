#!/usr/bin/env python3
"""Matched short-MD validation for regularized conservative IBI angle candidates.

The workflow is intentionally diagnostic-only.  It compares the selected current
IBI priors against one or more offline regularization candidates using the same
NVT protocol, structural histograms, and the historical coarse NVE timestep scan.
No candidate is promoted automatically.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import CubicHermiteSpline

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "simulation", ROOT / "ibi"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from nve_analysis import analyze_energy_series, read_energy_csv  # noqa: E402
from build_dbi_priors import load_continuation_priors  # noqa: E402
from conservative_spline import load_conservative_spline  # noqa: E402
from geometry_io import pool_requested, read_sampled_distributions  # noqa: E402
from ibi_core import histogram_density, normalize_density  # noqa: E402

SCHEMA_VERSION = 1


def _distribution_l1(a, b, grid) -> float:
    aa = normalize_density(np.asarray(a, dtype=float), np.asarray(grid, dtype=float))
    bb = normalize_density(np.asarray(b, dtype=float), np.asarray(grid, dtype=float))
    return float(np.trapezoid(np.abs(aa - bb), np.asarray(grid, dtype=float)))


def fit_sigma_range(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    good = [r for r in runs if r.get("status", "ok") == "ok"]
    good = sorted(good, key=lambda r: float(r["dt_ps"]))
    if len(good) < 3:
        return {"available": False, "reason": "fewer_than_three_successful_dt_values", "n_points": len(good)}
    dt = np.asarray([float(r["dt_ps"]) for r in good], dtype=float)
    sigma = np.asarray([float(r["sigma_E"]) for r in good], dtype=float)
    if np.any(dt <= 0) or np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        return {"available": False, "reason": "nonpositive_or_nonfinite_sigma", "n_points": len(good)}
    x = np.log(dt)
    y = np.log(sigma)
    p, logc = np.polyfit(x, y, 1)
    pred = p * x + logc
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= np.finfo(float).eps else 1.0 - ss_res / ss_tot
    c2 = sigma / (dt * dt)
    local = []
    for i in range(len(dt) - 1):
        lp = math.log(sigma[i + 1] / sigma[i]) / math.log(dt[i + 1] / dt[i])
        local.append({
            "dt_low_ps": float(dt[i]),
            "dt_high_ps": float(dt[i + 1]),
            "local_exponent_p": float(lp),
            "c2_low": float(c2[i]),
            "c2_high": float(c2[i + 1]),
        })
    return {
        "available": True,
        "n_points": int(len(dt)),
        "dt_ps": dt.tolist(),
        "sigma_E": sigma.tolist(),
        "sigma_over_dt2": c2.tolist(),
        "fit": {
            "model": "sigma_E = C * dt^p",
            "exponent_p": float(p),
            "prefactor_C": float(math.exp(logc)),
            "loglog_r2": float(r2),
        },
        "c2_spread_max_over_min": float(np.max(c2) / np.min(c2)),
        "adjacent_local_exponents": local,
        "max_clean_dt_factor_1p5": largest_clean_dt(dt, c2, 1.5),
        "max_clean_dt_factor_2": largest_clean_dt(dt, c2, 2.0),
    }


def largest_clean_dt(dt: np.ndarray, c2: np.ndarray, factor: float) -> float:
    """Largest contiguous dt from the smallest point with C2 within a factor of C2(dt_min)."""
    dt = np.asarray(dt, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    if dt.size == 0 or factor <= 1.0:
        raise ValueError("Need non-empty data and factor > 1")
    order = np.argsort(dt)
    dt = dt[order]
    c2 = c2[order]
    ref = c2[0]
    lo, hi = ref / factor, ref * factor
    best = float(dt[0])
    for d, value in zip(dt[1:], c2[1:]):
        if not (lo <= value <= hi):
            break
        best = float(d)
    return best


def _summarize_l1(group_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not group_rows:
        return {"n_groups": 0, "weighted_mean_l1": math.nan, "mean_l1": math.nan, "max_l1": math.nan}
    vals = np.asarray([float(v["l1_runtime_vs_target"]) for v in group_rows.values()], dtype=float)
    weights = np.asarray([int(v["target_samples"]) for v in group_rows.values()], dtype=float)
    return {
        "n_groups": int(len(vals)),
        "weighted_mean_l1": float(np.average(vals, weights=weights)),
        "mean_l1": float(np.mean(vals)),
        "max_l1": float(np.max(vals)),
    }


def structural_report(dataset: Path, priors_path: Path, ibi_config: Path, sample_npz: Path) -> dict[str, Any]:
    state = load_continuation_priors(
        dataset, priors_path, ibi_config=ibi_config, allow_conservative_spline=True
    )
    priors = state["priors"]
    sampled = read_sampled_distributions(sample_npz, priors)
    out: dict[str, Any] = {"bonds": {}, "angles": {}}

    for kind, key, sampled_values in (("bonds", "bonds", sampled[0]), ("angles", "angles", sampled[1])):
        target_groups = state["groups"][key]
        runtime_groups = pool_requested(priors, sampled_values, key)
        rows: dict[str, Any] = {}
        for name, target in sorted(target_groups.items()):
            runtime = runtime_groups.get(name)
            if runtime is None:
                raise RuntimeError(f"Runtime sample is missing {kind} group {name!r}")
            values = np.asarray(runtime["values"], dtype=float)
            _counts, density, centers = histogram_density(values, target["bins"])
            if not np.allclose(centers, target["hist_x"], atol=1e-12, rtol=0.0):
                raise RuntimeError(f"Histogram grid mismatch for {kind} group {name!r}")
            rows[name] = {
                "target_samples": int(len(target["values"])),
                "runtime_samples": int(len(values)),
                "l1_runtime_vs_target": _distribution_l1(density, target["target_density"], centers),
            }
        out[kind] = {"groups": rows, "summary": _summarize_l1(rows)}

    # Re-evaluate actual angle curvature on the NVT sample, because a candidate may
    # shift the sampled ensemble relative to the offline screen from step 30.
    angle_groups = pool_requested(priors, sampled[1], "angles")
    aggregate_u2: list[float] = []
    per_group_u2: dict[str, Any] = {}
    for name, group in sorted(angle_groups.items()):
        entries = [priors["angles"][idx] for idx in group["indices"]]
        paths = []
        for entry in entries:
            table = load_conservative_spline(entry, kind="angle", priors_path=priors_path)
            paths.append(table.path.resolve())
        if len(set(paths)) != 1:
            raise ValueError(f"Angle group {name!r} references multiple conservative tables")
        table = load_conservative_spline(entries[0], kind="angle", priors_path=priors_path)
        spline = CubicHermiteSpline(table.x, table.energy, table.derivative)
        values = np.asarray(group["values"], dtype=float)
        u2 = np.asarray(spline(values, 2), dtype=float)
        aggregate_u2.extend(u2.tolist())
        per_group_u2[name] = {
            "n": int(len(u2)),
            "p95_abs": float(np.percentile(np.abs(u2), 95)),
            "p99_abs": float(np.percentile(np.abs(u2), 99)),
            "max_abs": float(np.max(np.abs(u2))),
        }
    aggregate = np.asarray(aggregate_u2, dtype=float)
    out["angle_curvature_runtime"] = {
        "n": int(len(aggregate)),
        "p95_abs": float(np.percentile(np.abs(aggregate), 95)),
        "p99_abs": float(np.percentile(np.abs(aggregate), 99)),
        "max_abs": float(np.max(np.abs(aggregate))),
        "groups": per_group_u2,
    }
    return out


def nvt_energy_summary(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "E_kin" not in (reader.fieldnames or []):
            return {"available": False, "reason": "E_kin_column_missing"}
        vals = [float(row["E_kin"]) for row in reader]
    if not vals:
        return {"available": False, "reason": "no_rows"}
    a = np.asarray(vals, dtype=float)
    half = max(0, len(a) // 2)
    return {
        "available": True,
        "samples": int(len(a)),
        "mean_E_kin": float(np.mean(a)),
        "mean_E_kin_second_half": float(np.mean(a[half:])),
        "final_E_kin": float(a[-1]),
    }


def _parse_variant(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--variant must be NAME=PATH")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in name):
        raise argparse.ArgumentTypeError(f"Invalid variant name {name!r}")
    return name, Path(path).expanduser().resolve()


def _run(cmd: list[str], log_path: Path, *, allow_failure: bool = False) -> tuple[bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[CMD] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode == 0:
        return True, ""
    tail = "\n".join((proc.stdout or "").splitlines()[-30:])
    if allow_failure:
        print(f"[WARN] command failed ({proc.returncode}); keeping diagnostic failure and continuing", file=sys.stderr)
        if tail:
            print(tail, file=sys.stderr)
        return False, tail
    raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")


def _trace_complete(path: Path, steps: int, duration: float) -> bool:
    if not path.is_file():
        return False
    try:
        t, _e = read_energy_csv(path)
    except Exception:
        return False
    tol = max(1e-12, 1e-9 * max(1.0, duration))
    return len(t) == steps + 1 and abs(float(t[-1] - t[0]) - duration) <= tol


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pypresso", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--rb-info", type=Path, required=True)
    p.add_argument("--source-checkpoint", type=Path, required=True)
    p.add_argument("--ibi-config", type=Path, required=True)
    p.add_argument("--variant", action="append", required=True, help="NAME=PATH; first/current variant must be named current")
    p.add_argument("--dts", type=float, nargs="+", required=True)
    p.add_argument("--duration-ps", type=float, required=True)
    p.add_argument("--branch-dt", type=float, required=True)
    p.add_argument("--branch-duration-ps", type=float, required=True)
    p.add_argument("--kT", type=float, required=True)
    p.add_argument("--thermostat-seed", type=int, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--ml-precision", required=True)
    p.add_argument("--neighbor-search", choices=("verlet", "link-cell"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.model = args.model.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.dataset = args.dataset.expanduser().resolve()
    args.rb_info = args.rb_info.expanduser().resolve()
    args.source_checkpoint = args.source_checkpoint.expanduser().resolve()
    args.ibi_config = args.ibi_config.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    variants = [_parse_variant(v) for v in args.variant]
    if variants[0][0] != "current":
        raise ValueError("The first variant must be named current")
    if len({n for n, _ in variants}) != len(variants):
        raise ValueError("Variant names must be unique")
    for p in [args.model, args.config, args.dataset, args.rb_info, args.source_checkpoint, args.ibi_config, *[p for _, p in variants]]:
        if not p.is_file():
            raise FileNotFoundError(p)
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if len(args.dts) < 3 or min(args.dts) <= 0 or args.duration_ps <= 0 or args.branch_dt <= 0 or args.branch_duration_ps <= 0:
        raise ValueError("Positive durations and at least three positive dt values are required")

    branch_steps = int(round(args.branch_duration_ps / args.branch_dt))
    sample_start = branch_steps // 2
    nve_steps_per_variant = sum(int(round(args.duration_ps / dt)) for dt in args.dts)
    total_steps = len(variants) * (branch_steps + nve_steps_per_variant)

    print("[IBI ANGLE CANDIDATE MATCHED VALIDATION PLAN]")
    print("variants       : " + " / ".join(n for n, _ in variants))
    print("dt scan        : " + " ".join(f"{x:g}" for x in args.dts) + " ps")
    print(f"NVE duration   : {args.duration_ps:g} ps per dt")
    print(f"NVT branch     : {args.branch_duration_ps:g} ps at dt={args.branch_dt:g} ps kT={args.kT:g}")
    print(f"integration    : about {total_steps} total steps")
    print(f"output         : {args.output_dir}")
    print("[NOTE] Diagnostic-only. Candidates remain unvalidated; no priors are promoted.")
    if args.dry_run:
        return

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    elif args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; use --resume or --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ibi_angle_regularization_matched_validation",
        "diagnostic_only": True,
        "validated": False,
        "dts_ps": [float(x) for x in args.dts],
        "duration_ps": float(args.duration_ps),
        "branch_dt_ps": float(args.branch_dt),
        "branch_duration_ps": float(args.branch_duration_ps),
        "kT": float(args.kT),
        "estimated_total_steps": int(total_steps),
        "variants": {},
    }

    for variant_index, (name, priors) in enumerate(variants):
        vdir = args.output_dir / name
        vdir.mkdir(parents=True, exist_ok=True)
        checkpoint = vdir / "nvt_checkpoint.npz"
        sample = vdir / "nvt_structured_sample.npz"
        nvt_energy = vdir / "nvt_energy.csv"
        nvt_complete = args.resume and checkpoint.is_file() and sample.is_file() and nvt_energy.is_file()
        if nvt_complete:
            try:
                read_sampled_distributions(sample, json.loads(priors.read_text()))
            except Exception:
                nvt_complete = False
        if nvt_complete:
            print(f"[REUSE] {name}: existing NVT checkpoint/sample")
        else:
            if vdir.exists() and not args.resume:
                # directory is new/empty here; no action required
                pass
            nvt_cmd = [
                args.pypresso, str(ROOT / "simulation" / "run_cg_md.py"),
                "--model", str(args.model), "--disable_ml", "--config", str(args.config),
                "--priors", str(priors), "--rb_info", str(args.rb_info), "--dataset", str(args.dataset),
                "--checkpoint", str(args.source_checkpoint), "--allow_checkpoint_mismatch",
                "--dt", f"{args.branch_dt:.17g}", "--steps", str(branch_steps), "--log_interval", "1",
                "--device", args.device, "--ml_precision", args.ml_precision,
                "--neighbor_search", args.neighbor_search, "--energy_file", str(nvt_energy), "--no_vtf",
                "--kT", f"{args.kT:.17g}", "--thermostat_seed", str(args.thermostat_seed),
                "--out_checkpoint", str(checkpoint), "--sample_npz", str(sample),
                "--sample_start_step", str(sample_start),
            ]
            _run(nvt_cmd, vdir / "nvt.log")

        struct = structural_report(args.dataset, priors, args.ibi_config, sample)
        (vdir / "structural_report.json").write_text(json.dumps(struct, indent=2, sort_keys=True) + "\n")
        nvt_kin = nvt_energy_summary(nvt_energy)
        print(
            f"[STRUCT] {name:15s} angleL1={struct['angles']['summary']['weighted_mean_l1']:.6f} "
            f"bondL1={struct['bonds']['summary']['weighted_mean_l1']:.6f} "
            f"angleP99U2={struct['angle_curvature_runtime']['p99_abs']:.6g}"
        )

        nve_dir = vdir / "nve_coarse_scan"
        nve_dir.mkdir(parents=True, exist_ok=True)
        runs: list[dict[str, Any]] = []
        for dt in sorted(args.dts, reverse=True):
            steps = int(round(args.duration_ps / dt))
            actual_duration = steps * dt
            rdir = nve_dir / f"dt_{dt:.8g}".replace(".", "p")
            energy = rdir / "energy.csv"
            log = rdir / "run.log"
            reuse = args.resume and _trace_complete(energy, steps, actual_duration)
            if reuse:
                print(f"[REUSE] {name}: dt={dt:g} ps ({steps} steps)")
                ok = True
                error_tail = ""
            else:
                if rdir.exists():
                    shutil.rmtree(rdir)
                rdir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    args.pypresso, str(ROOT / "simulation" / "run_cg_md.py"),
                    "--model", str(args.model), "--disable_ml", "--config", str(args.config),
                    "--priors", str(priors), "--rb_info", str(args.rb_info), "--dataset", str(args.dataset),
                    "--checkpoint", str(checkpoint), "--dt", f"{dt:.17g}", "--steps", str(steps),
                    "--log_interval", "1", "--device", args.device, "--ml_precision", args.ml_precision,
                    "--neighbor_search", args.neighbor_search, "--energy_file", str(energy), "--no_vtf", "--nve",
                ]
                ok, error_tail = _run(cmd, log, allow_failure=True)
            if not ok or not _trace_complete(energy, steps, actual_duration):
                runs.append({
                    "status": "failed", "dt_ps": float(dt), "steps": int(steps),
                    "actual_duration_ps": float(actual_duration), "energy_csv": str(energy),
                    "run_log": str(log), "error_tail": error_tail,
                })
                print(f"[NVE-FAIL] {name:15s} dt={dt:g}")
                continue
            times, energies = read_energy_csv(energy)
            metrics = analyze_energy_series(times, energies)
            metrics.update({
                "status": "ok", "dt_ps": float(dt), "steps": int(steps),
                "actual_duration_ps": float(actual_duration), "energy_csv": str(energy), "run_log": str(log),
            })
            runs.append(metrics)
            print(
                f"[NVE] {name:15s} dt={dt:g} sigma_E={metrics['sigma_E']:.6g} "
                f"rel_block_drift={metrics['relative_block_mean_drift']:.3e}"
            )

        sigma = fit_sigma_range(runs)
        variant_report = {
            "priors": str(priors),
            "checkpoint": str(checkpoint),
            "sample": str(sample),
            "nvt_kinetic_energy": nvt_kin,
            "structural": struct,
            "nve_runs": runs,
            "sigma_range": sigma,
        }
        report["variants"][name] = variant_report
        (nve_dir / "coarse_scan_report.json").write_text(
            json.dumps({"schema_version": 1, "kind": "ibi_angle_candidate_coarse_nve", "variant": name, "runs": runs, "sigma_range": sigma}, indent=2, sort_keys=True) + "\n"
        )
        if sigma.get("available"):
            print(
                f"[RESULT] {name:15s} p={sigma['fit']['exponent_p']:.6f} "
                f"R2={sigma['fit']['loglog_r2']:.6f} C2spread={sigma['c2_spread_max_over_min']:.3f} "
                f"clean1.5x={sigma['max_clean_dt_factor_1p5']:.6g} ps"
            )
        else:
            print(f"[RESULT] {name:15s} NVE fit unavailable: {sigma.get('reason')}")

    current = report["variants"]["current"]
    comparisons: dict[str, Any] = {}
    cur_ang = float(current["structural"]["angles"]["summary"]["weighted_mean_l1"])
    cur_bond = float(current["structural"]["bonds"]["summary"]["weighted_mean_l1"])
    cur_u2 = float(current["structural"]["angle_curvature_runtime"]["p99_abs"])
    cur_sigma = current["sigma_range"]
    for name, _priors in variants[1:]:
        cand = report["variants"][name]
        cand_ang = float(cand["structural"]["angles"]["summary"]["weighted_mean_l1"])
        cand_bond = float(cand["structural"]["bonds"]["summary"]["weighted_mean_l1"])
        cand_u2 = float(cand["structural"]["angle_curvature_runtime"]["p99_abs"])
        cs = cand["sigma_range"]
        row: dict[str, Any] = {
            "delta_angle_weighted_l1": cand_ang - cur_ang,
            "delta_bond_weighted_l1": cand_bond - cur_bond,
            "runtime_angle_p99_curvature_reduction": cur_u2 / cand_u2 if cand_u2 > 0 else math.inf,
        }
        if cur_sigma.get("available") and cs.get("available"):
            row.update({
                "abs_p_minus_2_current": abs(float(cur_sigma["fit"]["exponent_p"]) - 2.0),
                "abs_p_minus_2_candidate": abs(float(cs["fit"]["exponent_p"]) - 2.0),
                "c2_spread_improvement_factor": float(cur_sigma["c2_spread_max_over_min"]) / float(cs["c2_spread_max_over_min"]),
                "current_max_clean_dt_factor_1p5": float(cur_sigma["max_clean_dt_factor_1p5"]),
                "candidate_max_clean_dt_factor_1p5": float(cs["max_clean_dt_factor_1p5"]),
            })
        comparisons[name] = row
        print(
            f"[COMPARE] {name:15s} dAngleL1={row['delta_angle_weighted_l1']:+.6f} "
            f"dBondL1={row['delta_bond_weighted_l1']:+.6f} P99U2red={row['runtime_angle_p99_curvature_reduction']:.3f}x"
            + (f" C2spreadImprove={row.get('c2_spread_improvement_factor', math.nan):.3f}x" if "c2_spread_improvement_factor" in row else "")
        )
    report["comparisons_vs_current"] = comparisons

    out = args.output_dir / "angle_candidate_validation_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[DONE] report: {out}")
    print("[NOTE] This is a matched diagnostic only. Do not promote a candidate from this report alone.")


if __name__ == "__main__":
    main()
