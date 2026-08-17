#!/usr/bin/env python3
"""Local matched-MD sweep around the useful IBI angle smoothing scale.

Step 31 showed that 0.01 rad smoothing improves the contiguous quadratic-energy
range, whereas more aggressive 0.02 rad smoothing lowers curvature further but
produces a less coherent sigma_E/dt^2 sequence.  This diagnostic therefore
samples a narrow neighborhood around 0.01 rad and ranks candidates primarily by
contiguous sigma_E/dt^2 behavior, not by the global log-log exponent alone.

The already-computed 0.01-rad result is reused from the step-31 report and is
re-fitted on exactly the same dt subset as the new candidates.  No candidate is
promoted automatically.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "simulation", ROOT / "ibi"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_angle_smoothing_candidate import generate_candidate, smoothing_name  # noqa: E402
from angle_regularization_diagnostics import sha256_file  # noqa: E402
from ibi_angle_candidate_validation import fit_sigma_range  # noqa: E402

SCHEMA_VERSION = 1
KIND = "ibi_angle_smoothing_local_sweep"


def _parse_sigma_list(values: list[float]) -> list[float]:
    out = sorted({float(v) for v in values})
    if not out or any((not math.isfinite(v) or v <= 0.0) for v in out):
        raise ValueError("All smoothing sigma values must be positive and finite")
    return out


def _dt_match(value: float, requested: float) -> bool:
    return math.isclose(float(value), float(requested), rel_tol=0.0, abs_tol=1e-12)


def subset_runs(runs: list[Mapping[str, Any]], dts: list[float]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dt in dts:
        matches = [dict(row) for row in runs if row.get("status", "ok") == "ok" and _dt_match(float(row["dt_ps"]), dt)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one successful baseline run at dt={dt:g}, found {len(matches)}")
        selected.append(matches[0])
    return selected


def clean_prefix_metrics(sigma_range: Mapping[str, Any], factor: float = 1.5) -> dict[str, Any]:
    if not sigma_range.get("available"):
        return {"available": False, "reason": sigma_range.get("reason", "sigma_range_unavailable")}
    dt = np.asarray(sigma_range["dt_ps"], dtype=float)
    sigma = np.asarray(sigma_range["sigma_E"], dtype=float)
    clean_dt = float(sigma_range["max_clean_dt_factor_1p5"] if factor == 1.5 else sigma_range["max_clean_dt_factor_2"])
    mask = dt <= clean_dt + 1e-15
    dtp = dt[mask]
    sigp = sigma[mask]
    c2 = sigp / (dtp * dtp)
    out: dict[str, Any] = {
        "available": True,
        "factor": float(factor),
        "max_dt_ps": clean_dt,
        "n_points": int(len(dtp)),
        "dt_ps": dtp.tolist(),
        "sigma_over_dt2": c2.tolist(),
        "c2_spread_max_over_min": float(np.max(c2) / np.min(c2)),
    }
    if len(dtp) >= 3:
        fit = fit_sigma_range([
            {"status": "ok", "dt_ps": float(d), "sigma_E": float(s)} for d, s in zip(dtp, sigp)
        ])
        out["fit"] = fit["fit"]
        out["abs_p_minus_2"] = abs(float(fit["fit"]["exponent_p"]) - 2.0)
    else:
        out["fit"] = None
        out["abs_p_minus_2"] = math.nan
    return out


def _candidate_row(variant: Mapping[str, Any], *, sigma_rad: float, source: str, current: Mapping[str, Any], dts: list[float]) -> dict[str, Any]:
    runs = subset_runs(list(variant["nve_runs"]), dts)
    sigma_range = fit_sigma_range(runs)
    struct = variant["structural"]
    current_struct = current["structural"]
    current_u2 = float(current_struct["angle_curvature_runtime"]["p99_abs"])
    cand_u2 = float(struct["angle_curvature_runtime"]["p99_abs"])
    row = {
        "sigma_rad": float(sigma_rad),
        "source": source,
        "priors": variant.get("priors"),
        "checkpoint": variant.get("checkpoint"),
        "nvt_kinetic_energy": variant.get("nvt_kinetic_energy"),
        "structural": struct,
        "nve_runs_common_grid": runs,
        "sigma_range_common_grid": sigma_range,
        "clean_prefix_factor_1p5": clean_prefix_metrics(sigma_range, 1.5),
        "delta_angle_weighted_l1_vs_current": (
            float(struct["angles"]["summary"]["weighted_mean_l1"])
            - float(current_struct["angles"]["summary"]["weighted_mean_l1"])
        ),
        "delta_bond_weighted_l1_vs_current": (
            float(struct["bonds"]["summary"]["weighted_mean_l1"])
            - float(current_struct["bonds"]["summary"]["weighted_mean_l1"])
        ),
        "runtime_angle_p99_curvature_reduction_vs_current": current_u2 / cand_u2 if cand_u2 > 0 else math.inf,
    }
    return row


def nve_rank_key(item: tuple[str, Mapping[str, Any]]) -> tuple[float, ...]:
    _name, row = item
    sr = row["sigma_range_common_grid"]
    prefix = row["clean_prefix_factor_1p5"]
    clean = float(sr["max_clean_dt_factor_1p5"])
    prefix_spread = float(prefix["c2_spread_max_over_min"])
    prefix_abs_p = float(prefix["abs_p_minus_2"])
    if not math.isfinite(prefix_abs_p):
        prefix_abs_p = 1.0e9
    global_abs_p = abs(float(sr["fit"]["exponent_p"]) - 2.0)
    global_r2 = float(sr["fit"]["loglog_r2"])
    global_spread = float(sr["c2_spread_max_over_min"])
    # Primary goal: extend a contiguous C2 plateau.  Only then prefer a flatter
    # plateau and cleaner second-order fit.  Structural deltas are reported but
    # intentionally not folded into this NVE-only ranking.
    return (-clean, prefix_spread, prefix_abs_p, global_spread, global_abs_p, -global_r2)


def _run_subvalidation(
    *,
    args: argparse.Namespace,
    name: str,
    priors: Path,
    output_dir: Path,
) -> dict[str, Any]:
    cmd = [
        args.python_bin,
        str(ROOT / "simulation" / "ibi_angle_candidate_validation.py"),
        "--pypresso", args.pypresso,
        "--model", str(args.model),
        "--config", str(args.config),
        "--dataset", str(args.dataset),
        "--rb-info", str(args.rb_info),
        "--source-checkpoint", str(args.source_checkpoint),
        "--ibi-config", str(args.ibi_config),
        "--variant", f"current={priors}",
        "--dts", *[f"{dt:.17g}" for dt in args.dts],
        "--duration-ps", f"{args.duration_ps:.17g}",
        "--branch-dt", f"{args.branch_dt:.17g}",
        "--branch-duration-ps", f"{args.branch_duration_ps:.17g}",
        "--kT", f"{args.kT:.17g}",
        "--thermostat-seed", str(args.thermostat_seed),
        "--device", args.device,
        "--ml-precision", args.ml_precision,
        "--neighbor-search", args.neighbor_search,
        "--output-dir", str(output_dir),
    ]
    if args.resume:
        cmd.append("--resume")
    elif args.overwrite:
        cmd.append("--overwrite")
    print(f"[RUN] {name}: matched NVT + coarse NVE", flush=True)
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Subvalidation failed for {name} with exit code {proc.returncode}")
    report_path = output_dir / "angle_candidate_validation_report.json"
    report = json.loads(report_path.read_text())
    return report["variants"]["current"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--pypresso", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--rb-info", type=Path, required=True)
    p.add_argument("--source-checkpoint", type=Path, required=True)
    p.add_argument("--ibi-config", type=Path, required=True)
    p.add_argument("--source-priors", type=Path, required=True)
    p.add_argument("--step31-report", type=Path, required=True)
    p.add_argument("--new-sigmas", type=float, nargs="+", required=True)
    p.add_argument("--reuse-sigma", type=float, required=True)
    p.add_argument("--reuse-variant", required=True)
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
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    args.new_sigmas = _parse_sigma_list(args.new_sigmas)
    args.dts = sorted({float(dt) for dt in args.dts})
    if len(args.dts) < 3 or min(args.dts) <= 0:
        raise ValueError("At least three positive dt values are required")
    if any(math.isclose(s, args.reuse_sigma, rel_tol=0.0, abs_tol=1e-12) for s in args.new_sigmas):
        raise ValueError("reuse_sigma must not also appear in --new-sigmas")
    for attr in ("model", "config", "dataset", "rb_info", "source_checkpoint", "ibi_config", "source_priors", "step31_report"):
        path = getattr(args, attr).expanduser().resolve()
        setattr(args, attr, path)
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir = args.output_dir.expanduser().resolve()

    step31 = json.loads(args.step31_report.read_text())
    if args.reuse_variant not in step31.get("variants", {}):
        raise KeyError(f"Step-31 report has no variant {args.reuse_variant!r}")
    if "current" not in step31.get("variants", {}):
        raise KeyError("Step-31 report has no current reference")
    if not math.isclose(float(step31.get("branch_dt_ps", math.nan)), args.branch_dt, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Step-31 branch_dt does not match requested sweep protocol")
    if not math.isclose(float(step31.get("branch_duration_ps", math.nan)), args.branch_duration_ps, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Step-31 branch duration does not match requested sweep protocol")
    if not math.isclose(float(step31.get("kT", math.nan)), args.kT, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Step-31 kT does not match requested sweep protocol")
    if not math.isclose(float(step31.get("duration_ps", math.nan)), args.duration_ps, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Step-31 NVE duration does not match requested sweep protocol")

    current_ref = step31["variants"]["current"]
    current_priors_path = Path(str(current_ref.get("priors", ""))).expanduser()
    if current_priors_path.is_file() and sha256_file(current_priors_path) != sha256_file(args.source_priors):
        raise ValueError("Step-31 current priors do not match the selected source priors")

    reused_variant = step31["variants"][args.reuse_variant]
    reused_priors_path = Path(str(reused_variant.get("priors", ""))).expanduser()
    if not reused_priors_path.is_file():
        raise FileNotFoundError(f"Reused step-31 candidate priors not found: {reused_priors_path}")
    reused_payload = json.loads(reused_priors_path.read_text())
    reused_meta = reused_payload.get("regularization_candidate", {})
    if not math.isclose(float(reused_meta.get("body_sigma_rad", math.nan)), args.reuse_sigma, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Reused step-31 candidate sigma does not match --reuse-sigma")
    expected_source_hash = reused_meta.get("source_priors_sha256")
    if expected_source_hash and str(expected_source_hash) != sha256_file(args.source_priors):
        raise ValueError("Reused step-31 candidate was generated from different source priors")

    # Fail now if the reused baseline does not contain the requested common grid.
    subset_runs(reused_variant["nve_runs"], args.dts)

    branch_steps = int(round(args.branch_duration_ps / args.branch_dt))
    nve_steps = sum(int(round(args.duration_ps / dt)) for dt in args.dts)
    new_steps = len(args.new_sigmas) * (branch_steps + nve_steps)
    names = [(sigma, smoothing_name(sigma).removesuffix("_wall_current")) for sigma in args.new_sigmas]

    print("[IBI ANGLE SMOOTHING LOCAL SWEEP PLAN]")
    print("new sigmas     : " + " ".join(f"{s:g}" for s in args.new_sigmas) + " rad")
    print(f"reused sigma   : {args.reuse_sigma:g} rad from {args.step31_report}")
    print("common dt grid : " + " ".join(f"{dt:g}" for dt in args.dts) + " ps")
    print(f"NVE duration   : {args.duration_ps:g} ps per dt")
    print(f"NVT branch     : {args.branch_duration_ps:g} ps at dt={args.branch_dt:g} ps kT={args.kT:g}")
    print(f"new integration: about {new_steps} total steps (reused 0.01 costs zero)")
    print(f"output         : {args.output_dir}")
    print("[NOTE] NVE ranking prioritizes contiguous sigma_E/dt^2 behavior; global p alone is not the selector.")
    print("[NOTE] Diagnostic-only. No candidate is promoted automatically.")
    if args.dry_run:
        return

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    elif args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; use --resume or --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    current = step31["variants"]["current"]
    candidates: dict[str, Any] = {}

    reused_name = smoothing_name(args.reuse_sigma).removesuffix("_wall_current")
    candidates[reused_name] = _candidate_row(
        reused_variant, sigma_rad=args.reuse_sigma, source="reused_step31", current=current, dts=args.dts
    )
    print(f"[REUSE] {reused_name}: step-31 NVT/NVE data re-fitted on common dt grid")

    for sigma, short_name in names:
        full_name = smoothing_name(sigma)
        cdir = args.output_dir / "candidates" / full_name
        if args.resume and (cdir / "cg_priors.json").is_file():
            payload = json.loads((cdir / "cg_priors.json").read_text())
            meta = payload.get("regularization_candidate", {})
            if not math.isclose(float(meta.get("body_sigma_rad", math.nan)), sigma, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"Existing candidate metadata mismatch in {cdir}")
            print(f"[REUSE] {short_name}: existing deterministic candidate priors")
        else:
            generate_candidate(
                source_priors=args.source_priors,
                ibi_config=args.ibi_config,
                body_sigma_rad=sigma,
                output_dir=cdir,
                candidate_name=full_name,
                overwrite=True,
            )
            print(f"[GENERATED] {short_name}: {cdir / 'cg_priors.json'}")
        vreport = _run_subvalidation(
            args=args,
            name=short_name,
            priors=cdir / "cg_priors.json",
            output_dir=args.output_dir / "runs" / short_name,
        )
        candidates[short_name] = _candidate_row(
            vreport, sigma_rad=sigma, source="new_step32", current=current, dts=args.dts
        )

    ranked = sorted(candidates.items(), key=nve_rank_key)
    ranking = []
    print("[LOCAL SWEEP SUMMARY -- COMMON DT GRID]")
    for rank, (name, row) in enumerate(ranked, 1):
        sr = row["sigma_range_common_grid"]
        prefix = row["clean_prefix_factor_1p5"]
        ranking.append({
            "rank_nve_only": rank,
            "name": name,
            "sigma_rad": row["sigma_rad"],
            "max_clean_dt_factor_1p5": sr["max_clean_dt_factor_1p5"],
            "clean_prefix_c2_spread": prefix["c2_spread_max_over_min"],
            "clean_prefix_abs_p_minus_2": prefix["abs_p_minus_2"],
            "global_p": sr["fit"]["exponent_p"],
            "global_r2": sr["fit"]["loglog_r2"],
            "global_c2_spread": sr["c2_spread_max_over_min"],
            "delta_angle_weighted_l1_vs_current": row["delta_angle_weighted_l1_vs_current"],
            "runtime_angle_p99_curvature_reduction_vs_current": row["runtime_angle_p99_curvature_reduction_vs_current"],
        })
        print(
            f"[SWEEP] #{rank} {name:15s} sigma={row['sigma_rad']:.4g} "
            f"clean1.5x={sr['max_clean_dt_factor_1p5']:.6g} ps "
            f"prefixC2={prefix['c2_spread_max_over_min']:.3f} "
            f"p={sr['fit']['exponent_p']:.4f} R2={sr['fit']['loglog_r2']:.4f} "
            f"C2spread={sr['c2_spread_max_over_min']:.3f} "
            f"dAngleL1={row['delta_angle_weighted_l1_vs_current']:+.5f} "
            f"P99U2red={row['runtime_angle_p99_curvature_reduction_vs_current']:.3f}x"
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "diagnostic_only": True,
        "validated": False,
        "source_priors": str(args.source_priors),
        "step31_report": str(args.step31_report),
        "common_dts_ps": args.dts,
        "duration_ps": args.duration_ps,
        "branch_dt_ps": args.branch_dt,
        "branch_duration_ps": args.branch_duration_ps,
        "kT": args.kT,
        "thermostat_seed": args.thermostat_seed,
        "estimated_new_integration_steps": new_steps,
        "current_reference": {
            "structural": current["structural"],
            "sigma_range_common_grid": fit_sigma_range(subset_runs(current["nve_runs"], args.dts)),
        },
        "candidates": candidates,
        "nve_only_ranking": ranking,
        "ranking_rule": (
            "lexicographic: largest contiguous factor-1.5 sigma_E/dt^2 range; then clean-prefix C2 spread; "
            "then clean-prefix |p-2|; then global C2 spread, global |p-2|, and global R2. "
            "Structural deltas are reported separately and must be inspected before any promotion decision."
        ),
        "note": "Local optimization diagnostic only; no candidate is validated or promoted automatically.",
    }
    out = args.output_dir / "angle_smoothing_sweep_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[DONE] report: {out}")
    print("[NOTE] Inspect structural deltas together with the NVE ranking before selecting a final candidate.")


if __name__ == "__main__":
    main()
