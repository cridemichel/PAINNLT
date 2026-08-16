#!/usr/bin/env python3
"""Short-time NVE state-convergence diagnostic against a fine reference trajectory."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from certify_nve import checkpoint_motion_summary, checkpoint_provenance_summary
from framework_utils import input_hashes as runtime_input_hashes


REQUIRED_STATE_KEYS = {
    "complete",
    "steps",
    "time_ps",
    "particle_ids",
    "rotation_flags",
    "positions",
    "velocities",
    "quaternions",
    "omega_body",
    "box",
    "metadata_json",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    found = shutil.which(value)
    if found:
        return found
    raise FileNotFoundError(f"Executable not found: {value}")


def _scalar_string(array: np.ndarray, name: str) -> str:
    raw = np.asarray(array)
    if raw.shape != ():
        raise ValueError(f"{name} must be a scalar string")
    return str(raw.item())


def load_state_trajectory(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(REQUIRED_STATE_KEYS.difference(data.files))
        if missing:
            raise ValueError(f"State trajectory {path} is missing keys: {missing}")
        complete = int(np.asarray(data["complete"]).item())
        if complete != 1:
            raise ValueError(f"State trajectory {path} is not marked complete")
        result = {
            "path": path,
            "steps": np.asarray(data["steps"], dtype=np.int64),
            "time_ps": np.asarray(data["time_ps"], dtype=float),
            "particle_ids": np.asarray(data["particle_ids"], dtype=np.int64),
            "rotation_flags": np.asarray(data["rotation_flags"], dtype=bool),
            "positions": np.asarray(data["positions"], dtype=float),
            "velocities": np.asarray(data["velocities"], dtype=float),
            "quaternions": np.asarray(data["quaternions"], dtype=float),
            "omega_body": np.asarray(data["omega_body"], dtype=float),
            "box": np.asarray(data["box"], dtype=float),
            "metadata": json.loads(_scalar_string(data["metadata_json"], "metadata_json")),
        }
    n_frames = result["time_ps"].size
    n_particles = result["particle_ids"].size
    if n_frames < 2:
        raise ValueError(f"State trajectory {path} needs at least two frames")
    if result["steps"].shape != (n_frames,):
        raise ValueError(f"State trajectory {path} has invalid steps shape")
    if np.any(np.diff(result["time_ps"]) <= 0.0):
        raise ValueError(f"State trajectory {path} times must be strictly increasing")
    if result["rotation_flags"].shape != (n_particles, 3):
        raise ValueError(f"State trajectory {path} has invalid rotation_flags shape")
    for key in ("positions", "velocities", "omega_body"):
        if result[key].shape != (n_frames, n_particles, 3):
            raise ValueError(f"State trajectory {path} has invalid {key} shape {result[key].shape}")
        if not np.isfinite(result[key]).all():
            raise ValueError(f"State trajectory {path} contains non-finite {key}")
    if result["quaternions"].shape != (n_frames, n_particles, 4):
        raise ValueError(f"State trajectory {path} has invalid quaternion shape")
    if not np.isfinite(result["quaternions"]).all():
        raise ValueError(f"State trajectory {path} contains non-finite quaternions")
    if result["box"].shape != (3,) or np.any(result["box"] <= 0.0):
        raise ValueError(f"State trajectory {path} has invalid box")
    return result


def minimum_image_delta(a: np.ndarray, b: np.ndarray, box: np.ndarray) -> np.ndarray:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return delta - np.asarray(box, dtype=float) * np.rint(delta / np.asarray(box, dtype=float))


def quaternion_angle_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    qa = np.asarray(a, dtype=float)
    qb = np.asarray(b, dtype=float)
    na = np.linalg.norm(qa, axis=-1)
    nb = np.linalg.norm(qb, axis=-1)
    if np.any(na <= 0.0) or np.any(nb <= 0.0):
        raise ValueError("Zero-norm quaternion in state trajectory")
    qa = qa / na[..., None]
    qb = qb / nb[..., None]
    raw_dots = np.sum(qa * qb, axis=-1)
    signs = np.where(raw_dots < 0.0, -1.0, 1.0)
    aligned_qb = qb * signs[..., None]
    # For unit quaternions, ||qa-qb|| = 2 sin(theta/4).  This form is
    # substantially more accurate than arccos(|qa.qb|) for tiny angles,
    # which is exactly the regime probed by the finest convergence runs.
    chord = np.linalg.norm(qa - aligned_qb, axis=-1)
    return 4.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))


def rms_vector(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


def state_error_at_frame(a: dict[str, Any], ia: int, b: dict[str, Any], ib: int) -> dict[str, float]:
    if not np.array_equal(a["particle_ids"], b["particle_ids"]):
        raise ValueError("State trajectories do not contain the same real-particle IDs")
    if not np.array_equal(a["rotation_flags"], b["rotation_flags"]):
        raise ValueError("State trajectories do not contain the same rotational degrees of freedom")
    if not np.allclose(a["box"], b["box"], atol=1.0e-12, rtol=0.0):
        raise ValueError("State trajectories do not use the same periodic box")
    pos_delta = minimum_image_delta(a["positions"][ia], b["positions"][ib], a["box"])
    vel_delta = a["velocities"][ia] - b["velocities"][ib]
    rotation_flags = np.asarray(a["rotation_flags"], dtype=bool)
    orient_active = np.any(rotation_flags, axis=1)
    if not np.any(orient_active):
        raise ValueError("State-convergence orientation metric requires at least one rotating rigid body")
    orient = quaternion_angle_error(
        a["quaternions"][ia][orient_active], b["quaternions"][ib][orient_active]
    )
    omega_delta = a["omega_body"][ia] - b["omega_body"][ib]
    omega_delta = np.where(rotation_flags, omega_delta, 0.0)
    omega_particle_delta = omega_delta[orient_active]
    return {
        "position_rms_nm": rms_vector(pos_delta),
        "velocity_rms_nm_per_ps": rms_vector(vel_delta),
        "orientation_rms_rad": float(np.sqrt(np.mean(orient * orient))),
        "omega_body_rms_per_ps": rms_vector(omega_particle_delta),
    }


def power_law_fit(x: list[float], y: list[float], label: str) -> dict[str, Any]:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    if xv.size < 3 or yv.size != xv.size:
        raise ValueError(f"{label}: need at least three equal-length points")
    if np.any(xv <= 0.0) or np.any(yv <= 0.0) or not np.isfinite(yv).all():
        raise ValueError(f"{label}: fit inputs must be finite and positive")
    lx = np.log(xv)
    ly = np.log(yv)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 if ss_tot <= np.finfo(float).eps else 1.0 - ss_res / ss_tot
    return {
        "observable": label,
        "model": f"{label} = C * dt^p",
        "n_points": int(xv.size),
        "dt_min_ps": float(np.min(xv)),
        "dt_max_ps": float(np.max(xv)),
        "exponent_p": float(slope),
        "prefactor_C": float(math.exp(intercept)),
        "loglog_r2": float(r2),
    }


def _frame_index_at_time(traj: dict[str, Any], time_ps: float, atol: float = 1.0e-12) -> int:
    idx = np.where(np.isclose(traj["time_ps"], float(time_ps), atol=atol, rtol=0.0))[0]
    if idx.size != 1:
        raise ValueError(
            f"Trajectory {traj['path']} does not contain exactly one frame at t={time_ps:.12g} ps"
        )
    return int(idx[0])


def analyze_state_convergence(
    trajectories: dict[float, dict[str, Any]],
    *,
    reference_dt: float,
    sample_times_ps: list[float],
    expected_order_min: float = 1.7,
    expected_order_max: float = 2.3,
    min_r2: float = 0.95,
) -> dict[str, Any]:
    dts = sorted(float(dt) for dt in trajectories)
    if reference_dt not in trajectories:
        raise ValueError("Reference dt is missing from trajectories")
    if dts[0] != reference_dt:
        raise ValueError("Reference dt must be the smallest timestep")
    # Richardson pair differences have a common proportionality factor only for
    # a fixed refinement ratio. Require the dyadic ladder used by this test.
    for lo, hi in zip(dts[:-1], dts[1:]):
        if not math.isclose(hi / lo, 2.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("State-convergence timesteps must form a dyadic dt, 2*dt ladder")

    reference = trajectories[reference_dt]
    expected_hashes = reference["metadata"].get("input_hashes")
    expected_mode = reference["metadata"].get("hamiltonian_mode")
    expected_source = reference["metadata"].get("source_checkpoint_sha256")
    for dt, traj in trajectories.items():
        if traj["metadata"].get("input_hashes") != expected_hashes:
            raise ValueError(f"State trajectory dt={dt} has different input hashes")
        if traj["metadata"].get("hamiltonian_mode") != expected_mode:
            raise ValueError(f"State trajectory dt={dt} has different Hamiltonian mode")
        if traj["metadata"].get("source_checkpoint_sha256") != expected_source:
            raise ValueError(f"State trajectory dt={dt} has different source checkpoint")

    metrics = (
        "position_rms_nm",
        "velocity_rms_nm_per_ps",
        "orientation_rms_rad",
        "omega_body_rms_per_ps",
    )
    time_reports: list[dict[str, Any]] = []
    fits_by_metric: dict[str, list[dict[str, Any]]] = {metric: [] for metric in metrics}

    for time_ps in sample_times_ps:
        indices = {dt: _frame_index_at_time(traj, time_ps) for dt, traj in trajectories.items()}
        reference_errors: list[dict[str, Any]] = []
        for dt in dts[1:]:
            err = state_error_at_frame(trajectories[dt], indices[dt], reference, indices[reference_dt])
            reference_errors.append({"dt_ps": dt, **err})

        pair_errors: list[dict[str, Any]] = []
        # Each row compares a coarse trajectory with its half-dt trajectory.
        for fine_dt, coarse_dt in zip(dts[:-1], dts[1:]):
            err = state_error_at_frame(
                trajectories[coarse_dt], indices[coarse_dt],
                trajectories[fine_dt], indices[fine_dt],
            )
            pair_errors.append({"coarse_dt_ps": coarse_dt, "fine_dt_ps": fine_dt, **err})

        fits: dict[str, Any] = {}
        coarse_dts = [float(row["coarse_dt_ps"]) for row in pair_errors]
        for metric in metrics:
            values = [float(row[metric]) for row in pair_errors]
            fit = power_law_fit(coarse_dts, values, f"Richardson {metric}")
            fits[metric] = fit
            fits_by_metric[metric].append(fit)

        time_reports.append({
            "time_ps": float(time_ps),
            "reference_errors": reference_errors,
            "richardson_pair_errors": pair_errors,
            "richardson_fits": fits,
        })

    summaries: dict[str, Any] = {}
    for metric in metrics:
        exponents = np.asarray([item["exponent_p"] for item in fits_by_metric[metric]], dtype=float)
        r2s = np.asarray([item["loglog_r2"] for item in fits_by_metric[metric]], dtype=float)
        median_p = float(np.median(exponents))
        median_r2 = float(np.median(r2s))
        summaries[metric] = {
            "median_exponent_p": median_p,
            "min_exponent_p": float(np.min(exponents)),
            "max_exponent_p": float(np.max(exponents)),
            "median_loglog_r2": median_r2,
            "min_loglog_r2": float(np.min(r2s)),
            "consistent_with_second_order": bool(
                expected_order_min <= median_p <= expected_order_max and median_r2 >= min_r2
            ),
        }

    return {
        "schema_version": 1,
        "kind": "conservative_ibi_nve_state_convergence_diagnostic",
        "diagnostic_only": True,
        "reference_dt_ps": float(reference_dt),
        "dts_ps": dts,
        "sample_times_ps": [float(x) for x in sample_times_ps],
        "hamiltonian_mode": expected_mode,
        "source_checkpoint_sha256": expected_source,
        "input_hashes": expected_hashes,
        "expected_second_order_window": {
            "slope_min": float(expected_order_min),
            "slope_max": float(expected_order_max),
            "min_median_loglog_r2": float(min_r2),
        },
        "times": time_reports,
        "metric_summary": summaries,
    }


def _dt_tag(dt: float) -> str:
    text = f"{dt:.10f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _exact_steps(duration_ps: float, dt_ps: float, name: str) -> int:
    raw = duration_ps / dt_ps
    steps = int(round(raw))
    if steps <= 0 or not math.isclose(steps * dt_ps, duration_ps, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{name}={duration_ps} ps must be an exact multiple of dt={dt_ps} ps")
    return steps


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypresso", default="pypresso")
    parser.add_argument("--runner", default=str(Path(__file__).with_name("run_cg_md.py")))
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--priors", required=True)
    parser.add_argument("--rb-info", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--require-checkpoint-hamiltonian-mode", default=None)
    parser.add_argument("--require-checkpoint-source", default=None)
    parser.add_argument(
        "--dts", nargs="+", type=float,
        default=[0.001, 0.0005, 0.00025, 0.000125],
        help="Non-reference timesteps; together with --reference-dt they must form a dyadic ladder",
    )
    parser.add_argument("--reference-dt", type=float, default=0.0000625)
    parser.add_argument("--duration-ps", type=float, default=0.096)
    parser.add_argument("--sample-interval-ps", type=float, default=0.012)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ml-precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--neighbor-search", choices=("verlet", "link-cell"), default="link-cell")
    parser.add_argument("--output-dir", default="nve_state_convergence_conservative_ibi_only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--order-min", type=float, default=1.7)
    parser.add_argument("--order-max", type=float, default=2.3)
    parser.add_argument("--min-r2", type=float, default=0.95)
    args = parser.parse_args()

    if args.duration_ps <= 0.0 or args.sample_interval_ps <= 0.0:
        raise ValueError("Duration and sample interval must be positive")
    _exact_steps(args.duration_ps, args.sample_interval_ps, "duration")

    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "runner": args.runner,
            "model": args.model,
            "config": args.config,
            "priors": args.priors,
            "rb_info": args.rb_info,
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
        }.items()
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    required_source = None
    if args.require_checkpoint_source:
        required_source = Path(args.require_checkpoint_source).expanduser().resolve()
        if not required_source.is_file():
            raise FileNotFoundError(f"required checkpoint source not found: {required_source}")

    summary = checkpoint_provenance_summary(
        paths["checkpoint"],
        dataset=paths["dataset"],
        config=paths["config"],
        priors=paths["priors"],
        rb_info=paths["rb_info"],
        model=paths["model"],
        required_hamiltonian_mode=args.require_checkpoint_hamiltonian_mode,
        required_source_checkpoint=required_source,
    )
    motion = checkpoint_motion_summary(paths["checkpoint"])
    if motion["velocity_rms"] <= 0.0:
        raise ValueError("Checkpoint has zero translational motion")

    all_dts = sorted(set([float(args.reference_dt), *map(float, args.dts)]))
    if len(all_dts) != len(args.dts) + 1 or all_dts[0] != float(args.reference_dt):
        raise ValueError("Reference dt must be unique and smaller than every --dts value")
    for lo, hi in zip(all_dts[:-1], all_dts[1:]):
        if not math.isclose(hi / lo, 2.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Timesteps must form a dyadic ladder: dt_ref, 2*dt_ref, 4*dt_ref, ...")

    run_specs = []
    sample_times = np.arange(
        args.sample_interval_ps,
        args.duration_ps + 0.5 * args.sample_interval_ps,
        args.sample_interval_ps,
        dtype=float,
    )
    for dt in sorted(all_dts, reverse=True):
        steps = _exact_steps(args.duration_ps, dt, "duration")
        log_interval = _exact_steps(args.sample_interval_ps, dt, "sample interval")
        run_specs.append({"dt_ps": dt, "steps": steps, "log_interval_steps": log_interval})

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and args.overwrite and not args.dry_run:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "schema_version": 1,
        "kind": "conservative_ibi_nve_state_convergence_plan",
        "diagnostic_only": True,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "checkpoint_provenance": summary,
        "checkpoint_motion": motion,
        "reference_dt_ps": float(args.reference_dt),
        "sample_interval_ps": float(args.sample_interval_ps),
        "duration_ps": float(args.duration_ps),
        "sample_times_ps": sample_times.tolist(),
        "runs": run_specs,
        "input_hashes": runtime_input_hashes(
            dataset=paths["dataset"], config=paths["config"], priors=paths["priors"],
            rb_info=paths["rb_info"], model=paths["model"],
        ),
    }
    (output_dir / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    print("[CONSERVATIVE IBI-ONLY NVE STATE-CONVERGENCE PLAN]")
    print(f"checkpoint   : {paths['checkpoint']}")
    print(f"dts          : {' '.join(f'{x:g}' for x in sorted(all_dts, reverse=True))} ps")
    print(f"reference dt : {args.reference_dt:g} ps")
    print(f"duration     : {args.duration_ps:g} ps")
    print(f"sample every : {args.sample_interval_ps:g} ps")
    print(f"sample times : {' '.join(f'{x:g}' for x in sample_times)} ps")
    print(f"output       : {output_dir}")
    print("[NOTE] Diagnostic-only: Richardson state convergence does not replace step-23 strict NVE certification.")

    pypresso = resolve_executable(args.pypresso) if not args.dry_run else args.pypresso
    if args.dry_run:
        for spec in run_specs:
            print(
                f"[PLAN] dt={spec['dt_ps']:g} ps steps={spec['steps']} "
                f"sample_every={spec['log_interval_steps']} steps"
            )
        print(f"[DONE] Dry-run plan: {output_dir / 'run_plan.json'}")
        return 0

    trajectories: dict[float, dict[str, Any]] = {}
    expected_source_sha = sha256_file(paths["checkpoint"])
    expected_hashes = plan["input_hashes"]
    for spec in run_specs:
        dt = float(spec["dt_ps"])
        run_dir = output_dir / f"dt_{_dt_tag(dt)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = run_dir / "state_samples.npz"
        run_log = run_dir / "run.log"
        cmd = [
            pypresso, str(paths["runner"]),
            "--model", str(paths["model"]), "--disable_ml",
            "--config", str(paths["config"]), "--priors", str(paths["priors"]),
            "--rb_info", str(paths["rb_info"]), "--dataset", str(paths["dataset"]),
            "--checkpoint", str(paths["checkpoint"]),
            "--dt", f"{dt:.12g}", "--steps", str(spec["steps"]),
            "--log_interval", str(spec["log_interval_steps"]),
            "--state_sample_npz", str(state_path), "--no_log", "--nve",
            "--device", args.device, "--ml_precision", args.ml_precision,
            "--neighbor_search", args.neighbor_search,
        ]
        print(
            f"[RUN] dt={dt:g} ps steps={spec['steps']} "
            f"sample_every={spec['log_interval_steps']} steps"
        )
        with run_log.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"NVE state trajectory failed for dt={dt:g} ps (exit={proc.returncode})\n{_tail(run_log)}"
            )
        traj = load_state_trajectory(state_path)
        meta = traj["metadata"]
        if meta.get("hamiltonian_mode") != args.require_checkpoint_hamiltonian_mode:
            raise ValueError(f"dt={dt:g}: state trajectory Hamiltonian mode mismatch")
        if meta.get("source_checkpoint_sha256") != expected_source_sha:
            raise ValueError(f"dt={dt:g}: state trajectory source checkpoint mismatch")
        if meta.get("input_hashes") != expected_hashes:
            raise ValueError(f"dt={dt:g}: state trajectory input hashes mismatch")
        expected_times = np.concatenate(([0.0], sample_times))
        if not np.allclose(traj["time_ps"], expected_times, atol=1.0e-12, rtol=0.0):
            raise ValueError(
                f"dt={dt:g}: sampled times differ from the common physical-time grid: "
                f"{traj['time_ps'].tolist()}"
            )
        trajectories[dt] = traj

    report = analyze_state_convergence(
        trajectories,
        reference_dt=float(args.reference_dt),
        sample_times_ps=sample_times.tolist(),
        expected_order_min=args.order_min,
        expected_order_max=args.order_max,
        min_r2=args.min_r2,
    )
    report.update({
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": expected_source_sha,
        "run_plan": str(output_dir / "run_plan.json"),
    })
    report_path = output_dir / "state_convergence_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for time_report in report["times"]:
        t = time_report["time_ps"]
        for metric, fit in time_report["richardson_fits"].items():
            print(
                f"[STATE ORDER FIT] t={t:g} ps {metric}: "
                f"p={fit['exponent_p']:.6f} R2={fit['loglog_r2']:.6f}"
            )
    print("[STATE ORDER SUMMARY]")
    for metric, item in report["metric_summary"].items():
        verdict = "ORDER2-LIKE" if item["consistent_with_second_order"] else "NOT-ORDER2-LIKE"
        print(
            f"{metric}: median_p={item['median_exponent_p']:.6f} "
            f"median_R2={item['median_loglog_r2']:.6f} {verdict}"
        )
    print(f"[DONE] State-convergence diagnostic report: {report_path}")
    print("[NOTE] This diagnostic does not change the strict step-23 certification decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
