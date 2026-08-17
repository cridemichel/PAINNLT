#!/usr/bin/env python3
"""Audit and generate *unvalidated* regularization candidates for conservative IBI angles.

The diagnostic is intentionally offline: it reads the mapped target geometry and
an existing structured runtime sample, analyzes the current conservative angle
splines, and writes self-contained candidate prior directories.  It never
modifies the selected IBI priors in place and it does not run MD.

Three effects are separated explicitly:

1. the quadratic endpoint walls added by the IBI builder;
2. short-wavelength structure in the angle potential body;
3. the C1 PCHIP/Hermite derivative choice used by the current conversion.

Candidate splines are built by smoothing only the de-walled potential body,
re-adding a selected wall, and using a C2 cubic spline to define the nodal first
derivatives.  The existing runtime still evaluates those nodes with the same
conservative Hermite representation; no ESPResSo/kernel change is required.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "preprocessing", ROOT / "ibi"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from conservative_spline import load_conservative_spline, save_conservative_spline  # noqa: E402
from build_dbi_priors import load_continuation_priors  # noqa: E402
from geometry_io import pool_requested, read_sampled_distributions  # noqa: E402
from ibi_core import histogram_density, normalize_density  # noqa: E402

SCHEMA_VERSION = 1
KIND = "ibi_angle_regularization_diagnostic"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    body_sigma_rad: float
    wall_width_rad: float
    wall_k: float
    note: str


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(value: str) -> str:
    out = "".join(c if (c.isalnum() or c in "_.-") else "_" for c in str(value))
    out = out.strip("._")
    return out or "unnamed"


def wall_energy_gradient_curvature(theta, width: float, k: float):
    """Return the exact quadratic wall U, dU/dtheta and d2U/dtheta2 arrays."""
    q = np.asarray(theta, dtype=float)
    u = np.zeros_like(q)
    du = np.zeros_like(q)
    u2 = np.zeros_like(q)
    left = q < width
    d = width - q[left]
    u[left] = 0.5 * k * d * d
    du[left] = -k * d
    u2[left] = k
    right_edge = np.pi - width
    right = q > right_edge
    d = q[right] - right_edge
    u[right] = 0.5 * k * d * d
    du[right] = k * d
    u2[right] = k
    return u, du, u2


def same_barrier_k(old_width: float, old_k: float, new_width: float) -> float:
    """Keep the endpoint wall energy 0.5*k*w^2 unchanged."""
    if old_width <= 0.0 or old_k < 0.0 or new_width <= 0.0:
        raise ValueError("Wall widths must be positive and wall_k non-negative")
    return float(old_k * (old_width / new_width) ** 2)


def candidate_specs_from_json(raw_json: str, wall_width: float, wall_k: float) -> list[CandidateSpec]:
    payload = json.loads(raw_json)
    if not isinstance(payload, list) or not payload:
        raise ValueError("--candidate-specs-json must be a non-empty JSON list")
    out = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("Each candidate spec must be an object")
        name = str(row["name"])
        sigma = float(row["body_sigma_rad"])
        scale = float(row["wall_width_scale"])
        if sigma < 0.0 or scale <= 0.0:
            raise ValueError(f"Invalid candidate spec {name!r}")
        width = wall_width * scale
        k = wall_k if np.isclose(scale, 1.0) else same_barrier_k(wall_width, wall_k, width)
        note = str(row.get("note", f"configured body smoothing sigma={sigma:g} rad; wall width scale={scale:g}"))
        out.append(CandidateSpec(name, sigma, width, k, note))
    names=[x.name for x in out]
    if len(names) != len(set(names)):
        raise ValueError("Candidate names must be unique")
    return out


def _summary_abs(values: Iterable[float]) -> dict[str, float | int]:
    a = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "p50_abs": math.nan, "p95_abs": math.nan, "p99_abs": math.nan, "max_abs": math.nan}
    av = np.abs(a)
    return {
        "n": int(a.size),
        "p50_abs": float(np.percentile(av, 50)),
        "p95_abs": float(np.percentile(av, 95)),
        "p99_abs": float(np.percentile(av, 99)),
        "max_abs": float(np.max(av)),
        "min_signed": float(np.min(a)),
        "max_signed": float(np.max(a)),
    }


def _distribution_l1(a, b, grid) -> float:
    aa = normalize_density(np.asarray(a, dtype=float), grid)
    bb = normalize_density(np.asarray(b, dtype=float), grid)
    return float(np.trapezoid(np.abs(aa - bb), np.asarray(grid, dtype=float)))


def _wall_fraction(values, width: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return math.nan
    return float(np.mean((values < width) | (values > np.pi - width)))


def _zone_curvature(values, curvature, width: float) -> dict[str, Any]:
    q = np.asarray(values, dtype=float)
    u2 = np.asarray(curvature, dtype=float)
    wall = (q < width) | (q > np.pi - width)
    return {
        "wall": _summary_abs(u2[wall]),
        "interior": _summary_abs(u2[~wall]),
    }


def _potential_perturbation(current, candidate, target_values, sample_values, kT: float) -> dict[str, Any]:
    target_values = np.asarray(target_values, dtype=float)
    sample_values = np.asarray(sample_values, dtype=float)
    d_target = np.asarray(candidate(target_values) - current(target_values), dtype=float)
    # Potential zero is arbitrary.  Use the target ensemble to align offsets.
    offset = float(np.mean(d_target)) if d_target.size else 0.0
    d_target = d_target - offset
    d_sample = np.asarray(candidate(sample_values) - current(sample_values), dtype=float) - offset

    def one(delta: np.ndarray) -> dict[str, float | int]:
        if delta.size == 0:
            return {"n": 0, "rms_kT": math.nan, "p95_abs_kT": math.nan, "p99_abs_kT": math.nan, "max_abs_kT": math.nan}
        x = delta / kT
        return {
            "n": int(x.size),
            "rms_kT": float(np.sqrt(np.mean(x * x))),
            "p95_abs_kT": float(np.percentile(np.abs(x), 95)),
            "p99_abs_kT": float(np.percentile(np.abs(x), 99)),
            "max_abs_kT": float(np.max(np.abs(x))),
        }

    return {"offset_kJmol": offset, "target": one(d_target), "sample": one(d_sample)}


def _knot_u2_jump(spline, x: np.ndarray) -> dict[str, float]:
    if len(x) < 3:
        return {"max_abs": 0.0, "p99_abs": 0.0}
    h = float(np.min(np.diff(x)))
    eps = max(h * 1.0e-7, 1.0e-12)
    knots = x[1:-1]
    jumps = np.asarray(spline(knots + eps, 2) - spline(knots - eps, 2), dtype=float)
    av = np.abs(jumps)
    return {"max_abs": float(np.max(av)), "p99_abs": float(np.percentile(av, 99))}


def _build_candidate(table, spec: CandidateSpec, current_wall_width: float, current_wall_k: float):
    x = np.asarray(table.x, dtype=float)
    u = np.asarray(table.energy, dtype=float)
    old_wall, _old_du, _old_u2 = wall_energy_gradient_curvature(x, current_wall_width, current_wall_k)
    body = u - old_wall
    h = float(np.median(np.diff(x)))
    if spec.body_sigma_rad > 0.0:
        body = gaussian_filter1d(body, sigma=spec.body_sigma_rad / h, mode="nearest")
    new_wall, _new_du, _new_u2 = wall_energy_gradient_curvature(x, spec.wall_width_rad, spec.wall_k)
    new_u = np.asarray(body + new_wall, dtype=float)
    new_u -= float(np.min(new_u))
    # CubicSpline is C2.  Exporting its nodal first derivatives into the existing
    # Hermite representation reconstructs the same piecewise cubic exactly.
    c2 = CubicSpline(x, new_u)
    return c2, new_u, np.asarray(c2(x, 1), dtype=float)


def _old_harmonic_k(old_priors: Mapping[str, Any], name: str) -> float:
    values = []
    for row in old_priors.get("angles", []):
        if str(row.get("name", "")) != name:
            continue
        if str(row.get("type", "")).lower() != "harmonic":
            raise ValueError(f"Reference angle group {name!r} is not harmonic")
        values.append(float(row["k"]))
    if not values:
        raise ValueError(f"Reference priors contain no angle group {name!r}")
    if not np.allclose(values, values[0], rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Reference angle group {name!r} has inconsistent k values: {values}")
    return float(values[0])


def _top_hotspots(spline, target_hist_x, target_density, sample_density, wall_width: float, *, n: int, min_sep: float):
    grid = np.linspace(0.0, np.pi, 16001)
    u2 = np.asarray(spline(grid, 2), dtype=float)
    order = np.argsort(np.abs(u2))[::-1]
    selected: list[int] = []
    for idx in order:
        if all(abs(float(grid[idx] - grid[j])) >= min_sep for j in selected):
            selected.append(int(idx))
        if len(selected) >= n:
            break
    out = []
    for idx in selected:
        q = float(grid[idx])
        out.append({
            "theta_rad": q,
            "theta_deg": float(np.degrees(q)),
            "u2": float(u2[idx]),
            "abs_u2": float(abs(u2[idx])),
            "wall_zone": bool(q < wall_width or q > np.pi - wall_width),
            "distance_to_endpoint_rad": float(min(q, np.pi - q)),
            "target_density": float(np.interp(q, target_hist_x, target_density, left=0.0, right=0.0)),
            "sample_density": float(np.interp(q, target_hist_x, sample_density, left=0.0, right=0.0)),
        })
    return out


def _copy_nonangle_tables(source_priors: Mapping[str, Any], source_path: Path, outdir: Path, payload: dict[str, Any]) -> None:
    copied: dict[Path, str] = {}
    for key in ("bonds", "dihedrals"):
        for idx, row in enumerate(payload.get(key, [])):
            if "file" not in row:
                continue
            src = Path(str(source_priors[key][idx]["file"])).expanduser()
            if not src.is_absolute():
                src = source_path.parent / src
            src = src.resolve()
            if not src.is_file():
                raise FileNotFoundError(src)
            if src not in copied:
                name = src.name
                dst = outdir / name
                if dst.exists() and sha256_file(dst) != sha256_file(src):
                    name = f"{src.stem}_{sha256_file(src)[:10]}{src.suffix}"
                    dst = outdir / name
                shutil.copy2(src, dst)
                copied[src] = name
            row["file"] = copied[src]


def _write_profile_csv(path: Path, x: np.ndarray, columns: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["theta_rad", *keys])
        for i, q in enumerate(x):
            w.writerow([f"{float(q):.17g}", *[f"{float(np.asarray(columns[k])[i]):.17g}" for k in keys]])


def diagnose_and_generate(
    *,
    dataset: str | Path,
    priors: str | Path,
    old_priors: str | Path,
    sample_npz: str | Path,
    ibi_config: str | Path,
    output_dir: str | Path,
    candidate_specs_json: str,
    hotspot_count: int,
    hotspot_min_separation_rad: float,
) -> dict[str, Any]:
    dataset = Path(dataset).expanduser().resolve()
    priors_path = Path(priors).expanduser().resolve()
    old_priors_path = Path(old_priors).expanduser().resolve()
    sample_npz = Path(sample_npz).expanduser().resolve()
    ibi_config = Path(ibi_config).expanduser().resolve()
    outdir = Path(output_dir).expanduser().resolve()
    for p in (dataset, priors_path, old_priors_path, sample_npz, ibi_config):
        if not p.is_file():
            raise FileNotFoundError(p)
    outdir.mkdir(parents=True, exist_ok=True)

    config = json.loads(ibi_config.read_text())
    kT = float(config["kT"])
    angle_cfg = config["angle"]
    wall_width = float(angle_cfg["wall_width"])
    wall_k = float(angle_cfg["wall_k"])
    specs = candidate_specs_from_json(candidate_specs_json, wall_width, wall_k)

    state = load_continuation_priors(dataset, priors_path, ibi_config=ibi_config, allow_conservative_spline=True)
    selected_priors = state["priors"]
    groups = state["groups"]["angles"]
    sampled = read_sampled_distributions(sample_npz, selected_priors)
    sampled_groups = pool_requested(selected_priors, sampled[1], "angles")
    old_payload = json.loads(old_priors_path.read_text())

    candidate_payloads: dict[str, dict[str, Any]] = {}
    candidate_dirs: dict[str, Path] = {}
    for spec in specs:
        cdir = outdir / "candidates" / spec.name
        cdir.mkdir(parents=True, exist_ok=True)
        payload = copy.deepcopy(selected_priors)
        _copy_nonangle_tables(selected_priors, priors_path, cdir, payload)
        payload["regularization_candidate"] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "unvalidated_ibi_angle_regularization_candidate",
            "source_priors": str(priors_path),
            "source_priors_sha256": sha256_file(priors_path),
            "candidate": spec.name,
            "body_sigma_rad": spec.body_sigma_rad,
            "wall_width_rad": spec.wall_width_rad,
            "wall_k": spec.wall_k,
            "note": spec.note,
            "validated": False,
        }
        candidate_payloads[spec.name] = payload
        candidate_dirs[spec.name] = cdir

    group_reports: dict[str, Any] = {}
    aggregate_current_target_u2: list[float] = []
    aggregate_current_sample_u2: list[float] = []
    aggregate_old_target_u2: list[float] = []
    aggregate_candidate: dict[str, dict[str, list[float]]] = {
        spec.name: {"target_u2": [], "sample_u2": [], "target_delta": [], "sample_delta": []}
        for spec in specs
    }

    for name, gs in sorted(groups.items()):
        sg = sampled_groups.get(name)
        if sg is None:
            raise RuntimeError(f"Runtime sample is missing angle group {name!r}")
        target_values = np.asarray(gs["values"], dtype=float)
        sample_values = np.asarray(sg["values"], dtype=float)
        _counts, sample_density, sample_hist_x = histogram_density(sample_values, gs["bins"])
        if not np.allclose(sample_hist_x, gs["hist_x"], rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"Histogram grid mismatch for angle group {name!r}")
        l1 = _distribution_l1(sample_density, gs["target_density"], gs["hist_x"])

        entries = [selected_priors["angles"][idx] for idx in gs["indices"]]
        paths = []
        for entry in entries:
            table = load_conservative_spline(entry, kind="angle", priors_path=priors_path)
            paths.append(table.path.resolve())
        if len(set(paths)) != 1:
            raise ValueError(f"Angle group {name!r} references multiple conservative tables: {paths}")
        table = load_conservative_spline(entries[0], kind="angle", priors_path=priors_path)
        x = np.asarray(table.x, dtype=float)
        current = CubicHermiteSpline(x, np.asarray(table.energy, dtype=float), np.asarray(table.derivative, dtype=float))
        old_k = _old_harmonic_k(old_payload, name)

        cur_target_u2 = np.asarray(current(target_values, 2), dtype=float)
        cur_sample_u2 = np.asarray(current(sample_values, 2), dtype=float)
        aggregate_current_target_u2.extend(cur_target_u2.tolist())
        aggregate_current_sample_u2.extend(cur_sample_u2.tolist())
        aggregate_old_target_u2.extend(np.full(target_values.shape, old_k).tolist())

        current_wall_u, current_wall_du, current_wall_u2_grid = wall_energy_gradient_curvature(x, wall_width, wall_k)
        dense_x = np.linspace(0.0, np.pi, 8001)
        dense_u2 = np.asarray(current(dense_x, 2), dtype=float)
        _dw, _ddw, dense_wall_u2 = wall_energy_gradient_curvature(dense_x, wall_width, wall_k)
        dense_body_u2 = dense_u2 - dense_wall_u2
        current_report = {
            "source_table": str(table.path),
            "source_table_sha256": sha256_file(table.path),
            "old_harmonic_k": old_k,
            "old_frequency_proxy_denominator_sqrt_k": math.sqrt(old_k),
            "distribution_l1_runtime_vs_target": l1,
            "target_samples": int(target_values.size),
            "runtime_samples": int(sample_values.size),
            "wall": {
                "width_rad": wall_width,
                "k": wall_k,
                "endpoint_barrier_kJmol": 0.5 * wall_k * wall_width * wall_width,
                "endpoint_barrier_kT": 0.5 * wall_k * wall_width * wall_width / kT,
                "target_fraction": _wall_fraction(target_values, wall_width),
                "runtime_fraction": _wall_fraction(sample_values, wall_width),
            },
            "curvature_target": _summary_abs(cur_target_u2),
            "curvature_runtime": _summary_abs(cur_sample_u2),
            "curvature_target_by_zone": _zone_curvature(target_values, cur_target_u2, wall_width),
            "curvature_runtime_by_zone": _zone_curvature(sample_values, cur_sample_u2, wall_width),
            "curvature_target_body_after_subtracting_explicit_wall": _summary_abs(
                cur_target_u2 - wall_energy_gradient_curvature(target_values, wall_width, wall_k)[2]
            ),
            "curvature_runtime_body_after_subtracting_explicit_wall": _summary_abs(
                cur_sample_u2 - wall_energy_gradient_curvature(sample_values, wall_width, wall_k)[2]
            ),
            "curvature_grid": _summary_abs(dense_u2),
            "curvature_grid_body_after_subtracting_explicit_wall": _summary_abs(dense_body_u2),
            "u2_knot_jump": _knot_u2_jump(current, x),
            "frequency_proxy_p99_vs_old_target": math.sqrt(float(np.percentile(np.abs(cur_target_u2), 99)) / old_k),
            "hotspots": _top_hotspots(current, gs["hist_x"], gs["target_density"], sample_density, wall_width, n=hotspot_count, min_sep=hotspot_min_separation_rad),
        }

        candidate_reports: dict[str, Any] = {}
        profile_columns: dict[str, np.ndarray] = {
            "target_density": np.interp(x, gs["hist_x"], gs["target_density"], left=0.0, right=0.0),
            "runtime_density": np.interp(x, gs["hist_x"], sample_density, left=0.0, right=0.0),
            "current_U": np.asarray(current(x), dtype=float),
            "current_dU": np.asarray(current(x, 1), dtype=float),
            "current_U2": np.asarray(current(x, 2), dtype=float),
            "explicit_wall_U": current_wall_u,
            "explicit_wall_dU": current_wall_du,
            "explicit_wall_U2": current_wall_u2_grid,
        }
        for spec in specs:
            c2, new_u, new_du = _build_candidate(table, spec, wall_width, wall_k)
            candidate_target_u2 = np.asarray(c2(target_values, 2), dtype=float)
            candidate_sample_u2 = np.asarray(c2(sample_values, 2), dtype=float)
            perturb = _potential_perturbation(current, c2, target_values, sample_values, kT)
            aggregate_candidate[spec.name]["target_u2"].extend(candidate_target_u2.tolist())
            aggregate_candidate[spec.name]["sample_u2"].extend(candidate_sample_u2.tolist())
            # Store centered deltas explicitly for aggregate RMS.
            dt = np.asarray(c2(target_values) - current(target_values), dtype=float)
            offset = float(np.mean(dt)) if dt.size else 0.0
            aggregate_candidate[spec.name]["target_delta"].extend((dt - offset).tolist())
            aggregate_candidate[spec.name]["sample_delta"].extend((np.asarray(c2(sample_values) - current(sample_values), dtype=float) - offset).tolist())

            cdir = candidate_dirs[spec.name]
            filename = f"angle_conservative_{_safe_name(name)}.dat"
            save_conservative_spline(cdir / filename, x, new_u, new_du)
            for idx in gs["indices"]:
                entry = candidate_payloads[spec.name]["angles"][idx]
                entry["file"] = filename
                entry["regularization"] = {
                    "kind": "angle_body_gaussian_plus_c2_v1",
                    "candidate": spec.name,
                    "body_sigma_rad": spec.body_sigma_rad,
                    "wall_width_rad": spec.wall_width_rad,
                    "wall_k": spec.wall_k,
                    "regularized_from_conservative_sha256": sha256_file(table.path),
                    "validated": False,
                }

            cand_p99 = float(np.percentile(np.abs(candidate_target_u2), 99))
            cur_p99 = float(np.percentile(np.abs(cur_target_u2), 99))
            candidate_reports[spec.name] = {
                "body_sigma_rad": spec.body_sigma_rad,
                "wall_width_rad": spec.wall_width_rad,
                "wall_k": spec.wall_k,
                "endpoint_barrier_kJmol": 0.5 * spec.wall_k * spec.wall_width_rad**2,
                "endpoint_barrier_kT": 0.5 * spec.wall_k * spec.wall_width_rad**2 / kT,
                "target_fraction_in_candidate_wall": _wall_fraction(target_values, spec.wall_width_rad),
                "runtime_fraction_in_candidate_wall": _wall_fraction(sample_values, spec.wall_width_rad),
                "curvature_target": _summary_abs(candidate_target_u2),
                "curvature_runtime": _summary_abs(candidate_sample_u2),
                "curvature_reduction_p99_target": cur_p99 / cand_p99 if cand_p99 > 0 else math.inf,
                "frequency_proxy_p99_vs_old_target": math.sqrt(cand_p99 / old_k) if old_k > 0 else math.nan,
                "potential_perturbation": perturb,
                "u2_knot_jump": _knot_u2_jump(c2, x),
                "note": spec.note,
            }
            profile_columns[f"{spec.name}_U"] = np.asarray(c2(x), dtype=float)
            profile_columns[f"{spec.name}_dU"] = np.asarray(c2(x, 1), dtype=float)
            profile_columns[f"{spec.name}_U2"] = np.asarray(c2(x, 2), dtype=float)

        _write_profile_csv(outdir / "profiles" / f"{_safe_name(name)}.csv", x, profile_columns)
        group_reports[name] = {"current": current_report, "candidates": candidate_reports}

    # Write candidate prior files only after every angle group has been processed.
    candidate_index: dict[str, Any] = {}
    for spec in specs:
        cdir = candidate_dirs[spec.name]
        priors_out = cdir / "cg_priors.json"
        priors_out.write_text(json.dumps(candidate_payloads[spec.name], indent=2, sort_keys=True) + "\n")
        candidate_index[spec.name] = {
            "priors": str(priors_out),
            "priors_sha256": sha256_file(priors_out),
            "body_sigma_rad": spec.body_sigma_rad,
            "wall_width_rad": spec.wall_width_rad,
            "wall_k": spec.wall_k,
            "note": spec.note,
            "validated": False,
        }

    current_target = np.asarray(aggregate_current_target_u2, dtype=float)
    current_sample = np.asarray(aggregate_current_sample_u2, dtype=float)
    old_target = np.asarray(aggregate_old_target_u2, dtype=float)
    aggregate = {
        "current": {
            "curvature_target": _summary_abs(current_target),
            "curvature_runtime": _summary_abs(current_sample),
            "old_harmonic_curvature_target": _summary_abs(old_target),
            "p99_curvature_ratio_vs_old": float(np.percentile(np.abs(current_target), 99) / np.percentile(np.abs(old_target), 99)),
            "sqrt_p99_ratio_frequency_proxy": float(math.sqrt(np.percentile(np.abs(current_target), 99) / np.percentile(np.abs(old_target), 99))),
        },
        "candidates": {},
    }
    current_p99 = float(np.percentile(np.abs(current_target), 99))
    old_p99 = float(np.percentile(np.abs(old_target), 99))
    for spec in specs:
        row = aggregate_candidate[spec.name]
        tu2 = np.asarray(row["target_u2"], dtype=float)
        su2 = np.asarray(row["sample_u2"], dtype=float)
        td = np.asarray(row["target_delta"], dtype=float) / kT
        sd = np.asarray(row["sample_delta"], dtype=float) / kT
        p99 = float(np.percentile(np.abs(tu2), 99))
        max_group_rms = max(
            float(group_reports[name]["candidates"][spec.name]["potential_perturbation"]["target"]["rms_kT"])
            for name in group_reports
        )
        aggregate["candidates"][spec.name] = {
            "curvature_target": _summary_abs(tu2),
            "curvature_runtime": _summary_abs(su2),
            "p99_curvature_reduction_vs_current": current_p99 / p99 if p99 > 0 else math.inf,
            "p99_curvature_ratio_vs_old": p99 / old_p99 if old_p99 > 0 else math.nan,
            "sqrt_p99_ratio_frequency_proxy_vs_old": math.sqrt(p99 / old_p99) if old_p99 > 0 and p99 >= 0 else math.nan,
            "potential_delta_target_rms_kT": float(np.sqrt(np.mean(td * td))) if td.size else math.nan,
            "potential_delta_target_p99_abs_kT": float(np.percentile(np.abs(td), 99)) if td.size else math.nan,
            "potential_delta_runtime_rms_kT": float(np.sqrt(np.mean(sd * sd))) if sd.size else math.nan,
            "max_group_potential_delta_target_rms_kT": max_group_rms,
            "candidate_priors": candidate_index[spec.name]["priors"],
            "validated": False,
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "diagnostic_only": True,
        "source_priors": str(priors_path),
        "source_priors_sha256": sha256_file(priors_path),
        "old_priors": str(old_priors_path),
        "dataset": str(dataset),
        "runtime_sample": str(sample_npz),
        "ibi_config": str(ibi_config),
        "kT": kT,
        "current_angle_wall": {
            "width_rad": wall_width,
            "k": wall_k,
            "endpoint_barrier_kJmol": 0.5 * wall_k * wall_width**2,
            "endpoint_barrier_kT": 0.5 * wall_k * wall_width**2 / kT,
        },
        "groups": group_reports,
        "aggregate": aggregate,
        "candidates": candidate_index,
        "interpretation": {
            "wall_hotspots": "large curvature localized in the wall zone implicates the artificial endpoint barrier",
            "interior_hotspots": "large occupied interior curvature implicates the IBI body/update smoothing rather than only the wall",
            "c2_raw_improves": "if c2_raw materially lowers curvature with tiny delta-U, derivative representation/C1 knot jumps contribute",
            "body_smoothing_improves": "if 10-20 mrad smoothing lowers occupied curvature with small delta-U, short-wavelength IBI structure is implicated",
            "wider_same_barrier_improves": "if a wider same-barrier wall lowers curvature but affects little target probability, wall redesign is plausible",
            "candidate_warning": "candidate priors are unvalidated and must not replace production priors before matched NVT structure and NVE scaling tests",
        },
    }
    report_path = outdir / "angle_regularization_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[ANGLE IBI REGULARIZATION AUDIT]")
    print(
        f"current aggregate target P99 |U''|={aggregate['current']['curvature_target']['p99_abs']:.6g} "
        f"ratio_vs_old={aggregate['current']['p99_curvature_ratio_vs_old']:.3g} "
        f"sqrt={aggregate['current']['sqrt_p99_ratio_frequency_proxy']:.3g}"
    )
    for name, row in group_reports.items():
        cur = row["current"]
        print(
            f"[ANGLE] {name:12s} L1={cur['distribution_l1_runtime_vs_target']:.4f} "
            f"wall(target/runtime)={cur['wall']['target_fraction']:.3%}/{cur['wall']['runtime_fraction']:.3%} "
            f"P99|U2| target={cur['curvature_target']['p99_abs']:.6g} "
            f"freq_proxy_vs_old={cur['frequency_proxy_p99_vs_old_target']:.3g}"
        )
    print("[CANDIDATE SCREEN -- OFFLINE ONLY]")
    for spec in specs:
        row = aggregate["candidates"][spec.name]
        print(
            f"{spec.name:36s} P99red={row['p99_curvature_reduction_vs_current']:.3g}x "
            f"freq_vs_old={row['sqrt_p99_ratio_frequency_proxy_vs_old']:.3g} "
            f"dU_rms_target={row['potential_delta_target_rms_kT']:.3g} kT "
            f"max_group_dU_rms={row['max_group_potential_delta_target_rms_kT']:.3g} kT"
        )
    print(f"[DONE] report: {report_path}")
    print(f"[DONE] profiles: {outdir / 'profiles'}")
    print(f"[DONE] unvalidated candidates: {outdir / 'candidates'}")
    print("[NOTE] Do not promote a candidate before matched structural + NVE validation.")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--priors", required=True)
    p.add_argument("--old-priors", required=True)
    p.add_argument("--sample-npz", required=True)
    p.add_argument("--ibi-config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--candidate-specs-json", required=True)
    p.add_argument("--hotspot-count", type=int, required=True)
    p.add_argument("--hotspot-min-separation-rad", type=float, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    required = [Path(x).expanduser().resolve() for x in (args.dataset, args.priors, args.old_priors, args.sample_npz, args.ibi_config)]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = json.loads(Path(args.ibi_config).read_text())
    w = float(cfg["angle"]["wall_width"])
    k = float(cfg["angle"]["wall_k"])
    specs = candidate_specs_from_json(args.candidate_specs_json, w, k)
    print("[IBI ANGLE STIFFNESS/REGULARIZATION PLAN]")
    print(f"priors        : {Path(args.priors).resolve()}")
    print(f"old priors    : {Path(args.old_priors).resolve()}")
    print(f"target dataset: {Path(args.dataset).resolve()}")
    print(f"runtime sample: {Path(args.sample_npz).resolve()}")
    print(f"wall          : width={w:g} rad k={k:g} endpoint barrier={0.5*k*w*w:.6g} kJ/mol")
    print("candidates    : " + " / ".join(s.name for s in specs))
    print(f"output        : {out}")
    print("[NOTE] Offline/read-only with respect to selected priors; candidate files are unvalidated.")
    if args.dry_run:
        return
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists; use --overwrite: {out}")
        shutil.rmtree(out)
    diagnose_and_generate(
        dataset=args.dataset,
        priors=args.priors,
        old_priors=args.old_priors,
        sample_npz=args.sample_npz,
        ibi_config=args.ibi_config,
        output_dir=out,
        candidate_specs_json=args.candidate_specs_json,
        hotspot_count=args.hotspot_count,
        hotspot_min_separation_rad=args.hotspot_min_separation_rad,
    )


if __name__ == "__main__":
    main()
