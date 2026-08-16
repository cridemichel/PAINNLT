#!/usr/bin/env python3
"""Pure helpers for conservative-IBI energy-scaling localization diagnostics.

The runtime tests live in separate scripts because they require ``pypresso``.
This module is intentionally importable with ordinary Python so spline
smoothness, prior variants, trajectory traces and report aggregation can be
unit tested without ESPResSo.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING = ROOT / "preprocessing"
import sys
if str(PREPROCESSING) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING))

from conservative_spline import (  # noqa: E402
    ConservativeSplinePrior,
    conservative_spline_value,
    load_conservative_spline,
)
from nve_analysis import analyze_energy_series, fit_metric_scaling  # noqa: E402


SCHEMA_VERSION = 1


def _hermite_second_derivative_endpoint(
    y0: float, y1: float, m0: float, m1: float, h: float, *, right: bool
) -> float:
    """Return d2U/dq2 at one endpoint of a cubic Hermite interval."""
    if h <= 0.0:
        raise ValueError("Hermite spacing must be positive")
    if right:
        # t=1: d2/dq2 = (6*y0 + 2*h*m0 - 6*y1 + 4*h*m1)/h^2
        return (6.0 * y0 + 2.0 * h * m0 - 6.0 * y1 + 4.0 * h * m1) / (h * h)
    # t=0: d2/dq2 = (-6*y0 - 4*h*m0 + 6*y1 - 2*h*m1)/h^2
    return (-6.0 * y0 - 4.0 * h * m0 + 6.0 * y1 - 2.0 * h * m1) / (h * h)


def curvature_jumps(table: ConservativeSplinePrior) -> dict[str, Any]:
    """Measure jumps of U'' at all internal knots of the C1 Hermite spline."""
    x = np.asarray(table.x, dtype=float)
    u = np.asarray(table.energy, dtype=float)
    du = np.asarray(table.derivative, dtype=float)
    if x.size < 3:
        raise ValueError("At least three spline nodes are required to inspect internal knots")
    h = float(x[1] - x[0])
    jumps = []
    for k in range(1, len(x) - 1):
        left = _hermite_second_derivative_endpoint(
            float(u[k - 1]), float(u[k]), float(du[k - 1]), float(du[k]), h, right=True
        )
        right = _hermite_second_derivative_endpoint(
            float(u[k]), float(u[k + 1]), float(du[k]), float(du[k + 1]), h, right=False
        )
        jumps.append({
            "knot_index": int(k),
            "q": float(x[k]),
            "u2_left": float(left),
            "u2_right": float(right),
            "jump": float(right - left),
            "abs_jump": float(abs(right - left)),
        })
    abs_jumps = np.asarray([row["abs_jump"] for row in jumps], dtype=float)
    scale = np.asarray(
        [max(abs(row["u2_left"]), abs(row["u2_right"]), 1.0e-30) for row in jumps],
        dtype=float,
    )
    relative = abs_jumps / scale
    return {
        "kind": table.kind,
        "file": str(table.path),
        "n_nodes": int(len(x)),
        "spacing": h,
        "n_internal_knots": int(len(jumps)),
        "max_abs_u2_jump": float(np.max(abs_jumps)) if len(jumps) else 0.0,
        "median_abs_u2_jump": float(np.median(abs_jumps)) if len(jumps) else 0.0,
        "max_relative_u2_jump": float(np.max(relative)) if len(jumps) else 0.0,
        "median_relative_u2_jump": float(np.median(relative)) if len(jumps) else 0.0,
        "knots": jumps,
    }


def unique_conservative_entries(priors: Mapping[str, Any]):
    seen: set[tuple[str, str]] = set()
    for json_key, kind in (("bonds", "bond"), ("angles", "angle")):
        for index, entry in enumerate(priors.get(json_key, [])):
            if str(entry.get("type", "")).lower() != "conservative_spline":
                continue
            key = (kind, str(entry.get("file", "")))
            if key in seen:
                continue
            seen.add(key)
            yield json_key, index, kind, entry


def inspect_prior_smoothness(priors_path: str | Path) -> dict[str, Any]:
    path = Path(priors_path).expanduser().resolve()
    priors = json.loads(path.read_text())
    rows = []
    for _json_key, _index, kind, entry in unique_conservative_entries(priors):
        table = load_conservative_spline(entry, kind=kind, priors_path=path)
        item = curvature_jumps(table)
        item["name"] = str(entry.get("name", Path(str(entry["file"])).stem))
        item["prior_file"] = str(entry["file"])
        rows.append(item)
    if not rows:
        raise ValueError(f"No conservative spline entries found in {path}")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "conservative_ibi_spline_smoothness",
        "priors": str(path),
        "tables": rows,
        "worst_max_abs_u2_jump": max(float(row["max_abs_u2_jump"]) for row in rows),
        "worst_max_relative_u2_jump": max(float(row["max_relative_u2_jump"]) for row in rows),
    }


def _absolutize_table_file(entry: dict[str, Any], source_priors: Path) -> dict[str, Any]:
    result = dict(entry)
    if "file" in result:
        table = Path(str(result["file"])).expanduser()
        if not table.is_absolute():
            table = source_priors.parent / table
        result["file"] = str(table.resolve())
    return result


def diagnostic_prior_variant(
    priors_path: str | Path,
    *,
    variant: str,
) -> dict[str, Any]:
    """Return a topology-preserving diagnostic variant of conservative priors.

    Disabled bonded terms are replaced by zero-strength analytic interactions,
    not deleted, so the WCA 1-2/1-3 exclusion topology remains unchanged.
    """
    source = Path(priors_path).expanduser().resolve()
    priors = json.loads(source.read_text())
    variant = str(variant).strip().lower()
    allowed = {"full", "no_ibi", "bonds_only", "angles_only"}
    if variant not in allowed:
        raise ValueError(f"Unknown diagnostic prior variant {variant!r}; expected {sorted(allowed)}")

    out = json.loads(json.dumps(priors))
    for key in ("bonds", "angles", "dihedrals"):
        out.setdefault(key, [])

    new_bonds = []
    for entry in out["bonds"]:
        etype = str(entry.get("type", "harmonic")).lower()
        keep = etype != "conservative_spline" or variant in {"full", "bonds_only"}
        if keep:
            new_bonds.append(_absolutize_table_file(entry, source))
        else:
            replacement = {
                k: v for k, v in entry.items()
                if k not in {"file", "min", "max", "spline_schema", "source_tabulated_sha256"}
            }
            replacement.update({"type": "harmonic", "k": 0.0, "r0": 1.0, "diagnostic_disabled_original_type": "conservative_spline"})
            new_bonds.append(replacement)
    out["bonds"] = new_bonds

    new_angles = []
    for entry in out["angles"]:
        etype = str(entry.get("type", "harmonic")).lower()
        keep = etype != "conservative_spline" or variant in {"full", "angles_only"}
        if keep:
            new_angles.append(_absolutize_table_file(entry, source))
        else:
            replacement = {
                k: v for k, v in entry.items()
                if k not in {"file", "min", "max", "spline_schema", "source_tabulated_sha256"}
            }
            replacement.update({"type": "harmonic", "k": 0.0, "theta0": math.pi / 2.0, "diagnostic_disabled_original_type": "conservative_spline"})
            new_angles.append(replacement)
    out["angles"] = new_angles
    out["diagnostic_variant"] = variant
    out["diagnostic_source_priors"] = str(source)
    return out


def write_diagnostic_prior_variants(priors_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for variant in ("no_ibi", "bonds_only", "angles_only", "full"):
        target = outdir / f"cg_priors_{variant}.json"
        target.write_text(json.dumps(diagnostic_prior_variant(priors_path, variant=variant), indent=2, sort_keys=True) + "\n")
        paths[variant] = str(target)
    return paths


def reverse_checkpoint_velocities(source: str | Path, target: str | Path) -> None:
    source = Path(source).expanduser().resolve()
    target = Path(target).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    for key in ("v", "omega"):
        if key not in payload:
            raise ValueError(f"Checkpoint {source} lacks required {key!r} array")
        payload[key] = -np.asarray(payload[key], dtype=float)
    if "metadata_json" in payload:
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        metadata["diagnostic_velocity_reversal_of"] = str(source)
        payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)


def _quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dots = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def compare_time_reversal(initial: str | Path, returned: str | Path) -> dict[str, float]:
    with np.load(initial, allow_pickle=False) as a, np.load(returned, allow_pickle=False) as b:
        for key in ("pos", "v", "quat", "omega", "particle_is_virtual"):
            if key not in a.files or key not in b.files:
                raise ValueError(f"Time-reversal checkpoint comparison requires {key!r}")
        mask = ~np.asarray(a["particle_is_virtual"], dtype=bool)
        if not np.array_equal(mask, ~np.asarray(b["particle_is_virtual"], dtype=bool)):
            raise ValueError("Time-reversal checkpoints have different particle identities")
        pos_a = np.asarray(a["pos"], dtype=float)[mask]
        pos_b = np.asarray(b["pos"], dtype=float)[mask]
        box = np.asarray(a["box_l"], dtype=float)
        dpos = pos_b - pos_a
        dpos -= box * np.round(dpos / box)
        # After forward evolution, velocity reversal, and the same forward
        # evolution, the returned velocities/omegas should be minus the initial ones.
        dv = np.asarray(b["v"], dtype=float)[mask] + np.asarray(a["v"], dtype=float)[mask]
        domega = np.asarray(b["omega"], dtype=float)[mask] + np.asarray(a["omega"], dtype=float)[mask]
        orient = _quat_angle(np.asarray(a["quat"], dtype=float)[mask], np.asarray(b["quat"], dtype=float)[mask])
    return {
        "position_rms_nm": float(np.sqrt(np.mean(dpos * dpos))),
        "position_max_nm": float(np.max(np.linalg.norm(dpos, axis=1))),
        "velocity_rms_nm_per_ps": float(np.sqrt(np.mean(dv * dv))),
        "omega_body_rms_per_ps": float(np.sqrt(np.mean(domega * domega))),
        "orientation_rms_rad": float(np.sqrt(np.mean(orient * orient))),
        "orientation_max_rad": float(np.max(orient)),
    }


def read_energy_rows(path: str | Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            numeric: dict[str, float] = {}
            for key, value in row.items():
                try:
                    numeric[key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(numeric)
    if len(rows) < 2:
        raise ValueError(f"Energy trace {path} has fewer than two rows")
    return rows


def _sample_site_lookup(sample: Mapping[str, np.ndarray]) -> dict[tuple[int, int], int]:
    mol = np.asarray(sample["site_molecule"], dtype=int)
    idx = np.asarray(sample["site_index"], dtype=int)
    return {(int(m), int(s)): i for i, (m, s) in enumerate(zip(mol, idx))}


def _endpoint_positions(sample: Mapping[str, np.ndarray], mol: int, site: int) -> np.ndarray:
    if int(site) == -1:
        return np.asarray(sample["com"], dtype=float)[:, int(mol), :]
    lookup = _sample_site_lookup(sample)
    key = (int(mol), int(site))
    if key not in lookup:
        raise ValueError(f"Structured sample lacks endpoint {key}")
    return np.asarray(sample["sites"], dtype=float)[:, lookup[key], :]


def _minimum_image(vec: np.ndarray, box: np.ndarray) -> np.ndarray:
    return vec - box * np.round(vec / box)


def _bond_coordinate(sample: Mapping[str, np.ndarray], entry: Mapping[str, Any]) -> np.ndarray:
    a = _endpoint_positions(sample, int(entry["mol_i"]), int(entry.get("site_i", -1)))
    b = _endpoint_positions(sample, int(entry["mol_j"]), int(entry.get("site_j", -1)))
    box = np.asarray(sample["box"], dtype=float)
    d = _minimum_image(b - a, box)
    return np.linalg.norm(d, axis=1)


def _angle_coordinate(sample: Mapping[str, np.ndarray], entry: Mapping[str, Any]) -> np.ndarray:
    a = _endpoint_positions(sample, int(entry["mol_i"]), int(entry.get("site_i", -1)))
    b = _endpoint_positions(sample, int(entry["mol_j"]), int(entry.get("site_j", -1)))
    c = _endpoint_positions(sample, int(entry["mol_k"]), int(entry.get("site_k", -1)))
    box = np.asarray(sample["box"], dtype=float)
    ba = _minimum_image(a - b, box)
    bc = _minimum_image(c - b, box)
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    cosv = np.sum(ba * bc, axis=1) / denom
    return np.arccos(np.clip(cosv, -1.0, 1.0))


def _cell_index(q: np.ndarray, table: ConservativeSplinePrior) -> np.ndarray:
    h = float(table.x[1] - table.x[0])
    scaled = np.floor((np.asarray(q, dtype=float) - table.minimum) / h).astype(int)
    return np.clip(scaled, 0, len(table.x) - 2)


def analyze_knot_trace(
    *,
    priors_path: str | Path,
    sample_npz: str | Path,
    energy_csv: str | Path,
) -> dict[str, Any]:
    """Correlate per-step energy increments with conservative-spline knot crossings."""
    priors_path = Path(priors_path).expanduser().resolve()
    priors = json.loads(priors_path.read_text())
    with np.load(sample_npz, allow_pickle=False) as data:
        sample = {key: np.asarray(data[key]) for key in data.files}
    times = np.asarray(sample["time_ps"], dtype=float)
    rows = read_energy_rows(energy_csv)
    e_time = np.asarray([row["Time_ps"] for row in rows], dtype=float)
    energy = np.asarray([row["E_tot"] for row in rows], dtype=float)
    if len(times) != len(e_time) or not np.allclose(times, e_time, rtol=0.0, atol=1.0e-12):
        raise ValueError("Structured sample and energy CSV must contain the same exact sampling times")

    crossing_count = np.zeros(len(times) - 1, dtype=int)
    weighted_jump = np.zeros(len(times) - 1, dtype=float)
    bond_cross = np.zeros_like(crossing_count)
    angle_cross = np.zeros_like(crossing_count)
    details = []

    for _json_key, _index, kind, entry in unique_conservative_entries(priors):
        table = load_conservative_spline(entry, kind=kind, priors_path=priors_path)
        q = _bond_coordinate(sample, entry) if kind == "bond" else _angle_coordinate(sample, entry)
        cells = _cell_index(q, table)
        smooth = curvature_jumps(table)
        jump_by_knot = {int(item["knot_index"]): float(item["abs_jump"]) for item in smooth["knots"]}
        local_cross = np.zeros(len(times) - 1, dtype=int)
        local_weight = np.zeros(len(times) - 1, dtype=float)
        for n, (c0, c1) in enumerate(zip(cells[:-1], cells[1:])):
            if c0 == c1:
                continue
            lo, hi = sorted((int(c0), int(c1)))
            knots = range(lo + 1, hi + 1)
            local_cross[n] = abs(int(c1) - int(c0))
            local_weight[n] = sum(jump_by_knot.get(k, 0.0) for k in knots)
        crossing_count += local_cross
        weighted_jump += local_weight
        if kind == "bond":
            bond_cross += local_cross
        else:
            angle_cross += local_cross
        details.append({
            "kind": kind,
            "name": str(entry.get("name", entry.get("file", ""))),
            "file": str(entry.get("file", "")),
            "total_crossings": int(np.sum(local_cross)),
            "steps_with_crossing": int(np.count_nonzero(local_cross)),
            "q_min": float(np.min(q)),
            "q_max": float(np.max(q)),
        })

    abs_de = np.abs(np.diff(energy))
    mask = crossing_count > 0
    no_mask = ~mask
    crossing_mean = float(np.mean(abs_de[mask])) if np.any(mask) else 0.0
    no_crossing_mean = float(np.mean(abs_de[no_mask])) if np.any(no_mask) else 0.0
    ratio = crossing_mean / no_crossing_mean if no_crossing_mean > 0.0 else math.inf
    corr_count = float(np.corrcoef(crossing_count, abs_de)[0, 1]) if np.std(crossing_count) > 0 and np.std(abs_de) > 0 else 0.0
    corr_weight = float(np.corrcoef(weighted_jump, abs_de)[0, 1]) if np.std(weighted_jump) > 0 and np.std(abs_de) > 0 else 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "conservative_ibi_knot_crossing_energy_correlation",
        "samples": int(len(times)),
        "steps": int(len(abs_de)),
        "steps_with_any_crossing": int(np.count_nonzero(mask)),
        "fraction_steps_with_crossing": float(np.mean(mask)),
        "total_crossings": int(np.sum(crossing_count)),
        "total_bond_crossings": int(np.sum(bond_cross)),
        "total_angle_crossings": int(np.sum(angle_cross)),
        "mean_abs_delta_E_crossing": crossing_mean,
        "mean_abs_delta_E_no_crossing": no_crossing_mean,
        "crossing_to_no_crossing_abs_delta_E_ratio": float(ratio),
        "pearson_crossing_count_vs_abs_delta_E": corr_count,
        "pearson_weighted_u2_jump_vs_abs_delta_E": corr_weight,
        "per_table": details,
    }


def analyze_energy_decomposition(energy_csv: str | Path) -> dict[str, Any]:
    rows = read_energy_rows(energy_csv)
    columns = [
        "E_tot", "E_kin", "E_kin_trans", "E_kin_rot", "E_class", "E_ml",
        "E_bonded", "E_non_bonded",
    ]
    out: dict[str, Any] = {}
    for column in columns:
        if column not in rows[0]:
            continue
        values = np.asarray([row[column] for row in rows], dtype=float)
        delta = np.diff(values)
        out[column] = {
            "mean": float(np.mean(values)),
            "sigma": float(np.std(values)),
            "rms_step_delta": float(np.sqrt(np.mean(delta * delta))) if len(delta) else 0.0,
            "max_abs_step_delta": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        }
    return {"schema_version": SCHEMA_VERSION, "kind": "energy_decomposition", "terms": out}


def summarize_scaling_runs(run_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    runs = [dict(row) for row in run_rows]
    for run in runs:
        if "sigma_E" not in run:
            metrics = analyze_energy_series(run["times_ps"], run["energies"])
            run.update(metrics)
    fit = fit_metric_scaling(runs, "sigma_E", label="sigma_E")
    return {"runs": runs, "fit": fit}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("smoothness")
    p.add_argument("--priors", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("make-priors")
    p.add_argument("--priors", required=True)
    p.add_argument("--output-dir", required=True)

    p = sub.add_parser("reverse-checkpoint")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("compare-reversal")
    p.add_argument("--initial", required=True)
    p.add_argument("--returned", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("knot-trace")
    p.add_argument("--priors", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--energy", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("energy-decomposition")
    p.add_argument("--energy", required=True)
    p.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "smoothness":
        report = inspect_prior_smoothness(args.priors)
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    elif args.command == "make-priors":
        report = write_diagnostic_prior_variants(args.priors, args.output_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "reverse-checkpoint":
        reverse_checkpoint_velocities(args.input, args.output)
    elif args.command == "compare-reversal":
        report = compare_time_reversal(args.initial, args.returned)
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "knot-trace":
        report = analyze_knot_trace(priors_path=args.priors, sample_npz=args.sample, energy_csv=args.energy)
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "energy-decomposition":
        report = analyze_energy_decomposition(args.energy)
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
