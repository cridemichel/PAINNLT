#!/usr/bin/env python3
"""Pure helpers for diagnosing the timestep range of conservative IBI priors.

This module intentionally has no ESPResSo dependency.  It builds matched prior
variants, evaluates local bonded curvature on structured trajectories, and
summarizes where sigma(E)/dt^2 departs from its fine-dt plateau.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING = ROOT / "preprocessing"
SIMULATION = ROOT / "simulation"
import sys
for path in (PREPROCESSING, SIMULATION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from conservative_spline import load_conservative_spline  # noqa: E402
from conservative_ibi_energy_diagnostics import _angle_coordinate, _bond_coordinate  # noqa: E402
from nve_analysis import fit_metric_scaling  # noqa: E402

SCHEMA_VERSION = 1


def _topology_key(kind: str, entry: Mapping[str, Any]) -> tuple[Any, ...]:
    if kind == "bond":
        return (
            int(entry["mol_i"]), int(entry["mol_j"]),
            int(entry.get("site_i", -1)), int(entry.get("site_j", -1)),
            str(entry.get("name", "")), bool(entry.get("exclude_wca", False)),
        )
    if kind == "angle":
        return (
            int(entry["mol_i"]), int(entry["mol_j"]), int(entry["mol_k"]),
            int(entry.get("site_i", -1)), int(entry.get("site_j", -1)), int(entry.get("site_k", -1)),
            str(entry.get("name", "")), bool(entry.get("exclude_wca", False)),
        )
    raise ValueError(kind)


def _absolutize_table(entry: Mapping[str, Any], source: Path) -> dict[str, Any]:
    out = dict(entry)
    if "file" in out:
        p = Path(str(out["file"]))
        if not p.is_absolute():
            p = (source.parent / p).resolve()
        out["file"] = str(p)
    return out


def build_matched_prior_variants(old_priors: str | Path, ibi_priors: str | Path) -> dict[str, dict[str, Any]]:
    """Return old / bond-IBI / angle-IBI / full-IBI variants with identical topology."""
    old_path = Path(old_priors).expanduser().resolve()
    ibi_path = Path(ibi_priors).expanduser().resolve()
    old = json.loads(old_path.read_text())
    ibi = json.loads(ibi_path.read_text())

    for key in ("wca", "wca_pairs", "wca_exclusions", "dihedrals"):
        if old.get(key) != ibi.get(key):
            raise ValueError(f"Cannot isolate bonded IBI: old and IBI priors differ in {key!r}")

    def aligned(key: str, kind: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        a = list(old.get(key, [])); b = list(ibi.get(key, []))
        if len(a) != len(b):
            raise ValueError(f"Topology mismatch in {key}: {len(a)} vs {len(b)}")
        amap = {_topology_key(kind, row): row for row in a}
        bmap = {_topology_key(kind, row): row for row in b}
        if set(amap) != set(bmap):
            raise ValueError(f"Topology keys differ between old and IBI {key}")
        keys = [_topology_key(kind, row) for row in a]
        return [dict(amap[k]) for k in keys], [dict(bmap[k]) for k in keys]

    old_bonds, ibi_bonds = aligned("bonds", "bond")
    old_angles, ibi_angles = aligned("angles", "angle")

    # Require unchanged Morse entries and paired harmonic -> conservative-spline replacements.
    for o, n in zip(old_bonds, ibi_bonds):
        ot, nt = str(o.get("type", "")).lower(), str(n.get("type", "")).lower()
        if ot == "morse":
            if o != n:
                raise ValueError(f"Common Morse term changed for {_topology_key('bond', o)}")
        elif not (ot == "harmonic" and nt == "conservative_spline"):
            raise ValueError(f"Unexpected bond replacement {ot!r} -> {nt!r}")
    for o, n in zip(old_angles, ibi_angles):
        ot, nt = str(o.get("type", "")).lower(), str(n.get("type", "")).lower()
        if not (ot == "harmonic" and nt == "conservative_spline"):
            raise ValueError(f"Unexpected angle replacement {ot!r} -> {nt!r}")

    def base_payload() -> dict[str, Any]:
        out = json.loads(json.dumps(old))
        out["diagnostic_kind"] = "matched_old_vs_conservative_ibi_timestep_range"
        out["diagnostic_old_priors"] = str(old_path)
        out["diagnostic_ibi_priors"] = str(ibi_path)
        return out

    variants: dict[str, dict[str, Any]] = {}
    for name, use_ibi_bonds, use_ibi_angles in (
        ("old_tel22", False, False),
        ("ibi_bonds_only", True, False),
        ("ibi_angles_only", False, True),
        ("full_ibi", True, True),
    ):
        out = base_payload()
        chosen_bonds = ibi_bonds if use_ibi_bonds else old_bonds
        chosen_angles = ibi_angles if use_ibi_angles else old_angles
        out["bonds"] = [
            _absolutize_table(row, ibi_path if str(row.get("type", "")).lower() == "conservative_spline" else old_path)
            for row in chosen_bonds
        ]
        out["angles"] = [
            _absolutize_table(row, ibi_path if str(row.get("type", "")).lower() == "conservative_spline" else old_path)
            for row in chosen_angles
        ]
        out["diagnostic_variant"] = name
        variants[name] = out
    return variants


def write_matched_prior_variants(old_priors: str | Path, ibi_priors: str | Path, output_dir: str | Path) -> dict[str, Path]:
    outdir = Path(output_dir).expanduser().resolve(); outdir.mkdir(parents=True, exist_ok=True)
    variants = build_matched_prior_variants(old_priors, ibi_priors)
    paths: dict[str, Path] = {}
    for name, payload in variants.items():
        path = outdir / f"cg_priors_{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        paths[name] = path
    return paths


def _hermite_second_derivative(table, q: np.ndarray) -> np.ndarray:
    x = np.asarray(table.x, dtype=float)
    y = np.asarray(table.energy, dtype=float)
    m = np.asarray(table.derivative, dtype=float)
    q = np.asarray(q, dtype=float)
    h = float(x[1] - x[0])
    idx = np.floor((q - x[0]) / h).astype(int)
    idx = np.clip(idx, 0, len(x) - 2)
    t = np.clip((q - x[idx]) / h, 0.0, 1.0)
    return (
        (12.0 * t - 6.0) * y[idx]
        + (6.0 * t - 4.0) * h * m[idx]
        + (-12.0 * t + 6.0) * y[idx + 1]
        + (6.0 * t - 2.0) * h * m[idx + 1]
    ) / (h * h)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "p50_abs": math.nan, "p95_abs": math.nan, "p99_abs": math.nan, "max_abs": math.nan}
    av = np.abs(values)
    return {
        "n": int(values.size),
        "p50_abs": float(np.percentile(av, 50)),
        "p95_abs": float(np.percentile(av, 95)),
        "p99_abs": float(np.percentile(av, 99)),
        "max_abs": float(np.max(av)),
        "min_signed": float(np.min(values)),
        "max_signed": float(np.max(values)),
    }


def visited_curvature_report(priors_path: str | Path, sample_npz: str | Path) -> dict[str, Any]:
    priors_path = Path(priors_path).expanduser().resolve()
    priors = json.loads(priors_path.read_text())
    with np.load(sample_npz, allow_pickle=False) as data:
        sample = {k: np.asarray(data[k]) for k in data.files}

    by_kind: dict[str, list[float]] = {"bond": [], "angle": []}
    by_name: dict[str, dict[str, Any]] = {}
    for key, kind in (("bonds", "bond"), ("angles", "angle")):
        for entry in priors.get(key, []):
            etype = str(entry.get("type", "")).lower()
            # Morse is common between variants and is intentionally excluded from the changed-bond stiffness comparison.
            if etype not in {"harmonic", "conservative_spline"}:
                continue
            q = _bond_coordinate(sample, entry) if kind == "bond" else _angle_coordinate(sample, entry)
            if etype == "harmonic":
                u2 = np.full_like(q, float(entry["k"]), dtype=float)
            else:
                table = load_conservative_spline(entry, kind=kind, priors_path=priors_path)
                u2 = _hermite_second_derivative(table, q)
            by_kind[kind].extend(np.asarray(u2, dtype=float).tolist())
            name = str(entry.get("name", "unnamed"))
            bucket = by_name.setdefault(name, {"kind": kind, "type": etype, "values": []})
            bucket["values"].extend(np.asarray(u2, dtype=float).tolist())

    groups = {
        name: {"kind": row["kind"], "type": row["type"], **_summary(np.asarray(row["values"], dtype=float))}
        for name, row in sorted(by_name.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "visited_bonded_curvature",
        "priors": str(priors_path),
        "sample_npz": str(Path(sample_npz).expanduser().resolve()),
        "bond": _summary(np.asarray(by_kind["bond"], dtype=float)),
        "angle": _summary(np.asarray(by_kind["angle"], dtype=float)),
        "groups": groups,
    }


def sigma_range_diagnostics(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted((dict(row) for row in runs), key=lambda r: float(r["dt_ps"]))
    if len(rows) < 3:
        raise ValueError("Need at least three timestep runs")
    fit = fit_metric_scaling(rows, "sigma_E")
    dt = np.asarray([float(r["dt_ps"]) for r in rows], dtype=float)
    sigma = np.asarray([float(r["sigma_E"]) for r in rows], dtype=float)
    c2 = sigma / (dt * dt)
    plateau = float(c2[0])
    local = []
    for i in range(len(rows) - 1):
        p = math.log(sigma[i + 1] / sigma[i]) / math.log(dt[i + 1] / dt[i])
        local.append({
            "dt_low_ps": float(dt[i]), "dt_high_ps": float(dt[i + 1]), "local_exponent_p": float(p),
            "c2_low": float(c2[i]), "c2_high": float(c2[i + 1]),
        })
    return {
        "fit": fit,
        "dt_ps": dt.tolist(),
        "sigma_E": sigma.tolist(),
        "sigma_over_dt2": c2.tolist(),
        "sigma_over_dt2_relative_to_smallest_dt": (c2 / plateau).tolist(),
        "c2_spread_max_over_min": float(np.max(c2) / np.min(c2)),
        "adjacent_local_exponents": local,
    }


def stiffness_ratios(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if "old_tel22" not in reports:
        raise ValueError("old_tel22 curvature report is required")
    out: dict[str, Any] = {}
    for variant, report in reports.items():
        if variant == "old_tel22":
            continue
        per_kind = {}
        for kind in ("bond", "angle"):
            base = float(reports["old_tel22"][kind]["p99_abs"])
            value = float(report[kind]["p99_abs"])
            ratio = value / base if base > 0.0 else math.nan
            per_kind[kind] = {
                "p99_abs_curvature_ratio_vs_old": ratio,
                "sqrt_ratio_frequency_proxy": math.sqrt(ratio) if ratio >= 0.0 and math.isfinite(ratio) else math.nan,
            }
        out[variant] = per_kind
    return out
