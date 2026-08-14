#!/usr/bin/env python3
"""Generate initial DBI tables for bonded IBI/DBI priors.

Seed ``cg_priors.json`` entries may use ``type: ibi`` or ``type: dbi``.
Entries sharing the same ``name`` are pooled into one target distribution.  The
output JSON converts those entries to ``type: tabulated`` and preserves
``ibi_mode`` for later iterative sampling.  Only bonded distance, angle and
dihedral priors are handled here; this command is not an RDF/nonbonded IBI
implementation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry_io import pool_requested, read_target_distributions, requested_mode  # noqa: E402
from ibi_core import (  # noqa: E402
    calculate_dbi_potential,
    histogram_density,
    load_ibi_settings,
    save_tabulated_potential,
)


def safe_group_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return value or "unnamed"


def _copy_fixed_tabulated_priors(output, source_priors: Path, outdir: Path, output_priors: Path):
    """Copy pre-existing fixed tables so the generated priors remain portable."""
    for json_key in ("bonds", "angles", "dihedrals"):
        for idx, entry in enumerate(output.get(json_key, [])):
            if str(entry.get("type", "")).lower() != "tabulated":
                continue
            if requested_mode(entry) is not None:
                # Converted IBI/DBI entries are regenerated below.
                continue
            if "file" not in entry:
                raise ValueError(f"Fixed tabulated {json_key}[{idx}] is missing 'file'")
            source = Path(str(entry["file"])).expanduser()
            if not source.is_absolute():
                source = source_priors.parent / source
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(
                    f"Missing fixed tabulated table for {json_key}[{idx}]: {source}"
                )
            suffix = source.suffix or ".dat"
            destination = outdir / f"fixed_{json_key}_{idx}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source != destination.resolve():
                shutil.copy2(source, destination)
            entry["file"] = os.path.relpath(destination, output_priors.parent)


def _grid_spec(kind: str, settings: dict, values):
    cfg = settings[kind]
    vals = np.asarray(values, dtype=float)
    if kind == "bond":
        bins = np.linspace(float(cfg["hist_min"]), float(cfg["hist_max"]), int(cfg["hist_edges"]))
        grid = np.linspace(float(cfg["table_min"]), float(cfg["table_max"]), int(cfg["table_points"]))
        periodic = False
    elif kind == "angle":
        bins = np.linspace(0.0, np.pi, int(cfg["hist_edges"]))
        grid = np.linspace(0.0, np.pi, int(cfg["table_points"]))
        periodic = False
    elif kind == "dihedral":
        vals = np.mod(vals, 2.0 * np.pi)
        bins = np.linspace(0.0, 2.0 * np.pi, int(cfg["hist_edges"]))
        grid = np.linspace(0.0, 2.0 * np.pi, int(cfg["table_points"]))
        periodic = True
    else:
        raise ValueError(f"Unsupported DBI kind {kind!r}")
    return vals, bins, grid, periodic


def build_initial_dbi_priors(
    dataset: str | Path,
    priors_path: str | Path,
    outdir: str | Path,
    *,
    output_priors: str | Path | None = None,
    ibi_config: str | Path | None = None,
):
    """Generate initial bonded DBI tables and return their metadata.

    The returned dictionary contains the converted priors, the output-priors
    path and per-group target histograms needed by the iterative IBI driver.
    """
    dataset = Path(dataset).resolve()
    priors_path = Path(priors_path).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    output_priors = (
        Path(output_priors).resolve()
        if output_priors is not None
        else outdir / "cg_priors_dbi.json"
    )
    settings = load_ibi_settings(ibi_config)
    priors = json.loads(priors_path.read_text())
    output = copy.deepcopy(priors)
    _copy_fixed_tabulated_priors(output, priors_path, outdir, output_priors)

    print(f"[INFO] Reading target geometry distributions from {dataset}")
    bond_values, angle_values, dihedral_values = read_target_distributions(dataset, priors)

    specs = [
        ("bonds", "bond", bond_values),
        ("angles", "angle", angle_values),
        ("dihedrals", "dihedral", dihedral_values),
    ]
    generated = 0
    group_state = {"bonds": {}, "angles": {}, "dihedrals": {}}
    used_filenames = set()

    for json_key, kind, values in specs:
        groups = pool_requested(priors, values, json_key)
        for name, group in groups.items():
            vals, bins, grid, periodic = _grid_spec(kind, settings, group["values"])
            if vals.size == 0:
                raise ValueError(f"No target samples for {kind} group {name!r}")

            x, energy, force, target_density, hist_x, counts, support = calculate_dbi_potential(
                vals,
                bins,
                grid,
                periodic=periodic,
                jacobian_type=kind,
                settings=settings,
            )

            stem = f"{kind}_tabulated_{safe_group_name(name)}.dat"
            if stem in used_filenames:
                raise ValueError(
                    f"IBI group names produce a duplicate table filename {stem!r}; "
                    "rename the groups so their sanitized names are unique"
                )
            used_filenames.add(stem)
            table_path = outdir / stem
            save_tabulated_potential(table_path, x, energy, force)
            rel_table = Path(os.path.relpath(table_path, output_priors.parent))

            for idx in group["indices"]:
                entry = output[json_key][idx]
                entry["ibi_mode"] = group["mode"]
                entry["type"] = "tabulated"
                entry["file"] = str(rel_table)
                entry["min"] = float(x[0])
                entry["max"] = float(x[-1])

            group_state[json_key][name] = {
                "kind": kind,
                "mode": group["mode"],
                "indices": list(group["indices"]),
                "values": vals,
                "bins": bins,
                "grid": x,
                "energy": energy,
                "force": force,
                "target_density": target_density,
                "hist_x": hist_x,
                "target_counts": counts,
                "support": support,
                "table_path": table_path,
            }
            print(
                f"[DBI] {kind} {name}: N={vals.size}, "
                f"support={support[0]:.6g}..{support[1]:.6g}, table={table_path}"
            )
            generated += 1

    if generated == 0:
        raise ValueError("No type=ibi or type=dbi bonded priors were requested")

    output_priors.parent.mkdir(parents=True, exist_ok=True)
    output_priors.write_text(json.dumps(output, indent=2) + "\n")
    return {
        "priors": output,
        "output_priors": output_priors,
        "settings": settings,
        "groups": group_state,
        "generated": generated,
    }


def load_continuation_priors(
    dataset: str | Path,
    priors_path: str | Path,
    *,
    ibi_config: str | Path | None = None,
):
    """Load an already-tabulated IBI/DBI prior set for continued sampling.

    Unlike :func:`build_initial_dbi_priors`, this function never performs a new
    Boltzmann inversion.  It reconstructs the target histograms from the mapped
    reference dataset while preserving the exact energy/force tables supplied
    by ``priors_path``.  This is the required initialization for a true IBI
    continuation from a previously evaluated prior set.
    """
    dataset = Path(dataset).resolve()
    priors_path = Path(priors_path).resolve()
    settings = load_ibi_settings(ibi_config)
    priors = json.loads(priors_path.read_text())

    print(f"[INFO] Reading target geometry distributions from {dataset}")
    bond_values, angle_values, dihedral_values = read_target_distributions(dataset, priors)
    specs = [
        ("bonds", "bond", bond_values),
        ("angles", "angle", angle_values),
        ("dihedrals", "dihedral", dihedral_values),
    ]
    group_state = {"bonds": {}, "angles": {}, "dihedrals": {}}
    generated = 0

    for json_key, kind, values in specs:
        groups = pool_requested(priors, values, json_key)
        for name, group in groups.items():
            entries = [priors[json_key][idx] for idx in group["indices"]]
            if any(str(entry.get("type", "")).lower() != "tabulated" for entry in entries):
                raise ValueError(
                    f"Continuation group {kind} {name!r} is not fully tabulated; "
                    "resume from an evaluated cg_priors.json, not from the original IBI seed"
                )
            if any(requested_mode(entry) != group["mode"] for entry in entries):
                raise ValueError(f"Continuation group {kind} {name!r} has inconsistent ibi_mode values")
            if any("file" not in entry for entry in entries):
                raise ValueError(f"Continuation group {kind} {name!r} is missing a table file")

            table_paths = []
            for entry in entries:
                table_path = Path(str(entry["file"])).expanduser()
                if not table_path.is_absolute():
                    table_path = priors_path.parent / table_path
                table_paths.append(table_path.resolve())
            if len(set(table_paths)) != 1:
                raise ValueError(
                    f"Continuation group {kind} {name!r} references multiple tables: {table_paths}"
                )
            table_path = table_paths[0]
            if not table_path.is_file():
                raise FileNotFoundError(f"Missing continuation table for {kind} {name!r}: {table_path}")

            table = np.loadtxt(table_path, dtype=float)
            if table.ndim == 1:
                table = table.reshape(1, -1)
            if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] < 2:
                raise ValueError(
                    f"Continuation table must contain at least two x/energy/force rows: {table_path}"
                )
            if not np.isfinite(table).all():
                raise ValueError(f"Continuation table contains non-finite values: {table_path}")
            grid, energy, force = (np.asarray(table[:, col], dtype=float) for col in range(3))
            spacing = np.diff(grid)
            if np.any(spacing <= 0.0) or not np.allclose(
                spacing, spacing[0], rtol=1.0e-10, atol=1.0e-12
            ):
                raise ValueError(f"Continuation table grid must be strictly increasing and uniform: {table_path}")

            for entry in entries:
                if "min" in entry and not np.isclose(float(entry["min"]), grid[0], rtol=0.0, atol=1.0e-10):
                    raise ValueError(f"Continuation table minimum disagrees with prior metadata for {kind} {name!r}")
                if "max" in entry and not np.isclose(float(entry["max"]), grid[-1], rtol=0.0, atol=1.0e-10):
                    raise ValueError(f"Continuation table maximum disagrees with prior metadata for {kind} {name!r}")

            vals, bins, _default_grid, periodic = _grid_spec(kind, settings, group["values"])
            if vals.size == 0:
                raise ValueError(f"No target samples for continuation {kind} group {name!r}")
            if kind == "angle" and not (
                np.isclose(grid[0], 0.0, atol=1.0e-10)
                and np.isclose(grid[-1], np.pi, atol=1.0e-10)
            ):
                raise ValueError(f"Continuation angle table must span [0, pi]: {table_path}")
            if kind == "dihedral" and not (
                np.isclose(grid[0], 0.0, atol=1.0e-10)
                and np.isclose(grid[-1], 2.0 * np.pi, atol=1.0e-10)
            ):
                raise ValueError(f"Continuation dihedral table must span [0, 2*pi]: {table_path}")

            target_counts, target_density, hist_x = histogram_density(vals, bins)
            group_state[json_key][name] = {
                "kind": kind,
                "mode": group["mode"],
                "indices": list(group["indices"]),
                "values": vals,
                "bins": bins,
                "grid": grid,
                "energy": energy,
                "force": force,
                "target_density": target_density,
                "hist_x": hist_x,
                "target_counts": target_counts,
                "table_path": table_path,
                "periodic": periodic,
            }
            print(
                f"[RESUME] {kind} {name}: N={vals.size}, mode={group['mode']}, "
                f"table={table_path}"
            )
            generated += 1

    if generated == 0:
        raise ValueError("Continuation priors contain no tabulated entries with ibi_mode=ibi/dbi")

    return {
        "priors": copy.deepcopy(priors),
        "output_priors": priors_path,
        "settings": settings,
        "groups": group_state,
        "generated": generated,
    }


def main():
    parser = argparse.ArgumentParser(description="Build initial bonded DBI/IBI tables from a CG dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--priors", required=True, help="Seed priors containing type=ibi/dbi entries")
    parser.add_argument("--outdir", default="ibi_priors")
    parser.add_argument("--output-priors", default=None)
    parser.add_argument("--ibi-config", default=None)
    args = parser.parse_args()

    result = build_initial_dbi_priors(
        args.dataset,
        args.priors,
        args.outdir,
        output_priors=args.output_priors,
        ibi_config=args.ibi_config,
    )
    print(f"[SUCCESS] Generated {result['generated']} pooled bonded tables")
    print(f"[SUCCESS] Priors: {result['output_priors']}")


if __name__ == "__main__":
    main()
