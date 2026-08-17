#!/usr/bin/env python3
"""Generate one unvalidated conservative IBI angle-body smoothing candidate.

This module is deliberately narrow.  It starts from an existing conservative
IBI prior set, subtracts the configured quadratic endpoint wall from each angle
table, Gaussian-smooths only the remaining potential body, re-adds the same
wall, and exports C2 cubic-spline nodal derivatives into the existing Hermite
runtime format.  Bonds/dihedrals are copied unchanged into a self-contained
candidate directory.

The output is *not* validated or promoted automatically.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "preprocessing", ROOT / "ibi"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from angle_regularization_diagnostics import (  # noqa: E402
    CandidateSpec,
    _build_candidate,
    _copy_nonangle_tables,
    _safe_name,
    sha256_file,
)
from conservative_spline import load_conservative_spline, save_conservative_spline  # noqa: E402

SCHEMA_VERSION = 1
KIND = "unvalidated_ibi_angle_smoothing_candidate"


def smoothing_name(sigma_rad: float, *, suffix: str = "wall_current") -> str:
    sigma = float(sigma_rad)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("body smoothing sigma must be positive and finite")
    text = f"{sigma:.8f}".rstrip("0").rstrip(".")
    if text.startswith("0."):
        text = "0p" + text[2:]
    else:
        text = text.replace(".", "p")
    return f"smooth_{text}_{suffix}"


def generate_candidate(
    *,
    source_priors: str | Path,
    ibi_config: str | Path,
    body_sigma_rad: float,
    output_dir: str | Path,
    candidate_name: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(source_priors).expanduser().resolve()
    config_path = Path(ibi_config).expanduser().resolve()
    outdir = Path(output_dir).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    sigma = float(body_sigma_rad)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("body_sigma_rad must be positive and finite")

    config = json.loads(config_path.read_text())
    angle_cfg = config["angle"]
    wall_width = float(angle_cfg["wall_width"])
    wall_k = float(angle_cfg["wall_k"])
    name = candidate_name or smoothing_name(sigma)
    spec = CandidateSpec(
        name=name,
        body_sigma_rad=sigma,
        wall_width_rad=wall_width,
        wall_k=wall_k,
        note=(
            f"{sigma:g} rad Gaussian smoothing of the de-walled IBI angle body; "
            "current endpoint wall retained; C2 cubic-spline nodal derivatives"
        ),
    )

    if outdir.exists() and any(outdir.iterdir()):
        if overwrite:
            shutil.rmtree(outdir)
        else:
            raise FileExistsError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    source = json.loads(source_path.read_text())
    payload = copy.deepcopy(source)
    _copy_nonangle_tables(source, source_path, outdir, payload)

    groups: dict[str, list[int]] = {}
    for idx, entry in enumerate(source.get("angles", [])):
        if str(entry.get("type", "")).lower() != "conservative_spline":
            raise ValueError(f"Angle entry {idx} is not conservative_spline")
        group_name = str(entry.get("name", "")).strip()
        if not group_name:
            raise ValueError(f"Angle entry {idx} has no group name")
        groups.setdefault(group_name, []).append(idx)
    if not groups:
        raise ValueError("No angle priors found")

    angle_tables: dict[str, Any] = {}
    for group_name, indices in sorted(groups.items()):
        tables = [load_conservative_spline(source["angles"][idx], kind="angle", priors_path=source_path) for idx in indices]
        paths = {table.path.resolve() for table in tables}
        if len(paths) != 1:
            raise ValueError(f"Angle group {group_name!r} references multiple source tables: {sorted(map(str, paths))}")
        table = tables[0]
        c2, new_u, new_du = _build_candidate(table, spec, wall_width, wall_k)
        filename = f"angle_conservative_{_safe_name(group_name)}.dat"
        output_table = outdir / filename
        save_conservative_spline(output_table, np.asarray(table.x), new_u, new_du)
        angle_tables[group_name] = {
            "source_table": str(table.path.resolve()),
            "source_table_sha256": sha256_file(table.path),
            "output_table": str(output_table),
            "output_table_sha256": sha256_file(output_table),
            "grid_points": int(len(table.x)),
            "body_sigma_rad": sigma,
            "wall_width_rad": wall_width,
            "wall_k": wall_k,
            "max_abs_u2_on_nodes": float(np.max(np.abs(c2(np.asarray(table.x), 2)))),
        }
        for idx in indices:
            row = payload["angles"][idx]
            row["file"] = filename
            row["regularization"] = {
                "kind": "angle_body_gaussian_plus_c2_v1",
                "candidate": name,
                "body_sigma_rad": sigma,
                "wall_width_rad": wall_width,
                "wall_k": wall_k,
                "regularized_from_conservative_sha256": sha256_file(table.path),
                "validated": False,
            }

    payload["regularization_candidate"] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "source_priors": str(source_path),
        "source_priors_sha256": sha256_file(source_path),
        "candidate": name,
        "body_sigma_rad": sigma,
        "wall_width_rad": wall_width,
        "wall_k": wall_k,
        "endpoint_barrier_kJmol": 0.5 * wall_k * wall_width * wall_width,
        "validated": False,
        "note": spec.note,
    }
    priors_out = outdir / "cg_priors.json"
    priors_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "candidate": name,
        "validated": False,
        "body_sigma_rad": sigma,
        "wall_width_rad": wall_width,
        "wall_k": wall_k,
        "source_priors": str(source_path),
        "source_priors_sha256": sha256_file(source_path),
        "candidate_priors": str(priors_out),
        "candidate_priors_sha256": sha256_file(priors_out),
        "angle_tables": angle_tables,
    }
    (outdir / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-priors", required=True)
    p.add_argument("--ibi-config", required=True)
    p.add_argument("--body-sigma-rad", type=float, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--candidate-name")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    name = args.candidate_name or smoothing_name(args.body_sigma_rad)
    print("[ANGLE SMOOTHING CANDIDATE PLAN]")
    print(f"source priors : {Path(args.source_priors).expanduser().resolve()}")
    print(f"IBI settings  : {Path(args.ibi_config).expanduser().resolve()}")
    print(f"sigma         : {args.body_sigma_rad:g} rad")
    print(f"candidate     : {name}")
    print(f"output        : {Path(args.output_dir).expanduser().resolve()}")
    print("[NOTE] Candidate is unvalidated and the source priors are never modified.")
    if args.dry_run:
        return
    manifest = generate_candidate(
        source_priors=args.source_priors,
        ibi_config=args.ibi_config,
        body_sigma_rad=args.body_sigma_rad,
        output_dir=args.output_dir,
        candidate_name=name,
        overwrite=args.overwrite,
    )
    print(f"[DONE] candidate priors: {manifest['candidate_priors']}")


if __name__ == "__main__":
    main()
