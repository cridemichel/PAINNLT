#!/usr/bin/env python3
"""Convert validated tabulated bonded IBI priors to conservative Hermite splines.

The scalar energy column of each source IBI table is treated as fundamental.
A shape-preserving PCHIP is fitted to ``U(q)`` and exported as nodal
``q, U, dU/dq`` values.  The runtime plugin and preprocessing reconstruct the
same cubic Hermite segments, so force is the exact analytical derivative of
that energy representation.

This tool is system-agnostic.  It converts bond, angle and periodic dihedral
tables.  Dihedral U(phi) is represented by a periodic cubic spline; runtime
uses the analytical Cartesian gradient of ESPResSo's signed dihedral angle
with that same dU/dphi.  Legacy force factors are used only for fidelity metrics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))

from conservative_spline import (  # noqa: E402
    SCHEMA,
    ConservativeSplinePrior,
    conservative_spline_value,
    load_conservative_spline,
    save_conservative_spline,
)
from prior_kernels import load_tabulated_prior, tabulated_value  # noqa: E402


SCHEMA_VERSION = 1
KIND = "ibi_conservative_spline_conversion"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_stem(path: Path, kind: str) -> str:
    stem = path.stem
    prefix = f"{kind}_tabulated_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return f"{kind}_conservative_{stem}.dat"


def _force_from_derivative(kind: str, grid: np.ndarray, derivative: np.ndarray) -> np.ndarray:
    derivative = np.asarray(derivative, dtype=float)
    if kind == "bond":
        return -derivative
    if kind == "angle":
        return derivative
    if kind == "dihedral":
        sin_phi = np.sin(np.asarray(grid, dtype=float))
        factor = np.zeros_like(derivative)
        regular = np.abs(sin_phi) > 1.0e-6
        factor[regular] = -derivative[regular] / sin_phi[regular]
        good = np.flatnonzero(regular)
        if good.size == 0:
            raise ValueError("Dihedral grid has no regular points for force-factor comparison")
        for idx in np.flatnonzero(~regular):
            nearest = good[np.argmin(np.abs(good - idx))]
            factor[idx] = factor[nearest]
        return factor
    raise ValueError(f"Unsupported conservative conversion kind: {kind}")


def _metrics(source, spline: ConservativeSplinePrior, *, dense_per_interval: int = 4):
    intervals = len(source.x) - 1
    n_dense = intervals * max(2, int(dense_per_interval)) + 1
    q = np.linspace(source.minimum, source.maximum, n_dense)
    old_u = np.interp(q, source.x, source.energy)
    old_f = np.interp(q, source.x, source.force)
    new_u = np.empty_like(q)
    new_du = np.empty_like(q)
    for i, value in enumerate(q):
        new_u[i], new_du[i] = conservative_spline_value(spline, float(value))
    new_f = _force_from_derivative(source.kind, q, new_du)
    du = new_u - old_u
    df = new_f - old_f
    force_scale = max(float(np.sqrt(np.mean(old_f * old_f))), 1.0e-12)
    energy_scale = max(float(np.ptp(old_u)), 1.0e-12)
    return {
        "dense_points": int(n_dense),
        "energy_rms_abs": float(np.sqrt(np.mean(du * du))),
        "energy_max_abs": float(np.max(np.abs(du))),
        "energy_rms_relative_to_range": float(np.sqrt(np.mean(du * du)) / energy_scale),
        "force_rms_abs": float(np.sqrt(np.mean(df * df))),
        "force_p99_abs": float(np.percentile(np.abs(df), 99.0)),
        "force_max_abs": float(np.max(np.abs(df))),
        "force_rms_relative": float(np.sqrt(np.mean(df * df)) / force_scale),
    }


def convert(priors_path: Path, output_dir: Path, *, overwrite: bool = False) -> dict:
    priors_path = priors_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not priors_path.is_file():
        raise FileNotFoundError(priors_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_priors_bytes = priors_path.read_bytes()
    source_priors_hash = hashlib.sha256(source_priors_bytes).hexdigest()
    source_data = json.loads(source_priors_bytes)
    converted = copy.deepcopy(source_data)
    table_cache: dict[tuple[str, str], dict] = {}
    passthrough_cache: dict[tuple[str, str], dict] = {}
    records: list[dict] = []
    passthrough_records: list[dict] = []
    output_sources: dict[str, str] = {}

    def reserve_output(filename: str, source_path: Path) -> Path:
        source_key = str(source_path.resolve())
        previous = output_sources.get(filename)
        if previous is not None and previous != source_key:
            raise ValueError(
                f"Conservative conversion output-name collision for {filename!r}: "
                f"{previous} vs {source_key}"
            )
        output_sources[filename] = source_key
        return output_dir / filename

    for json_key, kind in (("bonds", "bond"), ("angles", "angle"), ("dihedrals", "dihedral")):
        for idx, entry in enumerate(converted.get(json_key, [])):
            entry_type = str(entry.get("type", "")).lower()
            source_entry = source_data[json_key][idx]
            if entry_type == "conservative_spline":
                spline = load_conservative_spline(source_entry, kind=kind, priors_path=priors_path)
                source_path = spline.path.resolve()
                cache_key = (kind, str(source_path))
                if cache_key not in passthrough_cache:
                    out_name = source_path.name
                    out_path = reserve_output(out_name, source_path)
                    if source_path != out_path.resolve():
                        shutil.copy2(source_path, out_path)
                    source_hash = sha256_file(source_path)
                    output_hash = sha256_file(out_path)
                    if output_hash != source_hash:
                        raise RuntimeError(
                            f"Passthrough conservative spline was not copied byte-identically: {source_path}"
                        )
                    rec = {
                        "kind": kind,
                        "source_path": str(source_path),
                        "source_sha256": source_hash,
                        "output_file": out_name,
                        "output_path": str(out_path),
                        "output_sha256": output_hash,
                        "points": int(len(spline.x)),
                        "min": float(spline.minimum),
                        "max": float(spline.maximum),
                        "spline_schema": str(source_entry.get("spline_schema", SCHEMA)),
                        "byte_identical": True,
                    }
                    passthrough_cache[cache_key] = rec
                    passthrough_records.append(rec)
                rec = passthrough_cache[cache_key]
                entry["file"] = rec["output_file"]
                continue
            if entry_type != "tabulated":
                continue
            source = load_tabulated_prior(source_entry, kind=kind, priors_path=priors_path)
            cache_key = (kind, str(source.path.resolve()))
            if cache_key not in table_cache:
                if kind == "dihedral":
                    periodic_energy = np.asarray(source.energy, dtype=np.float64).copy()
                    scale = max(1.0, abs(float(periodic_energy[0])), abs(float(periodic_energy[-1])))
                    if abs(float(periodic_energy[-1] - periodic_energy[0])) > 1.0e-8 * scale:
                        raise ValueError(
                            f"Dihedral energy is not periodic at 0/2*pi: {source.path}; "
                            f"dU={periodic_energy[-1] - periodic_energy[0]:.6g}"
                        )
                    periodic_energy[-1] = periodic_energy[0]
                    interpolator = CubicSpline(source.x, periodic_energy, bc_type="periodic")
                    source_energy = periodic_energy
                else:
                    interpolator = PchipInterpolator(source.x, source.energy, extrapolate=False)
                    source_energy = source.energy
                derivative = np.asarray(interpolator(source.x, 1), dtype=np.float64)
                out_name = _safe_stem(source.path, kind)
                out_path = reserve_output(out_name, source.path)
                save_conservative_spline(out_path, source.x, source_energy, derivative)
                spline = ConservativeSplinePrior(
                    x=source.x.copy(), energy=np.asarray(source_energy).copy(), derivative=derivative,
                    minimum=source.minimum, maximum=source.maximum, kind=kind, path=out_path,
                )
                rec = {
                    "kind": kind,
                    "source_path": str(source.path.resolve()),
                    "source_sha256": sha256_file(source.path),
                    "output_file": out_name,
                    "output_path": str(out_path),
                    "output_sha256": sha256_file(out_path),
                    "points": int(len(source.x)),
                    "min": source.minimum,
                    "max": source.maximum,
                    "spline_schema": SCHEMA,
                    "fidelity": _metrics(source, spline),
                }
                table_cache[cache_key] = rec
                records.append(rec)
            rec = table_cache[cache_key]
            entry["type"] = "conservative_spline"
            entry["file"] = rec["output_file"]
            entry["min"] = float(source.minimum)
            entry["max"] = float(source.maximum)
            entry["spline_schema"] = SCHEMA
            entry["source_tabulated_sha256"] = rec["source_sha256"]

    if not records:
        raise ValueError("No tabulated bond/angle/dihedral priors were found for conservative conversion")

    output_priors = output_dir / "cg_priors.json"
    output_priors.write_text(json.dumps(converted, indent=2, sort_keys=False) + "\n")

    # Conversion is read-only with respect to the converged IBI artifacts.
    if hashlib.sha256(priors_path.read_bytes()).hexdigest() != source_priors_hash:
        raise RuntimeError("Source priors changed during conservative conversion")
    for rec in records:
        source = Path(rec["source_path"])
        if sha256_file(source) != rec["source_sha256"]:
            raise RuntimeError(f"Source IBI table changed during conversion: {source}")
    for rec in passthrough_records:
        source = Path(rec["source_path"])
        if sha256_file(source) != rec["source_sha256"]:
            raise RuntimeError(f"Source conservative spline changed during conversion: {source}")

    report = {
        "schema_version": SCHEMA_VERSION,
        "framework": "MLCG_Framework_v2",
        "kind": KIND,
        "source_priors": str(priors_path),
        "source_priors_sha256": source_priors_hash,
        "output_priors": str(output_priors),
        "output_priors_sha256": sha256_file(output_priors),
        "spline_schema": SCHEMA,
        "converted_unique_tables": len(records),
        "passthrough_unique_tables": len(passthrough_records),
        "records": records,
        "passthrough_records": passthrough_records,
        "source_artifacts_unchanged": True,
    }
    report_path = output_dir / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[CONSERVATIVE IBI CONVERSION]")
    print(f"source priors : {priors_path}")
    print(f"output priors : {output_priors}")
    print(f"converted tables   : {len(records)}")
    print(f"passthrough tables : {len(passthrough_records)}")
    for rec in records:
        m = rec["fidelity"]
        print(
            f"{rec['kind']:5s} {rec['output_file']}: "
            f"dU_rms/range={m['energy_rms_relative_to_range']:.3e} "
            f"dF_rms/scale={m['force_rms_relative']:.3e} "
            f"dF_p99={m['force_p99_abs']:.6g}"
        )
    print("[PASS] Source IBI artifacts were preserved; pre-existing conservative splines were copied byte-identically and converted artifacts were written separately.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priors", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert(args.priors, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
