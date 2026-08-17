#!/usr/bin/env python3
"""Generate test-only periodic dihedral candidates that localize an IBI update.

The tool compares the step-35 iteration-000 (DBI) and iteration-001 (one IBI
update) torsional potentials and generates conservative candidates by scaling
that *observed* update and optionally smoothing only the periodic update body.
Bond/angle conservative tables are copied byte-identically from the step-35
conservative candidate.  Production priors are never modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conservative_spline import SCHEMA, save_conservative_spline  # noqa: E402
from geometry_io import pool_requested, read_target_distributions  # noqa: E402


SCHEMA_VERSION = 1
KIND = "periodic_dihedral_ibi_update_localization_candidates"
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    update_fraction: float
    smooth_sigma_rad: float


DEFAULT_CANDIDATES = (
    CandidateSpec("frac_0p00_raw", 0.00, 0.0),
    CandidateSpec("frac_0p25_raw", 0.25, 0.0),
    CandidateSpec("frac_0p50_raw", 0.50, 0.0),
    CandidateSpec("frac_0p75_raw", 0.75, 0.0),
    CandidateSpec("frac_1p00_raw", 1.00, 0.0),
    CandidateSpec("frac_1p00_smooth_0p01", 1.00, 0.01),
    CandidateSpec("frac_1p00_smooth_0p02", 1.00, 0.02),
    CandidateSpec("frac_0p50_smooth_0p02", 0.50, 0.02),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_file(entry: dict, priors_path: Path) -> Path:
    path = Path(str(entry["file"])).expanduser()
    if not path.is_absolute():
        path = priors_path.resolve().parent / path
    return path.resolve()


def _load_priors(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _group_file_map(data: dict, priors_path: Path, *, expected_type: str | None = None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for idx, entry in enumerate(data.get("dihedrals", [])):
        name = str(entry.get("name", ""))
        if not name:
            raise ValueError(f"dihedrals[{idx}] is missing a pooled group name")
        if expected_type is not None and str(entry.get("type", "")).lower() != expected_type:
            raise ValueError(
                f"dihedrals[{idx}] {name!r} has type={entry.get('type')!r}; expected {expected_type!r}"
            )
        if "file" not in entry:
            raise ValueError(f"dihedrals[{idx}] {name!r} has no file")
        path = _resolve_file(entry, priors_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        previous = result.get(name)
        if previous is not None and previous != path:
            raise ValueError(f"Group {name!r} references multiple table files: {previous} vs {path}")
        result[name] = path
    if not result:
        raise ValueError(f"No dihedral groups found in {priors_path}")
    return result


def _load_tabulated(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[0] < 5 or data.shape[1] < 2:
        raise ValueError(f"Invalid tabulated dihedral table: {path}")
    x = np.asarray(data[:, 0], dtype=np.float64)
    u = np.asarray(data[:, 1], dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)):
        raise ValueError(f"Non-finite tabulated data: {path}")
    if not np.isclose(x[0], 0.0, atol=1.0e-12) or not np.isclose(x[-1], TWOPI, atol=1.0e-10):
        raise ValueError(f"Periodic dihedral table must span 0..2*pi: {path}")
    if not np.allclose(np.diff(x), np.diff(x)[0], rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"Dihedral grid is not uniform: {path}")
    scale = max(1.0, abs(float(u[0])), abs(float(u[-1])))
    if abs(float(u[-1] - u[0])) > 1.0e-8 * scale:
        raise ValueError(f"Dihedral energy is not periodic at the seam: {path}")
    u = u.copy()
    u[-1] = u[0]
    return x, u


def _smooth_periodic_update(delta_u: np.ndarray, x: np.ndarray, sigma_rad: float) -> np.ndarray:
    if sigma_rad <= 0.0:
        return np.asarray(delta_u, dtype=np.float64).copy()
    spacing = float(np.diff(x)[0])
    sigma_points = float(sigma_rad) / spacing
    unique = np.asarray(delta_u[:-1], dtype=np.float64)
    smooth = gaussian_filter1d(unique, sigma=sigma_points, mode="wrap")
    out = np.empty_like(delta_u, dtype=np.float64)
    out[:-1] = smooth
    out[-1] = smooth[0]
    return out


def _candidate_energy(u0: np.ndarray, u1: np.ndarray, x: np.ndarray, spec: CandidateSpec) -> np.ndarray:
    delta = np.asarray(u1 - u0, dtype=np.float64)
    delta[-1] = delta[0]
    filtered = _smooth_periodic_update(delta, x, spec.smooth_sigma_rad)
    if spec.update_fraction == 0.0 and spec.smooth_sigma_rad == 0.0:
        result = u0.copy()
    elif spec.update_fraction == 1.0 and spec.smooth_sigma_rad == 0.0:
        result = u1.copy()
    else:
        result = u0 + float(spec.update_fraction) * filtered
        result -= float(np.min(result))
        result[-1] = result[0]
    return result


def _curvature_metrics(spline: CubicSpline, target_phi: np.ndarray) -> dict:
    dense = np.linspace(0.0, TWOPI, 20001, endpoint=False)
    dense_u1 = np.abs(np.asarray(spline(dense, 1), dtype=float))
    dense_u2 = np.abs(np.asarray(spline(dense, 2), dtype=float))
    target_phi = np.mod(np.asarray(target_phi, dtype=float), TWOPI)
    target_u2 = np.abs(np.asarray(spline(target_phi, 2), dtype=float))
    return {
        "max_abs_dU_dphi": float(np.max(dense_u1)),
        "global_abs_U2_p95": float(np.percentile(dense_u2, 95.0)),
        "global_abs_U2_p99": float(np.percentile(dense_u2, 99.0)),
        "global_abs_U2_max": float(np.max(dense_u2)),
        "target_abs_U2_p50": float(np.percentile(target_u2, 50.0)),
        "target_abs_U2_p95": float(np.percentile(target_u2, 95.0)),
        "target_abs_U2_p99": float(np.percentile(target_u2, 99.0)),
        "target_abs_U2_max": float(np.max(target_u2)),
    }


def _copy_fixed_tables(template: dict, template_path: Path, candidate_dir: Path) -> list[dict]:
    records = []
    seen: dict[Path, str] = {}
    for key in ("bonds", "angles"):
        for idx, entry in enumerate(template.get(key, [])):
            if str(entry.get("type", "")).lower() != "conservative_spline":
                continue
            source = _resolve_file(entry, template_path)
            if source not in seen:
                destination = candidate_dir / source.name
                shutil.copy2(source, destination)
                if sha256_file(destination) != sha256_file(source):
                    raise RuntimeError(f"Fixed conservative table copy changed bytes: {source}")
                seen[source] = source.name
                records.append({
                    "kind": key[:-1],
                    "source": str(source),
                    "source_sha256": sha256_file(source),
                    "output": str(destination),
                    "output_sha256": sha256_file(destination),
                    "byte_identical": True,
                })
            entry["file"] = seen[source]
    return records


def generate(
    iteration0_priors: Path,
    iteration1_priors: Path,
    conservative_priors: Path,
    target_dataset: Path,
    output_dir: Path,
    *,
    ibi_report: Path | None = None,
    kT: float = 2.49,
    overwrite: bool = False,
) -> dict:
    iteration0_priors = iteration0_priors.expanduser().resolve()
    iteration1_priors = iteration1_priors.expanduser().resolve()
    conservative_priors = conservative_priors.expanduser().resolve()
    target_dataset = target_dataset.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    ibi_report = ibi_report.expanduser().resolve() if ibi_report is not None else None
    for path in (iteration0_priors, iteration1_priors, conservative_priors, target_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    if ibi_report is not None and not ibi_report.is_file():
        raise FileNotFoundError(ibi_report)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observed_alpha = 0.10
    if ibi_report is not None:
        ibi_meta = json.loads(ibi_report.read_text())
        observed_alpha = float(ibi_meta.get("settings", {}).get("alpha", observed_alpha))
        report_kT = float(ibi_meta.get("kT", kT))
        if not np.isclose(report_kT, kT, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"kT mismatch between arguments ({kT}) and IBI report ({report_kT})")

    p0 = _load_priors(iteration0_priors)
    p1 = _load_priors(iteration1_priors)
    template_source = _load_priors(conservative_priors)
    map0 = _group_file_map(p0, iteration0_priors, expected_type="tabulated")
    map1 = _group_file_map(p1, iteration1_priors, expected_type="tabulated")
    mapc = _group_file_map(template_source, conservative_priors, expected_type="conservative_spline")
    groups = sorted(map0)
    if groups != sorted(map1) or groups != sorted(mapc):
        raise ValueError(
            f"Dihedral group mismatch: iteration0={sorted(map0)} iteration1={sorted(map1)} conservative={sorted(mapc)}"
        )

    # Target torsional values are shared by all candidates because topology/grouping is unchanged.
    target_priors = copy.deepcopy(template_source)
    target_values = read_target_distributions(target_dataset, target_priors)[2]
    target_pooled = pool_requested(target_priors, target_values, "dihedrals")

    source_groups = {}
    for name in groups:
        x0, u0 = _load_tabulated(map0[name])
        x1, u1 = _load_tabulated(map1[name])
        if not np.allclose(x0, x1, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Grid changed across IBI update for {name}")
        delta = u1 - u0
        # An arbitrary additive offset in delta is dynamically irrelevant; report centered amplitudes.
        centered = delta[:-1] - float(np.mean(delta[:-1]))
        source_groups[name] = {
            "x": x0,
            "u0": u0,
            "u1": u1,
            "delta_u_rms_kT": float(np.sqrt(np.mean(centered * centered)) / kT),
            "delta_u_max_abs_kT": float(np.max(np.abs(centered)) / kT),
            "iteration0_table": str(map0[name]),
            "iteration1_table": str(map1[name]),
        }

    candidate_records = []
    for spec in DEFAULT_CANDIDATES:
        candidate_dir = output_dir / "candidates" / spec.name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_json = copy.deepcopy(template_source)
        fixed_records = _copy_fixed_tables(candidate_json, conservative_priors, candidate_dir)
        group_metrics = {}
        candidate_table_by_group = {}
        all_target_curvatures = []

        for name in groups:
            state = source_groups[name]
            x = state["x"]
            energy = _candidate_energy(state["u0"], state["u1"], x, spec)
            spline = CubicSpline(x, energy, bc_type="periodic")
            derivative = np.asarray(spline(x, 1), dtype=np.float64)
            out_name = f"dihedral_conservative_{name}.dat"
            out_path = candidate_dir / out_name
            save_conservative_spline(out_path, x, energy, derivative)
            candidate_table_by_group[name] = out_name

            target_group = target_pooled.get(name)
            if target_group is None:
                raise ValueError(f"Target dataset did not produce dihedral group {name!r}")
            phi = np.mod(np.asarray(target_group["values"], dtype=float), TWOPI)
            metrics = _curvature_metrics(spline, phi)
            target_u2 = np.abs(np.asarray(spline(phi, 2), dtype=float))
            all_target_curvatures.append(target_u2)
            metrics.update({
                "table": str(out_path),
                "table_sha256": sha256_file(out_path),
                "seam_energy_abs_gap": float(abs(energy[-1] - energy[0])),
                "seam_derivative_abs_gap": float(abs(derivative[-1] - derivative[0])),
                "target_samples": int(phi.size),
                "observed_update_rms_kT": state["delta_u_rms_kT"],
                "observed_update_max_abs_kT": state["delta_u_max_abs_kT"],
            })
            group_metrics[name] = metrics

        for entry in candidate_json.get("dihedrals", []):
            name = str(entry["name"])
            entry["type"] = "conservative_spline"
            entry["file"] = candidate_table_by_group[name]
            entry["min"] = 0.0
            entry["max"] = float(TWOPI)
            entry["spline_schema"] = SCHEMA
            entry.pop("source_tabulated_sha256", None)
            entry["dihedral_update_localization"] = {
                "test_only": True,
                "update_fraction": spec.update_fraction,
                "update_smoothing_sigma_rad": spec.smooth_sigma_rad,
            }

        candidate_json["dihedral_update_localization"] = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "test_only": True,
            "candidate": spec.name,
            "update_fraction": spec.update_fraction,
            "update_smoothing_sigma_rad": spec.smooth_sigma_rad,
            "source_iteration0_priors": str(iteration0_priors),
            "source_iteration1_priors": str(iteration1_priors),
            "source_step35_conservative_priors": str(conservative_priors),
        }
        candidate_priors = candidate_dir / "cg_priors.json"
        candidate_priors.write_text(json.dumps(candidate_json, indent=2, sort_keys=False) + "\n")

        aggregate_u2 = np.concatenate(all_target_curvatures)
        record = {
            "name": spec.name,
            "update_fraction": spec.update_fraction,
            "effective_alpha_if_linear_no_clip": float(observed_alpha * spec.update_fraction),
            "smooth_sigma_rad": spec.smooth_sigma_rad,
            "candidate_priors": str(candidate_priors),
            "candidate_priors_sha256": sha256_file(candidate_priors),
            "fixed_tables": fixed_records,
            "fixed_tables_byte_identical": all(r["byte_identical"] for r in fixed_records),
            "target_abs_U2_p50": float(np.percentile(aggregate_u2, 50.0)),
            "target_abs_U2_p95": float(np.percentile(aggregate_u2, 95.0)),
            "target_abs_U2_p99": float(np.percentile(aggregate_u2, 99.0)),
            "target_abs_U2_max": float(np.max(aggregate_u2)),
            "groups": group_metrics,
        }

        # The raw full-update candidate should reproduce the exact step-35 conservative tables.
        if spec.name == "frac_1p00_raw":
            exact = {}
            for name in groups:
                generated_path = candidate_dir / candidate_table_by_group[name]
                exact[name] = sha256_file(generated_path) == sha256_file(mapc[name])
            record["matches_step35_conservative_tables"] = exact
            record["all_step35_tables_byte_identical"] = bool(all(exact.values()))
            if not record["all_step35_tables_byte_identical"]:
                raise RuntimeError("Raw full-update reconstruction is not byte-identical to step-35 conservative dihedral tables")

        candidate_records.append(record)

    source_summary = {
        name: {
            key: value for key, value in state.items() if key not in {"x", "u0", "u1"}
        }
        for name, state in source_groups.items()
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "test_only": True,
        "production_modified": False,
        "iteration0_priors": str(iteration0_priors),
        "iteration0_priors_sha256": sha256_file(iteration0_priors),
        "iteration1_priors": str(iteration1_priors),
        "iteration1_priors_sha256": sha256_file(iteration1_priors),
        "step35_conservative_priors": str(conservative_priors),
        "step35_conservative_priors_sha256": sha256_file(conservative_priors),
        "target_dataset": str(target_dataset),
        "kT": float(kT),
        "observed_step35_alpha": float(observed_alpha),
        "ibi_report": str(ibi_report) if ibi_report is not None else None,
        "source_groups": source_summary,
        "candidates": candidate_records,
        "notes": [
            "Update fractions scale the observed step-35 iteration_000 -> iteration_001 energy correction.",
            "effective_alpha_if_linear_no_clip is diagnostic only; force/update clipping can break exact alpha linearity.",
            "Periodic smoothing is applied only to the observed delta-U, never to the DBI baseline potential.",
            "No candidate is promoted by this diagnostic.",
        ],
    }
    report_path = output_dir / "candidate_registry.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[DIHEDRAL IBI UPDATE LOCALIZATION CANDIDATES]")
    print(f"groups      : {len(groups)}")
    print(f"iteration 0 : {iteration0_priors}")
    print(f"iteration 1 : {iteration1_priors}")
    for rec in candidate_records:
        print(
            f"[CANDIDATE] {rec['name']:27s} fraction={rec['update_fraction']:.2f} "
            f"smooth={rec['smooth_sigma_rad']:.3f} rad target P99|U''|={rec['target_abs_U2_p99']:.6g}"
        )
    print(f"[DONE] registry: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration0-priors", required=True, type=Path)
    parser.add_argument("--iteration1-priors", required=True, type=Path)
    parser.add_argument("--conservative-priors", required=True, type=Path)
    parser.add_argument("--target-dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ibi-report", type=Path, default=None)
    parser.add_argument("--kT", type=float, default=2.49)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(
        args.iteration0_priors,
        args.iteration1_priors,
        args.conservative_priors,
        args.target_dataset,
        args.output_dir,
        ibi_report=args.ibi_report,
        kT=args.kT,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
