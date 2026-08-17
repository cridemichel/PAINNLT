#!/usr/bin/env python3
"""Finalize the test-only conservative-in-the-loop dihedral IBI experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))

from conservative_spline import load_conservative_spline  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mean_dihedral_l1(groups: dict) -> float:
    vals = [
        float(row["distribution_l1"])
        for key, row in groups.items()
        if str(row.get("kind")) == "dihedral"
    ]
    require(bool(vals), "IBI metrics contain no dihedral L1 values")
    return float(np.mean(vals))


def _inspect_final_priors(priors_path: Path) -> dict:
    data = json.loads(priors_path.read_text())
    active = []
    unique = {}
    legacy_active = []
    for idx, entry in enumerate(data.get("dihedrals", [])):
        if str(entry.get("ibi_mode", "")).lower() != "ibi":
            continue
        active.append(idx)
        if str(entry.get("type", "")).lower() != "conservative_spline":
            legacy_active.append(idx)
            continue
        if str(entry.get("ibi_runtime_representation", "")) != "conservative_spline":
            raise ValueError(f"Active dihedrals[{idx}] lacks conservative runtime marker")
        spline = load_conservative_spline(entry, kind="dihedral", priors_path=priors_path)
        unique[str(spline.path.resolve())] = {
            "sha256": sha256_file(spline.path.resolve()),
            "points": int(len(spline.x)),
            "min": float(spline.minimum),
            "max": float(spline.maximum),
        }
    require(bool(active), "Final priors contain no active IBI dihedrals")
    return {
        "active_dihedral_entries": len(active),
        "legacy_active_entries": legacy_active,
        "unique_conservative_tables": len(unique),
        "tables": unique,
        "pass": not legacy_active and len(unique) > 0,
    }


def _validate_final_sampling_protocol(ibi: dict, sampling: dict, *, final_priors: Path) -> dict:
    metrics = list(ibi.get("metrics", []))
    require(bool(metrics), "IBI report contains no completed sampling iterations")
    last_iteration = max(int(row["iteration"]) for row in metrics)
    expected_iteration = last_iteration + 1
    expected_velocity_seed = int(ibi["velocity_seed"]) + expected_iteration - 1
    expected_thermostat_seed = int(ibi["thermostat_seed"]) + expected_iteration - 1

    checks = {
        "kind": sampling.get("kind") == "matched_final_ibi_sampling_protocol",
        "matched_flag": bool(sampling.get("matched_to_ibi_loop", False)),
        "source_priors": Path(str(sampling.get("source_priors", ""))).resolve() == final_priors.resolve(),
        "sampled_iteration": int(sampling.get("sampled_iteration", -1)) == expected_iteration,
        "dt_ps": bool(np.isclose(float(sampling.get("dt_ps", np.nan)), float(ibi["dt_ps"]), rtol=0.0, atol=1.0e-15)),
        "burn_in_steps": int(sampling.get("burn_in_steps", -1)) == int(ibi["burn_in_steps"]),
        "production_steps": int(sampling.get("production_steps", -1)) == int(ibi["production_steps"]),
        "sample_interval": int(sampling.get("sample_interval", -1)) == int(ibi["sample_interval"]),
        "kT": bool(np.isclose(float(sampling.get("kT", np.nan)), float(ibi["kT"]), rtol=0.0, atol=1.0e-12)),
        "neighbor_search": str(sampling.get("neighbor_search", "")) == str(ibi["neighbor_search"]),
        "velocity_seed": int(sampling.get("velocity_seed", -1)) == expected_velocity_seed,
        "thermostat_seed": int(sampling.get("thermostat_seed", -1)) == expected_thermostat_seed,
        "no_checkpoint": sampling.get("checkpoint_used") is False,
        "ml_inactive": sampling.get("ml_active") is False,
        "dataset_start": sampling.get("starting_state") == "target_dataset_initial_frame_plus_initialized_velocities",
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "expected_iteration": expected_iteration,
        "expected_velocity_seed": expected_velocity_seed,
        "expected_thermostat_seed": expected_thermostat_seed,
    }


def build_report(
    *,
    ibi_report_path: Path,
    final_priors: Path,
    parity_report_path: Path,
    structure_report_path: Path,
    final_sampling_report_path: Path,
    output: Path,
    step35_report_path: Path | None = None,
    step37_report_path: Path | None = None,
    direction_flat_tolerance_l1: float,
    model_config_provenance: Path | None = None,
) -> dict:
    ibi = json.loads(ibi_report_path.read_text())
    parity = json.loads(parity_report_path.read_text())
    structure = json.loads(structure_report_path.read_text())
    final_sampling = json.loads(final_sampling_report_path.read_text())

    require(bool(ibi.get("conservative_dihedrals_in_loop")), "IBI report did not enable conservative dihedrals in loop")
    require(ibi.get("dihedral_runtime_representation") == "conservative_spline", "Unexpected dihedral runtime representation")
    require(int(ibi.get("ibi_groups", -1)) > 0, "No active IBI groups were reported")
    require(int(ibi.get("dbi_groups", -1)) == 0, "This test expects no DBI-only groups")

    sample_series = []
    for metric in ibi.get("metrics", []):
        groups = metric.get("groups", {})
        non_dihedral = [
            key for key, row in groups.items()
            if str(row.get("mode")) == "ibi" and str(row.get("kind")) != "dihedral"
        ]
        require(not non_dihedral, f"Non-dihedral groups were updated: {non_dihedral}")
        representations = {
            str(row.get("runtime_representation"))
            for row in groups.values()
            if str(row.get("kind")) == "dihedral" and str(row.get("mode")) == "ibi"
        }
        require(representations == {"conservative_spline"}, f"Non-conservative dihedral sampling representation: {representations}")
        sample_series.append({
            "iteration": int(metric["iteration"]),
            "mean_dihedral_l1": _mean_dihedral_l1(groups),
            "source_priors": str(metric.get("source_priors")),
        })
    require(bool(sample_series), "IBI report contains no completed sampling iterations")

    sampling_protocol = _validate_final_sampling_protocol(ibi, final_sampling, final_priors=final_priors)
    require(sampling_protocol["pass"], f"Final sampling protocol does not match IBI loop: {sampling_protocol['checks']}")

    final_l1 = float(structure.get("mean_l1_by_kind", {}).get("dihedral", np.nan))
    require(np.isfinite(final_l1), "Final runtime structure report lacks dihedral mean L1")
    l1_sequence = [row["mean_dihedral_l1"] for row in sample_series] + [final_l1]
    initial = float(l1_sequence[0])
    final = float(l1_sequence[-1])
    improvement = initial - final
    relative_improvement = improvement / max(initial, 1.0e-12)
    if improvement > direction_flat_tolerance_l1:
        direction = "improving"
    elif improvement < -direction_flat_tolerance_l1:
        direction = "worsening"
    else:
        direction = "flat_within_short_run"

    prior_check = _inspect_final_priors(final_priors)
    parity_f = float(parity.get("worst_force_abs_error", parity.get("runtime_max_force_abs_error", np.inf)))
    parity_e = float(parity.get("worst_energy_abs_error", parity.get("runtime_max_energy_abs_error", np.inf)))
    kernel_pass = bool(parity.get("pass", False)) and parity_f <= 1.0e-10 and parity_e <= 1.0e-10

    comparison = {}
    if step35_report_path is not None and step35_report_path.is_file():
        step35 = json.loads(step35_report_path.read_text())
        comparison["legacy_step35"] = {
            "pre_update_sampled_mean_l1": float(np.mean(list(step35.get("ibi_last_sampled_dihedral_l1", {}).values()))) if step35.get("ibi_last_sampled_dihedral_l1") else None,
            "post_conversion_runtime_mean_l1": step35.get("runtime_structure", {}).get("dihedral_mean_l1"),
        }
    if step37_report_path is not None and step37_report_path.is_file():
        step37 = json.loads(step37_report_path.read_text())
        comparison["legacy_conservativity_diagnostic"] = {
            "report": str(step37_report_path),
            "hint": step37.get("hint") or step37.get("diagnostic_hint"),
        }

    infrastructure_pass = bool(prior_check["pass"] and kernel_pass and sampling_protocol["pass"])
    report = {
        "schema_version": 1,
        "framework": "MLCG_Framework_v2",
        "kind": "conservative_in_loop_dihedral_ibi_test",
        "test_only": True,
        "ibi_report": str(ibi_report_path),
        "ibi_report_sha256": sha256_file(ibi_report_path),
        "final_priors": str(final_priors),
        "final_priors_sha256": sha256_file(final_priors),
        "parity_report": str(parity_report_path),
        "structure_report": str(structure_report_path),
        "final_sampling_report": str(final_sampling_report_path),
        "final_sampling_report_sha256": sha256_file(final_sampling_report_path),
        "final_sampling_protocol": sampling_protocol,
        "conservative_in_loop": True,
        "iterations_completed": int(ibi.get("iterations_completed", 0)),
        "sampled_l1_series": sample_series,
        "final_runtime_dihedral_mean_l1": final_l1,
        "l1_sequence": l1_sequence,
        "initial_to_final_l1_improvement": improvement,
        "initial_to_final_relative_improvement": relative_improvement,
        "convergence_direction": direction,
        "final_prior_check": prior_check,
        "kernel_parity": {
            "max_force_abs_error": parity_f,
            "max_energy_abs_error": parity_e,
            "pass": kernel_pass,
        },
        "comparison": comparison,
        "infrastructure_pass": infrastructure_pass,
        "direction_flat_tolerance_l1": float(direction_flat_tolerance_l1),
        "model_config_provenance": str(model_config_provenance.resolve()) if model_config_provenance else None,
        "promotion_ready": False,
        "notes": [
            "Every sampled active torsional IBI group used ConservativeSplineDihedral; legacy TabulatedDihedral is fail-closed in this mode.",
            "The final post-update prior is sampled with the same dataset-start, burn-in, production length, sampling interval, thermostat/init-kT, neighbor-search, and deterministic seed progression used inside the IBI loop.",
            "This short test diagnoses convergence direction only; it is not a structural or NVE certification and never promotes priors.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    series = " -> ".join(f"{x:.6f}" for x in l1_sequence)
    print("[CONSERVATIVE-IN-LOOP DIHEDRAL IBI TEST]")
    print(f"[LOOP] iterations={len(sample_series)} all_active_dihedrals=conservative_spline pass={prior_check['pass']}")
    print(f"[KERNEL] parityF={parity_f:.3e} parityE={parity_e:.3e} pass={kernel_pass}")
    print(f"[MATCHED SAMPLE] iteration={sampling_protocol['expected_iteration']} pass={sampling_protocol['pass']}")
    print(f"[STRUCT] mean L1 sequence: {series}")
    print(f"[STRUCT] initial->final delta={-improvement:+.6f} direction={direction}")
    print(f"[FINAL] infrastructure_pass={infrastructure_pass} promotion_ready=False")
    print(f"[DONE] report: {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibi-report", required=True, type=Path)
    parser.add_argument("--final-priors", required=True, type=Path)
    parser.add_argument("--parity-report", required=True, type=Path)
    parser.add_argument("--structure-report", required=True, type=Path)
    parser.add_argument("--final-sampling-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step35-report", type=Path, default=None)
    parser.add_argument("--step37-report", type=Path, default=None)
    parser.add_argument("--direction-flat-tolerance-l1", type=float, required=True)
    parser.add_argument("--model-config-provenance", type=Path, default=None)
    args = parser.parse_args()
    build_report(
        ibi_report_path=args.ibi_report.resolve(),
        final_priors=args.final_priors.resolve(),
        parity_report_path=args.parity_report.resolve(),
        structure_report_path=args.structure_report.resolve(),
        final_sampling_report_path=args.final_sampling_report.resolve(),
        output=args.output.resolve(),
        step35_report_path=args.step35_report.resolve() if args.step35_report else None,
        step37_report_path=args.step37_report.resolve() if args.step37_report else None,
        direction_flat_tolerance_l1=args.direction_flat_tolerance_l1,
        model_config_provenance=args.model_config_provenance,
    )


if __name__ == "__main__":
    main()
