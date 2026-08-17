#!/usr/bin/env python3
"""Aggregate short-NVT structure and offline stiffness for step-36 candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
KIND = "periodic_dihedral_ibi_update_localization"


def load(path: Path, label: str) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return json.loads(path.read_text())


def _mean_step35_preupdate_l1(step35: dict) -> float:
    values = [float(v) for v in step35.get("ibi_last_sampled_dihedral_l1", {}).values()]
    return float(np.mean(values)) if values else float("nan")


def finalize(
    registry_path: Path, step35_report_path: Path, structure_root: Path, output: Path, *,
    representation_dominance_factor: float, gain_dominance_factor: float, gain_floor: float,
    model_config_provenance: Path | None = None,
) -> dict:
    registry = load(registry_path, "candidate registry")
    step35 = load(step35_report_path, "step-35 report")
    structure_root = structure_root.expanduser().resolve()

    rows = []
    for rec in registry.get("candidates", []):
        name = str(rec["name"])
        structure_path = structure_root / name / "runtime_structure_report.json"
        structure = load(structure_path, f"structure report for {name}")
        l1 = float(structure.get("mean_l1_by_kind", {}).get("dihedral", float("nan")))
        if not np.isfinite(l1):
            raise ValueError(f"No finite dihedral mean L1 in {structure_path}")
        group_l1 = {
            key: float(item["distribution_l1"])
            for key, item in structure.get("groups", {}).items()
            if item.get("kind") == "dihedral"
        }
        rows.append({
            "name": name,
            "update_fraction": float(rec["update_fraction"]),
            "effective_alpha_if_linear_no_clip": float(rec["effective_alpha_if_linear_no_clip"]),
            "smooth_sigma_rad": float(rec["smooth_sigma_rad"]),
            "dihedral_mean_l1": l1,
            "dihedral_max_group_l1": max(group_l1.values()) if group_l1 else float("nan"),
            "target_abs_U2_p99": float(rec["target_abs_U2_p99"]),
            "target_abs_U2_p95": float(rec["target_abs_U2_p95"]),
            "target_abs_U2_max": float(rec["target_abs_U2_max"]),
            "candidate_priors": rec["candidate_priors"],
            "structure_report": str(structure_path),
            "group_l1": group_l1,
        })

    raw = sorted(
        [r for r in rows if r["smooth_sigma_rad"] == 0.0],
        key=lambda r: r["update_fraction"],
    )
    full_raw = next(r for r in raw if np.isclose(r["update_fraction"], 1.0))
    reduced_raw = [r for r in raw if r["update_fraction"] < 1.0]
    smooth_full = [r for r in rows if np.isclose(r["update_fraction"], 1.0) and r["smooth_sigma_rad"] > 0.0]
    best_reduced = min(reduced_raw, key=lambda r: r["dihedral_mean_l1"])
    best_smoothed_full = min(smooth_full, key=lambda r: r["dihedral_mean_l1"])

    amplitude_gain = full_raw["dihedral_mean_l1"] - best_reduced["dihedral_mean_l1"]
    smoothing_gain = full_raw["dihedral_mean_l1"] - best_smoothed_full["dihedral_mean_l1"]
    raw_l1 = np.asarray([r["dihedral_mean_l1"] for r in raw], dtype=float)
    raw_u2 = np.asarray([r["target_abs_U2_p99"] for r in raw], dtype=float)
    raw_monotone_l1 = bool(np.all(np.diff(raw_l1) >= -1.0e-12))
    raw_monotone_u2 = bool(np.all(np.diff(raw_u2) >= -1.0e-12))

    ranked = sorted(rows, key=lambda r: (r["dihedral_mean_l1"], r["target_abs_U2_p99"]))
    step35_pre = _mean_step35_preupdate_l1(step35)
    step35_post = float(step35.get("runtime_structure", {}).get("dihedral_mean_l1", float("nan")))
    zero_raw = next(r for r in raw if np.isclose(r["update_fraction"], 0.0))
    representation_jump = float(zero_raw["dihedral_mean_l1"] - step35_pre)

    # Heuristic only; report the underlying gains so the scientific interpretation
    # does not depend on this label.  A large jump already at fraction=0 means the
    # dominant change predates the IBI update amplitude: it is the legacy-tabulated
    # -> conservative representation change and must be diagnosed directly.
    gain_scale = max(abs(amplitude_gain), abs(smoothing_gain), gain_floor)
    if np.isfinite(representation_jump) and representation_jump > representation_dominance_factor * gain_scale:
        hint = "legacy_to_conservative_representation_change_dominant"
    elif amplitude_gain > gain_dominance_factor * max(smoothing_gain, 1.0e-12):
        hint = "update_amplitude_overshoot_dominant"
    elif smoothing_gain > gain_dominance_factor * max(amplitude_gain, 1.0e-12):
        hint = "update_roughness_dominant"
    else:
        hint = "mixed_or_not_separable_from_short_nvt"

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "test_only": True,
        "production_modified": False,
        "promotion_performed": False,
        "step35_reference": {
            "pre_update_sampled_mean_l1": step35_pre,
            "post_update_runtime_mean_l1": step35_post,
            "candidate_order2_through_configured_max_dt": bool(step35.get("candidate_order2_through_configured_max_dt", step35.get("candidate_order2_through_0p005", False))),
        },
        "candidate_results": rows,
        "ranking_by_structure_then_stiffness": [r["name"] for r in ranked],
        "localization": {
            "full_update_raw_l1": full_raw["dihedral_mean_l1"],
            "best_reduced_update_candidate": best_reduced["name"],
            "best_reduced_update_l1": best_reduced["dihedral_mean_l1"],
            "best_full_update_smoothed_candidate": best_smoothed_full["name"],
            "best_full_update_smoothed_l1": best_smoothed_full["dihedral_mean_l1"],
            "l1_gain_from_reducing_update_amplitude": float(amplitude_gain),
            "l1_gain_from_smoothing_full_update": float(smoothing_gain),
            "zero_update_conservative_l1": float(zero_raw["dihedral_mean_l1"]),
            "l1_jump_preupdate_sample_to_zero_update_conservative": representation_jump,
            "raw_fraction_l1_monotone_non_decreasing": raw_monotone_l1,
            "raw_fraction_target_p99_U2_monotone_non_decreasing": raw_monotone_u2,
            "diagnostic_hint": hint,
            "hint_is_gating": False,
        },
        "heuristic_policy": {"representation_dominance_factor": representation_dominance_factor, "gain_dominance_factor": gain_dominance_factor, "gain_floor": gain_floor},
        "model_config_provenance": str(model_config_provenance.resolve()) if model_config_provenance else None,
        "promotion_ready": False,
        "next_step": (
            "Diagnose legacy TabulatedDihedral energy/force consistency before any candidate NVE screen."
            if hint == "legacy_to_conservative_representation_change_dominant"
            else "Choose at most one or two promising candidates for a dedicated NVE screen; do not promote from step 36."
        ),
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[DIHEDRAL IBI UPDATE LOCALIZATION]")
    print(f"[STEP35] pre-update sampled mean L1={step35_pre:.6f} post-update runtime mean L1={step35_post:.6f}")
    for i, row in enumerate(ranked, 1):
        print(
            f"[RANK] #{i} {row['name']:27s} L1={row['dihedral_mean_l1']:.6f} "
            f"P99|U''|target={row['target_abs_U2_p99']:.6g} "
            f"fraction={row['update_fraction']:.2f} smooth={row['smooth_sigma_rad']:.3f}"
        )
    print(
        f"[LOCALIZE] representation_jump={representation_jump:+.6f} "
        f"amplitude_gain={amplitude_gain:+.6f} smoothing_gain={smoothing_gain:+.6f} "
        f"rawL1monotone={raw_monotone_l1} rawU2monotone={raw_monotone_u2}"
    )
    print(f"[HINT] {hint} (diagnostic only, non-gating)")
    print("[FINAL] promotion_ready=False")
    print(f"[DONE] report: {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-registry", required=True, type=Path)
    parser.add_argument("--step35-report", required=True, type=Path)
    parser.add_argument("--structure-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--representation-dominance-factor", type=float, required=True)
    parser.add_argument("--gain-dominance-factor", type=float, required=True)
    parser.add_argument("--gain-floor", type=float, required=True)
    parser.add_argument("--model-config-provenance", type=Path, default=None)
    args = parser.parse_args()
    finalize(
        args.candidate_registry, args.step35_report, args.structure_root, args.output,
        representation_dominance_factor=args.representation_dominance_factor,
        gain_dominance_factor=args.gain_dominance_factor, gain_floor=args.gain_floor,
        model_config_provenance=args.model_config_provenance,
    )


if __name__ == "__main__":
    main()
