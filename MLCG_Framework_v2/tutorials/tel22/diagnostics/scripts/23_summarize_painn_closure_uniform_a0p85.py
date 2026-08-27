#!/usr/bin/env python3
"""Summarize the short TEL22 PaiNN closure on uniform Morse a=0.255.

A = production priors (a=0.300) + old trained PaiNN
B = uniform a=0.255 priors + the same old trained PaiNN
C = uniform a=0.255 priors + PaiNN disabled

B is deliberately a decomposition-mismatch diagnostic. It is never promoted as
a production model; changed priors require rebuilding the residual target and
retraining PaiNN before accuracy claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

EXPECTED_DTS = (0.001, 0.0015, 0.002, 0.003, 0.004, 0.005)
EXPECTED_DURATION_PS = 2.0


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def local_exponents(runs: list[dict[str, Any]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for lo, hi in zip(runs[:-1], runs[1:]):
        dt0, dt1 = float(lo["dt_ps"]), float(hi["dt_ps"])
        s0, s1 = float(lo["sigma_E"]), float(hi["sigma_E"])
        p = math.log(s1 / s0) / math.log(dt1 / dt0)
        out.append({"dt_low_ps": dt0, "dt_high_ps": dt1, "local_exponent_p": p})
    return out


def _runtime_metadata(report: dict[str, Any], key: str) -> Any:
    """Read runtime metadata across current and legacy certification schemas."""
    definition = report.get("definition", {})
    top = report.get(key)
    nested = definition.get(key)
    if top is not None and nested is not None and top != nested:
        raise ValueError(f"Conflicting {key}: top-level={top!r}, definition={nested!r}")
    return top if top is not None else nested


def summarize_report(
    path: Path,
    expected_mode: str,
    *,
    allow_legacy_production_morse_defaults: bool = False,
) -> dict[str, Any]:
    report = load(path)
    definition = report.get("definition", {})
    mode = definition.get("hamiltonian_mode")
    if mode != expected_mode:
        raise ValueError(f"{path}: hamiltonian_mode={mode!r}, wanted {expected_mode!r}")
    if report.get("device") != "cpu":
        raise ValueError(f"{path}: closure reference requires CPU")
    if report.get("neighbor_search") != "link-cell":
        raise ValueError(f"{path}: closure reference requires link-cell")

    switch_mode = _runtime_metadata(report, "morse_switch_mode")
    morse_runtime = _runtime_metadata(report, "pair_specific_morse_runtime")
    inferred_runtime_metadata: list[str] = []
    if switch_mode is None and allow_legacy_production_morse_defaults:
        switch_mode = "switched"
        inferred_runtime_metadata.append("morse_switch_mode=switched")
    if morse_runtime is None and allow_legacy_production_morse_defaults:
        morse_runtime = "marker-nonbonded"
        inferred_runtime_metadata.append("pair_specific_morse_runtime=marker-nonbonded")
    if switch_mode != "switched":
        raise ValueError(f"{path}: closure requires switched Morse, got {switch_mode!r}")
    if morse_runtime != "marker-nonbonded":
        raise ValueError(f"{path}: closure requires marker-nonbonded Morse, got {morse_runtime!r}")

    runs = sorted(report.get("runs", []), key=lambda r: float(r["dt_ps"]))
    dts = tuple(float(r["dt_ps"]) for r in runs)
    if len(dts) != len(EXPECTED_DTS) or any(abs(a - b) > 1e-15 for a, b in zip(dts, EXPECTED_DTS)):
        raise ValueError(f"{path}: unexpected dt grid {dts}")
    for r in runs:
        dt = float(r["dt_ps"])
        expected_duration = round(EXPECTED_DURATION_PS / dt) * dt
        if abs(float(r["duration_ps"]) - expected_duration) > 1e-10:
            raise ValueError(f"{path}: dt={dt:g} duration mismatch")

    scaling = report["certification"]["scaling"]
    c2 = [float(r["sigma_E"]) / float(r["dt_ps"]) ** 2 for r in runs]
    coarse_median = statistics.median(c2[-3:])
    locals_ = local_exponents(runs)
    max_drift = max(float(r["relative_block_mean_drift"]) for r in runs)
    p = float(scaling["exponent_p"])
    return {
        "report": str(path.resolve()),
        "hamiltonian_mode": mode,
        "morse_switch_mode": switch_mode,
        "pair_specific_morse_runtime": morse_runtime,
        "legacy_inferred_runtime_metadata": inferred_runtime_metadata,
        "exponent_p": p,
        "abs_p_minus_2": abs(p - 2.0),
        "loglog_r2": float(scaling["loglog_r2"]),
        "c2_spread_max_over_min": max(c2) / min(c2),
        "c2_small_over_coarse_median": c2[0] / coarse_median,
        "c2_small_closeness_to_1": abs(c2[0] / coarse_median - 1.0),
        "sigma_E_dt_min": float(runs[0]["sigma_E"]),
        "c2_dt_min": c2[0],
        "c2_coarse_median": coarse_median,
        "adjacent_local_exponents": locals_,
        "local_exponent_range": max(x["local_exponent_p"] for x in locals_) - min(x["local_exponent_p"] for x in locals_),
        "max_relative_block_mean_drift": max_drift,
        "inputs_sha256": report.get("inputs_sha256", {}),
        "runs": [
            {
                "dt_ps": float(r["dt_ps"]),
                "sigma_E": float(r["sigma_E"]),
                "C2_sigma_over_dt2": c,
                "relative_block_mean_drift": float(r["relative_block_mean_drift"]),
            }
            for r, c in zip(runs, c2)
        ],
    }


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    paired = []
    for a, b in zip(reference["runs"], candidate["runs"]):
        if abs(a["dt_ps"] - b["dt_ps"]) > 1e-15:
            raise ValueError("paired dt mismatch")
        paired.append({
            "dt_ps": a["dt_ps"],
            "sigma_reference": a["sigma_E"],
            "sigma_candidate": b["sigma_E"],
            "sigma_ratio_candidate_over_reference": b["sigma_E"] / a["sigma_E"],
            "C2_reference": a["C2_sigma_over_dt2"],
            "C2_candidate": b["C2_sigma_over_dt2"],
        })
    return {
        "delta_p": candidate["exponent_p"] - reference["exponent_p"],
        "delta_abs_p_minus_2": candidate["abs_p_minus_2"] - reference["abs_p_minus_2"],
        "delta_r2": candidate["loglog_r2"] - reference["loglog_r2"],
        "c2_spread_ratio": candidate["c2_spread_max_over_min"] / reference["c2_spread_max_over_min"],
        "small_over_coarse_closeness_ratio": (
            candidate["c2_small_closeness_to_1"] / reference["c2_small_closeness_to_1"]
            if reference["c2_small_closeness_to_1"] > 0.0 else None
        ),
        "sigma_dt_min_ratio": candidate["sigma_E_dt_min"] / reference["sigma_E_dt_min"],
        "local_exponent_range_ratio": candidate["local_exponent_range"] / reference["local_exponent_range"],
        "paired_runs": paired,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--A-report", type=Path, required=True)
    ap.add_argument("--B-report", type=Path, required=True)
    ap.add_argument("--C-report", type=Path, required=True)
    ap.add_argument("--uniform-manifest", type=Path, required=True)
    ap.add_argument("--test22-summary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    manifest = load(args.uniform_manifest)
    if manifest.get("kind") != "tel22_morse_uniform_a0p85_inputs":
        raise ValueError("Unexpected uniform manifest kind")
    if int(manifest.get("morse_count", -1)) != 180 or abs(float(manifest.get("scaled_a", -1.0)) - 0.255) > 1e-15:
        raise ValueError("Uniform manifest is not 180 Morse at a=0.255")
    priors_path = Path(manifest["priors"])
    checkpoint_path = Path(manifest["checkpoint"])
    if not priors_path.is_file() or sha256_file(priors_path) != manifest.get("priors_sha256"):
        raise ValueError("Uniform priors hash mismatch")
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise ValueError("Uniform checkpoint hash mismatch")

    abc = load(args.test22_summary)
    if abc.get("kind") != "tel22_morse_stabilizer_abc_10ps_fullgrid":
        raise ValueError("Unexpected test-22 summary kind")
    if abc.get("recommended_numerical_stabilizer_for_next_step") != "C_uniform_a0p85":
        raise ValueError("Test 22 did not recommend uniform a=0.255")

    # A can be a pre-metadata historical production report. The shell runner
    # first validates it with 13_validate_full_baseline.py, and the hash checks
    # below bind it to the exact production priors/checkpoint/model. Only in
    # that legacy A arm may the historical production defaults be inferred.
    A = summarize_report(
        args.A_report,
        "model_active",
        allow_legacy_production_morse_defaults=True,
    )
    B = summarize_report(args.B_report, "model_active")
    C = summarize_report(args.C_report, "conservative_classical_model_provenance_ml_disabled")

    uniform_sha = str(manifest["priors_sha256"])
    if B["inputs_sha256"].get("priors") != uniform_sha or C["inputs_sha256"].get("priors") != uniform_sha:
        raise ValueError("B/C are not bound to the exact uniform priors selected by test 22")
    if B["inputs_sha256"].get("model") != C["inputs_sha256"].get("model"):
        raise ValueError("B/C model provenance mismatch")
    if B["inputs_sha256"].get("checkpoint") != C["inputs_sha256"].get("checkpoint"):
        raise ValueError("B/C checkpoint mismatch")
    if B["inputs_sha256"].get("checkpoint") != str(manifest["checkpoint_sha256"]):
        raise ValueError("B/C checkpoint is not the uniform derived checkpoint")
    if A["inputs_sha256"].get("model") != B["inputs_sha256"].get("model"):
        raise ValueError("A/B do not use the same old PaiNN model")
    if A["inputs_sha256"].get("priors") == uniform_sha:
        raise ValueError("A unexpectedly uses the uniform changed priors")
    source = manifest.get("source", {})
    if A["inputs_sha256"].get("priors") != source.get("priors_sha256"):
        raise ValueError("A priors hash does not match the source production priors of the uniform manifest")
    if A["inputs_sha256"].get("checkpoint") != source.get("checkpoint_sha256"):
        raise ValueError("A checkpoint hash does not match the source physical checkpoint of the uniform manifest")

    B_over_A = compare(A, B)
    C_over_A = compare(A, C)
    B_over_C = compare(C, B)

    # The closure answers an attribution question, not an accuracy question.
    # Use several independent diagnostics because a global p close to 2 can be
    # produced by compensating non-monotonic C2 points.
    prior_is_numerically_better = (
        C["abs_p_minus_2"] < A["abs_p_minus_2"]
        and C["c2_spread_max_over_min"] < A["c2_spread_max_over_min"]
        and C["c2_small_closeness_to_1"] < A["c2_small_closeness_to_1"]
    )
    old_painn_degrades_new_prior = (
        B["abs_p_minus_2"] > C["abs_p_minus_2"] + 0.03
        or B["c2_spread_max_over_min"] > C["c2_spread_max_over_min"] * 1.10
        or B["c2_small_closeness_to_1"] > C["c2_small_closeness_to_1"] + 0.08
        or B["sigma_E_dt_min"] > C["sigma_E_dt_min"] * 1.20
    )
    changed_prior_survives_old_painn = (
        B["abs_p_minus_2"] < A["abs_p_minus_2"]
        and B["c2_spread_max_over_min"] < A["c2_spread_max_over_min"]
        and B["c2_small_closeness_to_1"] < A["c2_small_closeness_to_1"]
    )

    if prior_is_numerically_better and old_painn_degrades_new_prior:
        interpretation = "uniform_priors_improve_nve_but_old_painn_reintroduces_nonideality_retrain_next"
    elif prior_is_numerically_better and changed_prior_survives_old_painn:
        interpretation = "uniform_priors_improvement_survives_old_painn_but_retraining_still_required"
    elif old_painn_degrades_new_prior:
        interpretation = "old_painn_degrades_uniform_prior_closure_retraining_required"
    else:
        interpretation = "mixed_painn_closure_review_metrics_before_pipeline_rebuild"

    out = {
        "schema_version": 1,
        "kind": "tel22_painn_closure_uniform_morse_a0p85_2ps",
        "scope": (
            "Historical six-dt 2 ps FP32 CPU/link-cell NVE closure. A reuses the exact production old-priors+old-PaiNN baseline. "
            "B uses uniform Morse a=0.255 with the same old PaiNN. C uses the same uniform priors/checkpoint with PaiNN disabled."
        ),
        "arms": {
            "A_old_priors_old_painn": A,
            "B_uniform_a0p85_old_painn": B,
            "C_uniform_a0p85_no_painn": C,
        },
        "comparisons": {
            "B_over_A_changed_priors_with_old_painn": B_over_A,
            "C_over_A_prior_change_without_painn": C_over_A,
            "B_over_C_old_painn_effect_on_uniform_priors": B_over_C,
        },
        "decision_checks": {
            "test22_uniform_prior_selected": True,
            "uniform_prior_is_numerically_better_than_old_prior_on_this_2ps_closure": prior_is_numerically_better,
            "old_painn_degrades_uniform_prior_nve_metrics": old_painn_degrades_new_prior,
            "uniform_prior_improvement_survives_old_painn": changed_prior_survives_old_painn,
            "B_and_C_same_uniform_priors_hash": True,
            "B_and_C_same_checkpoint_hash": True,
            "A_and_B_same_old_model_hash": True,
        },
        "interpretation": interpretation,
        "pipeline_decision": "rebuild_residual_target_and_retrain_painn_against_uniform_a0p85_priors",
        "caution": (
            "B is intentionally Hamiltonian-decomposition-inconsistent: the old PaiNN residual was trained against the old a=0.300 priors. "
            "Therefore B can diagnose numerical coupling/FP32-floor behavior but cannot validate forces, thermodynamics, or production accuracy. "
            "Regardless of B's numerical outcome, changed priors require residual regeneration and PaiNN retraining before production claims."
        ),
        "provenance": {
            "uniform_manifest": str(args.uniform_manifest.resolve()),
            "test22_summary": str(args.test22_summary.resolve()),
            "uniform_priors_sha256": uniform_sha,
            "uniform_checkpoint_sha256": str(manifest["checkpoint_sha256"]),
            "old_model_sha256": A["inputs_sha256"].get("model"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[TEL22 PAINN CLOSURE -- UNIFORM a=0.255, 2 ps]")
    print("arm                              p         R2        C2spread  C2small/coarse  sigma(dtmin)  localRange")
    for key in ("A_old_priors_old_painn", "B_uniform_a0p85_old_painn", "C_uniform_a0p85_no_painn"):
        x = out["arms"][key]
        print(
            f"{key:32s} {x['exponent_p']:.6f}  {x['loglog_r2']:.6f}  "
            f"{x['c2_spread_max_over_min']:.3f}     {x['c2_small_over_coarse_median']:.3f}           "
            f"{x['sigma_E_dt_min']:.6g}     {x['local_exponent_range']:.3f}"
        )
    print(f"[INTERPRETATION] {interpretation}")
    print("[NEXT] rebuild residual target and retrain PaiNN against uniform a=0.255 priors")
    print(f"[REPORT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
