#!/usr/bin/env python3
"""Finalize certification of the promoted smoothed conservative-IBI Hamiltonian.

Unlike the historical composite step 26, sigma_E scaling is gating here.  The
certification binds the production priors to the step-33 validated candidate,
fresh conservative-spline validation/runtime parity, a fresh NVE scan launched
from the production path, and fresh Richardson state convergence.  PaiNN and the
old residual labels are explicitly excluded from the certified Hamiltonian.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "training", ROOT / "simulation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from residual_input_provenance import referenced_prior_artifacts, validate_ibi_validation_report  # noqa: E402

FRAMEWORK = "MLCG_Framework_v2"
KIND = "promoted_conservative_ibi_hamiltonian_certification"
EXPECTED_MODE = "conservative_classical_model_provenance_ml_disabled"
STATE_METRIC_KEYS = {
    "position": ("position_rms_nm", "position"),
    "velocity": ("velocity_rms_nm_per_ps", "velocity"),
    "orientation": ("orientation_rms_rad", "orientation"),
    "omega_body": ("omega_body_rms_per_ps", "omega_body"),
}
REQUIRED_STATE_METRICS = tuple(STATE_METRIC_KEYS)


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _sigma_gate(
    strict: Mapping[str, Any],
    *,
    p_min: float,
    p_max: float,
    r2_min: float,
    c2_spread_max: float,
    full_dt_ps: float,
    max_relative_drift: float,
) -> dict[str, Any]:
    cert = strict.get("certification", {})
    scaling = cert.get("scaling", {})
    p = float(scaling.get("exponent_p", math.nan))
    r2 = float(scaling.get("loglog_r2", math.nan))
    runs = [r for r in strict.get("runs", []) if r.get("status", "ok") == "ok"]
    require(len(runs) >= 3, "Fresh strict NVE report contains fewer than three successful runs")
    dts = np.asarray([float(r["dt_ps"]) for r in runs], dtype=float)
    sigma = np.asarray([float(r["sigma_E"]) for r in runs], dtype=float)
    drifts = np.asarray([abs(float(r["relative_block_mean_drift"])) for r in runs], dtype=float)
    require(np.all(np.isfinite(dts)) and np.all(dts > 0), "Fresh NVE dt values are invalid")
    require(np.all(np.isfinite(sigma)) and np.all(sigma > 0), "Fresh NVE sigma_E values are invalid")
    c2 = sigma / (dts * dts)
    c2_spread = float(np.max(c2) / np.min(c2))
    max_dt = float(np.max(dts))
    max_drift = float(np.max(drifts))
    checks = {
        "strict_certifier_pass": bool(cert.get("pass")),
        "quadratic_exponent": p_min <= p <= p_max,
        "loglog_r2": r2 >= r2_min,
        "c2_spread": c2_spread <= c2_spread_max,
        "full_dt_reached": max_dt >= full_dt_ps - 1e-15,
        "relative_block_drift": max_drift <= max_relative_drift,
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "exponent_p": p,
        "loglog_r2": r2,
        "c2_spread_max_over_min": c2_spread,
        "max_dt_ps": max_dt,
        "max_relative_block_drift": max_drift,
        "dt_ps": dts.tolist(),
        "sigma_E": sigma.tolist(),
        "sigma_over_dt2": c2.tolist(),
        "thresholds": {
            "p_min": p_min,
            "p_max": p_max,
            "r2_min": r2_min,
            "c2_spread_max": c2_spread_max,
            "full_dt_ps": full_dt_ps,
            "max_relative_block_drift": max_relative_drift,
        },
    }


def _state_gate(state: Mapping[str, Any], *, p_min: float, p_max: float, r2_min: float) -> dict[str, Any]:
    require(state.get("kind") == "conservative_ibi_nve_state_convergence_diagnostic", "Unsupported state-convergence report")
    require(state.get("hamiltonian_mode") == EXPECTED_MODE, "State-convergence Hamiltonian mode mismatch")
    summary = state.get("metric_summary", {})
    metrics: dict[str, Any] = {}
    passed = True
    for name in REQUIRED_STATE_METRICS:
        source_key = next((key for key in STATE_METRIC_KEYS[name] if key in summary), None)
        require(
            source_key is not None,
            f"Missing state convergence metric: {name} (accepted keys: {', '.join(STATE_METRIC_KEYS[name])})",
        )
        row = summary[source_key]
        p = float(row.get("median_exponent_p", math.nan))
        r2 = float(row.get("median_loglog_r2", math.nan))
        ok = p_min <= p <= p_max and r2 >= r2_min and bool(row.get("consistent_with_second_order"))
        metrics[name] = {
            "pass": ok,
            "source_metric_key": source_key,
            "median_exponent_p": p,
            "median_loglog_r2": r2,
        }
        passed = passed and ok
    return {
        "pass": bool(passed),
        "metrics": metrics,
        "thresholds": {"p_min": p_min, "p_max": p_max, "r2_min": r2_min},
    }


def build_report(
    *,
    priors: Path,
    promotion_report: Path,
    step33_report: Path,
    validation_report: Path,
    runtime_parity_report: Path,
    preflight_report: Path,
    strict_nve_report: Path,
    state_report: Path,
    residual_ml_status: Path,
    expected_candidate_sha256: str,
    sigma_p_min: float,
    sigma_p_max: float,
    sigma_r2_min: float,
    sigma_c2_spread_max: float,
    full_dt_ps: float,
    max_relative_drift: float,
    state_p_min: float,
    state_p_max: float,
    state_r2_min: float,
    model_config_provenance: Path | None = None,
) -> dict[str, Any]:
    paths = {
        "priors": priors.resolve(),
        "promotion": promotion_report.resolve(),
        "step33": step33_report.resolve(),
        "validation": validation_report.resolve(),
        "runtime_parity": runtime_parity_report.resolve(),
        "preflight": preflight_report.resolve(),
        "strict_nve": strict_nve_report.resolve(),
        "state_convergence": state_report.resolve(),
        "residual_ml_status": residual_ml_status.resolve(),
    }
    if model_config_provenance is not None:
        model_config_provenance = model_config_provenance.resolve()
        require(model_config_provenance.is_file(), f"Missing model config provenance: {model_config_provenance}")
        paths["model_config_provenance"] = model_config_provenance
    data = {name: load(path) for name, path in paths.items() if name not in {"priors", "model_config_provenance"}}
    priors_sha = sha256_file(paths["priors"])
    artifacts = referenced_prior_artifacts(paths["priors"])

    promotion = data["promotion"]
    require(promotion.get("kind") == "validated_ibi_angle_candidate_promotion", "Unsupported promotion report")
    require(promotion.get("pass") is True, "Promotion report did not pass")
    require(promotion.get("candidate_priors_sha256") == expected_candidate_sha256, "Promotion candidate SHA mismatch")
    require(promotion.get("promoted_priors_sha256") == priors_sha, "Promoted priors changed after promotion")
    require(promotion.get("promoted_prior_artifact_sha256") == artifacts, "Promoted prior table artifacts changed after promotion")
    require(promotion.get("runtime_table_identity_with_validated_candidate") is True, "Promotion did not establish candidate/runtime table identity")
    require(promotion.get("candidate_table_sha256") == promotion.get("promoted_table_sha256"), "Promoted runtime tables differ from the validated candidate")
    require(promotion.get("step33_validation_report_sha256") == sha256_file(paths["step33"]), "Step-33 report changed after promotion")

    step33 = data["step33"]
    require(step33.get("kind") == "ibi_angle_final_candidate_validation", "Unsupported step-33 report")
    require(step33.get("pass") is True and step33.get("validated") is True, "Step-33 candidate validation did not pass")
    require(step33.get("candidate_priors_sha256") == expected_candidate_sha256, "Step-33 candidate SHA mismatch")
    rg = step33.get("replica_gate", {})
    sg = step33.get("long_structure", {}).get("gate", {})
    step33_gate = {
        "pass": bool(rg.get("pass") and sg.get("pass")),
        "replica_common_p": rg.get("common_fit", {}).get("exponent_p"),
        "replica_within_r2": rg.get("common_fit", {}).get("within_replica_r2"),
        "full_clean_replicas": rg.get("n_full_clean_replicas"),
        "n_replicas": len(step33.get("replicas", [])),
        "median_c2_spread": rg.get("median_c2_spread"),
        "structural_delta_angle_l1": sg.get("delta_angle_weighted_l1"),
        "structural_delta_bond_l1": sg.get("delta_bond_weighted_l1"),
        "angle_p99_curvature_reduction": sg.get("angle_p99_curvature_reduction"),
    }

    # This call validates both the FD report and runtime parity against the exact
    # production priors and all referenced table hashes.
    validated = validate_ibi_validation_report(
        paths["validation"], paths["priors"], runtime_parity_report=paths["runtime_parity"]
    )
    kernel_gate = {
        "pass": validated.get("mode") == "conservative_spline_validation",
        "max_fd_abs_dU_dq_error": validated.get("max_abs_dU_dq_error"),
        "runtime_max_force_abs_error": validated.get("runtime_max_force_abs_error"),
        "runtime_max_energy_abs_error": validated.get("runtime_max_energy_abs_error"),
    }

    preflight = data["preflight"]
    require(preflight.get("pass") is True, "Fresh conservative preflight did not pass")
    require(preflight.get("priors_sha256") == priors_sha, "Preflight priors SHA mismatch")
    require(preflight.get("prior_artifact_sha256") == artifacts, "Preflight prior artifact map mismatch")
    preflight_gate = {"pass": True}

    strict = data["strict_nve"]
    strict_inputs = strict.get("inputs_sha256", {})
    require(strict_inputs.get("priors") == priors_sha, "Fresh strict NVE used different priors")
    checkpoint_sha = strict_inputs.get("checkpoint")
    require(isinstance(checkpoint_sha, str) and len(checkpoint_sha) == 64, "Fresh strict NVE report lacks checkpoint SHA256")
    sigma_gate = _sigma_gate(
        strict,
        p_min=sigma_p_min,
        p_max=sigma_p_max,
        r2_min=sigma_r2_min,
        c2_spread_max=sigma_c2_spread_max,
        full_dt_ps=full_dt_ps,
        max_relative_drift=max_relative_drift,
    )

    state = data["state_convergence"]
    require(state.get("input_hashes", {}).get("priors_sha256") == priors_sha, "State-convergence used different priors")
    require(state.get("checkpoint_sha256") == checkpoint_sha, "Fresh NVE and Richardson diagnostics used different checkpoints")
    state_gate = _state_gate(state, p_min=state_p_min, p_max=state_p_max, r2_min=state_r2_min)

    stale = data["residual_ml_status"]
    stale_gate = {
        "pass": bool(
            stale.get("kind") == "residual_ml_staleness_after_prior_promotion"
            and stale.get("status") == "stale_for_ml_active_use"
            and stale.get("requires_rebuild_before_ml_active_use") is True
            and stale.get("promoted_priors_sha256") == priors_sha
        ),
        "status": stale.get("status"),
        "certified_use": stale.get("certified_use"),
    }

    gates = {
        "promotion_identity": {
            "pass": True,
            "candidate_priors_sha256": expected_candidate_sha256,
            "promoted_priors_sha256": priors_sha,
            "runtime_table_identity_with_validated_candidate": True,
        },
        "step33_replica_and_structure_validation": step33_gate,
        "conservative_kernel_and_runtime_parity": kernel_gate,
        "fresh_preflight": preflight_gate,
        "fresh_sigma_E_quadratic_scaling": sigma_gate,
        "fresh_richardson_state_order": state_gate,
        "ml_residual_exclusion": stale_gate,
    }
    final_pass = bool(all(row.get("pass") is True for row in gates.values()))
    return {
        "schema_version": 1,
        "framework": FRAMEWORK,
        "kind": KIND,
        "certification_basis": "post_promotion_identity_plus_gating_sigmaE_dt2_plus_richardson_v1",
        "pass": final_pass,
        "hamiltonian_mode": EXPECTED_MODE,
        "scope": "Configured classical nonbonded + conservative bonded IBI Hamiltonian; PaiNN disabled",
        "ml_active": False,
        "selected_priors_sha256": priors_sha,
        "validated_candidate_priors_sha256": expected_candidate_sha256,
        "gates": gates,
        "artifacts": {name: str(path) for name, path in paths.items()},
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "notes": [
            "sigma_E scaling is gating in this post-promotion certification.",
            "Step-33 three-replica/structure evidence is retained as candidate-validation evidence and is bridged to production by byte-identical runtime tables.",
            "A fresh production-path NVE scan and fresh Richardson state-convergence run are also required.",
            "The pre-promotion residual dataset and PaiNN model are stale for ML-active use and are not part of this certified Hamiltonian.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--priors", required=True, type=Path)
    p.add_argument("--promotion-report", required=True, type=Path)
    p.add_argument("--step33-report", required=True, type=Path)
    p.add_argument("--validation-report", required=True, type=Path)
    p.add_argument("--runtime-parity-report", required=True, type=Path)
    p.add_argument("--preflight-report", required=True, type=Path)
    p.add_argument("--strict-nve-report", required=True, type=Path)
    p.add_argument("--state-report", required=True, type=Path)
    p.add_argument("--residual-ml-status", required=True, type=Path)
    p.add_argument("--expected-candidate-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model-config-provenance", type=Path, default=None)
    p.add_argument("--sigma-p-min", type=float, required=True)
    p.add_argument("--sigma-p-max", type=float, required=True)
    p.add_argument("--sigma-r2-min", type=float, required=True)
    p.add_argument("--sigma-c2-spread-max", type=float, required=True)
    p.add_argument("--full-dt-ps", type=float, required=True)
    p.add_argument("--max-relative-drift", type=float, required=True)
    p.add_argument("--state-p-min", type=float, required=True)
    p.add_argument("--state-p-max", type=float, required=True)
    p.add_argument("--state-r2-min", type=float, required=True)
    args = p.parse_args()

    report = build_report(
        priors=args.priors,
        promotion_report=args.promotion_report,
        step33_report=args.step33_report,
        validation_report=args.validation_report,
        runtime_parity_report=args.runtime_parity_report,
        preflight_report=args.preflight_report,
        strict_nve_report=args.strict_nve_report,
        state_report=args.state_report,
        residual_ml_status=args.residual_ml_status,
        expected_candidate_sha256=args.expected_candidate_sha256,
        sigma_p_min=args.sigma_p_min,
        sigma_p_max=args.sigma_p_max,
        sigma_r2_min=args.sigma_r2_min,
        sigma_c2_spread_max=args.sigma_c2_spread_max,
        full_dt_ps=args.full_dt_ps,
        max_relative_drift=args.max_relative_drift,
        state_p_min=args.state_p_min,
        state_p_max=args.state_p_max,
        state_r2_min=args.state_r2_min,
        model_config_provenance=args.model_config_provenance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    g = report["gates"]
    sg = g["fresh_sigma_E_quadratic_scaling"]
    rg = g["fresh_richardson_state_order"]
    print("[PROMOTED IBI FINAL CERTIFICATION]")
    print(
        f"[SIGMA] p={sg['exponent_p']:.6f} R2={sg['loglog_r2']:.6f} "
        f"C2spread={sg['c2_spread_max_over_min']:.3f} maxdt={sg['max_dt_ps']:.6g} pass={sg['pass']}"
    )
    metric_text = " ".join(
        f"{name}:p={row['median_exponent_p']:.3f}/R2={row['median_loglog_r2']:.3f}"
        for name, row in rg["metrics"].items()
    )
    print(f"[RICHARDSON] {metric_text} pass={rg['pass']}")
    print(f"[FINAL] pass={report['pass']} ML_active=False")
    print(f"[DONE] report: {args.output}")
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
