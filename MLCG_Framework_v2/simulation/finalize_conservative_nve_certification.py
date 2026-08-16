#!/usr/bin/env python3
"""Assemble the final conservative-IBI NVE certification verdict.

The long-window sigma_E(dt) power-law fit is retained as a diagnostic because
shadow-energy oscillations over a finite trajectory need not provide a smooth,
monotonic estimator of the trajectory integrator order.  The gating order test
is the short-time Richardson convergence of the mechanical state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_HAMILTONIAN_MODE = "conservative_classical_model_provenance_ml_disabled"
REQUIRED_STATE_METRICS = (
    "position_rms_nm",
    "velocity_rms_nm_per_ps",
    "orientation_rms_rad",
    "omega_body_rms_per_ps",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_hash(label: str, expected: str | None, path: Path) -> str:
    actual = _sha256(path)
    if expected is not None:
        _require(actual == expected, f"{label} SHA256 mismatch: {path}")
    return actual


def build_final_certification(
    *,
    priors_path: Path,
    validation_report_path: Path,
    runtime_parity_report_path: Path,
    nve_preflight_report_path: Path,
    equilibration_report_path: Path,
    strict_nve_report_path: Path,
    state_convergence_report_path: Path,
    state_order_min: float = 1.7,
    state_order_max: float = 2.3,
    state_min_r2: float = 0.95,
    max_relative_drift: float = 1.0e-4,
) -> dict[str, Any]:
    paths = {
        "priors": Path(priors_path).resolve(),
        "validation": Path(validation_report_path).resolve(),
        "runtime_parity": Path(runtime_parity_report_path).resolve(),
        "nve_preflight": Path(nve_preflight_report_path).resolve(),
        "equilibration": Path(equilibration_report_path).resolve(),
        "strict_nve": Path(strict_nve_report_path).resolve(),
        "state_convergence": Path(state_convergence_report_path).resolve(),
    }
    for label, path in paths.items():
        _require(path.is_file(), f"Missing {label} artifact: {path}")

    validation = _load_json(paths["validation"])
    parity = _load_json(paths["runtime_parity"])
    preflight = _load_json(paths["nve_preflight"])
    equil = _load_json(paths["equilibration"])
    strict = _load_json(paths["strict_nve"])
    state = _load_json(paths["state_convergence"])

    priors_sha = _sha256(paths["priors"])
    _require(validation.get("kind") == "ibi_conservative_spline_validation", "Unsupported conservative validation report")
    _require(bool(validation.get("pass")), "Conservative spline validation did not pass")
    _require(validation.get("conservative_priors_sha256") == priors_sha, "Validation report is not bound to the selected priors")

    _require(parity.get("kind") == "ibi_conservative_spline_runtime_parity", "Unsupported runtime parity report")
    _require(bool(parity.get("pass")), "Runtime/preprocessing parity did not pass")
    _require(parity.get("priors_sha256") == priors_sha, "Runtime parity report is not bound to the selected priors")

    _require(preflight.get("kind") == "conservative_ibi_nve_preflight", "Unsupported conservative NVE preflight report")
    _require(bool(preflight.get("pass")), "Conservative NVE preflight did not pass")
    if preflight.get("priors_sha256") is not None:
        _require(preflight.get("priors_sha256") == priors_sha, "NVE preflight is not bound to the selected priors")

    _require(bool(equil.get("pass")), "IBI-only NVT equilibration report did not pass")
    _require(equil.get("hamiltonian_mode") == EXPECTED_HAMILTONIAN_MODE, "Equilibration Hamiltonian mode mismatch")
    _require(equil.get("sampling_ensemble") == "NVT_Langevin", "Equilibration report is not NVT_Langevin")
    _require(equil.get("ml_active") is False, "Equilibration report says ML was active")
    checkpoint_sha = str(equil.get("checkpoint_sha256", ""))
    _require(len(checkpoint_sha) == 64, "Equilibration report lacks a valid checkpoint SHA256")

    definition = strict.get("definition", {})
    _require(definition.get("hamiltonian_mode") == EXPECTED_HAMILTONIAN_MODE, "Strict NVE report Hamiltonian mode mismatch")
    _require(definition.get("thermostat") == "off (--nve)", "Strict NVE report does not record thermostat-off NVE")
    strict_inputs = strict.get("inputs_sha256", {})
    _require(strict_inputs.get("priors") == priors_sha, "Strict NVE report priors hash mismatch")
    _require(strict_inputs.get("checkpoint") == checkpoint_sha, "Strict NVE report checkpoint hash mismatch")

    _require(state.get("kind") == "conservative_ibi_nve_state_convergence_diagnostic", "Unsupported state-convergence report")
    _require(state.get("hamiltonian_mode") == EXPECTED_HAMILTONIAN_MODE, "State-convergence Hamiltonian mode mismatch")
    _require(state.get("checkpoint_sha256") == checkpoint_sha, "State-convergence report used a different checkpoint")
    state_inputs = state.get("input_hashes", {})
    _require(state_inputs.get("priors_sha256") == priors_sha, "State-convergence priors hash mismatch")

    # Bind the report files themselves.  This makes the final report a stable
    # audit artifact even if an upstream JSON is later replaced in place.
    artifact_sha256 = {label: _sha256(path) for label, path in paths.items()}

    fd_values = [
        float(item["max_abs_dU_dq_error"])
        for item in validation.get("finite_difference_checks", [])
        if "max_abs_dU_dq_error" in item
    ]
    conservative_kernel_pass = bool(validation.get("pass")) and bool(fd_values)
    kernel_detail = {
        "pass": conservative_kernel_pass,
        "max_fd_abs_dU_dq_error": max(fd_values) if fd_values else None,
        "n_tables_checked": len(fd_values),
    }

    runtime_parity_pass = bool(parity.get("pass"))
    parity_detail = {
        "pass": runtime_parity_pass,
        "worst_force_abs_error": parity.get("worst_force_abs_error"),
        "worst_energy_abs_error": parity.get("worst_energy_abs_error"),
        "force_atol": parity.get("force_atol"),
        "energy_atol": parity.get("energy_atol"),
    }

    runs = strict.get("runs", [])
    drifts = [float(run["relative_block_mean_drift"]) for run in runs if "relative_block_mean_drift" in run]
    _require(drifts, "Strict NVE report contains no relative block drift values")
    max_observed_drift = max(drifts)
    strict_cert = strict.get("certification", {})
    energy_drift_pass = bool(strict_cert.get("drift_pass")) and max_observed_drift <= max_relative_drift
    drift_detail = {
        "pass": energy_drift_pass,
        "max_observed_relative_block_mean_drift": max_observed_drift,
        "max_allowed_relative_block_mean_drift": max_relative_drift,
        "n_runs": len(drifts),
        "drift_failures": strict_cert.get("drift_failures", []),
    }

    metric_summary = state.get("metric_summary", {})
    state_metric_detail: dict[str, Any] = {}
    vv_second_order_pass = True
    for metric in REQUIRED_STATE_METRICS:
        _require(metric in metric_summary, f"State-convergence report lacks required metric: {metric}")
        item = metric_summary[metric]
        p = float(item["median_exponent_p"])
        r2 = float(item["median_loglog_r2"])
        numerical_pass = state_order_min <= p <= state_order_max and r2 >= state_min_r2
        report_pass = bool(item.get("consistent_with_second_order"))
        passed = numerical_pass and report_pass
        vv_second_order_pass = vv_second_order_pass and passed
        state_metric_detail[metric] = {
            "pass": passed,
            "median_exponent_p": p,
            "median_loglog_r2": r2,
            "reported_consistent_with_second_order": report_pass,
            "min_exponent_p": item.get("min_exponent_p"),
            "max_exponent_p": item.get("max_exponent_p"),
            "min_loglog_r2": item.get("min_loglog_r2"),
        }

    scaling = strict_cert.get("scaling", {})
    sigma_scaling = {
        "gating": False,
        "status": "diagnostic_only",
        "legacy_scaling_pass": bool(strict_cert.get("scaling_pass")),
        "exponent_p": scaling.get("exponent_p"),
        "loglog_r2": scaling.get("loglog_r2"),
        "observable": scaling.get("observable", "sigma_E"),
        "reason": (
            "Long-window sigma_E is retained as an energy/shadow-Hamiltonian diagnostic; "
            "the gating trajectory-order test is Richardson state convergence."
        ),
    }

    provenance_pass = (
        preflight.get("pass") is True
        and equil.get("hamiltonian_mode") == EXPECTED_HAMILTONIAN_MODE
        and strict_inputs.get("checkpoint") == checkpoint_sha
        and state.get("checkpoint_sha256") == checkpoint_sha
        and strict_inputs.get("priors") == priors_sha
        and state_inputs.get("priors_sha256") == priors_sha
    )

    gates = {
        "conservative_kernel": kernel_detail,
        "runtime_parity": parity_detail,
        "provenance_consistency": {"pass": provenance_pass},
        "energy_drift": drift_detail,
        "vv_second_order": {
            "pass": vv_second_order_pass,
            "order_window": [state_order_min, state_order_max],
            "min_median_loglog_r2": state_min_r2,
            "metrics": state_metric_detail,
        },
        "sigma_E_scaling": sigma_scaling,
    }
    final_pass = all(
        gates[name]["pass"]
        for name in (
            "conservative_kernel",
            "runtime_parity",
            "provenance_consistency",
            "energy_drift",
            "vv_second_order",
        )
    )

    return {
        "schema_version": 1,
        "framework": "MLCG_Framework_v2",
        "kind": "conservative_ibi_nve_composite_certification",
        "certification_basis": "conservative_kernel_parity_plus_drift_plus_richardson_state_order_v1",
        "pass": final_pass,
        "hamiltonian_mode": EXPECTED_HAMILTONIAN_MODE,
        "ml_active": False,
        "scope": "WCA + Morse + bonded conservative IBI; PaiNN disabled",
        "gates": gates,
        "artifacts": {label: str(path) for label, path in paths.items()},
        "artifact_sha256": artifact_sha256,
        "selected_priors_sha256": priors_sha,
        "checkpoint_sha256": checkpoint_sha,
        "legacy_step23_pass": bool(strict_cert.get("pass")),
        "legacy_step23_scaling_pass": bool(strict_cert.get("scaling_pass")),
        "legacy_step23_drift_pass": bool(strict_cert.get("drift_pass")),
        "notes": [
            "The original step-23 sigma_E scaling result is preserved and is not rewritten.",
            "sigma_E scaling is non-gating in this composite certification because Richardson state convergence directly measures the trajectory integrator order.",
            "This report certifies only the IBI-only Hamiltonian recorded above; it does not certify an ML-active Hamiltonian.",
        ],
    }


def _fmt(value: Any, fmt: str = ".6g") -> str:
    if value is None:
        return "n/a"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priors", required=True, type=Path)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--runtime-parity-report", required=True, type=Path)
    parser.add_argument("--nve-preflight-report", required=True, type=Path)
    parser.add_argument("--equilibration-report", required=True, type=Path)
    parser.add_argument("--strict-nve-report", required=True, type=Path)
    parser.add_argument("--state-convergence-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--state-order-min", type=float, default=1.7)
    parser.add_argument("--state-order-max", type=float, default=2.3)
    parser.add_argument("--state-min-r2", type=float, default=0.95)
    parser.add_argument("--max-relative-drift", type=float, default=1.0e-4)
    args = parser.parse_args()

    report = build_final_certification(
        priors_path=args.priors,
        validation_report_path=args.validation_report,
        runtime_parity_report_path=args.runtime_parity_report,
        nve_preflight_report_path=args.nve_preflight_report,
        equilibration_report_path=args.equilibration_report,
        strict_nve_report_path=args.strict_nve_report,
        state_convergence_report_path=args.state_convergence_report,
        state_order_min=args.state_order_min,
        state_order_max=args.state_order_max,
        state_min_r2=args.state_min_r2,
        max_relative_drift=args.max_relative_drift,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gates = report["gates"]
    print("[CONSERVATIVE IBI NVE COMPOSITE CERTIFICATION]")
    print(f"conservative_kernel : {'PASS' if gates['conservative_kernel']['pass'] else 'FAIL'} (FD max={_fmt(gates['conservative_kernel']['max_fd_abs_dU_dq_error'], '.3e')})")
    print(f"runtime_parity      : {'PASS' if gates['runtime_parity']['pass'] else 'FAIL'} (dF={_fmt(gates['runtime_parity']['worst_force_abs_error'], '.3e')} dE={_fmt(gates['runtime_parity']['worst_energy_abs_error'], '.3e')})")
    print(f"provenance          : {'PASS' if gates['provenance_consistency']['pass'] else 'FAIL'}")
    print(f"energy_drift        : {'PASS' if gates['energy_drift']['pass'] else 'FAIL'} (max={_fmt(gates['energy_drift']['max_observed_relative_block_mean_drift'], '.3e')})")
    print(f"VV_second_order     : {'PASS' if gates['vv_second_order']['pass'] else 'FAIL'}")
    for metric, item in gates["vv_second_order"]["metrics"].items():
        print(f"  {metric}: p={_fmt(item['median_exponent_p'])} R2={_fmt(item['median_loglog_r2'])} {'PASS' if item['pass'] else 'FAIL'}")
    sigma = gates["sigma_E_scaling"]
    print(f"sigma_E_scaling     : DIAGNOSTIC (legacy_pass={sigma['legacy_scaling_pass']} p={_fmt(sigma['exponent_p'])} R2={_fmt(sigma['loglog_r2'])})")
    print(f"[{'PASS' if report['pass'] else 'FAIL'}] Conservative IBI NVE certification")
    print(f"report: {args.output}")
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
