import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "ibi", ROOT / "simulation", ROOT / "training"):
    sys.path.insert(0, str(path))

from promote_validated_angle_candidate import promote, sha256_file, verify_promoted
from validate_conservative_spline import validate as validate_conservative
from finalize_promoted_ibi_certification import build_report
from residual_input_provenance import referenced_prior_artifacts


def _write_table(path: Path, x: np.ndarray, u: np.ndarray, du: np.ndarray) -> None:
    np.savetxt(path, np.column_stack([x, u, du]))


def _fixture(tmp_path: Path):
    current = tmp_path / "ibi_conservative"
    candidate_dir = tmp_path / "candidate"
    backup = tmp_path / "backup"
    current.mkdir()
    candidate_dir.mkdir()

    xb = np.linspace(0.01, 3.0, 101)
    ub = 50.0 * (xb - 1.2) ** 2
    db = 100.0 * (xb - 1.2)
    xa = np.linspace(0.0, np.pi, 101)
    ua = 40.0 * (xa - 1.7) ** 2
    da = 80.0 * (xa - 1.7)
    xd = np.linspace(0.0, 2.0 * np.pi, 129)
    ud = 3.0 * (1.0 - np.cos(xd))
    dd = 3.0 * np.sin(xd)
    _write_table(current / "bond.dat", xb, ub, db)
    _write_table(current / "angle.dat", xa, ua, da)
    _write_table(current / "dihedral.dat", xd, ud, dd)

    priors = {
        "bonds": [{
            "type": "conservative_spline", "name": "b", "file": "bond.dat",
            "min": 0.01, "max": 3.0, "spline_schema": "pchip_hermite_v1",
        }],
        "angles": [{
            "type": "conservative_spline", "name": "a", "file": "angle.dat",
            "min": 0.0, "max": float(np.pi), "spline_schema": "pchip_hermite_v1",
        }],
        "dihedrals": [{
            "type": "conservative_spline", "name": "d", "file": "dihedral.dat",
            "min": 0.0, "max": float(2.0 * np.pi), "spline_schema": "pchip_hermite_v1",
        }],
    }
    current_priors = current / "cg_priors.json"
    current_priors.write_text(json.dumps(priors, indent=2, sort_keys=True) + "\n")
    source_sha = sha256_file(current_priors)

    # Candidate keeps bond/dihedral byte-identical and changes only the angle table.
    (candidate_dir / "bond.dat").write_bytes((current / "bond.dat").read_bytes())
    (candidate_dir / "dihedral.dat").write_bytes((current / "dihedral.dat").read_bytes())
    ua2 = ua + 0.01 * np.sin(2.0 * xa)
    da2 = da + 0.02 * np.cos(2.0 * xa)
    _write_table(candidate_dir / "angle.dat", xa, ua2, da2)
    candidate = json.loads(json.dumps(priors))
    candidate["angles"][0]["regularization"] = {
        "kind": "angle_body_gaussian_plus_c2_v1",
        "candidate": "smooth_0p0075_wall_current",
        "body_sigma_rad": 0.0075,
        "wall_width_rad": 0.1,
        "wall_k": 5000.0,
        "validated": False,
    }
    candidate["regularization_candidate"] = {
        "schema_version": 1,
        "kind": "unvalidated_ibi_angle_smoothing_candidate",
        "source_priors": str(current_priors.resolve()),
        "source_priors_sha256": source_sha,
        "candidate": "smooth_0p0075_wall_current",
        "body_sigma_rad": 0.0075,
        "wall_width_rad": 0.1,
        "wall_k": 5000.0,
        "validated": False,
    }
    candidate_priors = candidate_dir / "cg_priors.json"
    candidate_priors.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    candidate_sha = sha256_file(candidate_priors)

    step33 = tmp_path / "step33.json"
    step33_payload = {
        "schema_version": 1,
        "kind": "ibi_angle_final_candidate_validation",
        "pass": True,
        "validated": True,
        "candidate_name": "smooth_0p0075",
        "candidate_sigma_rad": 0.0075,
        "candidate_priors": str(candidate_priors.resolve()),
        "candidate_priors_sha256": candidate_sha,
        "replicas": [{}, {}, {}],
        "replica_gate": {
            "pass": True,
            "common_fit": {"exponent_p": 1.95, "within_replica_r2": 0.985},
            "n_full_clean_replicas": 3,
            "median_c2_spread": 1.4,
        },
        "long_structure": {"gate": {
            "pass": True,
            "delta_angle_weighted_l1": 0.01,
            "delta_bond_weighted_l1": -0.02,
            "angle_p99_curvature_reduction": 2.4,
        }},
    }
    step33.write_text(json.dumps(step33_payload, indent=2, sort_keys=True) + "\n")
    return current, candidate_priors, backup, step33, candidate_sha


def _promote(tmp_path: Path):
    current, candidate, backup, step33, candidate_sha = _fixture(tmp_path)
    report = promote(
        current_dir=current,
        candidate_priors=candidate,
        final_report=step33,
        backup_dir=backup,
        expected_candidate_sha256=candidate_sha,
        expected_sigma_rad=0.0075,
        dataset=None,
        model=None,
    )
    return current, candidate, backup, step33, candidate_sha, report


def test_promotion_is_transactional_and_candidate_tables_are_identical(tmp_path):
    current, candidate, backup, step33, candidate_sha, report = _promote(tmp_path)
    assert backup.is_dir()
    assert report["candidate_priors_sha256"] == candidate_sha
    assert report["candidate_table_sha256"] == report["promoted_table_sha256"]
    promoted = json.loads((current / "cg_priors.json").read_text())
    assert promoted["regularization_candidate"]["validated"] is True
    assert promoted["regularization_candidate"]["promoted"] is True
    assert promoted["angles"][0]["regularization"]["validated"] is True
    assert (current / "dihedral.dat").read_bytes() == (candidate.parent / "dihedral.dat").read_bytes()
    assert json.loads((current / "residual_ml_status.json").read_text())["status"] == "stale_for_ml_active_use"
    assert verify_promoted(
        current_dir=current,
        expected_candidate_sha256=candidate_sha,
        expected_sigma_rad=0.0075,
    )["pass"] is True
    validation = validate_conservative(current / "conversion_report.json")
    assert validation["pass"] is True


def test_promotion_rejects_unreviewed_candidate_sha(tmp_path):
    current, candidate, backup, step33, _candidate_sha = _fixture(tmp_path)
    with pytest.raises(ValueError, match="Candidate priors SHA256 mismatch"):
        promote(
            current_dir=current,
            candidate_priors=candidate,
            final_report=step33,
            backup_dir=backup,
            expected_candidate_sha256="0" * 64,
            expected_sigma_rad=0.0075,
            dataset=None,
            model=None,
        )
    assert not backup.exists()


def _write_certification_inputs(tmp_path: Path):
    current, candidate, backup, step33, candidate_sha, promotion = _promote(tmp_path)
    priors = current / "cg_priors.json"
    priors_sha = sha256_file(priors)
    artifacts = referenced_prior_artifacts(priors)
    validation = validate_conservative(current / "conversion_report.json")
    assert validation["pass"] is True

    parity = {
        "schema_version": 1,
        "framework": "MLCG_Framework_v2",
        "kind": "ibi_conservative_spline_runtime_parity",
        "priors": str(priors.resolve()),
        "priors_sha256": priors_sha,
        "prior_artifact_sha256": artifacts,
        "force_atol": 1e-9,
        "energy_atol": 1e-10,
        "worst_force_abs_error": 1e-12,
        "worst_energy_abs_error": 1e-13,
        "results": [],
        "pass": True,
    }
    parity_path = current / "runtime_parity_report.json"
    parity_path.write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n")

    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({
        "pass": True,
        "priors_sha256": priors_sha,
        "prior_artifact_sha256": artifacts,
    }, indent=2, sort_keys=True) + "\n")

    runs = [
        {"status": "ok", "dt_ps": dt, "sigma_E": 600.0 * dt * dt, "relative_block_mean_drift": 1e-6}
        for dt in (0.001, 0.002, 0.003, 0.004, 0.005)
    ]
    strict = tmp_path / "strict.json"
    strict.write_text(json.dumps({
        "inputs_sha256": {"priors": priors_sha, "checkpoint": "a" * 64},
        "certification": {
            "pass": True,
            "drift_pass": True,
            "scaling": {"exponent_p": 2.0, "loglog_r2": 0.999},
        },
        "runs": runs,
    }, indent=2, sort_keys=True) + "\n")

    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "kind": "conservative_ibi_nve_state_convergence_diagnostic",
        "hamiltonian_mode": "conservative_classical_model_provenance_ml_disabled",
        "checkpoint_sha256": "a" * 64,
        "input_hashes": {"priors_sha256": priors_sha},
        "metric_summary": {
            name: {"median_exponent_p": 2.0, "median_loglog_r2": 0.99, "consistent_with_second_order": True}
            for name in (
                "position_rms_nm",
                "velocity_rms_nm_per_ps",
                "orientation_rms_rad",
                "omega_body_rms_per_ps",
            )
        },
    }, indent=2, sort_keys=True) + "\n")

    stale_path = current / "residual_ml_status.json"
    stale = json.loads(stale_path.read_text())
    stale["promoted_priors_sha256"] = priors_sha
    stale_path.write_text(json.dumps(stale, indent=2, sort_keys=True) + "\n")
    return current, step33, candidate_sha, parity_path, preflight, strict, state, stale_path


def test_finalizer_requires_gating_sigma_and_richardson(tmp_path):
    current, step33, candidate_sha, parity, preflight, strict, state, stale = _write_certification_inputs(tmp_path)
    report = build_report(
        priors=current / "cg_priors.json",
        promotion_report=current / "promotion_report.json",
        step33_report=step33,
        validation_report=current / "validation_report.json",
        runtime_parity_report=parity,
        preflight_report=preflight,
        strict_nve_report=strict,
        state_report=state,
        residual_ml_status=stale,
        expected_candidate_sha256=candidate_sha,
    )
    assert report["pass"] is True
    assert report["gates"]["fresh_sigma_E_quadratic_scaling"]["pass"] is True
    assert report["gates"]["fresh_richardson_state_order"]["pass"] is True
    assert report["ml_active"] is False


def test_finalizer_rejects_nonquadratic_fresh_sigma(tmp_path):
    current, step33, candidate_sha, parity, preflight, strict, state, stale = _write_certification_inputs(tmp_path)
    payload = json.loads(strict.read_text())
    payload["certification"]["scaling"]["exponent_p"] = 1.4
    payload["certification"]["pass"] = False
    strict.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report = build_report(
        priors=current / "cg_priors.json",
        promotion_report=current / "promotion_report.json",
        step33_report=step33,
        validation_report=current / "validation_report.json",
        runtime_parity_report=parity,
        preflight_report=preflight,
        strict_nve_report=strict,
        state_report=state,
        residual_ml_status=stale,
        expected_candidate_sha256=candidate_sha,
    )
    assert report["pass"] is False
    assert report["gates"]["fresh_sigma_E_quadratic_scaling"]["checks"]["quadratic_exponent"] is False
