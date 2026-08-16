#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SIMULATION = ROOT / "simulation"
sys.path.insert(0, str(SIMULATION))

from finalize_conservative_nve_certification import build_final_certification  # noqa: E402

MODE = "conservative_classical_model_provenance_ml_disabled"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


class FinalConservativeNVECertificationTests(unittest.TestCase):
    def make_artifacts(self, root: Path):
        priors = root / "cg_priors.json"
        priors.write_text('{"bonds": [], "angles": [], "dihedrals": []}\n', encoding="utf-8")
        priors_sha = sha(priors)

        validation = root / "validation.json"
        write_json(validation, {
            "kind": "ibi_conservative_spline_validation",
            "pass": True,
            "conservative_priors_sha256": priors_sha,
            "finite_difference_checks": [
                {"max_abs_dU_dq_error": 2.0e-6},
                {"max_abs_dU_dq_error": 1.0e-6},
            ],
        })
        parity = root / "parity.json"
        write_json(parity, {
            "kind": "ibi_conservative_spline_runtime_parity",
            "pass": True,
            "priors_sha256": priors_sha,
            "worst_force_abs_error": 2.0e-14,
            "worst_energy_abs_error": 3.0e-13,
            "force_atol": 1.0e-9,
            "energy_atol": 1.0e-10,
        })
        preflight = root / "preflight.json"
        write_json(preflight, {
            "kind": "conservative_ibi_nve_preflight",
            "pass": True,
            "priors_sha256": priors_sha,
        })
        checkpoint = root / "checkpoint.npz"
        checkpoint.write_bytes(b"checkpoint")
        checkpoint_sha = sha(checkpoint)
        equil = root / "equil.json"
        write_json(equil, {
            "pass": True,
            "hamiltonian_mode": MODE,
            "sampling_ensemble": "NVT_Langevin",
            "ml_active": False,
            "checkpoint_sha256": checkpoint_sha,
        })
        strict = root / "strict.json"
        write_json(strict, {
            "definition": {
                "hamiltonian_mode": MODE,
                "thermostat": "off (--nve)",
            },
            "inputs_sha256": {
                "priors": priors_sha,
                "checkpoint": checkpoint_sha,
            },
            "runs": [
                {"relative_block_mean_drift": 3.0e-5},
                {"relative_block_mean_drift": 4.0e-6},
            ],
            "certification": {
                "pass": False,
                "scaling_pass": False,
                "drift_pass": True,
                "drift_failures": [],
                "scaling": {
                    "observable": "sigma_E",
                    "exponent_p": 0.99,
                    "loglog_r2": 0.79,
                },
            },
        })
        state = root / "state.json"
        metric = lambda p: {
            "consistent_with_second_order": True,
            "median_exponent_p": p,
            "median_loglog_r2": 0.998,
            "min_exponent_p": p - 0.05,
            "max_exponent_p": p + 0.05,
            "min_loglog_r2": 0.99,
        }
        write_json(state, {
            "kind": "conservative_ibi_nve_state_convergence_diagnostic",
            "hamiltonian_mode": MODE,
            "checkpoint_sha256": checkpoint_sha,
            "input_hashes": {"priors_sha256": priors_sha},
            "metric_summary": {
                "position_rms_nm": metric(1.88),
                "velocity_rms_nm_per_ps": metric(1.90),
                "orientation_rms_rad": metric(1.99),
                "omega_body_rms_per_ps": metric(2.00),
            },
        })
        return priors, validation, parity, preflight, equil, strict, state

    def build(self, artifacts):
        priors, validation, parity, preflight, equil, strict, state = artifacts
        return build_final_certification(
            priors_path=priors,
            validation_report_path=validation,
            runtime_parity_report_path=parity,
            nve_preflight_report_path=preflight,
            equilibration_report_path=equil,
            strict_nve_report_path=strict,
            state_convergence_report_path=state,
        )

    def test_sigma_scaling_failure_is_non_gating_when_direct_state_order_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.build(self.make_artifacts(Path(tmp)))
            self.assertTrue(report["pass"])
            self.assertFalse(report["legacy_step23_pass"])
            self.assertFalse(report["gates"]["sigma_E_scaling"]["gating"])
            self.assertTrue(report["gates"]["vv_second_order"]["pass"])
            self.assertTrue(report["gates"]["energy_drift"]["pass"])

    def test_state_order_failure_is_gating(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.make_artifacts(Path(tmp))
            state = artifacts[-1]
            data = json.loads(state.read_text())
            data["metric_summary"]["position_rms_nm"]["median_exponent_p"] = 1.2
            data["metric_summary"]["position_rms_nm"]["consistent_with_second_order"] = False
            write_json(state, data)
            report = self.build(artifacts)
            self.assertFalse(report["pass"])
            self.assertFalse(report["gates"]["vv_second_order"]["pass"])

    def test_checkpoint_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.make_artifacts(Path(tmp))
            state = artifacts[-1]
            data = json.loads(state.read_text())
            data["checkpoint_sha256"] = "0" * 64
            write_json(state, data)
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                self.build(artifacts)

    def test_finalizer_records_composite_basis(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.build(self.make_artifacts(Path(tmp)))
            self.assertEqual(
                report["certification_basis"],
                "conservative_kernel_parity_plus_drift_plus_richardson_state_order_v1",
            )
            self.assertEqual(report["gates"]["sigma_E_scaling"]["status"], "diagnostic_only")


if __name__ == "__main__":
    unittest.main()
