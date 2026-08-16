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
TRAINING = ROOT / "training"
sys.path.insert(0, str(SIMULATION))
sys.path.insert(0, str(TRAINING))

from conservative_nve_preflight import validate_conservative_nve_inputs  # noqa: E402
from certify_nve import checkpoint_provenance_summary  # noqa: E402
from framework_utils import input_hashes  # noqa: E402
from residual_input_provenance import referenced_prior_artifacts  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ConservativeNVETests(unittest.TestCase):
    def make_artifacts(self, root: Path):
        source_table = root / "bond_tabulated_bb.dat"
        source_table.write_text("0.1 0.0 1.0\n0.2 0.0 0.0\n")
        source_priors = root / "source_priors.json"
        source_priors.write_text(json.dumps({"bonds": [{"type": "tabulated", "file": source_table.name}]}) + "\n")

        conservative_table = root / "bond_conservative_bb.dat"
        conservative_table.write_text("0.1 1.0 -2.0\n0.2 0.0 0.0\n")
        priors = root / "cg_priors.json"
        priors.write_text(json.dumps({
            "bonds": [{
                "type": "conservative_spline",
                "file": conservative_table.name,
                "min": 0.1,
                "max": 0.2,
            }],
            "angles": [],
            "dihedrals": [],
        }) + "\n")

        conversion = root / "conversion_report.json"
        conversion.write_text(json.dumps({
            "schema_version": 1,
            "framework": "MLCG_Framework_v2",
            "kind": "ibi_conservative_spline_conversion",
            "source_priors": str(source_priors.resolve()),
            "source_priors_sha256": sha(source_priors),
            "output_priors": str(priors.resolve()),
            "output_priors_sha256": sha(priors),
            "records": [{
                "kind": "bond",
                "source_path": str(source_table.resolve()),
                "source_sha256": sha(source_table),
                "output_file": conservative_table.name,
                "output_path": str(conservative_table.resolve()),
                "output_sha256": sha(conservative_table),
            }],
            "source_artifacts_unchanged": True,
        }) + "\n")

        validation = root / "validation_report.json"
        validation.write_text(json.dumps({
            "schema_version": 1,
            "framework": "MLCG_Framework_v2",
            "kind": "ibi_conservative_spline_validation",
            "conversion_report": str(conversion.resolve()),
            "source_priors": str(source_priors.resolve()),
            "source_priors_sha256": sha(source_priors),
            "conservative_priors": str(priors.resolve()),
            "conservative_priors_sha256": sha(priors),
            "finite_difference_checks": [{
                "kind": "bond",
                "file": conservative_table.name,
                "max_abs_dU_dq_error": 1.0e-7,
                "max_relative_error": 1.0e-8,
            }],
            "pass": True,
        }) + "\n")

        parity = root / "runtime_parity_report.json"
        parity.write_text(json.dumps({
            "schema_version": 1,
            "framework": "MLCG_Framework_v2",
            "kind": "ibi_conservative_spline_runtime_parity",
            "priors": str(priors.resolve()),
            "priors_sha256": sha(priors),
            "prior_artifact_sha256": referenced_prior_artifacts(priors),
            "force_atol": 1.0e-9,
            "energy_atol": 1.0e-10,
            "worst_force_abs_error": 1.0e-14,
            "worst_energy_abs_error": 1.0e-13,
            "pass": True,
        }) + "\n")
        return priors, validation, parity, conservative_table

    def test_preflight_accepts_bound_conservative_phase2_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            priors, validation, parity, _ = self.make_artifacts(Path(tmp))
            report = validate_conservative_nve_inputs(
                priors_path=priors,
                validation_report=validation,
                runtime_parity_report=parity,
            )
            self.assertTrue(report["pass"])
            self.assertEqual(report["mode"], "conservative_ibi_only")
            self.assertAlmostEqual(report["runtime_max_force_abs_error"], 1.0e-14)

    def test_preflight_fails_if_validated_spline_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            priors, validation, parity, table = self.make_artifacts(Path(tmp))
            table.write_text(table.read_text() + "0.3 0.0 0.0\n")
            with self.assertRaises(ValueError):
                validate_conservative_nve_inputs(
                    priors_path=priors,
                    validation_report=validation,
                    runtime_parity_report=parity,
                )


    def test_checkpoint_provenance_preflight_matches_runtime_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.bin"
            config = root / "config.json"
            priors = root / "priors.json"
            rb = root / "rb.json"
            model = root / "model.pt"
            manifest = root / "model.pt.manifest.json"
            for path in (dataset, config, priors, rb, model, manifest):
                path.write_text(path.name + "\n")
            hashes = input_hashes(
                dataset=dataset, config=config, priors=priors, rb_info=rb, model=model
            )
            checkpoint = root / "checkpoint.npz"
            import numpy as np
            np.savez_compressed(
                checkpoint,
                metadata_json=np.asarray(json.dumps({
                    "schema_version": 3,
                    "energy_gauge": "test",
                    "input_hashes": hashes,
                }, sort_keys=True)),
                v=np.asarray([[0.1, 0.0, 0.0]]),
                omega=np.zeros((1, 3)),
                particle_is_virtual=np.asarray([False]),
            )
            result = checkpoint_provenance_summary(
                checkpoint,
                dataset=dataset,
                config=config,
                priors=priors,
                rb_info=rb,
                model=model,
            )
            self.assertEqual(result["input_hashes"], hashes)

    def test_checkpoint_provenance_preflight_rejects_changed_priors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.bin"
            config = root / "config.json"
            priors = root / "priors.json"
            rb = root / "rb.json"
            model = root / "model.pt"
            manifest = root / "model.pt.manifest.json"
            for path in (dataset, config, priors, rb, model, manifest):
                path.write_text(path.name + "\n")
            hashes = input_hashes(
                dataset=dataset, config=config, priors=priors, rb_info=rb, model=model
            )
            checkpoint = root / "checkpoint.npz"
            import numpy as np
            np.savez_compressed(
                checkpoint,
                metadata_json=np.asarray(json.dumps({"input_hashes": hashes}, sort_keys=True)),
                v=np.asarray([[0.1, 0.0, 0.0]]),
                omega=np.zeros((1, 3)),
                particle_is_virtual=np.asarray([False]),
            )
            priors.write_text("changed\n")
            with self.assertRaises(ValueError):
                checkpoint_provenance_summary(
                    checkpoint,
                    dataset=dataset,
                    config=config,
                    priors=priors,
                    rb_info=rb,
                    model=model,
                )


    def test_checkpoint_mode_and_source_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.bin"
            config = root / "config.json"
            priors = root / "priors.json"
            rb = root / "rb.json"
            model = root / "model.pt"
            manifest = root / "model.pt.manifest.json"
            source = root / "source.npz"
            for path in (dataset, config, priors, rb, model, manifest):
                path.write_text(path.name + "\n")
            import numpy as np
            np.savez_compressed(source, marker=np.asarray([1]))
            hashes = input_hashes(
                dataset=dataset, config=config, priors=priors, rb_info=rb, model=model
            )
            import hashlib
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            checkpoint = root / "checkpoint.npz"
            np.savez_compressed(
                checkpoint,
                metadata_json=np.asarray(json.dumps({
                    "schema_version": 3,
                    "energy_gauge": "test",
                    "input_hashes": hashes,
                    "hamiltonian_mode": "conservative_classical_model_provenance_ml_disabled",
                    "sampling_ensemble": "NVT_Langevin",
                    "source_checkpoint_sha256": source_sha,
                }, sort_keys=True)),
                v=np.asarray([[0.1, 0.0, 0.0]]),
                omega=np.zeros((1, 3)),
                particle_is_virtual=np.asarray([False]),
            )
            result = checkpoint_provenance_summary(
                checkpoint,
                dataset=dataset,
                config=config,
                priors=priors,
                rb_info=rb,
                model=model,
                required_hamiltonian_mode="conservative_classical_model_provenance_ml_disabled",
                required_source_checkpoint=source,
            )
            self.assertEqual(result["sampling_ensemble"], "NVT_Langevin")
            with self.assertRaises(ValueError):
                checkpoint_provenance_summary(
                    checkpoint,
                    dataset=dataset,
                    config=config,
                    priors=priors,
                    rb_info=rb,
                    model=model,
                    required_hamiltonian_mode="painn_active",
                    required_source_checkpoint=source,
                )

    def test_step23_prepares_dedicated_ibi_only_checkpoint(self):
        source = ((ROOT / "tutorials") / ("tel" + "22_IBI") / "23_certify_conservative_ibi_nve.sh").read_text()
        runner = (SIMULATION / "run_cg_md.py").read_text()
        self.assertIn("NVE_EQ_DURATION_PS", source)
        self.assertIn("--disable_ml", source)
        self.assertIn("--out_checkpoint", source)
        self.assertIn("--require-checkpoint-hamiltonian-mode", source)
        self.assertIn("--require-checkpoint-source", source)
        self.assertIn('parser.add_argument("--out_checkpoint"', runner)
        self.assertIn('"hamiltonian_mode": hamiltonian_mode', runner)
        self.assertIn('"sampling_ensemble": "NVE" if args.nve else "NVT_Langevin"', runner)

    def test_certifier_supports_model_provenance_with_ml_disabled(self):
        source = (SIMULATION / "certify_nve.py").read_text()
        self.assertIn('"--disable-ml"', source)
        self.assertIn('command.append("--disable_ml")', source)
        self.assertIn('"conservative_classical_model_provenance_ml_disabled"', source)
        self.assertIn('"--provenance-artifact"', source)


    def test_step24_is_diagnostic_only_and_scans_fine_dt(self):
        source = ((ROOT / "tutorials") / ("tel" + "22_IBI") / "24_diagnose_conservative_ibi_nve_scaling.sh").read_text()
        certifier = (SIMULATION / "certify_nve.py").read_text()
        self.assertIn("NVE_DIAG_DTS", source)
        self.assertIn("0.00025 0.0005 0.00075", source)
        self.assertIn("--diagnostic-only", source)
        self.assertIn("--local-times-ps", source)
        self.assertIn("--diagnostic-only", certifier)
        self.assertIn("nve_diagnostic_report.json", certifier)
        self.assertIn("Diagnostic-only scan completed", certifier)



if __name__ == "__main__":
    unittest.main()
