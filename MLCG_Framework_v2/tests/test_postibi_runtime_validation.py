import csv
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "ibi"))

from analyze_nvt_smoke import analyze_nvt_smoke  # noqa: E402
from runtime_preflight import check_runtime_preflight, sha256_file  # noqa: E402
from residual_input_provenance import referenced_prior_artifacts, write_manifest  # noqa: E402
from validate_runtime_structure import validate_runtime_structure  # noqa: E402


class RuntimePreflightTests(unittest.TestCase):
    def _fixture(self, root: Path):
        model = root / "model.pt"
        config = root / "config.json"
        dataset = root / "dataset.bin"
        rb_info = root / "rb.json"
        priors = root / "priors.json"
        aa_top = root / "aa.gro"
        aa_traj = root / "aa.trr"
        mapping = root / "mapping.json"
        validation = root / "validation.json"
        residual_manifest = root / "residual_manifest.json"

        model.write_bytes(b"model-bytes")
        dataset.write_bytes(b"dataset-bytes")
        rb_info.write_text("{}\n")
        aa_top.write_text("gro\n")
        aa_traj.write_bytes(b"trajectory")
        mapping.write_text("{}\n")
        priors.write_text(json.dumps({"bonds": [], "angles": [], "dihedrals": []}) + "\n")
        cfg = {
            "architecture_variant": "painn_canonical_context_silu_v2",
            "num_species": 8,
            "hidden_channels": 64,
            "n_layers": 2,
            "num_rbf": 32,
            "cutoff": 1.2616,
            "toxvaerd_alpha": 0.1,
        }
        config.write_text(json.dumps(cfg) + "\n")
        validation.write_text(json.dumps({
            "schema_version": 1,
            "mode": "read_only_validation",
            "source_priors_unchanged": True,
            "priors": str(priors.resolve()),
            "source_artifact_sha256": referenced_prior_artifacts(priors),
            "mean_l1": 0.21,
            "max_l1": 0.35,
        }) + "\n")
        write_manifest(
            output=residual_manifest,
            dataset=dataset,
            rb_info=rb_info,
            priors=priors,
            aa_topology=aa_top,
            aa_trajectory=aa_traj,
            mapping_config=mapping,
            validation_report=validation,
        )
        model_manifest = {
            "schema_version": 3,
            "framework": "MLCG_Framework_v2",
            "energy_gauge": "isolated_species_zero_v1",
            "architecture": {
                "variant": cfg["architecture_variant"],
                "num_species": 8,
                "hidden_channels": 64,
                "n_layers": 2,
                "num_rbf": 32,
                "cutoff": 1.2616,
                "toxvaerd_alpha": 0.1,
            },
            "best_validation_loss": 1.5,
            "model_path": model.name,
            "model_file_size_bytes": model.stat().st_size,
            "model_sha256": sha256_file(model),
            "dataset_path": dataset.name,
            "dataset_file_size_bytes": dataset.stat().st_size,
            "dataset_sha256": sha256_file(dataset),
            "config_path": config.name,
            "config_file_size_bytes": config.stat().st_size,
            "config_sha256": sha256_file(config),
        }
        Path(f"{model}.manifest.json").write_text(json.dumps(model_manifest) + "\n")
        return model, config, dataset, priors, rb_info, residual_manifest

    def test_runtime_preflight_binds_model_to_residual_hamiltonian(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            report = check_runtime_preflight(
                model=paths[0], config=paths[1], dataset=paths[2], priors=paths[3],
                rb_info=paths[4], residual_manifest=paths[5],
            )
            self.assertTrue(report["pass"])
            self.assertEqual(report["artifacts"]["dataset"]["sha256"], sha256_file(paths[2]))

    def test_runtime_preflight_rejects_model_dataset_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            manifest_path = Path(f"{paths[0]}.manifest.json")
            manifest = json.loads(manifest_path.read_text())
            manifest["dataset_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest) + "\n")
            with self.assertRaisesRegex(ValueError, "dataset SHA256 mismatch"):
                check_runtime_preflight(
                    model=paths[0], config=paths[1], dataset=paths[2], priors=paths[3],
                    rb_info=paths[4], residual_manifest=paths[5],
                )

    def test_runtime_preflight_accepts_conservative_validation_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, config, dataset, priors, rb_info, residual_manifest = self._fixture(root)
            source_table = root / "source_bond.dat"
            source_table.write_text("0.0 1.0 -2.0\n1.0 0.0 0.0\n")
            source_priors = root / "source_priors.json"
            source_priors.write_text(json.dumps({
                "bonds": [{"type": "tabulated", "file": source_table.name}],
                "angles": [], "dihedrals": [],
            }) + "\n")
            spline = root / "bond_conservative_b.dat"
            spline.write_text("0.0 1.0 -2.0\n1.0 0.0 0.0\n")
            priors.write_text(json.dumps({
                "bonds": [{"type": "conservative_spline", "file": spline.name}],
                "angles": [], "dihedrals": [],
            }) + "\n")

            def digest(path):
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()

            conversion = root / "conversion_report.json"
            conversion.write_text(json.dumps({
                "schema_version": 1,
                "framework": "MLCG_Framework_v2",
                "kind": "ibi_conservative_spline_conversion",
                "source_priors": str(source_priors.resolve()),
                "source_priors_sha256": digest(source_priors),
                "output_priors": str(priors.resolve()),
                "output_priors_sha256": digest(priors),
                "records": [{
                    "source_path": str(source_table.resolve()),
                    "source_sha256": digest(source_table),
                    "output_file": spline.name,
                    "output_path": str(spline.resolve()),
                    "output_sha256": digest(spline),
                }],
                "source_artifacts_unchanged": True,
            }) + "\n")
            validation = root / "validation_conservative.json"
            validation.write_text(json.dumps({
                "schema_version": 1,
                "framework": "MLCG_Framework_v2",
                "kind": "ibi_conservative_spline_validation",
                "conversion_report": str(conversion.resolve()),
                "conservative_priors": str(priors.resolve()),
                "conservative_priors_sha256": digest(priors),
                "finite_difference_checks": [{"max_abs_dU_dq_error": 1.0e-7}],
                "pass": True,
            }) + "\n")
            parity = root / "runtime_parity_report.json"
            parity.write_text(json.dumps({
                "schema_version": 1,
                "framework": "MLCG_Framework_v2",
                "kind": "ibi_conservative_spline_runtime_parity",
                "priors": str(priors.resolve()),
                "priors_sha256": digest(priors),
                "prior_artifact_sha256": referenced_prior_artifacts(priors),
                "force_atol": 1.0e-9,
                "energy_atol": 1.0e-10,
                "worst_force_abs_error": 2.0e-14,
                "worst_energy_abs_error": 3.0e-13,
                "pass": True,
            }) + "\n")
            write_manifest(
                output=residual_manifest,
                dataset=dataset,
                rb_info=rb_info,
                priors=priors,
                aa_topology=root / "aa.gro",
                aa_trajectory=root / "aa.trr",
                mapping_config=root / "mapping.json",
                validation_report=validation,
                runtime_parity_report=parity,
            )

            report = check_runtime_preflight(
                model=model, config=config, dataset=dataset, priors=priors,
                rb_info=rb_info, residual_manifest=residual_manifest,
            )
            self.assertEqual(
                report["residual_validation"]["mode"],
                "conservative_spline_validation",
            )


class NvtSmokeAnalysisTests(unittest.TestCase):
    HEADER = [
        "Step", "Time_ps", "E_tot", "E_kin", "E_kin_trans", "E_kin_rot",
        "E_class", "E_ml", "min_dist", "min_pair", "min_pids", "f_max", "torque_max",
    ]

    def _write_csv(self, path: Path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.HEADER)
            writer.writerows(rows)

    def test_nvt_smoke_accepts_complete_finite_stable_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "energy.csv"
            self._write_csv(path, [
                [0, 0.0, -10, 100, 60, 40, -20, 10, 0.3, "0:1", "1:2", 300, 20],
                [100, 0.05, -9, 110, 65, 45, -19, 10, 0.25, "0:1", "1:2", 450, 35],
                [200, 0.1, -8, 105, 62, 43, -18, 10, 0.22, "0:1", "1:2", 500, 40],
            ])
            report = analyze_nvt_smoke(path, expected_steps=200)
            self.assertTrue(report["pass"])
            self.assertFalse(report["energy_conservation_certified"])
            self.assertEqual(report["final_step"], 200)

    def test_nvt_smoke_rejects_incomplete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "energy.csv"
            self._write_csv(path, [
                [0, 0.0, -10, 100, 60, 40, -20, 10, 0.3, "0:1", "1:2", 300, 20],
                [100, 0.05, -9, 110, 65, 45, -19, 10, 0.25, "0:1", "1:2", 450, 35],
            ])
            with self.assertRaisesRegex(ValueError, "did not reach"):
                analyze_nvt_smoke(path, expected_steps=200)


class RuntimeStructureTests(unittest.TestCase):
    def test_runtime_structure_reports_bond_l1_without_hard_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.bin"
            priors = root / "priors.json"
            table = root / "bond.dat"
            sample = root / "sample.npz"

            # One frame, two one-site molecules, 1.0 nm COM separation.
            with dataset.open("wb") as handle:
                handle.write(struct.pack("i", 1))
                handle.write(struct.pack("i", 2))
                handle.write(struct.pack("i", 2))
                handle.write(struct.pack("3f", 10.0, 10.0, 10.0))
                for mol, x in ((0, 0.0), (1, 1.0)):
                    handle.write(struct.pack("i", mol))
                    handle.write(struct.pack("i", 1))
                    handle.write(struct.pack("3f", x, 0.0, 0.0))
                    handle.write(struct.pack("3f", 0.0, 0.0, 0.0))
                    handle.write(struct.pack("3f", 0.0, 0.0, 0.0))
                    handle.write(struct.pack("i", mol))
                    handle.write(struct.pack("3f", x, 0.0, 0.0))

            np.savetxt(table, np.array([
                [0.01, 0.0, 0.0],
                [2.505, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]))
            priors.write_text(json.dumps({
                "bonds": [{
                    "type": "tabulated", "ibi_mode": "ibi", "name": "b",
                    "file": table.name, "min": 0.01, "max": 5.0,
                    "mol_i": 0, "mol_j": 1, "site_i": -1, "site_j": -1,
                }],
                "angles": [], "dihedrals": [],
            }) + "\n")
            com = np.array([
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ])
            sites = com.copy()
            np.savez_compressed(
                sample,
                schema_version=np.asarray(1, dtype=np.int32),
                complete=np.asarray(1, dtype=np.int8),
                steps=np.asarray([0, 100], dtype=np.int64),
                time_ps=np.asarray([0.0, 0.05]),
                com=com,
                sites=sites,
                site_molecule=np.asarray([0, 1], dtype=np.int32),
                site_index=np.asarray([0, 0], dtype=np.int32),
                box=np.asarray([10.0, 10.0, 10.0]),
            )
            report = validate_runtime_structure(
                dataset=dataset, priors=priors, sample_npz=sample
            )
            self.assertTrue(report["pass"])
            self.assertFalse(report["threshold_applied"])
            self.assertAlmostEqual(report["mean_l1"], 0.0, places=12)

    def test_runtime_structure_accepts_conservative_spline_priors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.bin"
            priors = root / "priors.json"
            spline = root / "bond_conservative_b.dat"
            sample = root / "sample.npz"

            with dataset.open("wb") as handle:
                handle.write(struct.pack("i", 1))
                handle.write(struct.pack("i", 2))
                handle.write(struct.pack("i", 2))
                handle.write(struct.pack("3f", 10.0, 10.0, 10.0))
                for mol, x in ((0, 0.0), (1, 1.0)):
                    handle.write(struct.pack("i", mol))
                    handle.write(struct.pack("i", 1))
                    handle.write(struct.pack("3f", x, 0.0, 0.0))
                    handle.write(struct.pack("3f", 0.0, 0.0, 0.0))
                    handle.write(struct.pack("3f", 0.0, 0.0, 0.0))
                    handle.write(struct.pack("i", mol))
                    handle.write(struct.pack("3f", x, 0.0, 0.0))

            np.savetxt(spline, np.array([
                [0.01, 0.0, 0.0],
                [2.505, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]))
            priors.write_text(json.dumps({
                "bonds": [{
                    "type": "conservative_spline", "ibi_mode": "ibi", "name": "b",
                    "file": spline.name, "min": 0.01, "max": 5.0,
                    "spline_schema": "pchip_hermite_v1",
                    "mol_i": 0, "mol_j": 1, "site_i": -1, "site_j": -1,
                }],
                "angles": [], "dihedrals": [],
            }) + "\n")
            com = np.array([
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ])
            np.savez_compressed(
                sample,
                schema_version=np.asarray(1, dtype=np.int32),
                complete=np.asarray(1, dtype=np.int8),
                steps=np.asarray([0, 100], dtype=np.int64),
                time_ps=np.asarray([0.0, 0.05]),
                com=com,
                sites=com.copy(),
                site_molecule=np.asarray([0, 1], dtype=np.int32),
                site_index=np.asarray([0, 0], dtype=np.int32),
                box=np.asarray([10.0, 10.0, 10.0]),
            )
            report = validate_runtime_structure(dataset=dataset, priors=priors, sample_npz=sample)
            self.assertTrue(report["pass"])
            self.assertAlmostEqual(report["mean_l1"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
