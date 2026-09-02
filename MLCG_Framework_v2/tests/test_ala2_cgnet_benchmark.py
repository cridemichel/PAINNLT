import importlib.util
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "tutorials" / "ala2_cgnet" / "diagnostics" / "scripts"


def load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = load_script("build_ala2_dataset.py")
downloader = load_script("download_cgnet_ala2.py")
official_downloader = load_script("download_official_cgnet.py")
validator = load_script("validate_ala2_benchmark.py")
runtime_preparer = load_script("prepare_ala2_runtime.py")
fes_analyzer = load_script("analyze_ala2_fes_ab.py")


class Ala2CgnetBenchmarkTests(unittest.TestCase):
    def synthetic_coordinates(self, frames=128):
        rng = np.random.default_rng(20260901)
        base = np.asarray(
            [
                [0.00, 0.00, 0.00],
                [0.13, 0.00, 0.00],
                [0.20, 0.12, 0.01],
                [0.34, 0.14, 0.08],
                [0.43, 0.24, 0.10],
            ],
            dtype=np.float64,
        )
        return base[None, :, :] + rng.normal(scale=0.006, size=(frames, 5, 3))

    def prior_energy(self, coordinates, bonds, angles):
        energy = 0.0
        for bond in bonds:
            i, j = bond["mol_i"], bond["mol_j"]
            distance = np.linalg.norm(coordinates[j] - coordinates[i])
            energy += 0.5 * bond["k"] * (distance - bond["r0"]) ** 2
        for angle in angles:
            i, j, k = angle["mol_i"], angle["mol_j"], angle["mol_k"]
            a = coordinates[i] - coordinates[j]
            b = coordinates[k] - coordinates[j]
            cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            theta = np.arccos(np.clip(cosine, -1.0, 1.0))
            energy += 0.5 * angle["k"] * (theta - angle["theta0"]) ** 2
        return energy

    def test_official_data_contract_is_pinned(self):
        self.assertEqual(downloader.EXPECTED_SHAPE, (10000, 5, 3))
        self.assertEqual(len(downloader.CGNET_COMMIT), 40)
        self.assertEqual(
            downloader.FILES["ala2_coordinates.npy"],
            "00f1f6b70fbc9473157511d53a73b6f629d284d3e08e79155b9d2bf546d6dc81",
        )
        self.assertEqual(
            downloader.FILES["ala2_forces.npy"],
            "de1936e1a431b789cb3366b6d5c0208913d2b516d81d199f09f5c914bb536f56",
        )

    def test_official_cgnet_source_contract_is_pinned_and_safe(self):
        self.assertEqual(official_downloader.CGNET_COMMIT, downloader.CGNET_COMMIT)
        self.assertEqual(len(official_downloader.ARCHIVE_SHA256), 64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                official_downloader.safe_extract(archive, root / "output")

    def test_unit_conversions_and_bead_types(self):
        self.assertEqual(builder.ANGSTROM_TO_NM, 0.1)
        self.assertEqual(builder.KCAL_PER_MOL_ANGSTROM_TO_KJ_PER_MOL_NM, 41.84)
        np.testing.assert_array_equal(builder.BEAD_TYPES, [6, 7, 6, 6, 7])

    def test_harmonic_priors_match_negative_energy_gradient(self):
        coordinates = self.synthetic_coordinates()
        bonds, angles = builder.fit_harmonic_priors(coordinates, 300.0)
        self.assertEqual(len(bonds), 4)
        self.assertEqual(len(angles), 3)
        self.assertTrue(all(item["k"] > 0.0 for item in bonds + angles))

        frame = coordinates[[0]].copy()
        analytic = builder.harmonic_prior_forces(frame, bonds, angles)[0]
        numerical = np.zeros_like(analytic)
        epsilon = 1.0e-7
        for bead in range(5):
            for dimension in range(3):
                plus = frame[0].copy()
                minus = frame[0].copy()
                plus[bead, dimension] += epsilon
                minus[bead, dimension] -= epsilon
                numerical[bead, dimension] = -(
                    self.prior_energy(plus, bonds, angles)
                    - self.prior_energy(minus, bonds, angles)
                ) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, numerical, rtol=2.0e-6, atol=2.0e-5)
        np.testing.assert_allclose(np.sum(analytic, axis=0), 0.0, atol=1.0e-9)

    def test_binary_writer_matches_trainer_layout(self):
        coordinates = self.synthetic_coordinates(frames=3) + 1.5
        forces = np.arange(45, dtype=np.float64).reshape(3, 5, 3)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset.bin"
            builder.write_dataset(output, coordinates, forces, 4.0)
            with output.open("rb") as handle:
                (frames,) = struct.unpack("<i", handle.read(4))
                self.assertEqual(frames, 3)
                for frame_index in range(frames):
                    molecules, sites, bx, by, bz = struct.unpack("<ii3f", handle.read(20))
                    self.assertEqual((molecules, sites), (5, 5))
                    self.assertEqual((bx, by, bz), (4.0, 4.0, 4.0))
                    for bead in range(5):
                        molecule_id, num_sites = struct.unpack("<ii", handle.read(8))
                        center = struct.unpack("<3f", handle.read(12))
                        target = struct.unpack("<3f", handle.read(12))
                        torque = struct.unpack("<3f", handle.read(12))
                        site_type, sx, sy, sz = struct.unpack("<i3f", handle.read(16))
                        self.assertEqual((molecule_id, num_sites), (bead, 1))
                        self.assertEqual(site_type, int(builder.BEAD_TYPES[bead]))
                        np.testing.assert_allclose(center, coordinates[frame_index, bead])
                        np.testing.assert_allclose((sx, sy, sz), center)
                        np.testing.assert_allclose(target, forces[frame_index, bead])
                        self.assertEqual(torque, (0.0, 0.0, 0.0))
                self.assertEqual(handle.read(1), b"")

    def test_config_and_skill_classification(self):
        config_path = (
            ROOT
            / "tutorials"
            / "ala2_cgnet"
            / "diagnostics"
            / "configs"
            / "ala2_training_config_50ep.json"
        )
        config = json.loads(config_path.read_text())
        self.assertEqual(config["validation_split_mode"], "tail")
        self.assertEqual(config["validation_tail_frames"], 2000)
        self.assertEqual(config["torque_weight"], 0.0)
        self.assertEqual(config["cutoff"], 0.5)
        self.assertEqual(config["hidden_channels"], 128)
        self.assertEqual(config["n_layers"], 5)
        self.assertEqual(config["num_rbf"], 50)
        self.assertEqual(config["batch_size"], 512)
        self.assertEqual(config["learning_rate"], 0.0003)
        self.assertEqual(validator.classify_skill(-0.01), "negative")
        self.assertEqual(validator.classify_skill(0.01), "weak")
        self.assertEqual(validator.classify_skill(0.07), "moderate")
        self.assertEqual(validator.classify_skill(0.15), "strong")

    def test_alltoall_spectral_diagnostic_is_controlled(self):
        config_path = (
            ROOT
            / "tutorials"
            / "ala2_cgnet"
            / "diagnostics"
            / "configs"
            / "ala2_training_config_painn_cgnetmatched_5ep.json"
        )
        baseline_path = config_path.with_name("ala2_training_config_50ep.json")
        config = json.loads(config_path.read_text())
        baseline = json.loads(baseline_path.read_text())
        changed = {key for key in set(config) | set(baseline) if config.get(key) != baseline.get(key)}
        self.assertEqual(
            changed,
            {
                "cutoff",
                "early_stopping_patience",
                "epoch_lr_decay_factor",
                "epochs",
                "grad_clip_norm",
                "hidden_channels",
                "learning_rate",
                "reduce_lr_patience",
                "spectral_projection_strength",
                "spectral_projection_power_iterations",
            },
        )
        self.assertEqual(config["cutoff"], 1.0)
        self.assertEqual(config["hidden_channels"], 160)
        self.assertEqual(config["learning_rate"], 0.003)
        self.assertEqual(config["epoch_lr_decay_factor"], 0.3)
        self.assertEqual(config["epochs"], 5)
        self.assertEqual(config["grad_clip_norm"], 0.0)
        self.assertEqual(config["spectral_projection_strength"], 4.0)
        self.assertEqual(config["spectral_projection_power_iterations"], 8)

        runner = (SCRIPTS / "04_test_ala2_painn_alltoall_spectral.sh").read_text()
        self.assertIn("ALA2_ALLTOALL_SOURCE_RUN_DIR", runner)
        self.assertIn("--require-all-to-all", runner)
        self.assertIn("--fes-only", runner)
        trainer = (ROOT / "training" / "train_painn.cpp").read_text()
        self.assertIn("project_dense_spectral_norms", trainer)
        self.assertIn('name == "embedding.weight"', trainer)
        self.assertIn("spectral_projection_strength", trainer)
        self.assertIn("epoch_lr_decay_factor", trainer)

    def test_ordered_geometry_head_is_chirality_aware_and_bound_to_runtime(self):
        config_path = (
            ROOT
            / "tutorials"
            / "ala2_cgnet"
            / "diagnostics"
            / "configs"
            / "ala2_training_config_painn_ordered_geometry_5ep.json"
        )
        config = json.loads(config_path.read_text())
        self.assertEqual(config["architecture_variant"], "painn_ordered_geometry_tanh_v2")
        self.assertEqual(config["ordered_geometry_nodes"], 5)
        self.assertEqual(config["ordered_geometry_head_layers"], 5)
        self.assertEqual(config["ordered_geometry_head_width"], 160)
        self.assertEqual(config["ordered_geometry_energy_scale_kj_mol"], 4.184)
        self.assertAlmostEqual(
            config["ordered_geometry_energy_scale_kj_mol"],
            builder.KCAL_PER_MOL_ANGSTROM_TO_KJ_PER_MOL_NM * 0.1,
            places=12,
        )
        feature_count = 5 * 4 // 2 + (5 - 2) + 2 * (5 - 3)
        self.assertEqual(feature_count, 17)

        coordinates = self.synthetic_coordinates(frames=1)[0]

        def signed_dihedral(points):
            b0 = points[1] - points[0]
            b1 = points[2] - points[1]
            b2 = points[3] - points[2]
            n1 = np.cross(b0, b1)
            n2 = np.cross(b1, b2)
            denominator = np.linalg.norm(n1) * np.linalg.norm(n2)
            cosine = np.dot(n1, n2) / denominator
            sine = np.dot(np.cross(n1, n2), b1 / np.linalg.norm(b1)) / denominator
            return cosine, sine

        reflected = coordinates.copy()
        reflected[:, 2] *= -1.0
        cosine, sine = signed_dihedral(coordinates[:4])
        reflected_cosine, reflected_sine = signed_dihedral(reflected[:4])
        self.assertAlmostEqual(cosine, reflected_cosine, places=12)
        self.assertAlmostEqual(sine, -reflected_sine, places=12)
        self.assertGreater(abs(sine), 1.0e-3)

        header = (ROOT / "training" / "PaiNN_Architecture.hpp").read_text()
        self.assertIn("PAINN_ORDERED_GEOMETRY_VARIANT", header)
        self.assertIn("ordered_geometry_features", header)
        self.assertIn("torch::linalg_cross(normal_1, normal_2, 1)", header)
        self.assertIn("ordered_geometry_mean", header)
        self.assertIn("* ordered_geometry_energy_scale", header)
        self.assertNotIn("return (raw - reference) * energy_scale;", header)
        trainer = (ROOT / "training" / "train_painn.cpp").read_text()
        self.assertIn("fit_ordered_geometry_statistics", trainer)
        self.assertIn("normalization fitted on TRAIN only", trainer)
        self.assertIn('effective_config["ordered_geometry_feature_mean"]', trainer)
        manifest_writer = (ROOT / "training" / "create_model_manifest.py").read_text()
        self.assertIn("required_statistics", manifest_writer)
        self.assertIn("Do not recreate this manifest from config alone", manifest_writer)
        runner = (SCRIPTS / "05_test_ala2_painn_ordered_geometry.sh").read_text()
        self.assertIn("--require-ordered-geometry", runner)
        runtime = (ROOT / "simulation" / "run_cg_md.py").read_text()
        self.assertIn('ordered_geometry_nodes=int(nn_config.get("ordered_geometry_nodes", 0))', runtime)
        self.assertIn('nn_config.get("ordered_geometry_energy_scale_kj_mol", 0.0)', runtime)
        cython_api = (ROOT / "simulation" / "espresso_plugin" / "painn.pyx").read_text()
        self.assertLess(
            cython_api.index('device: str = "auto"'),
            cython_api.index("ordered_geometry_nodes: int = 0"),
        )
        self.assertLess(
            cython_api.index('device: str = "auto"'),
            cython_api.index("ordered_geometry_energy_scale_kj_mol: float = 0.0"),
        )
        documentation = (ROOT / "tutorials" / "ala2_cgnet" / "ORDERED_GEOMETRY_HEAD.md").read_text()
        self.assertIn("Train-only normalization", documentation)
        self.assertIn("Fail-closed runtime contract", documentation)

    def test_runtime_documents_follow_framework_contract(self):
        coordinates = self.synthetic_coordinates()
        self.assertGreater(builder.minimum_nonbonded_distance(coordinates), 0.22)
        bonds, angles = builder.fit_harmonic_priors(coordinates, 300.0)
        priors = builder.build_priors_document(bonds, angles, 300.0, "harmonic")
        policy = priors["wca_exclusions"]
        self.assertEqual(policy["policy_version"], 3)
        self.assertEqual(policy["direct_pairs"], [[0, 1], [1, 2], [2, 3], [3, 4]])
        self.assertEqual(
            policy["direct_site_pairs"],
            [[0, 1, 0, 0], [1, 2, 0, 0], [2, 3, 0, 0], [3, 4, 0, 0]],
        )
        self.assertEqual(policy["one_three_pairs"], [[0, 2], [1, 3], [2, 4]])
        self.assertEqual(set(priors["wca_pairs"]), {"6_6", "6_7", "7_7"})
        self.assertTrue(
            all(item["cutoff_nm"] == 0.22 for item in priors["wca_pairs"].values())
        )
        self.assertTrue(all(item["site_i"] == 0 and item["site_j"] == 0 for item in bonds))

        rigid_bodies = builder.build_rigid_bodies_document()
        self.assertEqual(set(rigid_bodies), {"ALA2_C", "ALA2_N"})
        self.assertEqual(rigid_bodies["ALA2_C"]["sites"]["C"]["type"], 6)
        self.assertEqual(rigid_bodies["ALA2_N"]["sites"]["N"]["type"], 7)

    def test_runtime_preparer_upgrades_existing_training_priors(self):
        coordinates = self.synthetic_coordinates()
        bonds, angles = builder.fit_harmonic_priors(coordinates, 300.0)
        for item in bonds + angles:
            for key in list(item):
                if key.startswith("site_"):
                    item[key] = -1
        old_document = {
            "bonds": bonds,
            "angles": angles,
            "wca_pairs": {},
            "morse_type_pairs": [],
            "dihedrals": [],
        }
        upgraded = runtime_preparer.runtime_priors(old_document)
        self.assertTrue(all(item["site_i"] == 0 for item in upgraded["bonds"]))
        self.assertTrue(all(item["site_j"] == 0 for item in upgraded["angles"]))
        self.assertEqual(upgraded["wca_exclusions"]["direct_site_pair_count"], 4)
        self.assertEqual(set(upgraded["wca_pairs"]), {"6_6", "6_7", "7_7"})

    def test_replica_dataset_extraction_preserves_frames(self):
        coordinates = self.synthetic_coordinates(frames=5) + 1.5
        forces = np.zeros_like(coordinates)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            extracted = Path(temporary) / "one.bin"
            builder.write_dataset(source, coordinates, forces, 4.0)
            payloads = runtime_preparer.read_frame_payloads(source)
            self.assertEqual(len(payloads), 5)
            runtime_preparer.write_single_frame_dataset(extracted, payloads[3])
            one = runtime_preparer.read_frame_payloads(extracted)
            self.assertEqual(one, [payloads[3]])

    def test_fes_metrics_reward_matching_distribution(self):
        reference = np.asarray([[20.0, 2.0], [1.0, 12.0]])
        identical = fes_analyzer.surface_metrics(reference, reference, 0.5, 1)
        wrong = fes_analyzer.surface_metrics(reference, np.flip(reference, axis=0), 0.5, 1)
        self.assertAlmostEqual(identical["js_divergence_nats"], 0.0)
        self.assertAlmostEqual(identical["fes_mse_kbt2"], 0.0)
        self.assertGreater(wrong["js_divergence_nats"], identical["js_divergence_nats"])
        self.assertGreater(wrong["fes_mse_kbt2"], identical["fes_mse_kbt2"])

    def test_ab_runner_uses_matched_checkpoint_and_paper_metric(self):
        runner = (SCRIPTS / "02_test_ala2_free_energy_ab.sh").read_text()
        self.assertIn("--disable_ml", runner)
        self.assertEqual(runner.count('--checkpoint "${common_checkpoint}"'), 2)
        self.assertIn("--thermostat_seed \"$((4200 + replica))\"", runner)
        analyzer_source = (SCRIPTS / "analyze_ala2_fes_ab.py").read_text()
        self.assertIn('"fes_mse_kbt2"', analyzer_source)
        self.assertIn("paper_simulation_protocol", analyzer_source)

    def test_official_cgnet_comparator_is_reference_not_cgnet_like(self):
        source = (SCRIPTS / "run_official_cgnet_comparator.py").read_text()
        self.assertIn("from cgnet.network import (", source)
        self.assertIn("N_LAYERS = 5", source)
        self.assertIn("N_NODES = 160", source)
        self.assertIn("LEARNING_RATE = 0.003", source)
        self.assertIn("LIPSCHITZ_STRENGTH = 4.0", source)
        self.assertIn("Simulation(", source)
        self.assertIn("Brownian branch 1/2: harmonic prior only", source)
        self.assertIn("same_brownian_random_seed", source)
        self.assertIn("parameter.zero_()", source)
        self.assertIn("full_arch_state = state_dict_on_cpu(model.arch)", source)
        self.assertIn("model.arch.load_state_dict(full_arch_state)", source)
        self.assertNotIn("copy.deepcopy(model)", source)
        runner = (SCRIPTS / "03_compare_official_cgnet.sh").read_text()
        self.assertIn("--matched-runtime-samples", runner)
        self.assertIn("--cgnet-prior-samples", runner)
        self.assertIn("--cgnet-samples", runner)
        self.assertNotIn("PYRESSO=", runner)
        analyzer_source = (SCRIPTS / "analyze_ala2_fes_ab.py").read_text()
        self.assertIn('"matched_brownian_ab"', analyzer_source)
        self.assertIn('"cgnet_correction_improves_fes"', analyzer_source)

    def test_generic_paired_bootstrap_rewards_candidate(self):
        reference = np.asarray([[50.0, 1.0], [1.0, 25.0]])
        baseline = [np.asarray([[20.0, 20.0], [20.0, 20.0]]) for _ in range(4)]
        candidate = [reference.copy() for _ in range(4)]
        result = fes_analyzer.bootstrap_delta(reference, baseline, candidate, 0.5, 200)
        self.assertGreater(result["mean_js_improvement_nats"], 0.0)
        self.assertGreater(result["ci95_low_nats"], 0.0)


if __name__ == "__main__":
    unittest.main()
