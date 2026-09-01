import importlib.util
import json
import struct
import tempfile
import unittest
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
validator = load_script("validate_ala2_benchmark.py")


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


if __name__ == "__main__":
    unittest.main()
