import csv
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "simulation" / "diagnose_short_range_pair.py"
SPEC = importlib.util.spec_from_file_location("short_range_diag", MODULE_PATH)
DIAG = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(DIAG)


def write_dataset(path: Path):
    # Three single-site molecules. Runtime PIDs are COM/site = 0/1, 2/3, 4/5.
    with path.open("wb") as f:
        f.write(struct.pack("i", 1))
        f.write(struct.pack("i", 3))
        f.write(struct.pack("i", 3))
        f.write(struct.pack("3f", 5.0, 5.0, 5.0))
        for mol_idx, site_type in enumerate((4, 5, 5)):
            f.write(struct.pack("i", mol_idx))
            f.write(struct.pack("i", 1))
            f.write(struct.pack("3f", float(mol_idx), 0.0, 0.0))
            f.write(struct.pack("3f", 0.0, 0.0, 0.0))
            f.write(struct.pack("3f", 0.0, 0.0, 0.0))
            f.write(struct.pack("i", site_type))
            f.write(struct.pack("3f", float(mol_idx), 0.0, 0.0))


def write_energy(path: Path, pid_pair: str, type_pair: str = "4:5", distance: float = 0.18):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Step", "Time_ps", "min_dist", "min_pair", "min_pids"])
        writer.writeheader()
        writer.writerow({"Step": 10, "Time_ps": 0.1, "min_dist": distance, "min_pair": type_pair, "min_pids": pid_pair})


class ShortRangeDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset.bin"
        self.priors = self.root / "priors.json"
        self.energy = self.root / "energy.csv"
        write_dataset(self.dataset)
        self.prior_data = {
            "wca_pairs": {
                "4_5": {
                    "type_i": 4,
                    "type_j": 5,
                    "sigma_nm": 0.30,
                    "epsilon_kjmol": 3.0,
                    "cutoff_nm": 0.3367386145,
                }
            },
            "wca_exclusions": {
                "exclude_12": True,
                "exclude_13": True,
                "scope": "molecule_pair_all_sites",
                "pair_source": "explicit_topology_pairs_v2",
                "direct_pairs": [[0, 1]],
                "one_three_pairs": [[0, 2]],
                "direct_pair_count": 1,
                "one_three_pair_count": 1,
            },
            "bonds": [
                {"mol_i": 0, "mol_j": 1, "name": "b01", "type": "harmonic", "site_i": 0, "site_j": 0, "exclude_wca": True}
            ],
            "angles": [
                {"mol_i": 0, "mol_j": 1, "mol_k": 2, "name": "a012", "type": "harmonic", "site_i": 0, "site_j": 0, "site_k": 0, "exclude_wca": True}
            ],
        }
        self.priors.write_text(json.dumps(self.prior_data))

    def tearDown(self):
        self.tmp.cleanup()

    def test_direct_pair_is_classified_as_excluded_12(self):
        write_energy(self.energy, "1:3")
        report = DIAG.build_report(dataset=self.dataset, priors_path=self.priors, energy_csv=self.energy)
        item = report["minimum_observed"]
        self.assertEqual(item["topology"]["classification"], "1-2")
        self.assertTrue(item["topology"]["wca_excluded_by_topology"])
        self.assertFalse(item["wca_runtime_expected_active"])
        self.assertGreater(item["wca_nominal_at_observed_distance"]["force_magnitude_kjmol_nm"], 0.0)

    def test_one_three_pair_is_classified_as_excluded_13(self):
        # PID 1 = mol 0 type 4; PID 5 = mol 2 type 5.
        write_energy(self.energy, "1:5")
        report = DIAG.build_report(dataset=self.dataset, priors_path=self.priors, energy_csv=self.energy)
        item = report["minimum_observed"]
        self.assertEqual(item["topology"]["classification"], "1-3")
        self.assertTrue(item["topology"]["wca_excluded_by_topology"])
        self.assertEqual(len(item["topology"]["endpoint_angles"]), 1)

    def test_nonbonded_pair_would_have_active_wca(self):
        self.prior_data["wca_exclusions"]["one_three_pairs"] = []
        self.prior_data["wca_exclusions"]["one_three_pair_count"] = 0
        self.priors.write_text(json.dumps(self.prior_data))
        write_energy(self.energy, "1:5")
        report = DIAG.build_report(dataset=self.dataset, priors_path=self.priors, energy_csv=self.energy)
        item = report["minimum_observed"]
        self.assertEqual(item["topology"]["classification"], "nonbonded")
        self.assertFalse(item["topology"]["wca_excluded_by_topology"])
        self.assertTrue(item["wca_runtime_expected_active"])


if __name__ == "__main__":
    unittest.main()
