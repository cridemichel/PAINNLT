import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_HEADER = ROOT / "training" / "PaiNN_Architecture.hpp"
PLUGIN_HEADER = ROOT / "simulation" / "espresso_plugin" / "PaiNN_Architecture.hpp"
TRAINER = ROOT / "training" / "train_painn.cpp"
RUNTIME = ROOT / "simulation" / "run_cg_md.py"
CYTHON_API = ROOT / "simulation" / "espresso_plugin" / "painn.pyx"
CONFIG = (
    ROOT / "tutorials" / ("tel" + "22") / "diagnostics" / "configs"
    / "tel22_training_config_shared_head_15ep.json"
)
RUNNER = (
    ROOT / "tutorials" / ("tel" + "22") / "diagnostics" / "scripts"
    / "28_test_tel22_shared_head.sh"
)
DOC = ROOT / "tutorials" / ("tel" + "22") / "TEL22_CGNET_TRANSFER.md"
CUTOFF_PREPARER = RUNNER.parent / "prepare_shared_head_cutoff.py"


class Tel22SharedGeometryHeadTests(unittest.TestCase):
    def test_config_is_controlled_head_only_screen(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["architecture_variant"], "tel22_shared_geometry_tanh_v1")
        self.assertEqual(config["ordered_geometry_nodes"], 82)
        self.assertEqual(config["ordered_geometry_copies"], 10)
        self.assertEqual(config["ordered_geometry_head_layers"], 5)
        self.assertEqual(config["ordered_geometry_head_width"], 160)
        self.assertEqual(config["ordered_geometry_energy_scale_kj_mol"], 2.4943387854)
        self.assertEqual(config["epochs"], 15)
        self.assertEqual(config["split_seed"], 42)
        self.assertEqual(config["torque_weight"], 0.5)
        self.assertEqual(config["spectral_projection_strength"], 4.0)

    def test_feature_contract_and_shared_energy_are_compiled(self):
        header = TRAINING_HEADER.read_text(encoding="utf-8")
        self.assertIn('"tel22_shared_geometry_tanh_v1"', header)
        self.assertIn("TEL22_SHARED_FEATURES = 131", header)
        self.assertIn("TEL22_TETRADS", header)
        self.assertIn("TEL22_TETRAD_CONTACTS", header)
        self.assertIn("TEL22_STACKING_PAIRS", header)
        self.assertIn("ordered_geometry_copies", header)
        self.assertIn("reshape({num_frames, ordered_geometry_copies, 1}).sum(1)", header)
        self.assertIn("missing a mandatory topology edge", header)
        self.assertEqual(header, PLUGIN_HEADER.read_text(encoding="utf-8"))

    def test_training_contract_checks_sites_types_and_augments_topology_edges(self):
        trainer = TRAINER.read_text(encoding="utf-8")
        self.assertIn("tel22_shared_features_for_frame", trainer)
        self.assertIn("exactly 220 ordered molecules", trainer)
        self.assertIn("residue/site-type contract mismatch", trainer)
        self.assertIn("mandatory topology-edge augmentation added", trainer)
        self.assertIn("augment_required_topology_edges_v1", trainer)
        self.assertNotIn("mandatory shared-head edge exceeds cutoff", trainer)
        self.assertIn("TEL22_SHARED_GEOMETRY_VARIANT", trainer)

    def test_runtime_api_carries_copy_contract(self):
        runtime = RUNTIME.read_text(encoding="utf-8")
        cython = CYTHON_API.read_text(encoding="utf-8")
        self.assertIn('ordered_geometry_copies=int(nn_config.get("ordered_geometry_copies", 1))', runtime)
        self.assertIn("tel22_shared_geometry", runtime)
        self.assertIn("ordered_geometry_copies: int = 1", cython)
        self.assertIn("tel22_shared_geometry: bool = False", cython)

    def test_runner_reuses_variant_a_and_stops_before_md(self):
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn("TEL22_SHARED_SOURCE_RUN_DIR", runner)
        self.assertIn("tel22_dataset.bin", runner)
        self.assertIn("validate_shared_head_training.py", runner)
        self.assertIn("prepare_shared_head_cutoff.py", runner)
        self.assertNotIn("run_cg_md.py", runner)
        self.assertNotIn("equilibrate.py", runner)
        documentation = DOC.read_text(encoding="utf-8")
        self.assertIn("training diagnostic only", documentation)
        self.assertIn("not literature acceptance criteria", documentation)

    def test_cutoff_preflight_preserves_physical_cutoff_and_reports_excursions(self):
        source = CUTOFF_PREPARER.read_text(encoding="utf-8")
        self.assertIn('"augment_required_topology_edges_v1"', source)
        self.assertIn('"mandatory_edge_observations_outside_cutoff"', source)
        self.assertIn('"structural_excursion_above_1_5_nm"', source)
        self.assertNotIn("choose_cutoff", source)


if __name__ == "__main__":
    unittest.main()
