#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "training" / "train_painn.cpp"
TRAIN_HEADER = ROOT / "training" / "PaiNN_Architecture.hpp"
PLUGIN_HEADER = ROOT / "simulation" / "espresso_plugin" / "PaiNN_Architecture.hpp"
TRAIN_CONFIG = ROOT / "training" / "cg_model_config.json"
BUILDER = ROOT / "preprocessing" / "build_cg_dataset.py"


class TrainingSafetyInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = TRAINER.read_text(encoding="utf-8")
        cls.header = TRAIN_HEADER.read_text(encoding="utf-8")
        cls.plugin_header = PLUGIN_HEADER.read_text(encoding="utf-8")
        cls.train_config = json.loads(TRAIN_CONFIG.read_text())
        cls.builder = BUILDER.read_text(encoding="utf-8")

    def test_training_and_runtime_use_identical_painn_header(self):
        self.assertEqual(self.header, self.plugin_header)

    def test_canonical_painn_context_network_and_stable_norm(self):
        self.assertIn('PAINN_ARCHITECTURE_VARIANT = "painn_canonical_context_silu_v2"', self.header)
        self.assertIn('torch::nn::Linear(dim, dim)', self.header)
        self.assertIn('torch::nn::SiLU()', self.header)
        self.assertIn('torch::nn::Linear(dim, dim * 3)', self.header)
        self.assertIn('torch::sqrt(torch::sum(v_v * v_v, 1) + epsilon)', self.header)

    def test_all_forward_paths_use_row_minus_col_displacements(self):
        self.assertNotIn('batch.coordinates.index({col}) - batch.coordinates.index({row})', self.header)
        self.assertIn('batch.coordinates.index({row}) - batch.coordinates.index({col})', self.header)

    def test_generic_safe_training_defaults(self):
        c = self.train_config
        self.assertEqual(c["architecture_variant"], "painn_canonical_context_silu_v2")
        self.assertGreaterEqual(int(c["num_species"]), 1)
        self.assertEqual(int(c.get("diagnostic_overfit_frames", -1)), 0)
        self.assertTrue(c["physical_validation_only"])
        self.assertFalse(c["include_decoys_in_train"])
        self.assertTrue(c["shuffle_each_epoch"])
        self.assertEqual(int(c["split_seed"]), 42)
        self.assertIn('std::shuffle(train_dataset.begin(), train_dataset.end(), epoch_rng)', self.trainer)

    def test_unmasked_decoys_are_opt_in_only(self):
        self.assertIn('allow_unmasked_zero_target_decoys', self.builder)
        self.assertIn('binary schema has no per-molecule loss mask', self.builder)
        self.assertIn('decoy_target_fraction > 0 requests legacy whole-frame zero-target OOD decoys', self.builder)

    def test_single_site_mapping_is_forced_to_center(self):
        self.assertIn('Single-site CG body', self.builder)
        self.assertIn('must coincide with', self.builder)

    def test_binary_reader_has_integrity_checks(self):
        for marker in (
            'Truncated/corrupt CG dataset',
            'num_total_sites mismatch',
            'site_type out of range',
            'trailing bytes after declared frames',
            'Non-finite molecule data',
        ):
            self.assertIn(marker, self.trainer)

    def test_core_tests_do_not_depend_on_tel22_tutorial(self):
        test_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "tests").glob("test_*.py")
        )
        forbidden = 'tutorials' + '" / "' + 'tel' + '22'
        self.assertNotIn(forbidden, test_sources)


if __name__ == "__main__":
    unittest.main()
