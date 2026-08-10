#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "training" / "train_painn.cpp"
TRAIN_HEADER = ROOT / "training" / "PaiNN_Architecture.hpp"
PLUGIN_HEADER = ROOT / "simulation" / "espresso_plugin" / "PaiNN_Architecture.hpp"
TOPOLOGY = ROOT / "tutorials" / "tel22" / "tel22_topology.json"
TRAIN_CONFIG = ROOT / "tutorials" / "tel22" / "tel22_training_config.json"
BUILDER = ROOT / "preprocessing" / "build_cg_dataset.py"


class TrainingSafetyInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = TRAINER.read_text(encoding="utf-8")
        cls.header = TRAIN_HEADER.read_text(encoding="utf-8")
        cls.plugin_header = PLUGIN_HEADER.read_text(encoding="utf-8")
        cls.topology = json.loads(TOPOLOGY.read_text())
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

    def test_legacy_unmasked_decoys_are_disabled_for_tel22(self):
        self.assertEqual(float(self.topology.get("decoy_target_fraction", -1)), 0.0)
        self.assertFalse(bool(self.topology.get("allow_unmasked_zero_target_decoys", True)))
        self.assertFalse(bool(self.train_config.get("include_decoys_in_train", True)))
        self.assertIn('allow_unmasked_zero_target_decoys', self.builder)
        self.assertIn('binary schema has no per-molecule loss mask', self.builder)

    def test_tel22_one_site_beads_are_mapped_to_residue_com(self):
        residues = self.topology["mapping"]["residues"]
        self.assertEqual(residues["DA"]["CG_DA"], ["*"])
        self.assertEqual(residues["DT"]["CG_DT"], ["*"])
        self.assertIn('Single-site CG body', self.builder)

    def test_production_split_is_physical_only_and_epoch_shuffled(self):
        self.assertTrue(self.train_config["physical_validation_only"])
        self.assertTrue(self.train_config["shuffle_each_epoch"])
        self.assertEqual(int(self.train_config["split_seed"]), 42)
        self.assertIn('std::shuffle(train_dataset.begin(), train_dataset.end(), epoch_rng)', self.trainer)
        self.assertIn('Decoys excluded from optimization', self.trainer)

    def test_binary_reader_has_integrity_checks(self):
        for marker in (
            'Truncated/corrupt CG dataset',
            'num_total_sites mismatch',
            'site_type out of range',
            'trailing bytes after declared frames',
            'Non-finite molecule data',
        ):
            self.assertIn(marker, self.trainer)


if __name__ == "__main__":
    unittest.main()
