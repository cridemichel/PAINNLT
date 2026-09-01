#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tutorials"
    / "tel22"
    / "diagnostics"
    / "scripts"
    / "prepare_variant_a_topology.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_variant_a_topology", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sys.path.insert(0, str(MODULE_PATH.parent))
VARIANT_A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VARIANT_A)


class Tel22VariantATopologyTests(unittest.TestCase):
    def test_variant_a_removes_only_the_180_morse_contacts(self):
        source_path = ROOT / "tutorials" / "tel22" / "tel22_topology.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        original_bonds = list(source["bonds"])

        variant, removed = VARIANT_A.build_variant_a_data(source)

        self.assertEqual(removed, 180)
        self.assertEqual(source["bonds"], original_bonds)
        summary = VARIANT_A.validate_variant_a_topology_data(variant)
        self.assertEqual(summary["morse_contacts"], 0)
        self.assertEqual(summary["harmonic_bonds"], 210)
        self.assertEqual(summary["harmonic_angles"], 200)

    def test_variant_a_validator_rejects_a_reintroduced_morse(self):
        source = json.loads(
            (ROOT / "tutorials" / "tel22" / "tel22_topology.json").read_text(encoding="utf-8")
        )
        variant, _ = VARIANT_A.build_variant_a_data(source)
        variant["bonds"].append(
            {"mol_i": 1, "mol_j": 9, "type": "morse", "D": 50.0, "a": 0.3, "r0": 1.0}
        )
        with self.assertRaisesRegex(ValueError, "zero Morse bonds"):
            VARIANT_A.validate_variant_a_topology_data(variant)

    def test_training_profile_changes_only_run_length(self):
        current = json.loads(
            (
                ROOT
                / "tutorials"
                / "tel22"
                / "diagnostics"
                / "configs"
                / "tel22_training_config_pipeline40.json"
            ).read_text(encoding="utf-8")
        )
        variant = json.loads(
            (
                ROOT
                / "tutorials"
                / "tel22"
                / "diagnostics"
                / "configs"
                / "tel22_training_config_variant_a_15ep.json"
            ).read_text(encoding="utf-8")
        )
        differing = {key for key in current if current[key] != variant.get(key)}
        self.assertEqual(differing, {"epochs", "early_stopping_patience"})
        self.assertEqual(variant["epochs"], 15)
        self.assertGreater(variant["early_stopping_patience"], variant["epochs"])

    def test_regularized_profile_reduces_capacity_and_enables_early_stopping(self):
        baseline = json.loads(
            (
                ROOT
                / "tutorials"
                / "tel22"
                / "diagnostics"
                / "configs"
                / "tel22_training_config_variant_a_15ep.json"
            ).read_text(encoding="utf-8")
        )
        regularized = json.loads(
            (
                ROOT
                / "tutorials"
                / "tel22"
                / "diagnostics"
                / "configs"
                / "tel22_training_config_variant_ar_30ep.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(regularized["hidden_channels"], 32)
        self.assertEqual(regularized["n_layers"], baseline["n_layers"])
        self.assertEqual(regularized["weight_decay"], 1.0e-4)
        self.assertEqual(regularized["epochs"], 30)
        self.assertEqual(regularized["early_stopping_patience"], 8)
        self.assertEqual(regularized["reduce_lr_patience"], 4)
        for key in (
            "num_species",
            "num_rbf",
            "cutoff",
            "learning_rate",
            "batch_size",
            "torque_weight",
            "split_seed",
            "validation_fraction",
            "architecture_variant",
        ):
            self.assertEqual(regularized[key], baseline[key], key)


if __name__ == "__main__":
    unittest.main()
