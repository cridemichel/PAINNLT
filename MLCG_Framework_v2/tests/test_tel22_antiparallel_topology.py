#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tutorials" / "tel22" / "diagnostics" / "scripts" / "validate_antiparallel_topology.py"
SPEC = importlib.util.spec_from_file_location("validate_antiparallel_topology", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class Tel22AntiparallelTopologyTests(unittest.TestCase):
    def test_source_topologies_match_143d_and_reinfer_r0(self):
        for relative in (
            "tutorials/tel22/tel22_topology.json",
            "tutorials/tel22_IBI/tel22_topology.json",
        ):
            summary = VALIDATOR.validate_topology_file(
                ROOT / relative,
                r0_mode="auto",
                require_reference_metadata=True,
            )
            self.assertEqual(summary["morse_contacts"], 180)
            self.assertEqual(summary["tetrads_1based"], [[2, 10, 14, 22], [3, 9, 15, 21], [4, 8, 16, 20]])

    def test_legacy_parallel_register_is_rejected(self):
        source = json.loads((ROOT / "tutorials/tel22/tel22_topology.json").read_text())
        legacy_edges = []
        for group in VALIDATOR.LEGACY_TETRADS_1BASED:
            a, b, c, d = group
            legacy_edges.extend(((a, b), (b, c), (c, d), (d, a), (a, c), (b, d)))
        altered = copy.deepcopy(source)
        morse = [bond for bond in altered["bonds"] if bond["type"] == "morse"]
        for copy_index in range(10):
            offset = copy_index * 22
            for bond, (left, right) in zip(morse[copy_index * 18 : (copy_index + 1) * 18], legacy_edges):
                bond["mol_i"] = offset + left - 1
                bond["mol_j"] = offset + right - 1
        with self.assertRaisesRegex(ValueError, "legacy parallel-register Morse graph"):
            VALIDATOR.validate_topology_data(altered, r0_mode="auto")

    def test_model1_has_the_expected_hoogsteen_cycles_when_pdb_is_available(self):
        pdb = ROOT / "tutorials" / "tel22" / "143D.pdb"
        if not pdb.is_file():
            self.skipTest("143D.pdb is a downloaded tutorial input")
        summary = VALIDATOR.validate_pdb_model1(pdb)
        self.assertEqual(summary["hoogsteen_distances"], 24)
        self.assertLess(summary["max_hbond_angstrom"], 2.10)

    def test_pipeline_profile_runs_exactly_40_epochs(self):
        path = ROOT / "tutorials" / "tel22" / "diagnostics" / "configs" / "tel22_training_config_pipeline40.json"
        config = json.loads(path.read_text())
        self.assertEqual(config["epochs"], 40)
        self.assertGreater(config["early_stopping_patience"], config["epochs"])


if __name__ == "__main__":
    unittest.main()
