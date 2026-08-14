#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from multiseed_benchmark import (  # noqa: E402
    manifest_loss_matches_console,
    parse_training_log,
    summarize_records,
)


class MultiSeedBenchmarkTests(unittest.TestCase):
    def test_parse_training_log_finds_best_epoch(self):
        lines = [
            "Epoca [1/100]\n",
            "  [VAL]   Loss: 1.60 (F: 1.0, T: 1.2)\n",
            "Epoca [2/100]\n",
            "  [VAL]   Loss: 1.55 (F: 1.0, T: 1.1)\n",
            "Epoca [3/100]\n",
            "  [VAL]   Loss: 1.58 (F: 1.0, T: 1.2)\n",
        ]
        epoch, loss = parse_training_log(lines)
        self.assertEqual(epoch, 2)
        self.assertAlmostEqual(loss, 1.55)

    def test_manifest_loss_accepts_console_rounding_to_six_significant_digits(self):
        self.assertTrue(
            manifest_loss_matches_console(1.54686, 1.5468637943267822)
        )

    def test_manifest_loss_rejects_real_mismatch(self):
        self.assertFalse(
            manifest_loss_matches_console(1.54680, 1.5468637943267822)
        )

    def test_paired_summary_uses_same_seed_and_candidate_minus_reference(self):
        records = [
            {"case": "reference", "seed": 11, "best_validation_loss": 1.50},
            {"case": "reference", "seed": 42, "best_validation_loss": 1.60},
            {"case": "candidate", "seed": 11, "best_validation_loss": 1.40},
            {"case": "candidate", "seed": 42, "best_validation_loss": 1.65},
        ]
        summary = summarize_records(records, ["reference", "candidate"])
        paired = summary["paired_comparison"]
        self.assertEqual(paired["candidate_wins"], 1)
        self.assertEqual(paired["reference_wins"], 1)
        self.assertAlmostEqual(paired["pairs"][0]["candidate_minus_reference"], -0.10)
        self.assertAlmostEqual(paired["pairs"][1]["candidate_minus_reference"], 0.05)
        self.assertAlmostEqual(paired["mean_candidate_minus_reference"], -0.025)

    def test_paired_summary_rejects_unmatched_seed_sets(self):
        records = [
            {"case": "a", "seed": 1, "best_validation_loss": 1.0},
            {"case": "b", "seed": 2, "best_validation_loss": 1.0},
        ]
        with self.assertRaisesRegex(ValueError, "identical paired seed sets"):
            summarize_records(records, ["a", "b"])

    def test_duplicate_case_seed_is_rejected(self):
        records = [
            {"case": "a", "seed": 1, "best_validation_loss": 1.0},
            {"case": "a", "seed": 1, "best_validation_loss": 1.1},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate benchmark record"):
            summarize_records(records, ["a"])


if __name__ == "__main__":
    unittest.main()
