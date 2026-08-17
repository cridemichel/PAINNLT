import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ibi"))

from compare_runtime_structure import compare_runtime_structure_reports  # noqa: E402


class MatchedRuntimeStructureComparisonTests(unittest.TestCase):
    def _write_report(self, path: Path, values: dict[str, tuple[str, float]]) -> None:
        groups = {
            key: {"kind": kind, "distribution_l1": value, "samples": 100}
            for key, (kind, value) in values.items()
        }
        by_kind = {}
        for kind, value in values.values():
            by_kind.setdefault(kind, []).append(value)
        data = {
            "pass": True,
            "mean_l1": sum(v for _k, v in values.values()) / len(values),
            "max_l1": max(v for _k, v in values.values()),
            "mean_l1_by_kind": {
                kind: sum(items) / len(items) for kind, items in by_kind.items()
            },
            "groups": groups,
        }
        path.write_text(json.dumps(data) + "\n")

    def test_paired_comparison_reports_b_minus_a(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.json"
            b = root / "b.json"
            self._write_report(a, {
                "bonds:b": ("bond", 0.20),
                "angles:a": ("angle", 0.40),
            })
            self._write_report(b, {
                "bonds:b": ("bond", 0.30),
                "angles:a": ("angle", 0.35),
            })
            report = compare_runtime_structure_reports(
                a, b, label_a="IBI-only", label_b="IBI+PaiNN"
            )
            self.assertTrue(report["pass"])
            self.assertAlmostEqual(report["delta_mean_l1_b_minus_a"], 0.025)
            self.assertAlmostEqual(report["groups"]["bonds:b"]["delta_b_minus_a"], 0.10)
            self.assertAlmostEqual(report["groups"]["angles:a"]["delta_b_minus_a"], -0.05)
            self.assertEqual(report["group_wins"]["IBI-only"], 1)
            self.assertEqual(report["group_wins"]["IBI+PaiNN"], 1)

    def test_paired_comparison_rejects_different_group_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.json"
            b = root / "b.json"
            self._write_report(a, {"bonds:b": ("bond", 0.20)})
            self._write_report(b, {"angles:a": ("angle", 0.30)})
            with self.assertRaisesRegex(ValueError, "different bonded groups"):
                compare_runtime_structure_reports(a, b)


class MatchedRuntimeSourceInvariantTests(unittest.TestCase):
    def test_run_cg_md_supports_provenance_preserving_ml_disable(self):
        source = (ROOT / "simulation" / "run_cg_md.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--disable_ml"', source)
        self.assertIn('ml_active = bool(args.model and not args.disable_ml)', source)
        self.assertIn('if ml_active:', source)
        self.assertIn('model=args.model', source)
        self.assertIn('PaiNN disabled by --disable_ml', source)




if __name__ == "__main__":
    unittest.main()
