#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT.joinpath("tutorials", "archive_tel22_ibi_history.py")
spec = importlib.util.spec_from_file_location("archive_tel22_ibi_history", MODULE_PATH)
assert spec is not None and spec.loader is not None
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class Tel22HistoricalArchiveTests(unittest.TestCase):
    def _make_tree(self, root: Path) -> Path:
        tutorial = root.joinpath("tutorials", "tel22_IBI")
        tutorial.mkdir(parents=True)
        for name in ("ibi_dbi_preview", "ibi_run", "training_multiseed_benchmark", "ibi_ml_ab_validation"):
            path = tutorial / name
            path.mkdir()
            (path / "evidence.txt").write_text(f"{name}\n", encoding="utf-8")
        # Live historical dependencies must stay where configured workflows expect them.
        for name in ("ibi_run_16ps", "ibi_run_16ps_continue", "ibi_validation_best", "postibi_runtime_validation"):
            path = tutorial / name
            path.mkdir()
            (path / "keep.txt").write_text("live\n", encoding="utf-8")
        for name in ("tel22_dataset_ibi_residual.bin", "tel22_model_ibi.pt", "ibi_residual_build_manifest.json"):
            (tutorial / name).write_bytes(b"live-historical")
        # Representative GROMACS products specifically requested to remain in place.
        for name in ("md.trr", "md_whole.trr", "md.gro", "md.tpr", "nvt.gro", "npt.gro", "em.gro", "topol.top"):
            (tutorial / name).write_bytes(("gromacs:" + name).encode("ascii"))
        return tutorial

    def test_dry_run_moves_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tutorial = self._make_tree(Path(tmpdir))
            manifest = archive.execute(tutorial, run=False)
            self.assertFalse(manifest["executed"])
            self.assertTrue((tutorial / "ibi_run").exists())
            self.assertTrue((tutorial / "md.trr").exists())
            self.assertFalse((tutorial / "diagnostics" / "historical" / "phase3_archive" / "historical_ibi" / "ibi_run").exists())

    def test_run_archives_only_reviewed_terminal_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tutorial = self._make_tree(Path(tmpdir))
            archive.execute(tutorial, run=True)
            self.assertFalse((tutorial / "ibi_run").exists())
            self.assertFalse((tutorial / "ibi_dbi_preview").exists())
            self.assertTrue((tutorial / "diagnostics" / "historical" / "phase3_archive" / "historical_ibi" / "ibi_run" / "evidence.txt").exists())
            self.assertTrue(
                (tutorial / "diagnostics" / "historical" / "phase3_archive" / "ml_residual_experiments" / "ibi_ml_ab_validation" / "evidence.txt").exists()
            )
            # Live historical dependencies are intentionally untouched.
            self.assertTrue((tutorial / "ibi_run_16ps").exists())
            self.assertTrue((tutorial / "postibi_runtime_validation").exists())
            self.assertTrue((tutorial / "tel22_model_ibi.pt").exists())

    def test_gromacs_generated_products_are_preserved_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tutorial = self._make_tree(Path(tmpdir))
            names = ("md.trr", "md_whole.trr", "md.gro", "md.tpr", "nvt.gro", "npt.gro", "em.gro", "topol.top")
            before = {name: (tutorial / name).read_bytes() for name in names}
            archive.execute(tutorial, run=True)
            after = {name: (tutorial / name).read_bytes() for name in names}
            self.assertEqual(before, after)
            for name in names:
                self.assertTrue((tutorial / name).exists())

    def test_collision_fails_before_any_move(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tutorial = self._make_tree(Path(tmpdir))
            collision = tutorial / "diagnostics" / "historical" / "phase3_archive" / "historical_ibi" / "ibi_run"
            collision.mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                archive.execute(tutorial, run=True)
            # Fail-closed: another planned source was not moved before the collision was detected.
            self.assertTrue((tutorial / "ibi_dbi_preview").exists())

    def test_executed_manifest_records_preservation_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tutorial = self._make_tree(Path(tmpdir))
            archive.execute(tutorial, run=True)
            data = json.loads((tutorial / "diagnostics" / "historical" / "phase3_archive" / "archive_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(data["executed"])
            self.assertTrue(data["policy"]["gromacs_generated_preserved"])
            self.assertIn("md.trr", data["policy"]["protected_gromacs_toplevel"])
            self.assertIn("diagnostics/ml/postibi_runtime_validation", data["policy"]["live_historical_dependencies_preserved"])


if __name__ == "__main__":
    unittest.main()
