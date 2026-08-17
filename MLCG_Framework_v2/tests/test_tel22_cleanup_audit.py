#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tutorials" / "audit_tel22_dedup.py"
spec = importlib.util.spec_from_file_location("audit_tel22_dedup", MODULE_PATH)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Tel22CleanupAuditTests(unittest.TestCase):
    def test_duplicate_classification_is_hash_based_and_conservative(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a = root.joinpath("tutorials", "tel22")
            b = root.joinpath("tutorials", "tel22_IBI")
            (a / "mdp").mkdir(parents=True)
            (b / "mdp").mkdir(parents=True)

            # Immutable input candidate: identical by bytes.
            (a / "mdp" / "md.mdp").write_text("integrator = md\n")
            (b / "mdp" / "md.mdp").write_text("integrator = md\n")
            # Generated product: identical but should not be proposed for sharing.
            (a / "md.trr").write_bytes(b"trajectory")
            (b / "md.trr").write_bytes(b"trajectory")
            # Semantic artifact: identical today, but intentionally tutorial-local.
            (a / "cg_priors.json").write_text("{}\n")
            (b / "cg_priors.json").write_text("{}\n")
            # Same name but different content must stay separate.
            (a / "03_train_model.sh").write_text("echo legacy\n")
            (b / "03_train_model.sh").write_text("echo ibi\n")

            report = audit.build_report(root, a, b)
            classes = {x["relative_path"]: x["classification"] for x in report["duplicates"]}
            self.assertEqual(classes["mdp/md.mdp"], "SHARED_CANDIDATE")
            self.assertEqual(classes["md.trr"], "GENERATED_DUPLICATE")
            self.assertEqual(classes["cg_priors.json"], "KEEP_SEPARATE_DUPLICATE")
            self.assertEqual(classes["03_train_model.sh"], "KEEP_SEPARATE_DIFFERENT")

    def test_ibi_only_history_and_diagnostics_are_advisory_not_deleted(self):
        self.assertEqual(audit.classify_ibi_only("ibi_run/iteration_000/a.dat"), "HISTORICAL")
        self.assertEqual(
            audit.classify_ibi_only("ibi_dihedral_conservative_replica_matrix/report.json"),
            "DIAGNOSTIC",
        )
        self.assertEqual(audit.classify_ibi_only("ibi_conservative/cg_priors.json"), "KEEP")
        self.assertEqual(audit.classify_ibi_only("tel22_model_ibi.pt"), "HISTORICAL")

    def test_reference_audit_records_local_wrapper_hits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a = root.joinpath("tutorials", "tel22")
            b = root.joinpath("tutorials", "tel22_IBI")
            a.mkdir(parents=True)
            b.mkdir(parents=True)
            (a / "143D.pdb").write_text("ATOM\n")
            (b / "143D.pdb").write_text("ATOM\n")
            (a / "01_run_gromacs.sh").write_text("cp 143D.pdb work.pdb\n")
            (b / "01_run_gromacs.sh").write_text("cp 143D.pdb work.pdb\n")

            report = audit.build_report(root, a, b)
            item = next(x for x in report["duplicates"] if x["relative_path"] == "143D.pdb")
            self.assertEqual(item["classification"], "SHARED_CANDIDATE")
            files = {hit["file"] for hit in item["reference_hits"]}
            self.assertIn("tutorials/tel22/01_run_gromacs.sh", files)
            self.assertIn("tutorials/tel22_IBI/01_run_gromacs.sh", files)


if __name__ == "__main__":
    unittest.main()
