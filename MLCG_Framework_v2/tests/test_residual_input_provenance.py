#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from residual_input_provenance import (  # noqa: E402
    check_manifest,
    referenced_prior_artifacts,
    write_manifest,
)


class ResidualInputProvenanceTests(unittest.TestCase):
    def _fixture(self, root: Path):
        prior_dir = root / "best"
        prior_dir.mkdir()
        table = prior_dir / "bond.dat"
        table.write_text("0.0 1.0 2.0\n1.0 0.0 0.0\n")
        priors = prior_dir / "cg_priors.json"
        priors.write_text(json.dumps({
            "bonds": [{
                "type": "tabulated", "ibi_mode": "ibi", "file": "bond.dat",
                "mol_i": 0, "mol_j": 1, "site_i": 0, "site_j": 0,
            }],
            "angles": [], "dihedrals": [],
        }) + "\n")
        validation = root / "validation_report.json"
        validation.write_text(json.dumps({
            "schema_version": 1,
            "mode": "read_only_validation",
            "priors": str(priors.resolve()),
            "mean_l1": 0.22,
            "max_l1": 0.35,
            "source_artifact_sha256": referenced_prior_artifacts(priors),
            "source_priors_unchanged": True,
        }) + "\n")
        files = {}
        for name in ("dataset.bin", "rb.json", "aa.gro", "aa.trr", "mapping.json"):
            path = root / name
            path.write_bytes((name + "\n").encode())
            files[name] = path
        return priors, table, validation, files

    def _conservative_fixture(self, root: Path):
        source_dir = root / "best"
        source_dir.mkdir()
        source_table = source_dir / "bond_tabulated_b.dat"
        source_table.write_text("0.0 1.0 -2.0\n1.0 0.0 0.0\n")
        source_priors = source_dir / "cg_priors.json"
        source_priors.write_text(json.dumps({
            "bonds": [{"type": "tabulated", "file": source_table.name}],
            "angles": [], "dihedrals": [],
        }) + "\n")

        outdir = root / "ibi_conservative"
        outdir.mkdir()
        spline = outdir / "bond_conservative_b.dat"
        spline.write_text("0.0 1.0 -2.0\n1.0 0.0 0.0\n")
        priors = outdir / "cg_priors.json"
        priors.write_text(json.dumps({
            "bonds": [{"type": "conservative_spline", "file": spline.name}],
            "angles": [], "dihedrals": [],
        }) + "\n")
        conversion = outdir / "conversion_report.json"
        conversion.write_text(json.dumps({
            "schema_version": 1,
            "framework": "MLCG_Framework_v2",
            "kind": "ibi_conservative_spline_conversion",
            "source_priors": str(source_priors.resolve()),
            "source_priors_sha256": hashlib.sha256(source_priors.read_bytes()).hexdigest(),
            "output_priors": str(priors.resolve()),
            "output_priors_sha256": hashlib.sha256(priors.read_bytes()).hexdigest(),
            "records": [{
                "source_path": str(source_table.resolve()),
                "source_sha256": hashlib.sha256(source_table.read_bytes()).hexdigest(),
                "output_file": spline.name,
                "output_path": str(spline.resolve()),
                "output_sha256": hashlib.sha256(spline.read_bytes()).hexdigest(),
            }],
            "source_artifacts_unchanged": True,
        }) + "\n")
        validation = outdir / "validation_report.json"
        validation.write_text(json.dumps({
            "schema_version": 1,
            "framework": "MLCG_Framework_v2",
            "kind": "ibi_conservative_spline_validation",
            "conversion_report": str(conversion.resolve()),
            "conservative_priors": str(priors.resolve()),
            "conservative_priors_sha256": hashlib.sha256(priors.read_bytes()).hexdigest(),
            "finite_difference_checks": [{"max_abs_dU_dq_error": 1.0e-7}],
            "pass": True,
        }) + "\n")
        parity = outdir / "runtime_parity_report.json"
        parity.write_text(json.dumps({
            "schema_version": 1,
            "framework": "MLCG_Framework_v2",
            "kind": "ibi_conservative_spline_runtime_parity",
            "priors": str(priors.resolve()),
            "priors_sha256": hashlib.sha256(priors.read_bytes()).hexdigest(),
            "prior_artifact_sha256": referenced_prior_artifacts(priors),
            "force_atol": 1.0e-9,
            "energy_atol": 1.0e-10,
            "worst_force_abs_error": 2.0e-14,
            "worst_energy_abs_error": 3.0e-13,
            "pass": True,
        }) + "\n")
        files = {}
        for name in ("dataset.bin", "rb.json", "aa.gro", "aa.trr", "mapping.json"):
            path = root / name
            path.write_bytes((name + "\n").encode())
            files[name] = path
        return priors, spline, validation, parity, files

    def test_record_and_check_bind_exact_residual_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            priors, _table, validation, f = self._fixture(root)
            manifest = root / "residual_manifest.json"
            write_manifest(
                output=manifest,
                dataset=f["dataset.bin"], rb_info=f["rb.json"], priors=priors,
                aa_topology=f["aa.gro"], aa_trajectory=f["aa.trr"],
                mapping_config=f["mapping.json"], validation_report=validation,
            )
            checked = check_manifest(
                manifest_path=manifest, dataset=f["dataset.bin"],
                rb_info=f["rb.json"], priors=priors,
            )
            self.assertEqual(checked["kind"], "residual_training_inputs")

    def test_check_fails_if_dataset_changes_after_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            priors, _table, validation, f = self._fixture(root)
            manifest = root / "residual_manifest.json"
            write_manifest(
                output=manifest,
                dataset=f["dataset.bin"], rb_info=f["rb.json"], priors=priors,
                aa_topology=f["aa.gro"], aa_trajectory=f["aa.trr"],
                mapping_config=f["mapping.json"], validation_report=validation,
            )
            f["dataset.bin"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "dataset (size|SHA256) mismatch"):
                check_manifest(
                    manifest_path=manifest, dataset=f["dataset.bin"],
                    rb_info=f["rb.json"], priors=priors,
                )

    def test_check_fails_if_validated_table_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            priors, table, validation, f = self._fixture(root)
            manifest = root / "residual_manifest.json"
            write_manifest(
                output=manifest,
                dataset=f["dataset.bin"], rb_info=f["rb.json"], priors=priors,
                aa_topology=f["aa.gro"], aa_trajectory=f["aa.trr"],
                mapping_config=f["mapping.json"], validation_report=validation,
            )
            table.write_text("mutated\n")
            with self.assertRaisesRegex(ValueError, "Prior table provenance mismatch"):
                check_manifest(
                    manifest_path=manifest, dataset=f["dataset.bin"],
                    rb_info=f["rb.json"], priors=priors,
                )

    def test_conservative_record_and_check_bind_validation_and_runtime_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            priors, _spline, validation, parity, f = self._conservative_fixture(root)
            manifest = root / "residual_manifest.json"
            written = write_manifest(
                output=manifest,
                dataset=f["dataset.bin"], rb_info=f["rb.json"], priors=priors,
                aa_topology=f["aa.gro"], aa_trajectory=f["aa.trr"],
                mapping_config=f["mapping.json"], validation_report=validation,
                runtime_parity_report=parity,
            )
            self.assertEqual(
                written["build_inputs"]["ibi_validation_mode"],
                "conservative_spline_validation",
            )
            checked = check_manifest(
                manifest_path=manifest, dataset=f["dataset.bin"],
                rb_info=f["rb.json"], priors=priors,
            )
            self.assertEqual(checked["kind"], "residual_training_inputs")

    def test_conservative_record_requires_persisted_runtime_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            priors, _spline, validation, _parity, f = self._conservative_fixture(root)
            with self.assertRaisesRegex(ValueError, "persisted ESPResSo/runtime parity"):
                write_manifest(
                    output=root / "manifest.json",
                    dataset=f["dataset.bin"], rb_info=f["rb.json"], priors=priors,
                    aa_topology=f["aa.gro"], aa_trajectory=f["aa.trr"],
                    mapping_config=f["mapping.json"], validation_report=validation,
                )




if __name__ == "__main__":
    unittest.main()
