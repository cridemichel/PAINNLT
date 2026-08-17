#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "ibi"))
sys.path.insert(0, str(ROOT / "simulation"))

from conservative_spline import load_conservative_spline  # noqa: E402
from ibi_core import conservative_dihedral_from_potential  # noqa: E402
from run_ibi_loop import _assert_conservative_dihedral_loop, _write_iteration_priors  # noqa: E402
from finalize_conservative_dihedral_ibi_loop_test import build_report  # noqa: E402


class ConservativeDihedralIbiLoopTests(unittest.TestCase):
    def test_periodic_energy_primary_profile_returns_periodic_derivative(self):
        phi = np.linspace(0.0, 2.0 * np.pi, 257)
        potential = 1.7 * (1.0 - np.cos(phi)) + 0.15 * (1.0 - np.cos(3.0 * phi))
        energy, derivative = conservative_dihedral_from_potential(phi, potential)
        self.assertAlmostEqual(energy[0], energy[-1], places=13)
        self.assertAlmostEqual(derivative[0], derivative[-1], places=13)
        expected = 1.7 * np.sin(phi) + 0.45 * np.sin(3.0 * phi)
        self.assertLess(float(np.max(np.abs(derivative - expected))), 2.0e-5)

    def test_iteration_writer_emits_conservative_dihedral_and_copies_fixed_background(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            source_dir.mkdir()
            fixed = source_dir / "bond_conservative_fixed.dat"
            xb = np.linspace(0.5, 1.5, 9)
            np.savetxt(fixed, np.column_stack((xb, (xb - 1.0) ** 2, 2.0 * (xb - 1.0))), fmt="%.17g")
            source_priors = source_dir / "cg_priors.json"
            template = {
                "bonds": [{
                    "type": "conservative_spline", "file": fixed.name,
                    "min": 0.5, "max": 1.5, "spline_schema": "pchip_hermite_v1",
                }],
                "angles": [],
                "dihedrals": [{
                    "type": "ibi", "ibi_mode": "ibi", "name": "dih_test",
                    "mol_i": 0, "mol_j": 1, "mol_k": 2, "mol_l": 3,
                }],
            }
            source_priors.write_text(json.dumps(template) + "\n")
            phi = np.linspace(0.0, 2.0 * np.pi, 129)
            groups = {"bonds": {}, "angles": {}, "dihedrals": {
                "dih_test": {
                    "kind": "dihedral", "mode": "ibi", "indices": [0],
                    "grid": phi, "energy": 1.0 - np.cos(phi), "force": np.sin(phi),
                    "representation": "conservative_spline",
                }
            }}
            out_priors, out_path = _write_iteration_priors(
                template, groups, root / "iteration_000", source_priors_path=source_priors
            )
            entry = out_priors["dihedrals"][0]
            self.assertEqual(entry["type"], "conservative_spline")
            self.assertEqual(entry["ibi_runtime_representation"], "conservative_spline")
            spline = load_conservative_spline(entry, kind="dihedral", priors_path=out_path)
            self.assertTrue(np.allclose(spline.derivative, np.sin(phi), atol=1.0e-14))
            fixed_copy = out_path.parent / out_priors["bonds"][0]["file"]
            self.assertTrue(fixed_copy.is_file())
            self.assertEqual(fixed_copy.read_bytes(), fixed.read_bytes())
            _assert_conservative_dihedral_loop(out_priors, groups, out_path)

    def test_fail_closed_rejects_legacy_dihedral_in_conservative_loop(self):
        priors = {"dihedrals": [{"type": "tabulated", "ibi_mode": "ibi"}]}
        groups = {"dihedrals": {"x": {
            "kind": "dihedral", "mode": "ibi", "indices": [0], "representation": "tabulated"
        }}}
        with self.assertRaisesRegex(RuntimeError, "state 'x'"):
            _assert_conservative_dihedral_loop(priors, groups, Path("legacy.json"))

    def test_finalizer_tracks_l1_sequence_without_promoting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            phi = np.linspace(0.0, 2.0 * np.pi, 33)
            table = root / "dih.dat"
            np.savetxt(table, np.column_stack((phi, 1.0 - np.cos(phi), np.sin(phi))), fmt="%.17g")
            priors = root / "priors.json"
            priors.write_text(json.dumps({
                "bonds": [], "angles": [],
                "dihedrals": [{
                    "type": "conservative_spline", "ibi_mode": "ibi",
                    "ibi_runtime_representation": "conservative_spline",
                    "file": table.name, "min": 0.0, "max": float(2.0 * np.pi),
                    "spline_schema": "pchip_hermite_v1",
                }],
            }) + "\n")
            ibi_report = root / "ibi.json"
            ibi_report.write_text(json.dumps({
                "conservative_dihedrals_in_loop": True,
                "dihedral_runtime_representation": "conservative_spline",
                "ibi_groups": 1, "dbi_groups": 0, "iterations_completed": 2,
                "metrics": [
                    {"iteration": 1, "source_priors": "i0.json", "groups": {
                        "dihedrals:x": {"kind": "dihedral", "mode": "ibi", "distribution_l1": 1.4, "runtime_representation": "conservative_spline"}
                    }},
                    {"iteration": 2, "source_priors": "i1.json", "groups": {
                        "dihedrals:x": {"kind": "dihedral", "mode": "ibi", "distribution_l1": 1.1, "runtime_representation": "conservative_spline"}
                    }},
                ],
            }) + "\n")
            parity = root / "parity.json"
            parity.write_text(json.dumps({
                "pass": True, "worst_force_abs_error": 1.0e-13,
                "worst_energy_abs_error": 2.0e-13,
            }) + "\n")
            structure = root / "structure.json"
            structure.write_text(json.dumps({"mean_l1_by_kind": {"dihedral": 0.9}}) + "\n")
            sampling = root / "sampling.json"
            sampling.write_text(json.dumps({
                "kind": "matched_final_ibi_sampling_protocol",
                "matched_to_ibi_loop": True,
                "source_priors": str(priors.resolve()),
                "starting_state": "target_dataset_initial_frame_plus_initialized_velocities",
                "sampled_iteration": 3,
                "dt_ps": 0.0005,
                "burn_in_steps": 500,
                "production_steps": 2000,
                "sample_interval": 10,
                "kT": 2.49,
                "neighbor_search": "link-cell",
                "velocity_seed": 12,
                "thermostat_seed": 22,
                "checkpoint_used": False,
                "ml_active": False,
            }) + "\n")
            ibi_data = json.loads(ibi_report.read_text())
            ibi_data.update({
                "dt_ps": 0.0005, "burn_in_steps": 500, "production_steps": 2000,
                "sample_interval": 10, "kT": 2.49, "neighbor_search": "link-cell",
                "velocity_seed": 10, "thermostat_seed": 20,
            })
            ibi_report.write_text(json.dumps(ibi_data) + "\n")
            out = root / "report.json"
            report = build_report(
                ibi_report_path=ibi_report, final_priors=priors, parity_report_path=parity,
                structure_report_path=structure, final_sampling_report_path=sampling, output=out,
                direction_flat_tolerance_l1=0.02,
            )
            self.assertEqual(report["l1_sequence"], [1.4, 1.1, 0.9])
            self.assertEqual(report["convergence_direction"], "improving")
            self.assertTrue(report["infrastructure_pass"])
            self.assertTrue(report["final_sampling_protocol"]["pass"])
            self.assertFalse(report["promotion_ready"])

    def test_finalizer_rejects_mismatched_final_sampling_protocol(self):
        from finalize_conservative_dihedral_ibi_loop_test import _validate_final_sampling_protocol

        ibi = {
            "metrics": [{"iteration": 1}, {"iteration": 2}],
            "dt_ps": 0.0005, "burn_in_steps": 500, "production_steps": 2000,
            "sample_interval": 10, "kT": 2.49, "neighbor_search": "link-cell",
            "velocity_seed": 380001, "thermostat_seed": 380101,
        }
        final_priors = Path("/tmp/final.json")
        sampling = {
            "kind": "matched_final_ibi_sampling_protocol",
            "matched_to_ibi_loop": True,
            "source_priors": str(final_priors),
            "starting_state": "target_dataset_initial_frame_plus_initialized_velocities",
            "sampled_iteration": 3,
            "dt_ps": 0.0005, "burn_in_steps": 200, "production_steps": 800,
            "sample_interval": 10, "kT": 2.49, "neighbor_search": "link-cell",
            "velocity_seed": 380003, "thermostat_seed": 380103,
            "checkpoint_used": False, "ml_active": False,
        }
        check = _validate_final_sampling_protocol(ibi, sampling, final_priors=final_priors)
        self.assertFalse(check["pass"])
        self.assertFalse(check["checks"]["burn_in_steps"])
        self.assertFalse(check["checks"]["production_steps"])


if __name__ == "__main__":
    unittest.main()
