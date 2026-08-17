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

from convert_to_conservative_spline import convert  # noqa: E402
from prepare_dihedral_ibi_test_seed import prepare  # noqa: E402
from validate_conservative_spline import _five_point_fd_derivative, validate  # noqa: E402
from conservative_spline import ConservativeSplinePrior, conservative_spline_value  # noqa: E402
from finalize_dihedral_ibi_test import nve_metrics  # noqa: E402


class DihedralIbiTestWorkflowTests(unittest.TestCase):
    def test_seed_derives_consecutive_periodic_backbone_torsions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.json"
            angles = []
            names = ["ang_A_G_G", "ang_G_G_G", "ang_G_G_T"]
            for i, name in enumerate(names):
                angles.append({
                    "type": "harmonic", "name": name,
                    "mol_i": i, "mol_j": i + 1, "mol_k": i + 2,
                    "site_i": 0, "site_j": 0, "site_k": 0,
                    "k": 1.0, "theta0": 1.0, "ibi_mode": "ibi",
                })
            base.write_text(json.dumps({"bonds": [], "angles": angles, "dihedrals": []}) + "\n")
            output = root / "seed.json"
            report = prepare(base, output, grouping_strategy="consecutive_angle_types")
            seed = json.loads(output.read_text())
            self.assertEqual(report["dihedral_occurrences"], 2)
            self.assertEqual(report["unique_groups"], 2)
            self.assertEqual(
                [d["name"] for d in seed["dihedrals"]],
                ["dih_A_G_G_G", "dih_G_G_G_T"],
            )
            self.assertTrue(all(d["type"] == "ibi" and d["test_only"] for d in seed["dihedrals"]))
            self.assertTrue(all("ibi_mode" not in a for a in seed["angles"]))
            self.assertEqual(report["frozen_inherited_bond_angle_entries"], 3)

    def test_five_point_fd_handles_high_third_derivative_hermite_segment(self):
        h = 0.01
        x = np.linspace(0.0, 0.1, 11)
        q0 = 0.371 * h
        cubic = 2.0e7
        slope = 7.0
        u = cubic * (x - q0) ** 3 + slope * (x - q0)
        du = 3.0 * cubic * (x - q0) ** 2 + slope
        table = ConservativeSplinePrior(
            x=x, energy=u, derivative=du, minimum=0.0, maximum=0.1,
            kind="bond", path=Path("rough_cubic.dat"),
        )
        eps = 1.0e-6
        analytic = conservative_spline_value(table, q0)[1]
        up = conservative_spline_value(table, q0 + eps)[0]
        um = conservative_spline_value(table, q0 - eps)[0]
        three_point = (up - um) / (2.0 * eps)
        five_point = _five_point_fd_derivative(table, q0, eps)
        self.assertGreater(abs(three_point - analytic), 1.5e-5)
        self.assertLess(abs(five_point - analytic), 1.0e-8)

    def test_mixed_conversion_copies_existing_conservative_tables_byte_identically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xb = np.linspace(0.4, 2.0, 81)
            ub = 2.0 * (xb - 1.1) ** 2
            dub = 4.0 * (xb - 1.1)
            base_table = root / "bond_conservative_existing.dat"
            np.savetxt(base_table, np.column_stack([xb, ub, dub]), fmt="%.17g")
            original_bytes = base_table.read_bytes()

            xd = np.linspace(0.0, 2.0 * np.pi, 129)
            ud = 1.7 * (1.0 - np.cos(xd))
            force_factor = -1.7 * np.ones_like(xd)
            tab = root / "dihedral_tabulated_test.dat"
            np.savetxt(tab, np.column_stack([xd, ud, force_factor]), fmt="%.17g")

            priors = root / "priors.json"
            priors.write_text(json.dumps({
                "bonds": [{
                    "type": "conservative_spline", "file": base_table.name,
                    "min": float(xb[0]), "max": float(xb[-1]),
                    "spline_schema": "pchip_hermite_v1", "ibi_mode": "ibi",
                }],
                "angles": [],
                "dihedrals": [{
                    "type": "tabulated", "file": tab.name,
                    "min": 0.0, "max": float(2.0 * np.pi), "ibi_mode": "ibi",
                }],
            }) + "\n")

            out = root / "out"
            report = convert(priors, out)
            self.assertEqual(report["converted_unique_tables"], 1)
            self.assertEqual(report["passthrough_unique_tables"], 1)
            copied = out / base_table.name
            self.assertEqual(copied.read_bytes(), original_bytes)
            self.assertTrue(validate(out / "conversion_report.json")["pass"])

    def test_nve_test_gate_keeps_irregular_candidate_as_diagnostic_result(self):
        report = {
            "strict_reference": {
                "scaling": {"exponent_p": 1.4, "loglog_r2": 0.98},
            },
            "runs": [
                {"dt_ps": 0.001, "sigma_E": 1.0e-3, "relative_block_mean_drift": 1.0e-6},
                {"dt_ps": 0.002, "sigma_E": 2.5e-3, "relative_block_mean_drift": 1.0e-6},
                {"dt_ps": 0.005, "sigma_E": 8.0e-3, "relative_block_mean_drift": 1.0e-6},
            ],
        }
        result = nve_metrics(
            report, p_min=1.8, p_max=2.2, r2_min=0.95, c2_spread_max=2.0,
            max_relative_drift=2.0e-5, required_max_dt=0.005,
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["quadratic_exponent"])
        self.assertTrue(result["checks"]["full_dt_reached"])


if __name__ == "__main__":
    unittest.main()
