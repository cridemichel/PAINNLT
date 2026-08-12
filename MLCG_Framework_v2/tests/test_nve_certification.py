#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIMULATION = ROOT / "simulation"
sys.path.insert(0, str(SIMULATION))

from nve_analysis import analyze_energy_series, certify_metrics, fit_timestep_scaling  # noqa: E402


class NVECertificationTests(unittest.TestCase):
    def test_exact_quadratic_synthetic_scaling_passes(self):
        runs = []
        for dt in (0.002, 0.001, 0.0005, 0.00025):
            t = np.linspace(0.0, 5.0, 501)
            # Bounded error with amplitude exactly proportional to dt^2 and no
            # secular component.
            energy = -1000.0 + 2.0e7 * dt**2 * np.sin(2.0 * np.pi * t)
            metrics = analyze_energy_series(t, energy)
            metrics["dt_ps"] = dt
            runs.append(metrics)
        scaling = fit_timestep_scaling(runs)
        self.assertAlmostEqual(scaling["exponent_p"], 2.0, places=10)
        self.assertGreater(scaling["loglog_r2"], 0.999999999)
        result = certify_metrics(
            runs,
            slope_min=1.7,
            slope_max=2.3,
            min_r2=0.97,
            max_relative_drift=1e-4,
        )
        self.assertTrue(result["pass"])

    def test_linear_secular_drift_is_detected(self):
        t = np.linspace(0.0, 5.0, 501)
        energy = -1000.0 + 0.2 * np.sin(2.0 * np.pi * t) + 1.0 * t
        metrics = analyze_energy_series(t, energy)
        self.assertGreater(metrics["relative_block_mean_drift"], 1e-4)

    def test_runtime_source_has_explicit_nve_invariants(self):
        source = (SIMULATION / "run_cg_md.py").read_text(encoding="utf-8")
        self.assertIn("system.force_cap = 0.0", source)
        self.assertIn("system.integrator.set_vv()", source)
        self.assertIn("system.thermostat.turn_off()", source)
        self.assertIn("while simulation_ok and completed < args.steps", source)
        self.assertNotIn("while completed <= args.steps", source)
        self.assertIn("system.integrator.run(0, recalc_forces=True)", source)
        self.assertIn('"Time_ps"', source)

    def test_dummy_neighbor_interactions_exclude_com_types(self):
        for name in ("run_cg_md.py", "equilibrate.py"):
            source = (SIMULATION / name).read_text(encoding="utf-8")
            self.assertIn('for i in range(nn_config["num_species"]):', source)
            self.assertIn('for j in range(i, nn_config["num_species"]):', source)
            self.assertNotIn('range(nn_config["num_species"] + 2)', source)


if __name__ == "__main__":
    unittest.main()
