#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tutorials" / "ethanol" / "run_energy_scaling.py"
SPEC = importlib.util.spec_from_file_location("run_energy_scaling", MODULE_PATH)
SCALING = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCALING)


class EnergyScalingAnalysisTests(unittest.TestCase):
    def test_recovers_quadratic_synthetic_scaling_after_detrending(self):
        rng = np.random.default_rng(123)
        dts = np.asarray([0.004, 0.002, 0.001, 0.0005])
        base = np.sin(np.linspace(0.0, 16.0 * np.pi, 600, endpoint=False))
        stds = []
        for dt in dts:
            steps = np.arange(len(base), dtype=np.int64) * 10
            times = steps * dt
            energies = 100.0 + 0.03 * times + (dt**2) * base
            result = SCALING.analyse_series(
                steps, energies, dt, discard_fraction=0.1, bootstrap=100, rng=rng
            )
            self.assertGreaterEqual(result["block_length_samples"], 1)
            self.assertTrue(np.isfinite(result["std_ci_low"]))
            self.assertTrue(np.isfinite(result["std_ci_high"]))
            stds.append(result["detrended_std"])
        slope, r_squared = SCALING.fit_slope(dts, np.asarray(stds))
        self.assertAlmostEqual(slope, 2.0, places=8)
        self.assertAlmostEqual(r_squared, 1.0, places=12)


    def test_full_certification_uses_ci_r2_drift_and_ratios(self):
        rows = [
            {"dt": dt, "detrended_std": 3.0 * dt**2, "drift_to_std_over_run": 0.1}
            for dt in [0.004, 0.002, 0.001, 0.0005]
        ]
        thresholds = {
            "slope_target": 2.0,
            "slope_min": 1.8,
            "slope_max": 2.2,
            "slope_ci_min": 1.6,
            "slope_ci_max": 2.4,
            "max_slope_ci_width": 0.6,
            "min_r2": 0.98,
            "max_drift_to_std": 1.0,
            "ratio_relative_tolerance": 0.5,
            "min_ratio_pairs": 2,
        }
        result = SCALING.evaluate_certification(2.0, [1.9, 2.1], 0.999, rows, thresholds)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["median_normalized_ratio"], 1.0)
        self.assertAlmostEqual(result["max_normalized_ratio_deviation"], 0.0)
        self.assertTrue(result["checks"]["ratio_all_pairs"])
        self.assertEqual(len(result["ratios"]), 3)

        rows[0]["drift_to_std_over_run"] = 1.5
        failed = SCALING.evaluate_certification(2.0, [1.9, 2.1], 0.999, rows, thresholds)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["drift"])


    def test_certification_rejects_outlier_ratios_even_when_median_is_one(self):
        rows = [
            {"dt": 0.004, "detrended_std": 0.1, "drift_to_std_over_run": 0.1},
            {"dt": 0.002, "detrended_std": 0.25, "drift_to_std_over_run": 0.1},
            {"dt": 0.001, "detrended_std": 0.0625, "drift_to_std_over_run": 0.1},
            {"dt": 0.0005, "detrended_std": 0.015625, "drift_to_std_over_run": 0.1},
        ]
        thresholds = {
            "slope_target": 2.0,
            "slope_min": 1.8,
            "slope_max": 2.2,
            "slope_ci_min": 1.6,
            "slope_ci_max": 2.4,
            "max_slope_ci_width": 0.6,
            "min_r2": 0.98,
            "max_drift_to_std": 1.0,
            "ratio_relative_tolerance": 0.5,
            "min_ratio_pairs": 2,
        }
        result = SCALING.evaluate_certification(2.0, [1.9, 2.1], 0.999, rows, thresholds)
        self.assertFalse(result["checks"]["ratio_all_pairs"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
