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


if __name__ == "__main__":
    unittest.main()
