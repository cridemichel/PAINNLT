import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "simulation" / "nve_analysis.py"
SPEC = importlib.util.spec_from_file_location("nve_analysis_sigma_test", MODULE_PATH)
NVE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(NVE)


class TestNVESigmaScaling(unittest.TestCase):
    def sigma(self, energies):
        if hasattr(NVE, "energy_standard_deviation"):
            return NVE.energy_standard_deviation(energies)
        times = np.arange(len(energies), dtype=float)
        return NVE.analyze_energy_series(times, energies)["sigma_E"]

    def test_sigma_energy_is_population_std(self):
        energies = np.array([10.0, 11.0, 9.0, 10.5, 9.5], dtype=float)
        self.assertAlmostEqual(
            self.sigma(energies),
            float(np.std(energies, ddof=0)),
            places=14,
        )

    def test_sigma_scales_as_dt_squared(self):
        dts = np.array([0.001, 0.002, 0.005, 0.01], dtype=float)
        phase = np.linspace(0.0, 20.0 * math.pi, 20001)
        carrier = np.sin(phase) + 0.25 * np.cos(0.37 * phase)
        sigmas = []
        for dt in dts:
            energies = 2000.0 + 1.0e6 * dt * dt * carrier
            sigmas.append(self.sigma(energies))
        p = float(np.polyfit(np.log(dts), np.log(sigmas), 1)[0])
        self.assertAlmostEqual(p, 2.0, places=8)

    def test_certifier_overrides_sampling_every_step(self):
        source = (ROOT / "simulation" / "certify_nve.py").read_text(encoding="utf-8")
        self.assertIn(
            "log_every = 1  # NVE certification: sample energy every integration step",
            source,
        )
        self.assertIn("sigma_E", source)


if __name__ == "__main__":
    unittest.main()
