#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ibi"))
sys.path.insert(0, str(ROOT / "simulation"))

from generate_dihedral_update_localization_candidates import (  # noqa: E402
    CandidateSpec,
    _candidate_energy,
    _smooth_periodic_update,
)
from finalize_dihedral_update_localization import finalize  # noqa: E402


class DihedralUpdateLocalizationTests(unittest.TestCase):
    def test_periodic_update_smoothing_preserves_seam_and_reduces_short_wavelength_curvature(self):
        x = np.linspace(0.0, 2.0 * np.pi, 1441)
        u0 = 1.0 - np.cos(x)
        u1 = u0 + 0.1 * np.cos(45.0 * x)
        u1[-1] = u1[0]
        delta = u1 - u0
        smooth = _smooth_periodic_update(delta, x, 0.02)
        self.assertAlmostEqual(float(smooth[0]), float(smooth[-1]), places=14)
        raw = _candidate_energy(u0, u1, x, CandidateSpec("raw", 1.0, 0.0))
        reg = _candidate_energy(u0, u1, x, CandidateSpec("reg", 1.0, 0.02))
        raw_u2 = np.max(np.abs(CubicSpline(x, raw, bc_type="periodic")(x, 2)))
        reg_u2 = np.max(np.abs(CubicSpline(x, reg, bc_type="periodic")(x, 2)))
        self.assertLess(reg_u2, raw_u2)

    def test_update_fraction_interpolates_observed_energy_update(self):
        x = np.linspace(0.0, 2.0 * np.pi, 129)
        u0 = 1.2 * (1.0 - np.cos(x))
        u1 = u0 + 0.2 * (1.0 - np.cos(3.0 * x))
        half = _candidate_energy(u0, u1, x, CandidateSpec("half", 0.5, 0.0))
        expected = u0 + 0.5 * (u1 - u0)
        expected -= np.min(expected)
        expected[-1] = expected[0]
        np.testing.assert_allclose(half, expected, rtol=0.0, atol=1.0e-14)

    def test_finalizer_distinguishes_amplitude_gain_from_smoothing_gain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidates = [
                ("frac_0p00_raw", 0.0, 0.0, 0.60, 100.0),
                ("frac_0p50_raw", 0.5, 0.0, 0.80, 180.0),
                ("frac_1p00_raw", 1.0, 0.0, 1.40, 300.0),
                ("frac_1p00_smooth_0p02", 1.0, 0.02, 1.20, 140.0),
            ]
            registry = {
                "candidates": [
                    {
                        "name": n,
                        "update_fraction": f,
                        "effective_alpha_if_linear_no_clip": 0.1 * f,
                        "smooth_sigma_rad": s,
                        "target_abs_U2_p99": u2,
                        "target_abs_U2_p95": u2,
                        "target_abs_U2_max": u2,
                        "candidate_priors": str(root / n / "cg_priors.json"),
                    }
                    for n, f, s, _l1, u2 in candidates
                ]
            }
            regpath = root / "registry.json"
            regpath.write_text(json.dumps(registry))
            step35 = root / "step35.json"
            step35.write_text(json.dumps({
                "ibi_last_sampled_dihedral_l1": {"a": 0.6},
                "runtime_structure": {"dihedral_mean_l1": 1.4},
                "candidate_order2_through_0p005": False,
            }))
            sroot = root / "short"
            for name, _f, _s, l1, _u2 in candidates:
                p = sroot / name
                p.mkdir(parents=True)
                (p / "runtime_structure_report.json").write_text(json.dumps({
                    "mean_l1_by_kind": {"dihedral": l1},
                    "groups": {"d": {"kind": "dihedral", "distribution_l1": l1}},
                }))
            out = root / "out.json"
            report = finalize(regpath, step35, sroot, out)
            self.assertEqual(report["localization"]["diagnostic_hint"], "update_amplitude_overshoot_dominant")
            self.assertTrue(report["localization"]["raw_fraction_l1_monotone_non_decreasing"])
            self.assertFalse(report["promotion_ready"])


if __name__ == "__main__":
    unittest.main()
