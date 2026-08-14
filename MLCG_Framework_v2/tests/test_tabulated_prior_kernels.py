#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "ibi"))

from prior_kernels import (  # noqa: E402
    TabulatedPrior,
    espresso_dihedral_geometry,
    load_tabulated_prior,
    tabulated_angle_forces,
    tabulated_dihedral_forces,
    tabulated_distance_forces,
    tabulated_value,
)
from ibi_core import DEFAULT_IBI_SETTINGS, table_from_potential  # noqa: E402


def make_table(x, energy, force, kind="bond"):
    return TabulatedPrior(
        np.asarray(x, dtype=float),
        np.asarray(energy, dtype=float),
        np.asarray(force, dtype=float),
        float(x[0]),
        float(x[-1]),
        kind,
        Path("synthetic.dat"),
    )


def finite_difference_force(positions, energy_fn, eps=1.0e-7):
    positions = np.asarray(positions, dtype=float)
    result = np.zeros_like(positions)
    for i in range(len(positions)):
        for a in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[i, a] += eps
            minus[i, a] -= eps
            result[i, a] = -(energy_fn(plus) - energy_fn(minus)) / (2.0 * eps)
    return result


class TabulatedPriorKernelTests(unittest.TestCase):
    def test_table_lookup_is_linear_and_clamped(self):
        table = make_table([1.0, 2.0, 3.0], [0, 1, 4], [10, 20, 40])
        self.assertAlmostEqual(tabulated_value(table, 1.5), 15.0)
        self.assertAlmostEqual(tabulated_value(table, 0.5), 10.0)
        self.assertAlmostEqual(tabulated_value(table, 4.0), 40.0)

    def test_distance_force_matches_energy_gradient(self):
        k, r0 = 7.0, 1.2
        x = np.linspace(0.5, 2.5, 1001)
        energy = 0.5 * k * (x - r0) ** 2
        force = -k * (x - r0)
        table = make_table(x, energy, force, "bond")
        box = np.array([20.0, 20.0, 20.0])
        pos = np.array([[1.0, 2.0, 3.0], [2.7, 2.2, 3.0]])
        actual = np.vstack(tabulated_distance_forces(pos[0], pos[1], box, table))

        def e(coords):
            d = coords[1] - coords[0]
            d -= box * np.round(d / box)
            r = np.linalg.norm(d)
            return 0.5 * k * (r - r0) ** 2

        expected = finite_difference_force(pos, e)
        self.assertTrue(np.allclose(actual, expected, rtol=2e-6, atol=2e-7))

    def test_angle_force_matches_energy_gradient(self):
        k, theta0 = 5.0, 1.3
        x = np.linspace(0.0, np.pi, 2001)
        energy = 0.5 * k * (x - theta0) ** 2
        gradient = k * (x - theta0)
        table = make_table(x, energy, gradient, "angle")
        box = np.array([20.0, 20.0, 20.0])
        pos = np.array([[0.2, 0.4, 0.1], [1.0, 1.0, 1.0], [2.2, 1.4, 0.7]])
        actual = np.vstack(tabulated_angle_forces(pos[0], pos[1], pos[2], box, table))

        def e(coords):
            a = coords[0] - coords[1]
            b = coords[2] - coords[1]
            theta = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
            return 0.5 * k * (theta - theta0) ** 2

        expected = finite_difference_force(pos, e)
        self.assertTrue(np.allclose(actual, expected, rtol=4e-5, atol=3e-6))

    def test_dihedral_force_factor_matches_analytic_cosine_energy(self):
        K = 3.7
        x = np.linspace(0.0, 2.0 * np.pi, 4001)
        energy = K * (1.0 - np.cos(x))
        # For U=K(1-cos(phi)), ESPResSo's analytic dihedral geometry uses
        # fac=-K away from the phi singularities.
        force_factor = np.full_like(x, -K)
        table = make_table(x, energy, force_factor, "dihedral")
        box = np.array([30.0, 30.0, 30.0])
        pos = np.array([
            [0.2, 0.4, 0.1],
            [1.1, 0.9, 0.6],
            [2.0, 1.5, 1.2],
            [2.8, 2.2, 0.5],
        ])
        actual = np.vstack(tabulated_dihedral_forces(*pos, box, table))

        def e(coords):
            geom = espresso_dihedral_geometry(*coords, box)
            if geom is None:
                return 0.0
            return K * (1.0 - np.cos(geom[0]))

        expected = finite_difference_force(pos, e)
        self.assertTrue(np.allclose(actual, expected, rtol=5e-5, atol=5e-6))

    def test_ibi_dihedral_table_uses_espresso_force_factor(self):
        x = np.linspace(0.0, 2.0 * np.pi, 2001)
        K = 2.5
        potential = K * (1.0 - np.cos(x))
        settings = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_IBI_SETTINGS.items()}
        settings["dihedral"] = dict(DEFAULT_IBI_SETTINGS["dihedral"])
        settings["dihedral"]["force_max"] = 1000.0
        _energy, force = table_from_potential(x, potential, "dihedral", periodic=True, settings=settings)
        regular = np.abs(np.sin(x)) > 0.05
        self.assertLess(np.max(np.abs(force[regular] + K)), 2.0e-4)

    def test_load_rejects_noncanonical_angle_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            x = np.linspace(0.1, np.pi, 11)
            path = tmp / "angle.dat"
            np.savetxt(path, np.column_stack([x, x * 0, x * 0]))
            with self.assertRaisesRegex(ValueError, "0..pi"):
                load_tabulated_prior(
                    {"file": "angle.dat", "min": 0.1, "max": float(np.pi)},
                    kind="angle",
                    priors_path=tmp / "priors.json",
                )


if __name__ == "__main__":
    unittest.main()
