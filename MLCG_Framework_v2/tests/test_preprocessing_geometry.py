#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))

from geometry_utils import diagonalize_inertia_tensor, minimum_image_distance_matrix  # noqa: E402


class PreprocessingGeometryTests(unittest.TestCase):
    def test_minimum_image_across_periodic_face(self):
        positions = np.array([[0.05, 1.0, 1.0], [9.95, 1.0, 1.0]])
        distances = minimum_image_distance_matrix(positions, np.array([10.0, 10.0, 10.0]))
        self.assertAlmostEqual(distances[0, 1], 0.1, places=12)
        self.assertAlmostEqual(distances[1, 0], 0.1, places=12)

    def test_principal_axes_diagonalize_and_reconstruct(self):
        angle = 0.713
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        moments = np.array([1.0, 2.0, 5.0])
        tensor = rotation @ np.diag(moments) @ rotation.T
        values, axes = diagonalize_inertia_tensor(tensor)
        self.assertTrue(np.allclose(values, moments, atol=1e-12))
        self.assertGreater(np.linalg.det(axes), 0.0)
        self.assertTrue(np.allclose(axes @ np.diag(values) @ axes.T, tensor, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
