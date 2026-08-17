#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import math
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIMULATION = ROOT / "simulation"
sys.path.insert(0, str(SIMULATION))

from nve_state_convergence import (  # noqa: E402
    analyze_state_convergence,
    minimum_image_delta,
    quaternion_angle_error,
)


def quat_z(angle: float, n: int) -> np.ndarray:
    q = np.zeros((n, 4), dtype=float)
    q[:, 0] = math.cos(0.5 * angle)
    q[:, 3] = math.sin(0.5 * angle)
    return q


def synthetic_trajectory(dt: float, times: list[float]) -> dict:
    ids = np.asarray([0, 2, 5], dtype=np.int64)
    n = ids.size
    box = np.asarray([10.0, 10.0, 10.0], dtype=float)
    positions = []
    velocities = []
    quaternions = []
    omegas = []
    for t in times:
        exact_pos = np.asarray([
            [1.0 + 0.10 * t, 2.0, 3.0],
            [4.0, 5.0 + 0.20 * t, 6.0],
            [7.0, 8.0, 1.0 + 0.15 * t],
        ])
        exact_vel = np.asarray([
            [0.10, 0.0, 0.0],
            [0.0, 0.20, 0.0],
            [0.0, 0.0, 0.15],
        ])
        amp = dt * dt * (1.0 + 3.0 * t)
        positions.append(exact_pos + amp * np.asarray([[1.0, 0.2, 0.0], [0.0, -0.5, 0.1], [0.3, 0.0, -0.4]]))
        velocities.append(exact_vel + amp * np.asarray([[0.4, 0.0, 0.1], [0.0, -0.2, 0.0], [0.1, 0.0, 0.3]]))
        quaternions.append(quat_z(0.5 * t + 0.7 * amp, n))
        omegas.append(np.tile(np.asarray([0.0, 0.0, 0.5 + 0.9 * amp]), (n, 1)))
    return {
        "path": Path(f"synthetic_{dt:g}.npz"),
        "steps": np.arange(len(times), dtype=np.int64),
        "time_ps": np.asarray(times, dtype=float),
        "particle_ids": ids,
        "rotation_flags": np.ones((n, 3), dtype=bool),
        "positions": np.asarray(positions),
        "velocities": np.asarray(velocities),
        "quaternions": np.asarray(quaternions),
        "omega_body": np.asarray(omegas),
        "box": box,
        "metadata": {
            "input_hashes": {"dataset_sha256": "same"},
            "hamiltonian_mode": "conservative_classical_model_provenance_ml_disabled",
            "source_checkpoint_sha256": "source",
        },
    }


class NVEStateConvergenceTests(unittest.TestCase):
    def test_minimum_image_delta(self):
        a = np.asarray([[9.9, 0.0, 0.0]])
        b = np.asarray([[0.1, 0.0, 0.0]])
        d = minimum_image_delta(a, b, np.asarray([10.0, 10.0, 10.0]))
        self.assertTrue(np.allclose(d, [[-0.2, 0.0, 0.0]]))

    def test_quaternion_sign_is_irrelevant(self):
        q = quat_z(0.3, 4)
        err = quaternion_angle_error(q, -q)
        self.assertLess(float(np.max(err)), 1.0e-12)

    def test_richardson_recovers_second_order_for_all_state_metrics(self):
        times = [0.0, 0.012, 0.024, 0.048, 0.096]
        dts = [0.0000625, 0.000125, 0.00025, 0.0005, 0.001]
        trajectories = {dt: synthetic_trajectory(dt, times) for dt in dts}
        report = analyze_state_convergence(
            trajectories,
            reference_dt=0.0000625,
            sample_times_ps=times[1:],
            expected_order_min=1.7,
            expected_order_max=2.3,
            min_r2=0.99,
        )
        for metric, summary in report["metric_summary"].items():
            self.assertAlmostEqual(summary["median_exponent_p"], 2.0, places=5, msg=metric)
            self.assertGreater(summary["median_loglog_r2"], 0.999999)
            self.assertTrue(summary["consistent_with_second_order"])

    def test_non_dyadic_ladder_is_rejected(self):
        times = [0.0, 0.012, 0.024, 0.048]
        trajectories = {
            0.0000625: synthetic_trajectory(0.0000625, times),
            0.000125: synthetic_trajectory(0.000125, times),
            0.0003: synthetic_trajectory(0.0003, times),
            0.0006: synthetic_trajectory(0.0006, times),
        }
        with self.assertRaisesRegex(ValueError, "dyadic"):
            analyze_state_convergence(
                trajectories,
                reference_dt=0.0000625,
                sample_times_ps=times[1:],
            )

    def test_runner_contains_mechanical_state_sampling_interface(self):
        source = (SIMULATION / "run_cg_md.py").read_text(encoding="utf-8")
        self.assertIn("--state_sample_npz", source)
        self.assertIn("mlcg_real_particle_state_trajectory", source)
        self.assertIn("rotation_flags=state_sample_rotation_flags", source)
        self.assertIn("omega_body=np.asarray(state_sample_omegas", source)


if __name__ == "__main__":
    unittest.main()
