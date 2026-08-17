import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline, CubicSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ibi"))
sys.path.insert(0, str(ROOT / "preprocessing"))

from angle_regularization_diagnostics import (
    CandidateSpec,
    _build_candidate,
    _knot_u2_jump,
    candidate_specs_from_json,
    same_barrier_k,
    wall_energy_gradient_curvature,
)


class DummyTable:
    def __init__(self, x, energy, derivative):
        self.x = np.asarray(x, dtype=float)
        self.energy = np.asarray(energy, dtype=float)
        self.derivative = np.asarray(derivative, dtype=float)


def test_same_barrier_wall_scaling():
    old_w, old_k = 0.1, 5000.0
    for new_w in (0.15, 0.2, 0.3):
        new_k = same_barrier_k(old_w, old_k, new_w)
        assert 0.5 * old_k * old_w**2 == pytest.approx(0.5 * new_k * new_w**2)


def test_wall_energy_gradient_curvature_is_consistent():
    q = np.array([0.02, 0.1, 1.0, np.pi - 0.1, np.pi - 0.02])
    u, du, u2 = wall_energy_gradient_curvature(q, 0.1, 5000.0)
    assert u[2] == pytest.approx(0.0)
    assert du[2] == pytest.approx(0.0)
    assert u2[2] == pytest.approx(0.0)
    assert u2[0] == pytest.approx(5000.0)
    assert u2[-1] == pytest.approx(5000.0)
    eps = 1e-7
    up, _, _ = wall_energy_gradient_curvature(np.array([q[0] + eps]), 0.1, 5000.0)
    um, _, _ = wall_energy_gradient_curvature(np.array([q[0] - eps]), 0.1, 5000.0)
    fd = (up[0] - um[0]) / (2 * eps)
    assert fd == pytest.approx(du[0], rel=1e-7, abs=1e-7)


def test_c2_raw_preserves_nodal_energy_and_removes_u2_jumps():
    x = np.linspace(0.0, np.pi, 101)
    u = 4.0 * (x - 1.5) ** 2 + 0.15 * np.sin(17.0 * x)
    # Deliberately use a C1 Hermite derivative field that is not from one C2 spline.
    d = np.gradient(u, x)
    table = DummyTable(x, u, d)
    spec = CandidateSpec("raw", 0.0, 0.1, 0.0, "test")
    c2, new_u, new_du = _build_candidate(table, spec, 0.1, 0.0)
    np.testing.assert_allclose(new_u - np.min(new_u), u - np.min(u), atol=1e-12, rtol=0.0)
    current = CubicHermiteSpline(x, u, d)
    assert _knot_u2_jump(c2, x)["max_abs"] < 1e-5
    assert _knot_u2_jump(current, x)["max_abs"] > _knot_u2_jump(c2, x)["max_abs"]
    np.testing.assert_allclose(new_du, c2(x, 1), atol=0.0, rtol=0.0)


def test_body_smoothing_reduces_high_frequency_curvature():
    x = np.linspace(0.0, np.pi, 2001)
    base = 20.0 * (x - 1.4) ** 2
    noisy = base + 0.08 * np.sin(220.0 * x)
    current_c2 = CubicSpline(x, noisy)
    table = DummyTable(x, noisy, current_c2(x, 1))
    spec = CandidateSpec("smooth", 0.02, 0.1, 0.0, "test")
    cand, _, _ = _build_candidate(table, spec, 0.1, 0.0)
    q = np.linspace(0.2, np.pi - 0.2, 10000)
    cur_p99 = np.percentile(np.abs(current_c2(q, 2)), 99)
    new_p99 = np.percentile(np.abs(cand(q, 2)), 99)
    assert new_p99 < 0.25 * cur_p99


def test_configured_candidates_preserve_endpoint_barrier_for_widened_walls():
    specs = candidate_specs_from_json(
        """[{"name":"wide15_same_barrier","body_sigma_rad":0.0,"wall_width_scale":1.5},
             {"name":"wide20_same_barrier","body_sigma_rad":0.0,"wall_width_scale":2.0}]""",
        0.1, 5000.0,
    )
    old_barrier = 0.5 * 5000.0 * 0.1**2
    widened = [s for s in specs if "same_barrier" in s.name]
    assert len(widened) == 2
    for s in widened:
        assert 0.5 * s.wall_k * s.wall_width_rad**2 == pytest.approx(old_barrier)
