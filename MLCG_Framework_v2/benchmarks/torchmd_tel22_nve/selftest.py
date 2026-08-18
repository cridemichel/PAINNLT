#!/usr/bin/env python3
"""Fast dependency-light checks for the benchmark analysis and initial-state generator."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analysis import analyze_energy_series, certify_metrics  # noqa: E402
from run_certification import build_initial_state  # noqa: E402


def main() -> int:
    runs = []
    for dt in (0.001, 0.0015, 0.002, 0.003, 0.004, 0.005):
        t = np.linspace(0.0, 2.0, 2001)
        e = 1000.0 + 4.0e5 * dt * dt * np.cos(2.0 * np.pi * t)
        metrics = analyze_energy_series(t, e)
        metrics["dt_ps"] = dt
        runs.append(metrics)
    cert = certify_metrics(runs, slope_min=1.7, slope_max=2.3, min_r2=0.97, max_relative_drift=1e-4)
    assert cert["pass"]
    assert math.isclose(cert["scaling"]["exponent_p"], 2.0, abs_tol=1e-10)

    args = argparse.Namespace(
        seed=220526,
        particles=820,
        displacement_sigma_a=0.20,
        temperature_k=300.0,
        mass_g_mol=72.0,
        k_min=20.0,
        k_max=80.0,
    )
    a = build_initial_state(args)
    b = build_initial_state(args)
    assert a["positions"].shape == (820, 3)
    assert a["masses"].shape == (820, 1)
    assert np.array_equal(a["positions"], b["positions"])
    assert np.array_equal(a["velocities"], b["velocities"])
    assert np.max(np.abs(a["velocities"].mean(axis=0))) < 1e-14
    print("[PASS] TorchMD NVE benchmark self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
