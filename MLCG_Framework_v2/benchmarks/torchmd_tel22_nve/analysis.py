#!/usr/bin/env python3
"""Unit-neutral NVE energy analysis for the isolated TorchMD benchmark."""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def analyze_energy_series(times_ps: Iterable[float], energies: Iterable[float]) -> dict[str, float]:
    t = np.asarray(list(times_ps), dtype=np.float64)
    e = np.asarray(list(energies), dtype=np.float64)
    if t.ndim != 1 or e.ndim != 1 or t.size != e.size or t.size < 3:
        raise ValueError("times and energies must be aligned 1D arrays with at least three samples")
    if not np.isfinite(t).all() or not np.isfinite(e).all():
        raise ValueError("NVE energy series contains NaN or Inf")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("sample times must be strictly increasing")

    duration = float(t[-1] - t[0])
    if duration <= 0.0:
        raise ValueError("NVE duration must be positive")

    e0 = float(e[0])
    delta = e - e0
    sigma = float(np.std(e, ddof=0))
    rms_delta = float(np.sqrt(np.mean(delta * delta)))
    peak_to_peak = float(np.ptp(e))
    max_abs_delta = float(np.max(np.abs(delta)))

    slope, intercept = np.polyfit(t - t[0], e, 1)
    drift_span = float(slope * duration)
    block_size = max(1, int(math.floor(0.20 * t.size)))
    first_block_mean = float(np.mean(e[:block_size]))
    last_block_mean = float(np.mean(e[-block_size:]))
    block_mean_drift = last_block_mean - first_block_mean
    energy_scale = max(abs(e0), float(np.mean(np.abs(e))), 1.0)

    return {
        "samples": int(t.size),
        "duration_ps": duration,
        "E0": e0,
        "E_mean": float(np.mean(e)),
        "energy_scale": energy_scale,
        "rms_delta_E": rms_delta,
        "sigma_E": sigma,
        "peak_to_peak_E": peak_to_peak,
        "max_abs_delta_E": max_abs_delta,
        "linear_drift_energy_per_ps": float(slope),
        "linear_fit_intercept": float(intercept),
        "linear_drift_span": drift_span,
        "relative_linear_drift_span": abs(drift_span) / energy_scale,
        "drift_block_fraction": 0.20,
        "first_block_mean_E": first_block_mean,
        "last_block_mean_E": last_block_mean,
        "block_mean_drift": block_mean_drift,
        "relative_block_mean_drift": abs(block_mean_drift) / energy_scale,
        "drift_span_over_rms_delta": abs(block_mean_drift) / max(rms_delta, np.finfo(float).tiny),
    }


def fit_timestep_scaling(run_metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    runs = sorted(list(run_metrics), key=lambda item: float(item["dt_ps"]))
    if len(runs) < 3:
        raise ValueError("at least three time steps are required")

    dts = np.asarray([float(item["dt_ps"]) for item in runs], dtype=np.float64)
    sigma = np.asarray([float(item["sigma_E"]) for item in runs], dtype=np.float64)
    if np.any(dts <= 0.0) or np.any(sigma <= 0.0) or not np.isfinite(sigma).all():
        raise ValueError("dt and sigma_E must be finite and positive")

    x = np.log(dts)
    y = np.log(sigma)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= np.finfo(float).eps else 1.0 - ss_res / ss_tot

    return {
        "observable": "sigma_E",
        "model": "sigma_E = C * dt^p",
        "exponent_p": float(slope),
        "prefactor_C": float(math.exp(intercept)),
        "loglog_r2": float(r2),
    }


def certify_metrics(
    run_metrics: Iterable[dict[str, Any]],
    *,
    slope_min: float,
    slope_max: float,
    min_r2: float,
    max_relative_drift: float,
) -> dict[str, Any]:
    runs = list(run_metrics)
    scaling = fit_timestep_scaling(runs)
    scaling_pass = slope_min <= scaling["exponent_p"] <= slope_max and scaling["loglog_r2"] >= min_r2
    drift_failures = [
        {
            "dt_ps": float(item["dt_ps"]),
            "relative_block_mean_drift": float(item["relative_block_mean_drift"]),
        }
        for item in runs
        if float(item["relative_block_mean_drift"]) > max_relative_drift
    ]
    c2 = [float(item["sigma_E"]) / float(item["dt_ps"]) ** 2 for item in runs]
    c2_spread = float(max(c2) / min(c2))
    return {
        "pass": bool(scaling_pass and not drift_failures),
        "scaling_pass": bool(scaling_pass),
        "drift_pass": not drift_failures,
        "thresholds": {
            "slope_min": float(slope_min),
            "slope_max": float(slope_max),
            "min_loglog_r2": float(min_r2),
            "max_relative_block_mean_drift": float(max_relative_drift),
        },
        "scaling": scaling,
        "c2_spread_max_over_min": c2_spread,
        "drift_failures": drift_failures,
    }
