#!/usr/bin/env python3
"""Pure numerical helpers for NVE energy-conservation certification."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def read_energy_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read Time_ps and E_tot from a run_cg_md.py energy CSV."""
    times: list[float] = []
    energies: list[float] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"Time_ps", "E_tot"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Energy CSV {path} must contain columns {sorted(required)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            times.append(float(row["Time_ps"]))
            energies.append(float(row["E_tot"]))
    return np.asarray(times, dtype=float), np.asarray(energies, dtype=float)


def analyze_energy_series(times_ps: Iterable[float], energies: Iterable[float]) -> dict[str, float]:
    """Measure bounded energy error plus block-mean and linear drift diagnostics."""
    t = np.asarray(list(times_ps), dtype=float)
    e = np.asarray(list(energies), dtype=float)
    if t.ndim != 1 or e.ndim != 1 or t.size != e.size:
        raise ValueError("times and energies must be one-dimensional arrays of equal length")
    if t.size < 3:
        raise ValueError("At least three energy samples are required")
    if not np.isfinite(t).all() or not np.isfinite(e).all():
        raise ValueError("NVE energy series contains non-finite values")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("NVE energy sample times must be strictly increasing")

    duration = float(t[-1] - t[0])
    if duration <= 0.0:
        raise ValueError("NVE duration must be positive")

    e0 = float(e[0])
    delta = e - e0
    rms_delta = float(np.sqrt(np.mean(delta * delta)))
    centered_rms = float(np.std(e))
    peak_to_peak = float(np.ptp(e))
    max_abs_delta = float(np.max(np.abs(delta)))

    slope, intercept = np.polyfit(t - t[0], e, 1)
    slope = float(slope)
    intercept = float(intercept)
    drift_span = float(slope * duration)

    # A linear fit to an oscillatory bounded error can report an apparent slope
    # when the oscillation phase correlates with time. For the certification
    # drift metric, compare endpoint block means instead.
    block_size = max(1, int(math.floor(0.20 * t.size)))
    first_block_mean = float(np.mean(e[:block_size]))
    last_block_mean = float(np.mean(e[-block_size:]))
    block_mean_drift = last_block_mean - first_block_mean

    # Total energy can legitimately be close to zero. Use an O(energy) scale
    # that does not become singular in that case.
    energy_scale = max(abs(e0), float(np.mean(np.abs(e))), 1.0)
    relative_rms = rms_delta / energy_scale
    relative_peak_to_peak = peak_to_peak / energy_scale
    relative_linear_drift = abs(drift_span) / energy_scale
    relative_block_mean_drift = abs(block_mean_drift) / energy_scale
    drift_to_rms = abs(block_mean_drift) / max(rms_delta, np.finfo(float).tiny)

    return {
        "samples": int(t.size),
        "duration_ps": duration,
        "E0": e0,
        "E_mean": float(np.mean(e)),
        "energy_scale": energy_scale,
        "rms_delta_E": rms_delta,
        "centered_rms_E": centered_rms,
        "peak_to_peak_E": peak_to_peak,
        "max_abs_delta_E": max_abs_delta,
        "relative_rms_delta_E": relative_rms,
        "relative_peak_to_peak_E": relative_peak_to_peak,
        "linear_drift_kjmol_per_ps": slope,
        "linear_fit_intercept": intercept,
        "linear_drift_span_kjmol": drift_span,
        "relative_linear_drift_span": relative_linear_drift,
        "drift_block_fraction": 0.20,
        "first_block_mean_E": first_block_mean,
        "last_block_mean_E": last_block_mean,
        "block_mean_drift_kjmol": block_mean_drift,
        "relative_block_mean_drift": relative_block_mean_drift,
        "drift_span_over_rms_delta": drift_to_rms,
    }


def fit_timestep_scaling(run_metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Fit RMS energy error = C * dt**p across multiple NVE runs."""
    runs = sorted(run_metrics, key=lambda item: float(item["dt_ps"]))
    if len(runs) < 3:
        raise ValueError("At least three time steps are required for scaling certification")

    dts = np.asarray([float(item["dt_ps"]) for item in runs], dtype=float)
    rms = np.asarray([float(item["rms_delta_E"]) for item in runs], dtype=float)
    if np.any(dts <= 0.0) or np.any(rms <= 0.0) or not np.isfinite(rms).all():
        raise ValueError("Time steps and RMS energy errors must be finite and positive")

    x = np.log(dts)
    y = np.log(rms)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= np.finfo(float).eps else 1.0 - ss_res / ss_tot

    adjacent = []
    for lo, hi in zip(runs[:-1], runs[1:]):
        dt_ratio = float(hi["dt_ps"]) / float(lo["dt_ps"])
        observed = float(hi["rms_delta_E"]) / float(lo["rms_delta_E"])
        adjacent.append({
            "dt_low_ps": float(lo["dt_ps"]),
            "dt_high_ps": float(hi["dt_ps"]),
            "dt_ratio": dt_ratio,
            "observed_rms_ratio": observed,
            "quadratic_expected_ratio": dt_ratio * dt_ratio,
        })

    return {
        "exponent_p": float(slope),
        "prefactor_C": float(math.exp(intercept)),
        "loglog_r2": float(r2),
        "adjacent_ratios": adjacent,
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
    scaling_pass = (
        slope_min <= scaling["exponent_p"] <= slope_max
        and scaling["loglog_r2"] >= min_r2
    )
    drift_failures = [
        {
            "dt_ps": float(item["dt_ps"]),
            "relative_block_mean_drift": float(item["relative_block_mean_drift"]),
        }
        for item in runs
        if float(item["relative_block_mean_drift"]) > max_relative_drift
    ]
    drift_pass = not drift_failures
    return {
        "pass": bool(scaling_pass and drift_pass),
        "scaling_pass": bool(scaling_pass),
        "drift_pass": bool(drift_pass),
        "thresholds": {
            "slope_min": float(slope_min),
            "slope_max": float(slope_max),
            "min_loglog_r2": float(min_r2),
            "max_relative_block_mean_drift": float(max_relative_drift),
        },
        "scaling": scaling,
        "drift_failures": drift_failures,
    }
