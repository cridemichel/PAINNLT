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
        "sigma_E": centered_rms,
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
    """Fit sigma_E = C * dt**p across multiple fixed-duration NVE runs."""
    runs = sorted(run_metrics, key=lambda item: float(item["dt_ps"]))
    if len(runs) < 3:
        raise ValueError("At least three time steps are required for scaling certification")

    dts = np.asarray([float(item["dt_ps"]) for item in runs], dtype=float)
    sigma = np.asarray([float(item["sigma_E"]) for item in runs], dtype=float)
    if np.any(dts <= 0.0) or np.any(sigma <= 0.0) or not np.isfinite(sigma).all():
        raise ValueError("Time steps and sigma_E values must be finite and positive")

    x = np.log(dts)
    y = np.log(sigma)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= np.finfo(float).eps else 1.0 - ss_res / ss_tot

    adjacent = []
    for lo, hi in zip(runs[:-1], runs[1:]):
        dt_ratio = float(hi["dt_ps"]) / float(lo["dt_ps"])
        observed = float(hi["sigma_E"]) / float(lo["sigma_E"])
        adjacent.append({
            "dt_low_ps": float(lo["dt_ps"]),
            "dt_high_ps": float(hi["dt_ps"]),
            "dt_ratio": dt_ratio,
            "observed_sigma_ratio": observed,
            "quadratic_expected_ratio": dt_ratio * dt_ratio,
        })

    return {
        "observable": "sigma_E",
        "model": "sigma_E = C * dt^p",
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



def fit_metric_scaling(
    run_metrics: Iterable[dict[str, Any]],
    metric: str,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    """Fit a positive per-run metric to C*dt**p without imposing certification thresholds."""
    runs = sorted(run_metrics, key=lambda item: float(item["dt_ps"]))
    if len(runs) < 3:
        raise ValueError("At least three time steps are required for a diagnostic power-law fit")
    dts = np.asarray([float(item["dt_ps"]) for item in runs], dtype=float)
    values = np.asarray([float(item[metric]) for item in runs], dtype=float)
    if np.any(dts <= 0.0) or np.any(values <= 0.0) or not np.isfinite(values).all():
        raise ValueError(f"Time steps and {metric} values must be finite and positive")

    x = np.log(dts)
    y = np.log(values)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= np.finfo(float).eps else 1.0 - ss_res / ss_tot
    return {
        "observable": label or metric,
        "metric_key": metric,
        "model": f"{label or metric} = C * dt^p",
        "n_points": int(len(runs)),
        "dt_min_ps": float(np.min(dts)),
        "dt_max_ps": float(np.max(dts)),
        "exponent_p": float(slope),
        "prefactor_C": float(math.exp(intercept)),
        "loglog_r2": float(r2),
    }


def analyze_local_energy_windows(
    times_ps: Iterable[float],
    energies: Iterable[float],
    windows_ps: Iterable[float],
) -> dict[str, dict[str, float]]:
    """Measure short-time energy error at exact, common physical-time windows.

    The requested windows must coincide with sampled times.  This is deliberate:
    interpolating E(t) would introduce a second numerical approximation into the
    timestep-order diagnostic.
    """
    t = np.asarray(list(times_ps), dtype=float)
    e = np.asarray(list(energies), dtype=float)
    if t.ndim != 1 or e.ndim != 1 or t.size != e.size or t.size < 3:
        raise ValueError("times and energies must contain at least three aligned samples")
    if not np.isfinite(t).all() or not np.isfinite(e).all():
        raise ValueError("Local NVE diagnostic contains non-finite values")
    elapsed = t - t[0]
    if np.any(np.diff(elapsed) <= 0.0):
        raise ValueError("NVE energy sample times must be strictly increasing")

    result: dict[str, dict[str, float]] = {}
    for requested in windows_ps:
        requested = float(requested)
        if requested <= 0.0 or requested > float(elapsed[-1]):
            raise ValueError(
                f"Local diagnostic window {requested:g} ps is outside trajectory duration "
                f"{elapsed[-1]:g} ps"
            )
        idx = int(np.argmin(np.abs(elapsed - requested)))
        tolerance = max(1.0e-12, 1.0e-9 * max(1.0, requested))
        if abs(float(elapsed[idx]) - requested) > tolerance:
            raise ValueError(
                f"Local diagnostic time {requested:g} ps is not sampled exactly; nearest "
                f"sample is {elapsed[idx]:.17g} ps. Choose windows commensurate with every "
                "fine-regime timestep."
            )
        if idx < 2:
            raise ValueError(
                f"Local diagnostic window {requested:g} ps contains fewer than three samples"
            )
        segment = e[: idx + 1]
        delta = segment - segment[0]
        key = format(requested, ".12g")
        result[key] = {
            "requested_time_ps": requested,
            "actual_time_ps": float(elapsed[idx]),
            "samples": int(idx + 1),
            "endpoint_abs_delta_E": float(abs(segment[-1] - segment[0])),
            "prefix_rms_delta_E": float(np.sqrt(np.mean(delta * delta))),
            "prefix_sigma_E": float(np.std(segment)),
            "prefix_max_abs_delta_E": float(np.max(np.abs(delta))),
        }
    return result


def build_nve_diagnostics(
    run_metrics: Iterable[dict[str, Any]],
    *,
    fine_max_dt: float,
    coarse_min_dt: float,
    local_times_ps: Iterable[float],
) -> dict[str, Any]:
    """Build global/fine/coarse and short-time power-law fits for diagnosis only."""
    runs = sorted(list(run_metrics), key=lambda item: float(item["dt_ps"]))
    fine = [item for item in runs if float(item["dt_ps"]) <= float(fine_max_dt)]
    coarse = [item for item in runs if float(item["dt_ps"]) >= float(coarse_min_dt)]
    if len(fine) < 3:
        raise ValueError("Fine-regime diagnostic requires at least three dt values")
    if len(coarse) < 3:
        raise ValueError("Coarse-regime diagnostic requires at least three dt values")

    split_fits: dict[str, dict[str, Any]] = {}
    for name, subset in (("global", runs), ("fine", fine), ("coarse", coarse)):
        split_fits[name] = {
            "dt_values_ps": [float(item["dt_ps"]) for item in subset],
            "sigma_E": fit_metric_scaling(subset, "sigma_E", label="sigma_E"),
            "rms_delta_E": fit_metric_scaling(subset, "rms_delta_E", label="rms_delta_E"),
        }

    def safe_fit(rows, metric, label):
        try:
            result = fit_metric_scaling(rows, metric, label=label)
            result["available"] = True
            return result
        except ValueError as exc:
            return {
                "available": False,
                "observable": label,
                "metric_key": metric,
                "reason": str(exc),
            }

    local_fits: dict[str, Any] = {}
    for requested in local_times_ps:
        key = format(float(requested), ".12g")
        rows = []
        for item in fine:
            local_map = item.get("local_energy_windows")
            if not isinstance(local_map, dict) or key not in local_map:
                raise ValueError(f"Fine-regime run dt={item['dt_ps']} lacks local window {key} ps")
            local = local_map[key]
            rows.append({"dt_ps": float(item["dt_ps"]), **local})
        local_fits[key] = {
            "time_ps": float(requested),
            "dt_values_ps": [float(item["dt_ps"]) for item in rows],
            "endpoint_abs_delta_E": safe_fit(
                rows, "endpoint_abs_delta_E", f"|Delta E({key} ps)|"
            ),
            "prefix_rms_delta_E": safe_fit(
                rows, "prefix_rms_delta_E", f"RMS Delta E[0,{key} ps]"
            ),
            "prefix_sigma_E": safe_fit(
                rows, "prefix_sigma_E", f"Sigma E[0,{key} ps]"
            ),
        }

    return {
        "purpose": "diagnostic_only_not_a_certification_gate",
        "fine_max_dt_ps": float(fine_max_dt),
        "coarse_min_dt_ps": float(coarse_min_dt),
        "split_fits": split_fits,
        "local_energy_fits": local_fits,
    }


def energy_standard_deviation(energies):
    """Population standard deviation of the sampled total energy time series.

    For a fixed physical NVE trajectory duration this is the primary quantity
    used for the velocity-Verlet timestep-scaling test, sigma_E ~ dt**2.
    """
    values = np.asarray(energies, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("At least three energy samples are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("Energy series contains NaN or Inf")
    return float(np.std(values, ddof=0))
