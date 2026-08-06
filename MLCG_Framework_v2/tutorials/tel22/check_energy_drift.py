#!/usr/bin/env python3
"""Analyse long-run NVE energy drift from an ESPResSo ``energy.csv`` file.

The script expects at least the columns:

    Step,E_tot

Typical use:

    uv run --with numpy --with matplotlib python check_energy_drift.py \
        energy.csv --dt 0.001 --strict

The analysis:
  * discards an optional initial transient;
  * fits E_tot(t) = intercept + slope * t;
  * computes raw and detrended energy fluctuations;
  * estimates a confidence interval for the slope with a circular
    moving-block bootstrap, preserving short-range time correlation;
  * reports the total fitted drift over the analysed trajectory;
  * compares that drift with the detrended standard deviation;
  * saves a JSON report and a diagnostic plot.

Exit status:
  0: analysis completed and checks passed, or --strict not requested
  2: analysis completed but one or more strict checks failed
  1: invalid input or analysis error
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DriftResult:
    source_file: str
    dt_ps: float
    samples_total: int
    samples_analysed: int
    discarded_fraction: float
    analysed_time_ps: float
    mean_energy_kj_mol: float
    raw_std_kj_mol: float
    detrended_std_kj_mol: float
    energy_range_kj_mol: float
    slope_kj_mol_ps: float
    slope_ci_low_kj_mol_ps: float
    slope_ci_high_kj_mol_ps: float
    fitted_drift_over_run_kj_mol: float
    abs_drift_over_detrended_std: float
    relative_drift_over_mean: float
    r_squared_linear_trend: float
    block_length_samples: int
    bootstrap_samples: int
    confidence: float
    checks: dict[str, bool]
    passed: bool


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1)")
    return parsed


def read_energy_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(f"{path}: missing CSV header")

        required = {"Step", "E_tot"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise RuntimeError(
                f"{path}: missing required columns {sorted(missing)}; "
                f"available columns are {reader.fieldnames}"
            )

        steps: list[int] = []
        energies: list[float] = []

        for line_number, row in enumerate(reader, start=2):
            try:
                step = int(row["Step"])
                energy = float(row["E_tot"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid Step or E_tot value"
                ) from exc

            if not math.isfinite(energy):
                raise RuntimeError(
                    f"{path}:{line_number}: non-finite E_tot={energy}"
                )

            steps.append(step)
            energies.append(energy)

    if len(steps) < 30:
        raise RuntimeError(
            f"{path}: only {len(steps)} samples; at least 30 are required"
        )

    step_array = np.asarray(steps, dtype=np.int64)
    energy_array = np.asarray(energies, dtype=np.float64)

    differences = np.diff(step_array)
    if np.any(differences <= 0):
        raise RuntimeError(f"{path}: Step values must be strictly increasing")

    return step_array, energy_array


def linear_fit(
    time_ps: np.ndarray,
    energy: np.ndarray,
) -> tuple[float, float, np.ndarray, float]:
    slope, intercept = np.polyfit(time_ps, energy, 1)
    fitted = slope * time_ps + intercept
    residual = energy - fitted

    residual_ss = float(np.sum(residual**2))
    centered = energy - np.mean(energy)
    total_ss = float(np.sum(centered**2))
    r_squared = 1.0 if total_ss == 0.0 else 1.0 - residual_ss / total_ss

    return float(slope), float(intercept), residual, r_squared


def estimate_block_length(values: np.ndarray) -> int:
    """Estimate a conservative moving-block length from autocorrelation.

    The first non-positive autocorrelation crossing is used when available.
    The result is bounded to avoid pathological blocks for short trajectories.
    """
    centered = values - np.mean(values)
    variance = float(np.dot(centered, centered))
    n_values = len(values)

    if variance == 0.0:
        return max(2, int(round(math.sqrt(n_values))))

    max_lag = min(n_values // 4, 1000)
    crossing = None

    for lag in range(1, max_lag + 1):
        correlation = float(
            np.dot(centered[:-lag], centered[lag:]) / variance
        )
        if correlation <= 0.0:
            crossing = lag
            break

    if crossing is None:
        crossing = int(round(math.sqrt(n_values)))

    return max(2, min(crossing, max(2, n_values // 4)))


def bootstrap_slope_interval(
    time_ps: np.ndarray,
    energy: np.ndarray,
    *,
    slope: float,
    intercept: float,
    residual: np.ndarray,
    block_length: int,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray]:
    """Residual moving-block bootstrap for the fitted slope."""
    n_values = len(energy)
    n_blocks = math.ceil(n_values / block_length)
    offsets = np.arange(block_length, dtype=np.int64)
    fitted = slope * time_ps + intercept

    slopes = np.empty(samples, dtype=np.float64)

    for index in range(samples):
        starts = rng.integers(0, n_values, size=n_blocks)
        indices = (
            (starts[:, None] + offsets[None, :]) % n_values
        ).reshape(-1)[:n_values]

        synthetic = fitted + residual[indices]
        slopes[index] = np.polyfit(time_ps, synthetic, 1)[0]

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(slopes, [alpha, 1.0 - alpha])
    return float(low), float(high), slopes


def analyse(
    path: Path,
    *,
    dt: float,
    discard_fraction: float,
    bootstrap_samples: int,
    confidence: float,
    block_length_override: int | None,
    max_drift_sigma: float,
    max_relative_drift: float,
    rng: np.random.Generator,
) -> tuple[DriftResult, np.ndarray, np.ndarray, np.ndarray]:
    steps, energies = read_energy_csv(path)
    total_samples = len(energies)

    discard_count = int(math.floor(total_samples * discard_fraction))
    steps = steps[discard_count:]
    energies = energies[discard_count:]

    if len(energies) < 30:
        raise RuntimeError(
            "fewer than 30 samples remain after discarding the transient"
        )

    time_ps = (steps - steps[0]).astype(np.float64) * dt
    analysed_time = float(time_ps[-1] - time_ps[0])

    if analysed_time <= 0.0:
        raise RuntimeError("analysed trajectory has zero physical duration")

    slope, intercept, residual, r_squared = linear_fit(time_ps, energies)

    raw_std = float(np.std(energies, ddof=1))
    detrended_std = float(np.std(residual, ddof=1))
    energy_range = float(np.ptp(energies))
    mean_energy = float(np.mean(energies))
    drift_over_run = float(slope * analysed_time)

    if detrended_std > 0.0:
        drift_to_std = abs(drift_over_run) / detrended_std
    else:
        drift_to_std = 0.0 if drift_over_run == 0.0 else math.inf

    relative_drift = abs(drift_over_run) / max(
        abs(mean_energy), np.finfo(np.float64).tiny
    )

    if block_length_override is None:
        block_length = estimate_block_length(residual)
    else:
        block_length = block_length_override

    if block_length < 1 or block_length > len(residual) // 2:
        raise RuntimeError(
            f"invalid block length {block_length}; expected 1 to "
            f"{len(residual) // 2}"
        )

    ci_low, ci_high, _ = bootstrap_slope_interval(
        time_ps,
        energies,
        slope=slope,
        intercept=intercept,
        residual=residual,
        block_length=block_length,
        samples=bootstrap_samples,
        confidence=confidence,
        rng=rng,
    )

    checks = {
        "slope_ci_contains_zero": ci_low <= 0.0 <= ci_high,
        "abs_drift_over_std_within_limit": drift_to_std <= max_drift_sigma,
        "relative_drift_within_limit": relative_drift <= max_relative_drift,
        "all_values_finite": bool(
            np.all(np.isfinite(energies))
            and np.all(np.isfinite(residual))
        ),
    }

    result = DriftResult(
        source_file=str(path.resolve()),
        dt_ps=dt,
        samples_total=total_samples,
        samples_analysed=len(energies),
        discarded_fraction=discard_fraction,
        analysed_time_ps=analysed_time,
        mean_energy_kj_mol=mean_energy,
        raw_std_kj_mol=raw_std,
        detrended_std_kj_mol=detrended_std,
        energy_range_kj_mol=energy_range,
        slope_kj_mol_ps=slope,
        slope_ci_low_kj_mol_ps=ci_low,
        slope_ci_high_kj_mol_ps=ci_high,
        fitted_drift_over_run_kj_mol=drift_over_run,
        abs_drift_over_detrended_std=drift_to_std,
        relative_drift_over_mean=relative_drift,
        r_squared_linear_trend=r_squared,
        block_length_samples=block_length,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        checks=checks,
        passed=all(checks.values()),
    )

    fitted = slope * time_ps + intercept
    return result, time_ps, energies, fitted


def save_plot(
    path: Path,
    *,
    result: DriftResult,
    time_ps: np.ndarray,
    energies: np.ndarray,
    fitted: np.ndarray,
) -> None:
    residual = energies - fitted

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(time_ps, energies, linewidth=1.0, label="E_tot")
    axes[0].plot(
        time_ps,
        fitted,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"linear fit: {result.slope_kj_mol_ps:.3e} "
            "kJ mol$^{-1}$ ps$^{-1}$"
        ),
    )
    axes[0].set_ylabel("E_tot [kJ/mol]")
    axes[0].set_title("NVE total-energy drift analysis")
    axes[0].grid(True, linestyle=":", alpha=0.7)
    axes[0].legend()

    axes[1].plot(time_ps, residual, linewidth=1.0)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Time [ps]")
    axes[1].set_ylabel("Detrended E_tot [kJ/mol]")
    axes[1].grid(True, linestyle=":", alpha=0.7)

    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check long-run drift of total NVE energy in energy.csv."
    )
    parser.add_argument(
        "energy_csv",
        nargs="?",
        type=Path,
        default=Path("energy.csv"),
        help="CSV file produced by run_cg_md.py; default: energy.csv",
    )
    parser.add_argument(
        "--dt",
        required=True,
        type=positive_float,
        help="MD timestep in ps.",
    )
    parser.add_argument(
        "--discard-fraction",
        type=fraction,
        default=0.10,
        help="Initial fraction discarded as transient; default: 0.10",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of moving-block bootstrap replicates; default: 2000",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Bootstrap confidence level; default: 0.95",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=None,
        help="Override automatic bootstrap block length in samples.",
    )
    parser.add_argument(
        "--max-drift-sigma",
        type=positive_float,
        default=1.0,
        help=(
            "Maximum absolute fitted drift over the analysed run, measured "
            "in detrended standard deviations; default: 1.0"
        ),
    )
    parser.add_argument(
        "--max-relative-drift",
        type=positive_float,
        default=1.0e-6,
        help=(
            "Maximum absolute fitted drift divided by |mean energy|; "
            "default: 1e-6"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260806,
        help="Random seed for bootstrap.",
    )
    parser.add_argument(
        "--output-prefix",
        default="energy_drift",
        help="Prefix for JSON and PNG output files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit status 2 if any drift check fails.",
    )
    args = parser.parse_args()

    if args.bootstrap_samples < 200:
        parser.error("--bootstrap-samples must be at least 200")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")

    path = args.energy_csv.resolve()
    if not path.is_file():
        parser.error(f"file not found: {path}")

    try:
        result, time_ps, energies, fitted = analyse(
            path,
            dt=args.dt,
            discard_fraction=args.discard_fraction,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            block_length_override=args.block_length,
            max_drift_sigma=args.max_drift_sigma,
            max_relative_drift=args.max_relative_drift,
            rng=np.random.default_rng(args.seed),
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    prefix = Path(args.output_prefix)
    if not prefix.is_absolute():
        prefix = path.parent / prefix

    json_path = prefix.with_suffix(".json")
    plot_path = prefix.with_suffix(".png")

    json_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_plot(
        plot_path,
        result=result,
        time_ps=time_ps,
        energies=energies,
        fitted=fitted,
    )

    print("\n=== NVE total-energy drift analysis ===")
    print(f"file                         : {result.source_file}")
    print(f"analysed samples             : {result.samples_analysed}")
    print(f"analysed duration            : {result.analysed_time_ps:.9g} ps")
    print(f"mean(E_tot)                  : {result.mean_energy_kj_mol:.12e} kJ/mol")
    print(f"raw std(E_tot)               : {result.raw_std_kj_mol:.12e} kJ/mol")
    print(
        "detrended std(E_tot)         : "
        f"{result.detrended_std_kj_mol:.12e} kJ/mol"
    )
    print(f"range(E_tot)                 : {result.energy_range_kj_mol:.12e} kJ/mol")
    print(
        "linear drift slope           : "
        f"{result.slope_kj_mol_ps:.12e} kJ/mol/ps"
    )
    print(
        f"{result.confidence:.1%} bootstrap slope CI     : "
        f"[{result.slope_ci_low_kj_mol_ps:.12e}, "
        f"{result.slope_ci_high_kj_mol_ps:.12e}] kJ/mol/ps"
    )
    print(
        "fitted drift over run        : "
        f"{result.fitted_drift_over_run_kj_mol:.12e} kJ/mol"
    )
    print(
        "|drift| / detrended std      : "
        f"{result.abs_drift_over_detrended_std:.6f}"
    )
    print(
        "|drift| / |mean energy|      : "
        f"{result.relative_drift_over_mean:.12e}"
    )
    print(
        "linear-trend R^2             : "
        f"{result.r_squared_linear_trend:.8f}"
    )
    print(f"bootstrap block length       : {result.block_length_samples} samples")

    print("\nChecks:")
    for name, passed in result.checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    print(f"\n[INFO] JSON report: {json_path}")
    print(f"[INFO] Plot:        {plot_path}")

    if result.passed:
        print("[PASS] No significant total-energy drift detected.")
        return 0

    print("[WARN] One or more energy-drift checks failed.")
    return 2 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
