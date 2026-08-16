import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from ibi_angle_candidate_validation import fit_sigma_range, largest_clean_dt, _summarize_l1


def test_fit_sigma_range_recovers_quadratic_scaling():
    runs = []
    for dt in [0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]:
        runs.append({"status": "ok", "dt_ps": dt, "sigma_E": 7.5 * dt * dt})
    result = fit_sigma_range(runs)
    assert result["available"] is True
    assert math.isclose(result["fit"]["exponent_p"], 2.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(result["fit"]["loglog_r2"], 1.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(result["c2_spread_max_over_min"], 1.0, rel_tol=0, abs_tol=1e-12)
    assert result["max_clean_dt_factor_1p5"] == 0.005


def test_largest_clean_dt_stops_at_first_noncontiguous_c2_failure():
    dt = np.array([0.001, 0.0015, 0.002, 0.003])
    c2 = np.array([10.0, 12.0, 16.0, 11.0])
    assert largest_clean_dt(dt, c2, 1.5) == 0.0015
    assert largest_clean_dt(dt, c2, 2.0) == 0.003


def test_fit_sigma_range_ignores_failed_runs():
    runs = [
        {"status": "ok", "dt_ps": 0.001, "sigma_E": 1e-6},
        {"status": "ok", "dt_ps": 0.002, "sigma_E": 4e-6},
        {"status": "failed", "dt_ps": 0.003},
        {"status": "ok", "dt_ps": 0.004, "sigma_E": 16e-6},
    ]
    result = fit_sigma_range(runs)
    assert result["available"] is True
    assert result["n_points"] == 3
    assert abs(result["fit"]["exponent_p"] - 2.0) < 1e-12


def test_structural_l1_summary_is_target_count_weighted():
    rows = {
        "a": {"l1_runtime_vs_target": 0.2, "target_samples": 100},
        "b": {"l1_runtime_vs_target": 0.8, "target_samples": 300},
    }
    out = _summarize_l1(rows)
    assert out["n_groups"] == 2
    assert math.isclose(out["weighted_mean_l1"], 0.65)
    assert math.isclose(out["mean_l1"], 0.5)
    assert math.isclose(out["max_l1"], 0.8)
