import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))
sys.path.insert(0, str(ROOT / "ibi"))
sys.path.insert(0, str(ROOT / "preprocessing"))

from conservative_spline import load_conservative_spline, save_conservative_spline
from generate_angle_smoothing_candidate import generate_candidate, smoothing_name
from ibi_angle_candidate_validation import fit_sigma_range
from ibi_angle_smoothing_sweep import clean_prefix_metrics, nve_rank_key, subset_runs


def test_smoothing_name_is_stable_for_local_grid():
    assert smoothing_name(0.0075) == "smooth_0p0075_wall_current"
    assert smoothing_name(0.01) == "smooth_0p01_wall_current"
    assert smoothing_name(0.0125) == "smooth_0p0125_wall_current"
    assert smoothing_name(0.015) == "smooth_0p015_wall_current"


def test_generate_candidate_is_self_contained_and_unvalidated(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    x = np.linspace(0.0, np.pi, 101)
    u_angle = 0.5 * 80.0 * (x - 1.8) ** 2 + 0.08 * np.sin(90.0 * x)
    du_angle = np.gradient(u_angle, x)
    save_conservative_spline(source_dir / "angle.dat", x, u_angle, du_angle)
    xb = np.linspace(0.8, 1.2, 41)
    ub = 0.5 * 300.0 * (xb - 1.0) ** 2
    dub = 300.0 * (xb - 1.0)
    save_conservative_spline(source_dir / "bond.dat", xb, ub, dub)
    priors = {
        "bonds": [{"type": "conservative_spline", "name": "bb_X_X", "file": "bond.dat", "min": 0.8, "max": 1.2}],
        "angles": [{"type": "conservative_spline", "name": "ang_X_X_X", "file": "angle.dat", "min": 0.0, "max": float(np.pi)}],
        "dihedrals": [],
    }
    source_priors = source_dir / "cg_priors.json"
    source_priors.write_text(json.dumps(priors))
    cfg = tmp_path / "ibi_settings.json"
    cfg.write_text(json.dumps({"angle": {"wall_width": 0.1, "wall_k": 5000.0}}))
    out = tmp_path / "candidate"
    manifest = generate_candidate(
        source_priors=source_priors,
        ibi_config=cfg,
        body_sigma_rad=0.0125,
        output_dir=out,
    )
    payload = json.loads((out / "cg_priors.json").read_text())
    assert manifest["validated"] is False
    assert payload["regularization_candidate"]["validated"] is False
    assert math.isclose(payload["regularization_candidate"]["body_sigma_rad"], 0.0125)
    assert (out / payload["bonds"][0]["file"]).is_file()
    assert (out / payload["angles"][0]["file"]).is_file()
    table = load_conservative_spline(payload["angles"][0], kind="angle", priors_path=out / "cg_priors.json")
    assert np.all(np.isfinite(table.energy))
    assert np.all(np.isfinite(table.derivative))


def test_subset_runs_and_clean_prefix_use_common_grid():
    runs = [
        {"status": "ok", "dt_ps": 0.001, "sigma_E": 10 * 0.001**2},
        {"status": "ok", "dt_ps": 0.0015, "sigma_E": 99.0},
        {"status": "ok", "dt_ps": 0.002, "sigma_E": 11 * 0.002**2},
        {"status": "ok", "dt_ps": 0.003, "sigma_E": 12 * 0.003**2},
        {"status": "ok", "dt_ps": 0.004, "sigma_E": 30 * 0.004**2},
    ]
    common = subset_runs(runs, [0.001, 0.002, 0.003, 0.004])
    assert [r["dt_ps"] for r in common] == [0.001, 0.002, 0.003, 0.004]
    fit = fit_sigma_range(common)
    assert fit["max_clean_dt_factor_1p5"] == 0.003
    prefix = clean_prefix_metrics(fit)
    assert prefix["n_points"] == 3
    assert prefix["max_dt_ps"] == 0.003
    assert prefix["c2_spread_max_over_min"] == 1.2


def test_nve_ranking_prefers_longer_contiguous_c2_plateau_over_global_p():
    def row(clean, prefix_spread, prefix_p, global_spread, global_p, r2):
        return {
            "sigma_range_common_grid": {
                "max_clean_dt_factor_1p5": clean,
                "fit": {"exponent_p": global_p, "loglog_r2": r2},
                "c2_spread_max_over_min": global_spread,
            },
            "clean_prefix_factor_1p5": {
                "c2_spread_max_over_min": prefix_spread,
                "abs_p_minus_2": abs(prefix_p - 2.0),
            },
        }
    globally_pretty_but_short = ("pretty", row(0.002, 1.02, 2.0, 1.1, 2.0, 0.999))
    longer_plateau = ("plateau", row(0.004, 1.30, 1.8, 2.0, 1.8, 0.95))
    assert nve_rank_key(longer_plateau) < nve_rank_key(globally_pretty_but_short)
