import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))
sys.path.insert(0, str(ROOT / "preprocessing"))

from ibi_timestep_range_diagnostics import (
    _hermite_second_derivative,
    build_matched_prior_variants,
    sigma_range_diagnostics,
    stiffness_ratios,
)


class DummyTable:
    def __init__(self):
        self.x = np.array([0.0, 1.0])
        # U=x^2 represented exactly by cubic Hermite data on [0,1].
        self.energy = np.array([0.0, 1.0])
        self.derivative = np.array([0.0, 2.0])


def _bond(kind="harmonic"):
    row = {"mol_i": 0, "mol_j": 1, "site_i": 0, "site_j": 0, "name": "b", "exclude_wca": True}
    if kind == "harmonic": row.update(type="harmonic", k=100.0, r0=0.5)
    elif kind == "conservative_spline": row.update(type="conservative_spline", file="b.dat", min=0.0, max=1.0, spline_schema="pchip_hermite_v1")
    return row


def _angle(kind="harmonic"):
    row = {"mol_i": 0, "mol_j": 1, "mol_k": 2, "site_i": 0, "site_j": 0, "site_k": 0, "name": "a", "exclude_wca": True}
    if kind == "harmonic": row.update(type="harmonic", k=50.0, theta0=1.0)
    elif kind == "conservative_spline": row.update(type="conservative_spline", file="a.dat", min=0.0, max=np.pi, spline_schema="pchip_hermite_v1")
    return row


def test_matched_variants_restore_old_harmonics(tmp_path):
    common = {"wca": {"epsilon": 1}, "wca_pairs": [], "wca_exclusions": [], "dihedrals": []}
    old = {**common, "bonds": [_bond()], "angles": [_angle()]}
    ibi = {**common, "bonds": [_bond("conservative_spline")], "angles": [_angle("conservative_spline")]}
    (tmp_path / "old.json").write_text(json.dumps(old))
    (tmp_path / "ibi.json").write_text(json.dumps(ibi))
    variants = build_matched_prior_variants(tmp_path / "old.json", tmp_path / "ibi.json")
    assert variants["reference"]["bonds"][0]["type"] == "harmonic"
    assert variants["ibi_bonds_only"]["bonds"][0]["type"] == "conservative_spline"
    assert variants["ibi_bonds_only"]["angles"][0]["type"] == "harmonic"
    assert variants["ibi_angles_only"]["bonds"][0]["type"] == "harmonic"
    assert variants["ibi_angles_only"]["angles"][0]["type"] == "conservative_spline"
    assert variants["full_ibi"]["bonds"][0]["type"] == "conservative_spline"
    assert Path(variants["full_ibi"]["bonds"][0]["file"]).is_absolute()


def test_hermite_second_derivative_exact_quadratic():
    q = np.linspace(0.0, 1.0, 17)
    got = _hermite_second_derivative(DummyTable(), q)
    np.testing.assert_allclose(got, 2.0, rtol=0.0, atol=1e-13)


def test_sigma_range_recovers_quadratic_and_c2_plateau():
    dts = [0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]
    runs = [{"dt_ps": dt, "sigma_E": 123.0 * dt**2} for dt in dts]
    report = sigma_range_diagnostics(runs)
    assert report["fit"]["exponent_p"] == pytest.approx(2.0, abs=1e-12)
    assert report["fit"]["loglog_r2"] == pytest.approx(1.0, abs=1e-12)
    assert report["c2_spread_max_over_min"] == pytest.approx(1.0, abs=1e-12)
    assert all(row["local_exponent_p"] == pytest.approx(2.0, abs=1e-12) for row in report["adjacent_local_exponents"])


def test_stiffness_ratio_frequency_proxy():
    reports = {
        "reference": {"bond": {"p99_abs": 100.0}, "angle": {"p99_abs": 25.0}},
        "full_ibi": {"bond": {"p99_abs": 2500.0}, "angle": {"p99_abs": 100.0}},
    }
    out = stiffness_ratios(reports)["full_ibi"]
    assert out["bond"]["p99_abs_curvature_ratio_vs_reference"] == pytest.approx(25.0)
    assert out["bond"]["sqrt_ratio_frequency_proxy"] == pytest.approx(5.0)
    assert out["angle"]["sqrt_ratio_frequency_proxy"] == pytest.approx(2.0)
