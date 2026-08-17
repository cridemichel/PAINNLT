import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from ibi_angle_final_validation import fixed_effects_sigma_slope, replica_gate, structural_gate


def _rep(name, amp=1.0, p=2.0, clean=0.005, spread=1.2, drift=1e-6):
    dts = [0.001, 0.002, 0.003, 0.004, 0.005]
    sig = [amp * d**p for d in dts]
    return {
        "name": name,
        "max_relative_block_drift": drift,
        "sigma_range": {
            "available": True,
            "dt_ps": dts,
            "sigma_E": sig,
            "max_clean_dt_factor_1p5": clean,
            "c2_spread_max_over_min": spread,
        },
    }


def test_fixed_effects_slope_ignores_replica_amplitudes():
    fit = fixed_effects_sigma_slope([_rep("a", 1.0), _rep("b", 17.0), _rep("c", 0.03)])
    assert fit["exponent_p"] == pytest.approx(2.0, abs=1e-12)
    assert fit["within_replica_r2"] == pytest.approx(1.0, abs=1e-12)


def test_replica_gate_requires_reproducible_clean_range():
    rows = [_rep("a"), _rep("b"), _rep("c", clean=0.003)]
    gate = replica_gate(
        rows,
        common_p_min=1.8, common_p_max=2.2, common_r2_min=0.95,
        full_clean_dt=0.005, min_clean_dt=0.003, min_full_clean_replicas=2,
        median_c2_spread_max=2.0, max_relative_block_drift=2e-5,
    )
    assert gate["pass"] is True
    rows[1]["sigma_range"]["max_clean_dt_factor_1p5"] = 0.002
    gate2 = replica_gate(
        rows,
        common_p_min=1.8, common_p_max=2.2, common_r2_min=0.95,
        full_clean_dt=0.005, min_clean_dt=0.003, min_full_clean_replicas=2,
        median_c2_spread_max=2.0, max_relative_block_drift=2e-5,
    )
    assert gate2["pass"] is False
    assert gate2["checks"]["all_replicas_min_clean_dt"] is False


def _long(angle=1.0, bond=0.7, kin=100.0, group_shift=0.0, u2=1000.0):
    return {
        "nvt_kinetic_energy": {"mean_E_kin_second_half": kin},
        "structural": {
            "angle_curvature_runtime": {"p99_abs": u2},
            "angles": {
                "summary": {"weighted_mean_l1": angle},
                "groups": {
                    "a": {"l1_runtime_vs_target": angle + group_shift},
                    "b": {"l1_runtime_vs_target": angle - group_shift},
                },
            },
            "bonds": {"summary": {"weighted_mean_l1": bond}},
        },
    }


def test_structural_gate_matched_deltas():
    cur = _long()
    cand = _long(angle=1.02, bond=0.69, kin=101.0, group_shift=0.02, u2=600.0)
    gate = structural_gate(
        cur, cand,
        weighted_angle_delta_max=0.03, weighted_bond_delta_max=0.03,
        max_group_angle_delta_max=0.10, kinetic_relative_delta_max=0.03, curvature_reduction_min=1.2,
    )
    assert gate["pass"] is True
    bad = _long(angle=1.05, bond=0.69, kin=101.0, u2=600.0)
    gate2 = structural_gate(
        cur, bad,
        weighted_angle_delta_max=0.03, weighted_bond_delta_max=0.03,
        max_group_angle_delta_max=0.10, kinetic_relative_delta_max=0.03, curvature_reduction_min=1.2,
    )
    assert gate2["pass"] is False
