#!/usr/bin/env python3
"""Self-contained checks for test-22 uniform Morse A/B/C diagnostic."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

prep_spec = importlib.util.spec_from_file_location("uniform_prep", HERE / "22_prepare_morse_uniform_a0p85.py")
prep = importlib.util.module_from_spec(prep_spec)
assert prep_spec.loader is not None
prep_spec.loader.exec_module(prep)

sum_spec = importlib.util.spec_from_file_location("abc_summary", HERE / "22_summarize_morse_uniform_abc.py")
summ = importlib.util.module_from_spec(sum_spec)
assert sum_spec.loader is not None
sum_spec.loader.exec_module(summ)

# 180 Morse + one harmonic: all and only Morse a values must change.
priors = {
    "bonds": [
        {"type": "morse", "D": 50.0, "a": 0.3, "r0": 1.0 + 0.001 * i, "r_cut": 15.0, "i": i, "j": i + 1}
        for i in range(180)
    ] + [{"type": "harmonic", "k": 12.0, "r0": 1.2, "i": 1000, "j": 1001}],
    "morse_type_pairs": [],
}
derived, changed = prep.build_uniform_variant(priors)
assert changed == list(range(180))
for i in range(180):
    assert derived["bonds"][i]["D"] == 50.0
    assert derived["bonds"][i]["r0"] == priors["bonds"][i]["r0"]
    assert math.isclose(derived["bonds"][i]["a"], 0.255, abs_tol=1e-15)
assert derived["bonds"][180] == priors["bonds"][180]
assert math.isclose(prep.SCALE * prep.SCALE, 0.7225, abs_tol=1e-15)

# Perfect second-order synthetic rows stay perfect through the imported test-21 metrics.
rows = []
for dt in summ.robust.EXPECTED_DTS:
    sigma = 11.0 * dt * dt
    rows.append({
        "dt_ps": dt,
        "sigma_E": sigma,
        "C2_sigma_over_dt2": 11.0,
        "relative_block_mean_drift": 1e-8,
    })
metrics = summ.robust.summarize_rows(rows)
assert math.isclose(metrics["exponent_p"], 2.0, abs_tol=1e-12)
assert math.isclose(metrics["c2_spread_max_over_min"], 1.0, abs_tol=1e-12)
assert math.isclose(metrics["local_exponent_range"], 0.0, abs_tol=1e-12)

runner = (HERE / "22_test_nve_morse_uniform_abc.sh").read_text(encoding="utf-8")
for token in (
    "A production",
    "B selective",
    "C uniform",
    "180/180 Morse a=0.255",
    "0.001 0.0015 0.002 0.003 0.004 0.005",
    "NVE_DURATION_PS:-10.0",
    "--disable-ml",
    "marker-nonbonded",
    "--dry-run --reuse-existing",
    "only C performs new MD",
):
    assert token in runner, token

print("[PASS] Morse uniform-a0.85 A/B/C diagnostic selftest")
