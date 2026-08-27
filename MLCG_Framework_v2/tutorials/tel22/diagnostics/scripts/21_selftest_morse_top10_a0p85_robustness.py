#!/usr/bin/env python3
"""Self-contained checks for test-21 Morse a=0.85 robustness diagnostic."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "morse_a085_robust_summary", HERE / "21_summarize_morse_top10_a0p85_robustness.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

assert mod.EXPECTED_DTS == [0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]
rows = []
for dt in mod.EXPECTED_DTS:
    sigma = 7.5 * dt * dt
    rows.append({
        "dt_ps": dt,
        "sigma_E": sigma,
        "C2_sigma_over_dt2": sigma / (dt * dt),
        "relative_block_mean_drift": 1e-8,
    })
s = mod.summarize_rows(rows)
assert math.isclose(s["exponent_p"], 2.0, abs_tol=1e-12)
assert math.isclose(s["loglog_r2"], 1.0, abs_tol=1e-12)
assert math.isclose(s["c2_spread_max_over_min"], 1.0, abs_tol=1e-12)
assert math.isclose(s["local_exponent_range"], 0.0, abs_tol=1e-12)

runner = (HERE / "21_test_nve_morse_top10_a0p85_robustness.sh").read_text(encoding="utf-8")
for token in (
    "0.001 0.0015 0.002 0.003 0.004 0.005",
    "NVE_DURATION_PS:-10.0",
    "top10_a0p850",
    "marker-nonbonded",
    "--disable-ml",
    "--dry-run --reuse-existing",
    "Morse role",
):
    assert token in runner, token
print("[PASS] Morse a=0.85 10 ps full-grid robustness diagnostic selftest")
