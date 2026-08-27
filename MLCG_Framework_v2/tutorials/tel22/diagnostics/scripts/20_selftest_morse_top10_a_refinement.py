#!/usr/bin/env python3
"""Self-contained checks for TEL22 Morse top-10% a-refinement diagnostic."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent

pspec = importlib.util.spec_from_file_location("morse_refine_prepare", HERE / "20_prepare_morse_top10_a_refinement.py")
prep = importlib.util.module_from_spec(pspec)
assert pspec.loader is not None
pspec.loader.exec_module(prep)

sspec = importlib.util.spec_from_file_location("morse_refine_summary", HERE / "20_summarize_morse_top10_a_refinement.py")
summ = importlib.util.module_from_spec(sspec)
assert sspec.loader is not None
sspec.loader.exec_module(summ)

assert prep.SCALE_SPECS == (
    ("top10_a0p950", 0.950),
    ("top10_a0p925", 0.925),
    ("top10_a0p875", 0.875),
    ("top10_a0p850", 0.850),
)
assert prep.CENTER_SCALE == 0.900

fixture = {
    "bonds": [
        {"type": "morse", "D": 50.0, "a": 0.3, "r0": 1.0, "mol_i": 0, "mol_j": 1},
        {"type": "harmonic", "k": 100.0, "r0": 1.0, "mol_i": 1, "mol_j": 2},
        {"type": "morse", "D": 50.0, "a": 0.3, "r0": 1.2, "mol_i": 2, "mol_j": 3},
    ],
    "angles": [{"type": "harmonic", "k": 3.0}],
    "dihedrals": [],
}
for _name, scale in prep.SCALE_SPECS:
    out = prep.base.build_scaled_variant(fixture, [0, 2], scale)
    assert abs(out["bonds"][0]["a"] - 0.3 * scale) < 1e-15
    assert abs(out["bonds"][2]["a"] - 0.3 * scale) < 1e-15
    assert out["bonds"][0]["D"] == 50.0 and out["bonds"][2]["D"] == 50.0
    assert out["bonds"][0]["r0"] == 1.0 and out["bonds"][2]["r0"] == 1.2
    assert out["bonds"][1] == fixture["bonds"][1]
    assert out["angles"] == fixture["angles"]

# The selection policy must prefer flatter C2/local structure over p alone.
a = {"c2_spread_max_over_min": 1.20, "local_exponent_range": 0.80, "loglog_r2": 0.99, "abs_p_minus_2": 0.20}
b = {"c2_spread_max_over_min": 1.30, "local_exponent_range": 0.10, "loglog_r2": 1.00, "abs_p_minus_2": 0.00}
assert summ.regularity_key(a) < summ.regularity_key(b)

runner19 = (HERE / "19_test_nve_morse_top10_a_softening.sh").read_text(encoding="utf-8")
assert 'dry-run) cmd+=(--dry-run --reuse-existing) ;;' in runner19
assert '("${MODE}" == "resume" || "${MODE}" == "dry-run")' in runner19

runner20 = (HERE / "20_test_nve_morse_top10_a_refinement.sh").read_text(encoding="utf-8")
for token in (
    "top10_a0p950", "top10_a0p925", "top10_a0p875", "top10_a0p850",
    "top10_a0p90", "bonded-analytic", "--disable-ml", "--dry-run --reuse-existing",
):
    assert token in runner20
print("[PASS] Morse top-10% a-refinement diagnostic selftest")
