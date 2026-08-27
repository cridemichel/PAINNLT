#!/usr/bin/env python3
"""Static/self-contained checks for TEL22 Morse top-10% a-softening diagnostic."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = Path(__file__).with_name("19_prepare_morse_top10_a_softening.py")
spec = importlib.util.spec_from_file_location("morse_a_prepare", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

assert mod.SCALES == (0.90, 0.80, 0.70)
assert [mod.variant_name(x) for x in mod.SCALES] == ["top10_a0p90", "top10_a0p80", "top10_a0p70"]

fixture = {
    "bonds": [
        {"type": "morse", "D": 50.0, "a": 0.3, "r0": 1.0, "mol_i": 0, "mol_j": 1},
        {"type": "harmonic", "k": 100.0, "r0": 1.0, "mol_i": 1, "mol_j": 2},
        {"type": "morse", "D": 50.0, "a": 0.3, "r0": 1.2, "mol_i": 2, "mol_j": 3},
    ],
    "angles": [{"type": "harmonic", "k": 3.0}],
    "dihedrals": [],
}
for scale in mod.SCALES:
    out = mod.build_scaled_variant(fixture, [0, 2], scale)
    assert out["bonds"][0]["a"] == 0.3 * scale
    assert out["bonds"][2]["a"] == 0.3 * scale
    assert out["bonds"][0]["D"] == 50.0 and out["bonds"][2]["D"] == 50.0
    assert out["bonds"][0]["r0"] == 1.0 and out["bonds"][2]["r0"] == 1.2
    assert out["bonds"][1] == fixture["bonds"][1]
    assert out["angles"] == fixture["angles"]
    assert fixture["bonds"][0]["a"] == 0.3

priors = ROOT / "tutorials/tel22/cg_priors.json"
ranking = ROOT / "tutorials/tel22/diagnostics/nve/nve_morse_curvature_quantiles_coarse_5ps/inputs/curvature_quantile_inputs.json"
if priors.is_file() and ranking.is_file():
    p = json.loads(priors.read_text(encoding="utf-8"))
    r = json.loads(ranking.read_text(encoding="utf-8"))
    selected = mod.selected_indices_from_ranking(r, mod.sha256_file(priors))
    assert len(selected) == 18
    for idx in selected:
        e = p["bonds"][idx]
        assert str(e.get("type", "harmonic")).lower() == "morse"
        assert float(e["D"]) == 50.0 and float(e["a"]) == 0.3

runner = Path(__file__).with_name("19_test_nve_morse_top10_a_softening.sh").read_text(encoding="utf-8")
for token in ("top10_a0p90", "top10_a0p80", "top10_a0p70", "bonded-analytic", "--disable-ml"):
    assert token in runner
print("[PASS] Morse top-10% a-softening diagnostic selftest")
