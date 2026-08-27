#!/usr/bin/env python3
"""Static/self-contained checks for the TEL22 Morse curvature-quantile diagnostic."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = Path(__file__).with_name("18_prepare_morse_curvature_quantiles.py")
spec = importlib.util.spec_from_file_location("morse_curv_prepare", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def U(r, D, a, r0):
    q = math.exp(-a * (r - r0))
    return D * (q * q - 2 * q)


for r in (0.8, 1.0, 1.4, 2.0):
    D, a, r0 = 50.0, 0.3, 1.2
    m = mod.local_curvature_metrics({"D": D, "a": a, "r0": r0}, r)
    h = 1e-5
    d1 = (U(r + h, D, a, r0) - U(r - h, D, a, r0)) / (2 * h)
    d2 = (U(r + h, D, a, r0) - 2 * U(r, D, a, r0) + U(r - h, D, a, r0)) / (h * h)
    assert abs(m["dU_dr"] - d1) < 2e-7
    assert abs(m["radial_curvature"] - d2) < 2e-4

priors = ROOT / "tutorials/tel22/cg_priors.json"
if priors.is_file():
    import json
    p = json.loads(priors.read_text())
    ms = [e for e in p.get("bonds", []) if str(e.get("type", "harmonic")).lower() == "morse"]
    assert len(ms) == 180
    Ds = {float(e["D"]) for e in ms}
    As = {float(e["a"]) for e in ms}
    K = {2 * float(e["D"]) * float(e["a"]) ** 2 for e in ms}
    assert Ds == {50.0} and As == {0.3} and K == {9.0}, (Ds, As, K)

runner = Path(__file__).with_name("18_test_nve_morse_curvature_quantiles.sh").read_text()
assert "bonded-analytic" in runner
assert "--disable-ml" in runner
assert "top_05pct_zeroD" in runner and "top_10pct_zeroD" in runner and "top_20pct_zeroD" in runner
print("[PASS] Morse curvature-quantile diagnostic selftest")
