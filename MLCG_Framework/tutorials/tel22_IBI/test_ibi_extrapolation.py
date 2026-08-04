#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "ibi" / "run_ibi_loop.py"
spec = importlib.util.spec_from_file_location("run_ibi_loop", MODULE)
ibi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ibi)

settings = ibi.load_ibi_settings(Path(__file__).with_name("ibi_extrapolation_config.json"))
rng = np.random.default_rng(20260804)
target = rng.normal(1.25, 0.12, 50000)
target = target[(target > 0.2) & (target < 2.2)]
bins = np.linspace(settings["bond"]["hist_min"], settings["bond"]["hist_max"], settings["bond"]["hist_edges"])
grid = np.linspace(settings["bond"]["table_min"], settings["bond"]["table_max"], settings["bond"]["table_points"])

x, u0, f0, p0, hx, c0, support = ibi.calculate_dbi_potential(
    target, bins, grid, jacobian_type="bond", settings=settings
)
ibi.validate_extrapolated_table(x, u0, f0, "bond", support)
assert x[-1] == settings["bond"]["table_max"]
assert np.mean(f0[:20]) > 0.0
assert np.mean(f0[-20:]) < 0.0
assert u0[-1] > u0[np.searchsorted(x, support[1])]

simulated = rng.normal(1.31, 0.14, 40000)
sim_counts, p_sim, _ = ibi.histogram_density(simulated, bins)
_, u1, f1 = ibi.update_ibi_potential(
    u0, p_sim, p0, hx, x, c0, sim_counts,
    periodic=False, target_type="bond", settings=settings
)
ibi.validate_extrapolated_table(x, u1, f1, "bond", support)
assert np.max(np.abs(u1 - u0)) > 1e-6
assert np.mean(f1[-20:]) < 0.0
assert np.isfinite(u1).all() and np.isfinite(f1).all()

print("PASS: support-aware IBI update and C1 exponential bond tails")
print(f"support={support[0]:.6f}..{support[1]:.6f} nm")
print(f"edge forces: left={np.mean(f1[:20]):.6f}, right={np.mean(f1[-20:]):.6f}")
