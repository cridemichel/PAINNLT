#!/usr/bin/env python3
"""Apply the NVE sigma(E) timestep-scaling update to MLCG_Framework_v2.

Changes:
  * fixed physical duration remains 5 ps (existing default)
  * default dt range becomes 0.001, 0.002, 0.005, 0.01 ps
  * energy is logged every integration step
  * timestep scaling uses sigma_E = std(E(t)) as the primary error metric
  * rms_dE remains available from nve_analysis.py as a secondary diagnostic
  * adds a regression test for sigma_E ~ dt^2

The patcher is intentionally strict and aborts instead of silently guessing if the
expected current NVE source layout is not found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path.cwd()
SIM = ROOT / "simulation"
TESTS = ROOT / "tests"
TEL22 = ROOT / "tutorials" / "tel22"


def die(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


def read(path: Path) -> str:
    if not path.is_file():
        die(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def write_if_changed(path: Path, old: str, new: str) -> bool:
    if new == old:
        print(f"[SKIP] {path}: already patched")
        return False
    path.write_text(new, encoding="utf-8")
    print(f"[PATCH] {path}")
    return True


# ---------------------------------------------------------------------------
# simulation/nve_analysis.py
# ---------------------------------------------------------------------------
analysis_path = SIM / "nve_analysis.py"
analysis_old = read(analysis_path)
analysis_new = analysis_old

if "def energy_standard_deviation(" not in analysis_new:
    if not re.search(r"^import numpy as np\s*$", analysis_new, flags=re.M):
        # nve_analysis currently uses NumPy in the NVE certification implementation;
        # keep this fallback for robustness.
        m = list(re.finditer(r"^(?:from\s+\S+\s+import\s+.*|import\s+.*)\n", analysis_new, flags=re.M))
        if not m:
            die("could not locate import block in simulation/nve_analysis.py")
        pos = m[-1].end()
        analysis_new = analysis_new[:pos] + "import numpy as np\n" + analysis_new[pos:]

    helper = r'''


def energy_standard_deviation(energies):
    """Population standard deviation of the sampled total energy time series.

    For a fixed physical NVE trajectory duration this is the primary quantity
    used for the velocity-Verlet timestep-scaling test, sigma_E ~ dt**2.
    """
    values = np.asarray(energies, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("At least three energy samples are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("Energy series contains NaN or Inf")
    return float(np.std(values, ddof=0))
'''
    analysis_new = analysis_new.rstrip() + helper + "\n"

write_if_changed(analysis_path, analysis_old, analysis_new)


# ---------------------------------------------------------------------------
# simulation/certify_nve.py
# ---------------------------------------------------------------------------
cert_path = SIM / "certify_nve.py"
cert_old = read(cert_path)
cert_new = cert_old

# Import the new metric helper.
if "energy_standard_deviation" not in cert_new:
    m = re.search(r"^from nve_analysis import ([^\n]+)$", cert_new, flags=re.M)
    if m:
        names = m.group(1).strip()
        repl = f"from nve_analysis import {names}, energy_standard_deviation"
        cert_new = cert_new[:m.start()] + repl + cert_new[m.end():]
    else:
        # Fallback if imports are grouped differently.
        m = list(re.finditer(r"^(?:from\s+\S+\s+import\s+.*|import\s+.*)\n", cert_new, flags=re.M))
        if not m:
            die("could not locate import block in simulation/certify_nve.py")
        pos = m[-1].end()
        cert_new = cert_new[:pos] + "from nve_analysis import energy_standard_deviation\n" + cert_new[pos:]

# Default dt grid: span one decade without entering sub-femtosecond territory.
old_dt_literals = [
    "0.002, 0.001, 0.0005, 0.00025",
    "0.002 0.001 0.0005 0.00025",
]
for lit in old_dt_literals:
    if lit in cert_new:
        cert_new = cert_new.replace(lit, "0.001, 0.002, 0.005, 0.01" if "," in lit else "0.001 0.002 0.005 0.01")

# Force one energy sample per integration step.  The current implementation
# computes log_every from steps (historically steps//5); replace the assignment
# irrespective of the exact RHS.
log_matches = list(re.finditer(r"^(\s*)log_every\s*=\s*[^\n]+$", cert_new, flags=re.M))
if not log_matches:
    die("could not locate log_every assignment in simulation/certify_nve.py")
# Only the run-planning assignment should exist; if multiple are present, patch
# all assignments to preserve the invariant that certification samples every step.
cert_new = re.sub(
    r"^(\s*)log_every\s*=\s*[^\n]+$",
    r"\1log_every = 1  # NVE certification: sample total energy every integration step",
    cert_new,
    flags=re.M,
)

# Add sigma_E to each analyzed run immediately after the existing analysis call.
if 'metrics["sigma_E"]' not in cert_new:
    pat = re.compile(r"^(\s*)metrics\s*=\s*analyze_energy_series\(([^\n]+)\)\s*$", flags=re.M)
    m = pat.search(cert_new)
    if not m:
        die("could not locate analyze_energy_series(...) call in simulation/certify_nve.py")
    line = m.group(0)
    indent = m.group(1)
    insertion = line + f'\n{indent}metrics["sigma_E"] = energy_standard_deviation(energies)'
    cert_new = cert_new[:m.start()] + insertion + cert_new[m.end():]

# The certification fit and terminal summary previously used rms_dE.  Switch all
# certifier-side references to sigma_E. nve_analysis still retains rms_dE in its
# per-run metrics as a secondary diagnostic.
cert_new = cert_new.replace("rms_dE", "sigma_E")

# Make the metric explicit in the JSON report if there is a report dict with a
# thresholds/fit section; this insertion is optional and harmless if absent.
if '"scaling_metric"' not in cert_new:
    # Insert beside the first duration field in a dict literal if recognizable.
    cert_new, _ = re.subn(
        r'(?m)^(\s*)"duration_ps"\s*:\s*args\.duration_ps,\s*$',
        r'\1"duration_ps": args.duration_ps,\n\1"scaling_metric": "sigma_E_std_total_energy",\n\1"energy_log_every_steps": 1,',
        cert_new,
        count=1,
    )

write_if_changed(cert_path, cert_old, cert_new)


# ---------------------------------------------------------------------------
# tutorials/tel22/06_certify_nve.sh
# ---------------------------------------------------------------------------
wrapper_path = TEL22 / "06_certify_nve.sh"
wrapper_old = read(wrapper_path)
wrapper_new = wrapper_old
wrapper_new = wrapper_new.replace(
    "0.002 0.001 0.0005 0.00025",
    "0.001 0.002 0.005 0.01",
)
wrapper_new = wrapper_new.replace(
    "0.002,0.001,0.0005,0.00025",
    "0.001,0.002,0.005,0.01",
)
write_if_changed(wrapper_path, wrapper_old, wrapper_new)


# ---------------------------------------------------------------------------
# Regression test: add a separate file to avoid brittle edits to existing tests.
# ---------------------------------------------------------------------------
test_path = TESTS / "test_nve_sigma_scaling.py"
test_text = r'''import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIMULATION = ROOT / "simulation"
if str(SIMULATION) not in sys.path:
    sys.path.insert(0, str(SIMULATION))

from nve_analysis import energy_standard_deviation, fit_timestep_scaling


class TestNveSigmaScaling(unittest.TestCase):
    def test_energy_standard_deviation(self):
        values = np.array([1.0, 2.0, 3.0], dtype=float)
        self.assertAlmostEqual(
            energy_standard_deviation(values),
            float(np.std(values, ddof=0)),
            places=15,
        )

    def test_sigma_energy_recovers_dt_squared(self):
        dts = np.array([0.001, 0.002, 0.005, 0.01], dtype=float)
        phase = np.linspace(0.0, 40.0 * np.pi, 20001)
        carrier = np.sin(phase) + 0.35 * np.cos(0.37 * phase)
        sigmas = []
        for dt in dts:
            energies = 2500.0 + (dt ** 2) * carrier
            sigmas.append(energy_standard_deviation(energies))

        fit = fit_timestep_scaling(dts, sigmas)
        self.assertAlmostEqual(fit["p"], 2.0, places=8)
        self.assertGreater(fit["r2"], 0.9999999999)

    def test_certifier_samples_every_step_and_uses_sigma(self):
        source = (SIMULATION / "certify_nve.py").read_text(encoding="utf-8")
        self.assertIn("log_every = 1", source)
        self.assertIn('metrics["sigma_E"]', source)
        self.assertIn("energy_standard_deviation", source)
        self.assertNotIn("rms_dE=", source)


if __name__ == "__main__":
    unittest.main()
'''
if test_path.exists() and test_path.read_text(encoding="utf-8") == test_text:
    print(f"[SKIP] {test_path}: already patched")
else:
    test_path.write_text(test_text, encoding="utf-8")
    print(f"[PATCH] {test_path}")

print("[DONE] NVE sigma(E) scaling patch applied")
print("       default dt: 0.001 0.002 0.005 0.01 ps")
print("       duration:   existing 5 ps default")
print("       sampling:   every integration step")
print("       fit metric: sigma_E = std(E_total)")
