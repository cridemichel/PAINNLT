#!/usr/bin/env python3
"""Static self-test for TEL22 marker/non-bonded vs bonded Morse diagnostic."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SIM = ROOT / "simulation"
SCRIPT_DIR = Path(__file__).resolve().parent
TUTORIAL = SCRIPT_DIR.parents[1]

runner = (SIM / "run_cg_md.py").read_text(encoding="utf-8")
certifier = (SIM / "certify_nve.py").read_text(encoding="utf-8")
interactions = (SIM / "espresso_interactions.py").read_text(encoding="utf-8")
shell = (SCRIPT_DIR / "17_test_nve_morse_runtime_ab.sh").read_text(encoding="utf-8")
summarizer = (SCRIPT_DIR / "17_summarize_morse_runtime_ab.py").read_text(encoding="utf-8")

assert '--pair_specific_morse_runtime' in runner
assert 'default="marker-nonbonded"' in runner
assert 'configure_pair_specific_morse_bonds(' in runner
assert 'marker_nonbonded_morse = bool(' in runner
assert '--pair-specific-morse-runtime' in certifier
assert 'default="marker-nonbonded"' in certifier
assert 'def configure_pair_specific_morse_bonds(' in interactions
assert 'make_analytic_morse_bond(interactions, contact)' in interactions
assert '--pair-specific-morse-runtime bonded-analytic' in shell
assert '--disable-ml' in shell
assert 'NVE_DURATION_PS="${NVE_DURATION_PS:-5.0}"' in shell
assert 'NVE_DTS="${NVE_DTS:-0.002 0.003 0.004 0.005}"' in shell
assert 'same equilibrated.npz' in shell
assert 'energy_gauge_note' in summarizer
assert 'first_block_mean_E' in summarizer
assert 'block_mean_drift_abs = None' in summarizer

priors = json.loads((TUTORIAL / "cg_priors.json").read_text(encoding="utf-8"))
morse = [x for x in priors.get("bonds", []) if str(x.get("type", "harmonic")).lower() == "morse"]
assert len(morse) == 180
assert not priors.get("morse_type_pairs", [])
assert all(math.isclose(float(x.get("r_cut", 15.0)), 15.0) for x in morse)

# Below r_switch, bonded Morse and production Morse have identical forces and
# differ only by the constant +D energy gauge.
for item in morse:
    D = float(item["D"]); a = float(item["a"]); r0 = float(item["r0"])
    rc = float(item.get("r_cut", 15.0))
    rs = float(item.get("r_switch", r0 + 0.75 * (rc - r0)))
    r = 0.5 * (r0 + rs)
    y = math.exp(-a * (r - r0))
    e_marker = D * (y * y - 2.0 * y)
    e_bonded = D * (1.0 - y) ** 2
    f_marker = 2.0 * D * a * y * (y - 1.0)
    f_bonded = -2.0 * a * D * (1.0 - y) * y
    assert math.isclose(f_marker, f_bonded, rel_tol=1e-14, abs_tol=1e-14)
    assert math.isclose(e_bonded - e_marker, D, rel_tol=1e-14, abs_tol=1e-14)

print("[PASS] TEL22 Morse runtime A/B self-test")
print("       production default remains marker-nonbonded")
print("       candidate uses same 180 D/a/r0/r_cut on physical endpoints")
print("       below r_switch: force parity exact; energy differs only by +D/contact")
