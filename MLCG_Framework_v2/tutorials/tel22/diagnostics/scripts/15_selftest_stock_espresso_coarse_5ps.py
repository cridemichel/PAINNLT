#!/usr/bin/env python3
"""Source-level checks for the TEL22 stock-ESPResSo coarse NVE diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TUTORIAL = ROOT / "tutorials" / "tel22"
SCRIPTS = TUTORIAL / "diagnostics" / "scripts"


def main() -> int:
    runner = (SCRIPTS / "15_test_nve_stock_espresso_coarse_5ps.sh").read_text(encoding="utf-8")
    summarizer = (SCRIPTS / "15_summarize_stock_espresso_coarse_5ps.py").read_text(encoding="utf-8")
    md = (ROOT / "simulation" / "run_cg_md.py").read_text(encoding="utf-8")
    priors = json.loads((TUTORIAL / "cg_priors.json").read_text(encoding="utf-8"))

    bond_types = [str(x.get("type", "harmonic")).lower() for x in priors.get("bonds", [])]
    angle_types = [str(x.get("type", "harmonic")).lower() for x in priors.get("angles", [])]
    assert bond_types.count("harmonic") == 210
    assert bond_types.count("morse") == 180
    assert set(angle_types) == {"harmonic"} and len(angle_types) == 200
    assert priors.get("dihedrals", []) == []
    assert len(priors.get("wca_pairs", {})) == 36

    assert "11_prepare_prior_ablation.py" in runner
    assert 'm["variants"]["no_morse"]' in runner
    assert "--disable-ml" in runner
    assert "NVE_DTS=\"${NVE_DTS:-0.002 0.003 0.004 0.005}\"" in runner
    assert "NVE_DURATION_PS=\"${NVE_DURATION_PS:-5.0}\"" in runner
    assert "wca_pairs" in runner
    assert "stock ESPResSo LennardJones" in runner
    assert "system.non_bonded_inter[type_i, type_j].lennard_jones.set_params" in md
    assert "espressomd.interactions.HarmonicBond" in md
    assert "conservative_classical_model_provenance_ml_disabled" in summarizer
    assert "custom_morse_active\": False" in summarizer
    print("[PASS] TEL22 stock-ESPResSo coarse-dt diagnostic self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
