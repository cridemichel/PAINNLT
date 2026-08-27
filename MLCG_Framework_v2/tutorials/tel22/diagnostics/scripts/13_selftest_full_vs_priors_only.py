#!/usr/bin/env python3
"""Lightweight source-level checks for the final TEL22 full/prior-only A/B."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "tutorials" / "tel22" / "diagnostics" / "scripts"


def main() -> int:
    runner = (SCRIPTS / "13_test_nve_priors_only.sh").read_text(encoding="utf-8")
    compare = (SCRIPTS / "13_compare_full_vs_priors_only.py").read_text(encoding="utf-8")
    validator = (SCRIPTS / "13_validate_full_baseline.py").read_text(encoding="utf-8")
    certifier = (ROOT / "simulation" / "certify_nve.py").read_text(encoding="utf-8")
    md = (ROOT / "simulation" / "run_cg_md.py").read_text(encoding="utf-8")

    assert "--disable-ml" in runner
    assert "--disable-ml" in certifier
    assert "--disable_ml" in certifier
    assert "--disable_ml" in md
    assert "cg_priors.json" in runner
    assert "equilibrated.npz" in runner
    assert "tel22_model.pt" in runner
    assert "conservative_classical_model_provenance_ml_disabled" in compare
    assert "run_plan.json" in validator
    assert "--ml_precision" in validator
    print("[PASS] TEL22 full vs priors-only diagnostic self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
