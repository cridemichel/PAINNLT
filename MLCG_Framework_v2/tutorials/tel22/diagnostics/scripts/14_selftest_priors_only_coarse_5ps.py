#!/usr/bin/env python3
"""Source-level checks for the TEL22 priors-only 5 ps coarse-dt diagnostic."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "tutorials" / "tel22" / "diagnostics" / "scripts"


def main() -> int:
    runner = (SCRIPTS / "14_test_nve_priors_only_coarse_5ps.sh").read_text(encoding="utf-8")
    summarizer = (SCRIPTS / "14_summarize_priors_only_coarse_5ps.py").read_text(encoding="utf-8")
    certifier = (ROOT / "simulation" / "certify_nve.py").read_text(encoding="utf-8")
    md = (ROOT / "simulation" / "run_cg_md.py").read_text(encoding="utf-8")

    assert "NVE_DTS=\"${NVE_DTS:-0.002 0.003 0.004 0.005}\"" in runner
    assert "NVE_DURATION_PS=\"${NVE_DURATION_PS:-5.0}\"" in runner
    assert "--disable-ml" in runner
    assert "\n    --ml-precision" not in runner
    assert "ML precision         : not applicable" in runner
    assert "conservative_classical_model_provenance_ml_disabled" in summarizer
    assert "--disable-ml" in certifier
    assert "ml_active = bool(args.model and not args.disable_ml)" in md
    assert "PaiNN disabled by --disable_ml" in md
    print("[PASS] TEL22 priors-only 5 ps coarse-dt diagnostic self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
