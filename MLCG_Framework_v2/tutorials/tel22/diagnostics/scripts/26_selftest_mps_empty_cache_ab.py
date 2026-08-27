#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "26_summarize_mps_empty_cache_ab.py"
spec = importlib.util.spec_from_file_location("mps_emptycache_summary_26", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def report(*, cadence: int, peak: float, growth: float, slope: float, elapsed: float):
    return {
        "run": {"complete": True},
        "environment": {mod.ENV_NAME: str(cadence)},
        "provenance_inputs": {"model": {"sha256": "same"}},
        "sampling": {"elapsed_seconds": elapsed},
        "memory_analysis": {
            "physical_footprint": {
                "peak_mib": peak,
                "at_last_step_mib": peak,
                "peak_minus_warmup_mib": growth,
                "slope_mib_per_1000_steps": slope,
            },
            "peak_rss_mib": 2000.0,
            "rss_at_last_step_mib": 1900.0,
            "rss_slope_mib_per_1000_steps": 10.0,
        },
    }


def main() -> int:
    baseline = report(cadence=0, peak=27000, growth=1600, slope=600, elapsed=300)
    effective = report(cadence=100, peak=12000, growth=200, slope=50, elapsed=330)
    good = mod.build_report(baseline, effective, 100)
    assert good["gates"]["effective"] is True
    assert good["decision"] == "empty_cache_candidate_worth_longer_validation"

    slow = report(cadence=100, peak=12000, growth=200, slope=50, elapsed=420)
    costly = mod.build_report(baseline, slow, 100)
    assert costly["decision"] == "memory_effective_but_tune_cadence_for_performance"

    ineffective = report(cadence=100, peak=26800, growth=1500, slope=580, elapsed=305)
    bad = mod.build_report(baseline, ineffective, 100)
    assert bad["gates"]["effective"] is False
    assert bad["decision"] == "do_not_promote_empty_cache_no_material_memory_benefit"

    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_log = Path(tmpdir) / "baseline.log"
        candidate_log = Path(tmpdir) / "candidate.log"
        baseline_log.write_text(
            "[PaiNN] MPS diagnostic emptyCache cadence: 0 successful force calls\n",
            encoding="utf-8",
        )
        candidate_log.write_text(
            "[PaiNN] MPS diagnostic emptyCache cadence: 100 successful force calls\n",
            encoding="utf-8",
        )
        mod.require_attestation(baseline_log, 0)
        mod.require_attestation(candidate_log, 100)

    print("[PASS] effective, costly, and ineffective emptyCache decisions")
    print("[PASS] A/B provenance and environment policy are explicit")
    print("[PASS] flushed runtime policy attestations are required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
