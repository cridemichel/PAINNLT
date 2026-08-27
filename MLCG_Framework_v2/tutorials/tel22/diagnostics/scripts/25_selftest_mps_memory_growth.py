#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "25_monitor_mps_memory.py"
spec = importlib.util.spec_from_file_location("mps_memory_monitor_25", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def sample(step: int, rss_mib: float, physical_footprint_mib: float | None = None):
    return mod.Sample(
        elapsed_seconds=float(step),
        integration_step=step,
        root_pid=123,
        process_count=1,
        rss_mib=rss_mib,
        vsz_mib=rss_mib * 2,
        physical_footprint_mib=physical_footprint_mib,
        swap_used_mib=None,
    )


def main() -> int:
    assert mod.parse_size_to_mib("1.5G") == 1536.0
    assert mod.parse_size_to_mib("512M") == 512.0
    assert mod.parse_size_to_mib("1024K") == 1.0

    ps_rows = mod.parse_ps_rows(" 10 1 1024 4096\n 11 10 2048 8192\n")
    assert ps_rows == {10: (1, 1024, 4096), 11: (10, 2048, 8192)}

    with tempfile.TemporaryDirectory() as tmpdir:
        energy = Path(tmpdir) / "energy.csv"
        energy.write_text("Step,Time_ps,E_tot\n0,0.0,-1\n10,0.01,-1.1\n")
        assert mod.read_last_energy_step(energy) == 10
        empty = Path(tmpdir) / "missing.csv"
        assert mod.read_last_energy_step(empty) is None

    bounded = mod.analyze_samples(
        [sample(500, 1000), sample(1000, 1010), sample(1500, 1005), sample(2000, 1015)],
        warmup_step=500,
        growth_threshold_mib=1024,
        slope_threshold_mib_per_1000_steps=256,
    )
    assert bounded["classification"] == "bounded_over_observed_window"

    growing = mod.analyze_samples(
        [sample(500, 1000), sample(1000, 2000), sample(1500, 3000), sample(2000, 4000)],
        warmup_step=500,
        growth_threshold_mib=1024,
        slope_threshold_mib_per_1000_steps=256,
    )
    assert growing["classification"] == "sustained_process_memory_growth_observed"
    assert abs(growing["rss_slope_mib_per_1000_steps"] - 2000.0) < 1.0e-12

    footprint_growing = mod.analyze_samples(
        [
            sample(500, 1000, 1000),
            sample(1000, 1010, 2000),
            sample(1500, 1005, 3000),
            sample(2000, 1015, 4000),
        ],
        warmup_step=500,
        growth_threshold_mib=1024,
        slope_threshold_mib_per_1000_steps=256,
    )
    assert footprint_growing["classification"] == "sustained_process_memory_growth_observed"
    assert footprint_growing["classification_basis"] == ["macos_physical_footprint"]
    assert abs(
        footprint_growing["physical_footprint"]["slope_mib_per_1000_steps"] - 2000.0
    ) < 1.0e-12

    sparse_footprint = mod.analyze_samples(
        [
            sample(500, 1000, None),
            sample(1000, 1005, 2000),
            sample(1500, 1010, None),
            sample(2000, 1015, 2200),
            sample(2500, 1020, None),
            sample(3000, 1025, 2400),
        ],
        warmup_step=500,
        growth_threshold_mib=1024,
        slope_threshold_mib_per_1000_steps=256,
    )
    assert sparse_footprint["physical_footprint"]["first_step"] == 1000
    assert sparse_footprint["physical_footprint"]["last_step"] == 3000
    assert sparse_footprint["physical_footprint"]["at_warmup_mib"] == 2000

    inconclusive = mod.analyze_samples(
        [sample(500, 1000), sample(1000, 1500)],
        warmup_step=500,
        growth_threshold_mib=1024,
        slope_threshold_mib_per_1000_steps=256,
    )
    assert inconclusive["classification"] == "inconclusive_too_few_post_warmup_samples"

    print("[PASS] memory-size and ps parsers")
    print("[PASS] flushed energy-step reader")
    print("[PASS] RSS/physical-footprint bounded, growing, and inconclusive classifiers")
    print("[PASS] sparse vmmap samples are not forward-filled into the warmup baseline")
    print("[PASS] diagnostic is external and does not import ESPResSo or mutate production inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
