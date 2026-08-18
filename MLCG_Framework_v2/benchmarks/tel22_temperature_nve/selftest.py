#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from analyze_temperature_sweep import build_summary, summarize_report
from make_scaled_checkpoint import R_KJ_MOL_K, rescale_checkpoint, sha256_file


def fake_report(path: Path, p: float, sigma: list[float]) -> None:
    dts = [0.001, 0.002, 0.004]
    payload = {
        "certification": {
            "pass": True,
            "scaling_pass": True,
            "drift_pass": True,
            "scaling": {"exponent_p": p, "loglog_r2": 0.999},
        },
        "runs": [
            {"dt_ps": dt, "sigma_E": value, "relative_block_mean_drift": 1e-6 * (i + 1)}
            for i, (dt, value) in enumerate(zip(dts, sigma))
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        source = tmpdir / "source.npz"
        output = tmpdir / "scaled.npz"
        pos = np.arange(18, dtype=np.float64).reshape(6, 3)
        quat = np.arange(24, dtype=np.float64).reshape(6, 4)
        v = np.full((6, 3), 2.0, dtype=np.float64)
        omega = np.full((6, 3), 4.0, dtype=np.float64)
        metadata = {
            "schema_version": 3,
            "created_with_kT_kJ_mol": R_KJ_MOL_K * 300.0,
            "input_hashes": {"dummy": "unchanged"},
        }
        np.savez_compressed(
            source,
            pos=pos,
            quat=quat,
            v=v,
            omega=omega,
            particle_is_virtual=np.asarray([False, False, True, False, True, False]),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        source_hash = sha256_file(source)
        summary = rescale_checkpoint(
            source, output, source_temperature_k=300.0, target_temperature_k=75.0
        )
        assert math.isclose(summary["velocity_scale"], 0.5)
        with np.load(output, allow_pickle=False) as scaled:
            assert np.array_equal(scaled["pos"], pos)
            assert np.array_equal(scaled["quat"], quat)
            assert np.array_equal(scaled["v"], v * 0.5)
            assert np.array_equal(scaled["omega"], omega * 0.5)
            out_meta = json.loads(str(np.asarray(scaled["metadata_json"]).item()))
        diag = out_meta["temperature_sweep_diagnostic"]
        assert diag["source_checkpoint_sha256"] == source_hash
        assert diag["source_temperature_K"] == 300.0
        assert diag["target_temperature_K"] == 75.0
        assert math.isclose(out_meta["created_with_kT_kJ_mol"], R_KJ_MOL_K * 75.0)
        assert out_meta["input_hashes"] == metadata["input_hashes"]

        report300 = tmpdir / "r300.json"
        report30 = tmpdir / "r30.json"
        fake_report(report300, 1.89, [1.2, 4.0, 16.0])
        fake_report(report30, 1.99, [1.0, 4.0, 16.0])
        row300 = summarize_report(report300, 300.0, "float32")
        row30 = summarize_report(report30, 30.0, "float32")
        aggregate = build_summary([row300, row30])
        trend = aggregate["trends"]["float32"]
        assert trend["improvement_in_abs_p_minus_2"] > 0.09
        assert row300["small_dt_C2_over_coarse_median"] > row30["small_dt_C2_over_coarse_median"]

    print("[PASS] TEL22 temperature NVE benchmark self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
