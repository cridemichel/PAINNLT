#!/usr/bin/env python3
"""Self-test for the TEL22 uniform-Morse PaiNN closure diagnostic."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "tutorials" / "tel22" / "diagnostics" / "scripts"


def synthetic_report(mode: str, c2: list[float], p: float = 2.0) -> dict:
    dts = [0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]
    runs = []
    for dt, C2 in zip(dts, c2):
        duration = round(2.0 / dt) * dt
        runs.append({
            "dt_ps": dt,
            "duration_ps": duration,
            "sigma_E": C2 * dt * dt,
            "relative_block_mean_drift": 1e-7,
        })
    return {
        "definition": {
            "hamiltonian_mode": mode,
            "morse_switch_mode": "switched",
            "pair_specific_morse_runtime": "marker-nonbonded",
        },
        "device": "cpu",
        "neighbor_search": "link-cell",
        "morse_switch_mode": "switched",
        "pair_specific_morse_runtime": "marker-nonbonded",
        "inputs_sha256": {"priors": "p", "checkpoint": "c", "model": "m"},
        "runs": runs,
        "certification": {"scaling": {"exponent_p": p, "loglog_r2": 0.999}},
    }


def main() -> int:
    runner = (SCRIPTS / "23_test_nve_painn_closure_uniform_a0p85.sh").read_text(encoding="utf-8")
    summarizer_path = SCRIPTS / "23_summarize_painn_closure_uniform_a0p85.py"
    summarizer_src = summarizer_path.read_text(encoding="utf-8")

    assert "NVE_DURATION_PS=\"${NVE_DURATION_PS:-2.0}\"" in runner
    assert "0.001 0.0015 0.002 0.003 0.004 0.005" in runner
    assert "B_uniform_a0p85_old_painn" in runner
    assert "C_uniform_a0p85_no_painn" in runner
    assert "--disable-ml" in runner
    assert "--ml-precision \"${NVE_ML_PRECISION}\"" in runner
    assert "--pair-specific-morse-runtime marker-nonbonded" in runner
    assert "--morse-switch-mode switched" in runner
    assert "uniform_a0p85_supports_next_painn_closure" in runner
    assert "morse_uniform_abc_summary" in runner
    assert "rebuild_residual_target_and_retrain_painn_against_uniform_a0p85_priors" in summarizer_src
    assert "intentionally Hamiltonian-decomposition-inconsistent" in summarizer_src

    spec = importlib.util.spec_from_file_location("closure", summarizer_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # A perfectly flat C2 curve must be recognized as exact quadratic scaling
    # by the local-exponent diagnostic, independent of the externally supplied
    # global p field in this synthetic schema fixture.
    r = synthetic_report("model_active", [150.0] * 6, p=2.0)
    tmp = Path(__file__).with_name(".23_selftest_tmp_report.json")
    try:
        import json
        tmp.write_text(json.dumps(r), encoding="utf-8")
        m = mod.summarize_report(tmp, "model_active")
        assert math.isclose(m["c2_spread_max_over_min"], 1.0, abs_tol=1e-15)
        assert math.isclose(m["c2_small_over_coarse_median"], 1.0, abs_tol=1e-15)
        assert max(abs(x["local_exponent_p"] - 2.0) for x in m["adjacent_local_exponents"]) < 1e-12

        # Historical A reports predate explicit Morse runtime metadata. They are
        # allowed to infer the then-production defaults only when the caller opts
        # in; current B/C reports remain strict.
        legacy = synthetic_report("model_active", [150.0] * 6, p=2.0)
        legacy.pop("morse_switch_mode")
        legacy.pop("pair_specific_morse_runtime")
        legacy["definition"].pop("morse_switch_mode")
        legacy["definition"].pop("pair_specific_morse_runtime")
        tmp.write_text(json.dumps(legacy), encoding="utf-8")
        try:
            mod.summarize_report(tmp, "model_active")
        except ValueError:
            pass
        else:
            raise AssertionError("current-schema validation must reject missing Morse runtime metadata")
        lm = mod.summarize_report(
            tmp,
            "model_active",
            allow_legacy_production_morse_defaults=True,
        )
        assert lm["morse_switch_mode"] == "switched"
        assert lm["pair_specific_morse_runtime"] == "marker-nonbonded"
        assert lm["legacy_inferred_runtime_metadata"] == [
            "morse_switch_mode=switched",
            "pair_specific_morse_runtime=marker-nonbonded",
        ]
    finally:
        tmp.unlink(missing_ok=True)

    print("[PASS] TEL22 uniform-a0.85 PaiNN closure self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
