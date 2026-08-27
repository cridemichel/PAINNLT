#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = SCRIPT_DIR / "16_summarize_morse_switch_ab.py"

spec = importlib.util.spec_from_file_location("morse_switch_summary", SUMMARY_PATH)
summary = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(summary)


def fake_report(path: Path, mode: str | None, factor: float = 1.0) -> None:
    dts = [0.002, 0.003, 0.004, 0.005]
    runs = []
    for dt in dts:
        sigma = factor * 180.0 * dt * dt
        runs.append({
            "dt_ps": dt,
            "duration_ps": 5.0,
            "sigma_E": sigma,
            "relative_block_mean_drift": 1.0e-7,
        })
    definition = {
        "hamiltonian_mode": "conservative_classical_model_provenance_ml_disabled",
        "neighbor_search": "link-cell",
    }
    payload = {
        "definition": definition,
        "neighbor_search": "link-cell",
        "runs": runs,
        "certification": {"scaling": {"exponent_p": 2.0, "loglog_r2": 1.0}},
    }
    if mode is not None:
        payload["morse_switch_mode"] = mode
        payload["definition"]["morse_switch_mode"] = mode
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> int:
    priors = json.loads((ROOT / "tutorials/tel22/cg_priors.json").read_text(encoding="utf-8"))
    morse = [x for x in priors.get("bonds", []) if str(x.get("type", "harmonic")).lower() == "morse"]
    assert len(morse) == 180
    assert sum("r_switch" in x for x in morse) == 0
    assert sum("r_cut" in x for x in morse) == 0
    inv = summary.morse_inventory(ROOT / "tutorials/tel22/cg_priors.json")
    assert inv["r_switch_nm"]["min"] > inv["r0_nm"]["max"]
    assert math.isclose(inv["r_cut_nm"]["median"], 15.0)

    runner = (ROOT / "simulation/run_cg_md.py").read_text(encoding="utf-8")
    certifier = (ROOT / "simulation/certify_nve.py").read_text(encoding="utf-8")
    interactions = (ROOT / "simulation/espresso_interactions.py").read_text(encoding="utf-8")
    assert '"--morse_switch_mode"' in runner
    assert 'switch_mode=args.morse_switch_mode' in runner
    assert '"--morse-switch-mode"' in certifier
    assert '"--morse_switch_mode", args.morse_switch_mode' in certifier
    assert 'return float(item["r_switch"]) if switch_mode == "switched" else -1.0' in interactions

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        switched = td / "switched.json"
        stock = td / "stock.json"
        fake_report(switched, None, 1.0)  # legacy report: absent flag means switched
        fake_report(stock, "stock-shifted", 1.0)
        a = summary.summarize_report(switched, "switched")
        b = summary.summarize_report(stock, "stock-shifted")
        assert math.isclose(a["exponent_p"], 2.0)
        assert math.isclose(b["c2_spread_max_over_min"], 1.0)
        assert all(
            math.isclose(x["sigma_E"], y["sigma_E"], rel_tol=0.0, abs_tol=1.0e-15)
            for x, y in zip(a["runs"], b["runs"])
        )

    print("[PASS] TEL22 Morse switch A/B diagnostic self-test")
    print(
        "[INFO] production derived switch range: "
        f"{inv['r_switch_nm']['min']:.6g}..{inv['r_switch_nm']['max']:.6g} nm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
