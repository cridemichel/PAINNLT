#!/usr/bin/env python3
"""Summarize refined TEL22 top-10%-local-curvature Morse-a NVE sweep."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_DTS = [0.002, 0.003, 0.004, 0.005]
CANDIDATES = ["top10_a0p950", "top10_a0p925", "top10_a0p875", "top10_a0p850"]
CENTER_NAME = "top10_a0p900_reference"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_report(path: Path) -> dict[str, Any]:
    r = load(path)
    d = r.get("definition", {})
    if d.get("hamiltonian_mode") != "conservative_classical_model_provenance_ml_disabled":
        raise ValueError(f"{path} is not a priors-only --disable-ml report")
    runtime = r.get("pair_specific_morse_runtime", d.get("pair_specific_morse_runtime"))
    if runtime != "bonded-analytic":
        raise ValueError(f"{path} Morse runtime={runtime!r}, expected bonded-analytic")
    runs = sorted(r["runs"], key=lambda x: float(x["dt_ps"]))
    dts = [float(x["dt_ps"]) for x in runs]
    if len(runs) != 4 or any(abs(a - b) > 1e-12 for a, b in zip(dts, EXPECTED_DTS)):
        raise ValueError(f"Unexpected dt grid in {path}: {dts}")
    if any(abs(float(x["duration_ps"]) - 5.0) > 0.5 * float(x["dt_ps"]) + 1e-12 for x in runs):
        raise ValueError(f"{path} is not a 5 ps coarse-grid report")
    rows = []
    for x in runs:
        dt = float(x["dt_ps"])
        sigma = float(x["sigma_E"])
        rows.append({
            "dt_ps": dt,
            "sigma_E": sigma,
            "C2_sigma_over_dt2": sigma / (dt * dt),
            "relative_block_mean_drift": float(x["relative_block_mean_drift"]),
        })
    local = []
    for lo, hi in zip(rows, rows[1:]):
        local.append({
            "dt_low_ps": lo["dt_ps"],
            "dt_high_ps": hi["dt_ps"],
            "local_exponent_p": math.log(hi["sigma_E"] / lo["sigma_E"]) / math.log(hi["dt_ps"] / lo["dt_ps"]),
        })
    sc = r["certification"]["scaling"]
    p = float(sc["exponent_p"])
    c2 = [x["C2_sigma_over_dt2"] for x in rows]
    lv = [x["local_exponent_p"] for x in local]
    return {
        "report": str(path.resolve()),
        "exponent_p": p,
        "abs_p_minus_2": abs(p - 2.0),
        "loglog_r2": float(sc["loglog_r2"]),
        "c2_spread_max_over_min": max(c2) / min(c2),
        "local_exponent_range": max(lv) - min(lv),
        "max_relative_block_mean_drift": max(x["relative_block_mean_drift"] for x in rows),
        "runs": rows,
        "adjacent_local_exponents": local,
    }


def regularity_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    # C2 flatness and local-slope regularity dominate. R2 and |p-2| are tie-breakers.
    return (
        float(metrics["c2_spread_max_over_min"]),
        float(metrics["local_exponent_range"]),
        -float(metrics["loglog_r2"]),
        float(metrics["abs_p_minus_2"]),
    )


def compare(reference: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    paired = []
    max_rel = 0.0
    for a, b in zip(reference["runs"], cand["runs"]):
        ratio = b["sigma_E"] / a["sigma_E"]
        max_rel = max(max_rel, abs(ratio - 1.0))
        paired.append({
            "dt_ps": a["dt_ps"],
            "sigma_reference": a["sigma_E"],
            "sigma_variant": b["sigma_E"],
            "sigma_ratio_variant_over_reference": ratio,
            "C2_reference": a["C2_sigma_over_dt2"],
            "C2_variant": b["C2_sigma_over_dt2"],
        })
    return {
        "delta_p": cand["exponent_p"] - reference["exponent_p"],
        "delta_abs_p_minus_2": cand["abs_p_minus_2"] - reference["abs_p_minus_2"],
        "c2_spread_ratio": cand["c2_spread_max_over_min"] / reference["c2_spread_max_over_min"],
        "local_exponent_range_ratio": cand["local_exponent_range"] / reference["local_exponent_range"],
        "max_relative_sigma_difference": max_rel,
        "paired_runs": paired,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-reference-report", type=Path, required=True)
    ap.add_argument("--center-reference-report", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    for name in CANDIDATES:
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    inputs = load(args.inputs)
    if inputs.get("kind") != "tel22_morse_top10_local_curvature_a_refinement_inputs":
        raise ValueError("Unexpected input manifest kind")
    full = summarize_report(args.full_reference_report)
    center = summarize_report(args.center_reference_report)
    candidates = {name: summarize_report(getattr(args, name)) for name in CANDIDATES}

    variants: dict[str, Any] = {
        CENTER_NAME: {
            **center,
            "a_scale": 0.900,
            "scaled_a": 0.270,
            "k_at_r0_ratio": 0.810,
            "source": "reused_test19",
        }
    }
    for name, metrics in candidates.items():
        meta = inputs["variants"][name]
        variants[name] = {
            **metrics,
            "a_scale": float(meta["a_scale"]),
            "scaled_a": float(meta["scaled_a"]),
            "k_at_r0_ratio": float(meta["k_at_r0_ratio"]),
            "source": "test20_candidate",
        }

    comparisons_to_full = {name: compare(full, row) for name, row in variants.items()}
    comparisons_to_center = {
        name: compare(center, row)
        for name, row in variants.items()
        if name != CENTER_NAME
    }

    ranked = sorted(variants, key=lambda name: regularity_key(variants[name]))
    best = ranked[0]
    if best == CENTER_NAME:
        interpretation = "a0p90_remains_best_by_c2_regularity_in_refined_grid"
    else:
        vs_center = comparisons_to_center[best]
        if vs_center["c2_spread_ratio"] < 1.0 and vs_center["local_exponent_range_ratio"] <= 1.05:
            interpretation = "refined_a_scale_improves_c2_regularity_over_a0p90"
        else:
            interpretation = "refined_grid_shows_tradeoff_no_clear_dominant_scale"

    out = {
        "schema_version": 1,
        "kind": "tel22_morse_top10_local_curvature_a_refinement_coarse_5ps",
        "scope": "PaiNN disabled; bonded-analytic Morse runtime; fixed top-10%-local-curvature 18-contact subset from test 18. Full a=1.00 and center a=0.90 reports are reused; only Morse a changes in test-20 candidates.",
        "full_reference": full,
        "center_reference": variants[CENTER_NAME],
        "variants": variants,
        "comparisons_to_full": comparisons_to_full,
        "comparisons_to_a0p90": comparisons_to_center,
        "ranking_by_c2_regularity": ranked,
        "best_by_c2_regularity": best,
        "input_manifest": str(args.inputs.resolve()),
        "interpretation": interpretation,
        "selection_policy": "Lexicographic: C2 spread, local-exponent range, -R2, |p-2|. Global p is deliberately secondary.",
        "caution": "Diagnostic parameter refinement only; numerical regularity does not establish a physically validated production reparameterization.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[TEL22 MORSE TOP10 a-REFINEMENT -- 5 ps COARSE DT]")
    print(f"full a=1.000 : p={full['exponent_p']:.8f} R2={full['loglog_r2']:.8f} C2spread={full['c2_spread_max_over_min']:.6f} localRange={full['local_exponent_range']:.6f}")
    for name in sorted(variants, key=lambda n: -float(variants[n]["a_scale"])):
        s = variants[name]
        c = comparisons_to_full[name]
        tag = "REUSE" if name == CENTER_NAME else "NEW"
        print(
            f"{name:23s}: [{tag}] aScale={s['a_scale']:.3f} kRatio={s['k_at_r0_ratio']:.6f} "
            f"p={s['exponent_p']:.8f} R2={s['loglog_r2']:.8f} "
            f"C2spread={s['c2_spread_max_over_min']:.6f} spread/full={c['c2_spread_ratio']:.4f} "
            f"local/full={c['local_exponent_range_ratio']:.4f}"
        )
    print(f"[BEST C2 REGULARITY] {best}")
    print(f"[INTERPRETATION] {interpretation}")
    print(f"[REPORT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
