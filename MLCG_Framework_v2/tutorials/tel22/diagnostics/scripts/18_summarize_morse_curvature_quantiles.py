#!/usr/bin/env python3
"""Summarize TEL22 Morse local-curvature quantile NVE ablations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_DTS = [0.002, 0.003, 0.004, 0.005]
VARIANTS = ["top_05pct_zeroD", "top_10pct_zeroD", "top_20pct_zeroD"]


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-report", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    for name in VARIANTS:
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    ap.add_argument("--no-morse-summary", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    manifest = load(args.inputs)
    if manifest.get("kind") != "tel22_morse_checkpoint_local_curvature_quantile_inputs":
        raise ValueError("Unexpected input manifest kind")
    reference = summarize_report(args.reference_report)
    variants = {name: summarize_report(getattr(args, name)) for name in VARIANTS}

    comparisons = {}
    for name, cand in variants.items():
        paired = []
        rel = []
        for a, b in zip(reference["runs"], cand["runs"]):
            ratio = b["sigma_E"] / a["sigma_E"]
            rel.append(abs(ratio - 1.0))
            paired.append({
                "dt_ps": a["dt_ps"],
                "sigma_full": a["sigma_E"],
                "sigma_variant": b["sigma_E"],
                "sigma_ratio_variant_over_full": ratio,
                "C2_full": a["C2_sigma_over_dt2"],
                "C2_variant": b["C2_sigma_over_dt2"],
            })
        comparisons[name] = {
            "zeroed_count": int(manifest["variants"][name]["selected_count"]),
            "fraction": float(manifest["variants"][name]["fraction"]),
            "delta_p": cand["exponent_p"] - reference["exponent_p"],
            "delta_abs_p_minus_2": cand["abs_p_minus_2"] - reference["abs_p_minus_2"],
            "c2_spread_ratio": cand["c2_spread_max_over_min"] / reference["c2_spread_max_over_min"],
            "local_exponent_range_ratio": cand["local_exponent_range"] / reference["local_exponent_range"],
            "max_relative_sigma_difference": max(rel),
            "paired_runs": paired,
        }

    no_morse = None
    if args.no_morse_summary and args.no_morse_summary.is_file():
        x = load(args.no_morse_summary)
        no_morse = {
            "summary": str(args.no_morse_summary.resolve()),
            "exponent_p": float(x["exponent_p"]),
            "c2_spread_max_over_min": float(x["c2_spread_max_over_min"]),
            "runs": x.get("runs", []),
        }

    c5 = comparisons["top_05pct_zeroD"]
    c20 = comparisons["top_20pct_zeroD"]
    if c5["c2_spread_ratio"] <= 0.90 and c5["local_exponent_range_ratio"] <= 0.80:
        interpretation = "small_high_local_curvature_morse_subset_dominates_coarse_structure"
    elif c20["c2_spread_ratio"] <= 0.90 or c20["local_exponent_range_ratio"] <= 0.80:
        interpretation = "broader_high_local_curvature_morse_subset_contributes_to_coarse_structure"
    elif c20["max_relative_sigma_difference"] < 0.03:
        interpretation = "coarse_structure_not_localized_to_top20pct_local_curvature_morse_subset"
    else:
        interpretation = "curvature_ablation_changes_dynamics_without_clean_localization_review_c2_and_local_exponents"

    out = {
        "schema_version": 1,
        "kind": "tel22_morse_local_curvature_quantile_ablation_coarse_5ps",
        "scope": "PaiNN disabled; bonded-analytic Morse runtime in every full/quantile branch; same checkpoint mechanical state and particle set. Selected contacts are retained but set D=0 in nested top local-curvature subsets.",
        "ranking_metric": manifest["ranking_metric"],
        "source_uniform_k_at_r0": manifest["source"]["unique_k_at_r0"],
        "reference_full_morse": reference,
        "variants": variants,
        "comparisons": comparisons,
        "no_morse_control": no_morse,
        "input_manifest": str(args.inputs.resolve()),
        "interpretation": interpretation,
        "caution": "Diagnostic force ablation only. D=0 variants are not reparameterized TEL22 models and are not candidates for production promotion.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[TEL22 MORSE LOCAL-CURVATURE QUANTILES -- 5 ps COARSE DT]")
    print(f"full Morse : p={reference['exponent_p']:.8f} R2={reference['loglog_r2']:.8f} C2spread={reference['c2_spread_max_over_min']:.6f} localRange={reference['local_exponent_range']:.6f}")
    for name in VARIANTS:
        s = variants[name]
        c = comparisons[name]
        print(f"{name:17s}: n={c['zeroed_count']:3d} p={s['exponent_p']:.8f} R2={s['loglog_r2']:.8f} C2spread={s['c2_spread_max_over_min']:.6f} spreadRatio={c['c2_spread_ratio']:.4f} localRatio={c['local_exponent_range_ratio']:.4f}")
    print(f"[INTERPRETATION] {interpretation}")
    print(f"[REPORT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
