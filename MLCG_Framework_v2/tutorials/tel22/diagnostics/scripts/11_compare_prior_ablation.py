#!/usr/bin/env python3
"""Compare TEL22 NVE scaling across Morse/dihedral prior-ablation branches."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ORDER = ("baseline", "no_morse", "no_dihedrals", "no_morse_no_dihedrals")


def metrics(report: dict[str, Any]) -> dict[str, Any]:
    cert = report["certification"]
    scaling = cert["scaling"]
    runs = sorted(report["runs"], key=lambda row: float(row["dt_ps"]))
    c2 = [float(row["sigma_E"]) / float(row["dt_ps"]) ** 2 for row in runs]
    coarse = c2[-3:] if len(c2) >= 3 else c2
    worst = max(runs, key=lambda row: float(row["relative_block_mean_drift"]))
    return {
        "pass": bool(cert["pass"]),
        "scaling_pass": bool(cert["scaling_pass"]),
        "drift_pass": bool(cert["drift_pass"]),
        "exponent_p": float(scaling["exponent_p"]),
        "abs_p_minus_2": abs(float(scaling["exponent_p"]) - 2.0),
        "loglog_r2": float(scaling["loglog_r2"]),
        "c2_spread_max_over_min": max(c2) / min(c2),
        "dt_min_ps": float(runs[0]["dt_ps"]),
        "sigma_E_dt_min": float(runs[0]["sigma_E"]),
        "c2_dt_min": c2[0],
        "c2_coarse_median": statistics.median(coarse),
        "small_dt_c2_over_coarse_median": c2[0] / statistics.median(coarse),
        "max_relative_block_mean_drift": float(worst["relative_block_mean_drift"]),
        "max_drift_dt_ps": float(worst["dt_ps"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    variants = inputs["variants"]
    out: dict[str, Any] = {
        "schema_version": 1,
        "kind": "tel22_nve_morse_dihedral_prior_ablation_comparison",
        "scope": inputs["scope"],
        "source_counts": inputs["source"],
        "variants": {},
    }

    for name in ORDER:
        spec = variants[name]
        canonical = spec["alias_of"] or name
        report_path = args.results_dir / canonical / "nve_certification_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing completed NVE report for {name}: {report_path}")
        data = metrics(json.loads(report_path.read_text(encoding="utf-8")))
        data.update({
            "canonical_result": canonical,
            "alias_of": spec["alias_of"],
            "removed_morse_entries": int(spec["removed_morse_entries"]),
            "removed_dihedral_entries": int(spec["removed_dihedral_entries"]),
        })
        out["variants"][name] = data

    base = out["variants"]["baseline"]
    for name in ORDER[1:]:
        item = out["variants"][name]
        item["delta_p_vs_baseline"] = item["exponent_p"] - base["exponent_p"]
        item["improvement_in_abs_p_minus_2_vs_baseline"] = base["abs_p_minus_2"] - item["abs_p_minus_2"]
        item["sigma_E_dt_min_ratio_vs_baseline"] = item["sigma_E_dt_min"] / base["sigma_E_dt_min"]
        item["small_dt_c2_ratio_vs_baseline"] = (
            item["small_dt_c2_over_coarse_median"] / base["small_dt_c2_over_coarse_median"]
        )

    no_morse = out["variants"]["no_morse"]
    dihedral_count = int(inputs["source"]["dihedral_entries"])
    morse_improvement = float(no_morse["improvement_in_abs_p_minus_2_vs_baseline"])
    if no_morse["abs_p_minus_2"] <= 0.03 and morse_improvement >= 0.05:
        morse_interpretation = "strong_evidence_morse_contributes_to_nonideal_scaling"
    elif morse_improvement >= 0.03:
        morse_interpretation = "evidence_morse_contributes_to_nonideal_scaling"
    elif morse_improvement <= -0.03:
        morse_interpretation = "removing_morse_worsens_scaling"
    else:
        morse_interpretation = "morse_ablation_does_not_materially_change_global_exponent"

    out["interpretation"] = {
        "morse": morse_interpretation,
        "dihedrals": (
            "not_testable_production_tel22_has_zero_dihedral_priors"
            if dihedral_count == 0 else "compare_no_dihedrals_and_no_morse_no_dihedrals_branches"
        ),
        "warning": (
            "Ablation changes the Hamiltonian while retaining the trained residual PaiNN; "
            "this is a numerical diagnostic, not a physically reparameterized TEL22 model."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[TEL22 MORSE / DIHEDRAL NVE ABLATION]")
    print("variant                    p         R2        C2spread  C2small/coarse  max_drift")
    for name in ORDER:
        item = out["variants"][name]
        alias = f" (={item['alias_of']})" if item["alias_of"] else ""
        print(
            f"{name:26s} {item['exponent_p']:.6f}  {item['loglog_r2']:.6f}  "
            f"{item['c2_spread_max_over_min']:.3f}     "
            f"{item['small_dt_c2_over_coarse_median']:.3f}           "
            f"{item['max_relative_block_mean_drift']:.3e}{alias}"
        )
    print(f"[MORSE] {morse_interpretation}")
    if dihedral_count == 0:
        print("[DIHEDRAL] production TEL22 contains zero dihedral priors; removal is a no-op.")
    print(f"[REPORT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
