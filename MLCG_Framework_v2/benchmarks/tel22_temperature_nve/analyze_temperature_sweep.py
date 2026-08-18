#!/usr/bin/env python3
"""Aggregate TEL22 temperature-sweep NVE reports without changing their pass/fail gates."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any

RUN_RE = re.compile(r"^T(?P<temp>[0-9]+(?:p[0-9]+)?)K_(?P<precision>float32|float64)$")


def _temperature_from_label(value: str) -> float:
    return float(value.replace("p", "."))


def _kinetic_summary(report_path: Path, run: dict[str, Any]) -> dict[str, float | None]:
    candidates = []
    raw = run.get("energy_csv")
    if raw:
        candidates.append(Path(str(raw)))
        candidates.append(report_path.parent / Path(str(raw)).parent.name / "energy.csv")
    energy_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if energy_path is None:
        return {
            "Ekin_initial_kj_mol": None,
            "Ekin_mean_kj_mol": None,
            "Ekin_final_block_mean_kj_mol": None,
            "Ekin_mean_over_initial": None,
        }
    values = []
    with energy_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "E_kin" not in reader.fieldnames:
            return {
                "Ekin_initial_kj_mol": None,
                "Ekin_mean_kj_mol": None,
                "Ekin_final_block_mean_kj_mol": None,
                "Ekin_mean_over_initial": None,
            }
        for row in reader:
            values.append(float(row["E_kin"]))
    if not values:
        return {
            "Ekin_initial_kj_mol": None,
            "Ekin_mean_kj_mol": None,
            "Ekin_final_block_mean_kj_mol": None,
            "Ekin_mean_over_initial": None,
        }
    block = max(1, int(math.ceil(0.2 * len(values))))
    initial = values[0]
    mean_value = sum(values) / len(values)
    final_mean = sum(values[-block:]) / block
    return {
        "Ekin_initial_kj_mol": initial,
        "Ekin_mean_kj_mol": mean_value,
        "Ekin_final_block_mean_kj_mol": final_mean,
        "Ekin_mean_over_initial": mean_value / initial if initial != 0.0 else math.nan,
    }


def summarize_report(path: Path, temperature_k: float, precision: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    certification = report["certification"]
    scaling = certification["scaling"]
    runs = sorted(report["runs"], key=lambda item: float(item["dt_ps"]))
    c2 = [float(item["sigma_E"]) / float(item["dt_ps"]) ** 2 for item in runs]
    coarse_count = min(3, len(c2))
    coarse_median = median(c2[-coarse_count:])
    small_ratio = c2[0] / coarse_median if coarse_median > 0.0 else math.nan
    worst = max(runs, key=lambda item: float(item["relative_block_mean_drift"]))
    kinetic = _kinetic_summary(path, runs[0])
    return {
        "temperature_K": temperature_k,
        "precision": precision,
        "report": str(path.resolve()),
        "pass": bool(certification["pass"]),
        "scaling_pass": bool(certification["scaling_pass"]),
        "drift_pass": bool(certification["drift_pass"]),
        "exponent_p": float(scaling["exponent_p"]),
        "loglog_r2": float(scaling["loglog_r2"]),
        "abs_p_minus_2": abs(float(scaling["exponent_p"]) - 2.0),
        "dt_min_ps": float(runs[0]["dt_ps"]),
        "sigma_E_dt_min": float(runs[0]["sigma_E"]),
        "C2_dt_min": c2[0],
        "C2_coarse_median": coarse_median,
        "small_dt_C2_over_coarse_median": small_ratio,
        "C2_spread_max_over_min": max(c2) / min(c2),
        "max_relative_block_mean_drift": float(worst["relative_block_mean_drift"]),
        "max_drift_dt_ps": float(worst["dt_ps"]),
        **kinetic,
    }


def collect(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not results_dir.is_dir():
        raise FileNotFoundError(results_dir)
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        match = RUN_RE.match(child.name)
        if not match:
            continue
        report = child / "nve_certification_report.json"
        if not report.is_file():
            continue
        rows.append(
            summarize_report(
                report,
                _temperature_from_label(match.group("temp")),
                match.group("precision"),
            )
        )
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_precision: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_precision.setdefault(str(row["precision"]), []).append(row)
    trends: dict[str, Any] = {}
    for precision, group in by_precision.items():
        ordered = sorted(group, key=lambda item: float(item["temperature_K"]), reverse=True)
        trend: dict[str, Any] = {"temperatures_high_to_low_K": [x["temperature_K"] for x in ordered]}
        if len(ordered) >= 2:
            high = ordered[0]
            low = ordered[-1]
            trend.update({
                "high_temperature_K": high["temperature_K"],
                "low_temperature_K": low["temperature_K"],
                "delta_p_low_minus_high": low["exponent_p"] - high["exponent_p"],
                "improvement_in_abs_p_minus_2": high["abs_p_minus_2"] - low["abs_p_minus_2"],
                "small_dt_C2_ratio_low_over_high": (
                    low["small_dt_C2_over_coarse_median"]
                    / high["small_dt_C2_over_coarse_median"]
                    if high["small_dt_C2_over_coarse_median"] != 0.0 else math.nan
                ),
            })
        trends[precision] = trend
    return {
        "kind": "tel22_iso_configurational_temperature_nve_summary",
        "scope": (
            "Same TEL22 coordinates/orientations at all temperatures; only translational velocities "
            "and body-frame angular velocities are scaled by sqrt(T/T_source). This is an amplitude "
            "diagnostic, not an independently equilibrated canonical-temperature comparison."
        ),
        "rows": sorted(rows, key=lambda x: (str(x["precision"]), -float(x["temperature_K"]))),
        "trends": trends,
        "interpretation": {
            "p_moves_toward_2_when_T_decreases": (
                "Consistent with amplitude/configuration-dependent stiffness or anharmonic regions becoming less sampled; "
                "not by itself proof of a harmonic high-frequency mode, whose frequency is temperature independent."
            ),
            "p_stays_similar_when_T_decreases": (
                "No evidence from this test that kinetic amplitude is the main cause; intrinsic stiffness or another "
                "TEL22-specific numerical mechanism remains possible."
            ),
            "small_dt_C2_excess_grows_when_T_decreases": (
                "Consistent with a timestep-independent numerical/FP32 floor becoming more visible as physical energy "
                "fluctuations shrink."
            ),
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n[TEL22 ISO-CONFIGURATIONAL TEMPERATURE NVE SUMMARY]")
    print("precision  T[K]     p          R2         C2spread   C2small/coarse  max_drift   Ekin_mean/Ekin0")
    for row in summary["rows"]:
        kinetic_ratio = row["Ekin_mean_over_initial"]
        kinetic_text = "n/a" if kinetic_ratio is None else f"{float(kinetic_ratio):.3f}"
        print(
            f"{row['precision']:<10} {row['temperature_K']:<8g} "
            f"{row['exponent_p']:<10.6f} {row['loglog_r2']:<10.6f} "
            f"{row['C2_spread_max_over_min']:<10.3f} "
            f"{row['small_dt_C2_over_coarse_median']:<15.3f} "
            f"{row['max_relative_block_mean_drift']:.3e}   {kinetic_text}"
        )
    for precision, trend in summary["trends"].items():
        if "delta_p_low_minus_high" not in trend:
            continue
        print(
            f"[TREND {precision}] {trend['high_temperature_K']:g}K -> {trend['low_temperature_K']:g}K: "
            f"delta_p={trend['delta_p_low_minus_high']:+.6f}; "
            f"improvement |p-2|={trend['improvement_in_abs_p_minus_2']:+.6f}; "
            f"small-dt C2-ratio low/high={trend['small_dt_C2_ratio_low_over_high']:.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = collect(args.results_dir.expanduser().resolve())
    if not rows:
        raise RuntimeError(f"No completed NVE certification reports found under {args.results_dir}")
    summary = build_summary(rows)
    output = args.output or (args.results_dir / "temperature_nve_summary.json")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"[REPORT] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
