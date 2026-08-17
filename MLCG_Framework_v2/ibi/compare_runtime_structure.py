#!/usr/bin/env python3
"""Compare two matched bonded-runtime structural validation reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def _load_report(path: str | Path) -> dict:
    path = Path(path)
    data = json.loads(path.read_text())
    if not data.get("pass", False):
        raise ValueError(f"Runtime structure report did not pass: {path}")
    groups = data.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(f"Runtime structure report has no groups: {path}")
    return data


def compare_runtime_structure_reports(
    report_a: str | Path,
    report_b: str | Path,
    *,
    label_a: str = "A",
    label_b: str = "B",
    output: str | Path | None = None,
) -> dict:
    a = _load_report(report_a)
    b = _load_report(report_b)
    keys_a = set(a["groups"])
    keys_b = set(b["groups"])
    if keys_a != keys_b:
        missing_b = sorted(keys_a - keys_b)
        missing_a = sorted(keys_b - keys_a)
        raise ValueError(
            "Matched reports contain different bonded groups; "
            f"missing in {label_b}: {missing_b}; missing in {label_a}: {missing_a}"
        )

    groups = {}
    deltas = []
    deltas_by_kind: dict[str, list[float]] = {}
    wins_a = wins_b = ties = 0
    for key in sorted(keys_a):
        ga = a["groups"][key]
        gb = b["groups"][key]
        if ga.get("kind") != gb.get("kind"):
            raise ValueError(f"Kind mismatch for group {key}: {ga.get('kind')} vs {gb.get('kind')}")
        la = float(ga["distribution_l1"])
        lb = float(gb["distribution_l1"])
        delta = lb - la
        deltas.append(delta)
        deltas_by_kind.setdefault(str(ga["kind"]), []).append(delta)
        tol = 1.0e-12
        if delta < -tol:
            wins_b += 1
        elif delta > tol:
            wins_a += 1
        else:
            ties += 1
        groups[key] = {
            "kind": ga["kind"],
            "l1_a": la,
            "l1_b": lb,
            "delta_b_minus_a": delta,
            "samples_a": int(ga.get("samples", 0)),
            "samples_b": int(gb.get("samples", 0)),
        }

    deltas_arr = np.asarray(deltas, dtype=float)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": "matched_runtime_structure_ab",
        "label_a": label_a,
        "label_b": label_b,
        "report_a": str(Path(report_a).resolve()),
        "report_b": str(Path(report_b).resolve()),
        "mean_l1_a": float(a["mean_l1"]),
        "mean_l1_b": float(b["mean_l1"]),
        "delta_mean_l1_b_minus_a": float(b["mean_l1"] - a["mean_l1"]),
        "max_l1_a": float(a["max_l1"]),
        "max_l1_b": float(b["max_l1"]),
        "mean_l1_by_kind_a": a.get("mean_l1_by_kind", {}),
        "mean_l1_by_kind_b": b.get("mean_l1_by_kind", {}),
        "paired_group_delta_mean": float(np.mean(deltas_arr)),
        "paired_group_delta_std": float(np.std(deltas_arr, ddof=1)) if deltas_arr.size > 1 else 0.0,
        "paired_group_delta_by_kind": {
            kind: float(np.mean(values)) for kind, values in sorted(deltas_by_kind.items())
        },
        "group_wins": {label_a: wins_a, label_b: wins_b, "ties": ties},
        "groups": groups,
        "pass": True,
        "threshold_applied": False,
        "note": (
            "Lower L1 is better. delta_b_minus_a < 0 favors B. "
            "This matched A/B report is diagnostic and imposes no universal structural threshold."
        ),
    }

    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("[IBI/ML MATCHED RUNTIME STRUCTURE COMPARISON]")
    print(
        f"{label_a}: mean={summary['mean_l1_a']:.6f} max={summary['max_l1_a']:.6f} | "
        f"{label_b}: mean={summary['mean_l1_b']:.6f} max={summary['max_l1_b']:.6f}"
    )
    print(f"delta_mean({label_b}-{label_a})={summary['delta_mean_l1_b_minus_a']:+.6f}")
    for kind in sorted(set(summary["mean_l1_by_kind_a"]) | set(summary["mean_l1_by_kind_b"])):
        va = float(summary["mean_l1_by_kind_a"].get(kind, float("nan")))
        vb = float(summary["mean_l1_by_kind_b"].get(kind, float("nan")))
        print(f"{kind}: {label_a}={va:.6f} {label_b}={vb:.6f} delta={vb-va:+.6f}")
    for key, item in groups.items():
        print(
            f"[A/B] {key}: {label_a}={item['l1_a']:.6f} {label_b}={item['l1_b']:.6f} "
            f"delta={item['delta_b_minus_a']:+.6f}"
        )
    print(
        f"paired group delta mean={summary['paired_group_delta_mean']:+.6f} "
        f"std={summary['paired_group_delta_std']:.6f} | "
        f"wins {label_a}={wins_a} {label_b}={wins_b} ties={ties}"
    )
    print(f"[NOTE] Lower L1 is better; negative delta favors {label_b}. No hard threshold is imposed.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-a", required=True)
    parser.add_argument("--report-b", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    compare_runtime_structure_reports(
        args.report_a,
        args.report_b,
        label_a=args.label_a,
        label_b=args.label_b,
        output=args.output,
    )


if __name__ == "__main__":
    main()
