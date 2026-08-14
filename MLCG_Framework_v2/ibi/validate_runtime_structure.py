#!/usr/bin/env python3
"""Compare a sampled full-runtime trajectory with bonded IBI target distributions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dbi_priors import load_continuation_priors  # noqa: E402
from geometry_io import pool_requested, read_sampled_distributions  # noqa: E402
from ibi_core import histogram_density, normalize_density  # noqa: E402

SCHEMA_VERSION = 1


def distribution_l1(sim_density, target_density, grid):
    sim = normalize_density(sim_density, grid)
    target = normalize_density(target_density, grid)
    return float(np.trapezoid(np.abs(sim - target), grid))


def validate_runtime_structure(
    *,
    dataset: str | Path,
    priors: str | Path,
    sample_npz: str | Path,
    ibi_config: str | Path | None = None,
    output: str | Path | None = None,
):
    state = load_continuation_priors(dataset, priors, ibi_config=ibi_config)
    selected_priors = state["priors"]
    groups = state["groups"]
    sampled = read_sampled_distributions(sample_npz, selected_priors)
    sampled_by_key = {"bonds": sampled[0], "angles": sampled[1], "dihedrals": sampled[2]}

    metrics = {}
    by_kind: dict[str, list[float]] = {}
    for json_key, category in groups.items():
        sim_groups = pool_requested(selected_priors, sampled_by_key[json_key], json_key)
        for name, group_state in category.items():
            sim_group = sim_groups.get(name)
            if sim_group is None:
                raise RuntimeError(f"Missing sampled group {json_key}:{name}")
            values = np.asarray(sim_group["values"], dtype=float)
            if values.size == 0:
                raise RuntimeError(f"No samples for {json_key}:{name}")
            if group_state["kind"] == "dihedral":
                values = np.mod(values, 2.0 * np.pi)
            _counts, density, hist_x = histogram_density(values, group_state["bins"])
            if not np.allclose(hist_x, group_state["hist_x"], rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"Histogram grid mismatch for {json_key}:{name}")
            l1 = distribution_l1(density, group_state["target_density"], hist_x)
            metrics[f"{json_key}:{name}"] = {
                "kind": group_state["kind"],
                "samples": int(values.size),
                "distribution_l1": l1,
                "sample_min": float(np.min(values)),
                "sample_max": float(np.max(values)),
            }
            by_kind.setdefault(group_state["kind"], []).append(l1)
            print(
                f"[RUNTIME STRUCTURE] {group_state['kind']} {name}: "
                f"N={values.size}, L1(Psim,Ptarget)={l1:.6g}"
            )

    all_l1 = [item["distribution_l1"] for item in metrics.values()]
    if not all_l1:
        raise ValueError("No IBI/DBI bonded groups were available for runtime structural validation")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": "full_runtime_bonded_structure",
        "dataset": str(Path(dataset).resolve()),
        "priors": str(Path(priors).resolve()),
        "sample_npz": str(Path(sample_npz).resolve()),
        "mean_l1": float(np.mean(all_l1)),
        "max_l1": float(np.max(all_l1)),
        "mean_l1_by_kind": {
            kind: float(np.mean(values)) for kind, values in sorted(by_kind.items())
        },
        "groups": metrics,
        "pass": True,
        "threshold_applied": False,
        "note": (
            "Structural L1 is reported diagnostically for the complete IBI+ML Hamiltonian. "
            "No universal pass threshold is imposed by the generic core."
        ),
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("[POST-IBI RUNTIME STRUCTURE SUMMARY]")
    print(f"mean={summary['mean_l1']:.6f} max={summary['max_l1']:.6f}")
    for kind, value in summary["mean_l1_by_kind"].items():
        print(f"{kind}={value:.6f}")
    print("[PASS] Full-runtime bonded distributions were sampled and compared to the mapped target.")
    print("[NOTE] Structural L1 is diagnostic here; no generic hard threshold is imposed.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--priors", required=True)
    parser.add_argument("--sample-npz", required=True)
    parser.add_argument("--ibi-config", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    validate_runtime_structure(
        dataset=args.dataset,
        priors=args.priors,
        sample_npz=args.sample_npz,
        ibi_config=args.ibi_config,
        output=args.output,
    )


if __name__ == "__main__":
    main()
