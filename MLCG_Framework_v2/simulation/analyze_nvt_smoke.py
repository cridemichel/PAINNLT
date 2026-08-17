#!/usr/bin/env python3
"""Validate an NVT smoke trajectory written by simulation/run_cg_md.py.

This is a stability/finite-value diagnostic, not an energy-conservation test.
Explicit tabulated priors are therefore fully supported.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

SCHEMA_VERSION = 1
REQUIRED_COLUMNS = {
    "Step", "Time_ps", "E_tot", "E_kin", "E_kin_trans", "E_kin_rot",
    "E_class", "E_ml", "min_dist", "f_max", "torque_max",
}


def _float(row, key, row_number):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {key!r} at CSV row {row_number}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key!r} at CSV row {row_number}: {value}")
    return value


def analyze_nvt_smoke(
    energy_csv: str | Path,
    *,
    expected_steps: int,
    min_distance_nm: float = 0.15,
    max_force: float = 10000.0,
    max_kinetic: float = 5000.0,
    output: str | Path | None = None,
):
    path = Path(energy_csv).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_steps <= 0:
        raise ValueError("expected_steps must be positive")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Energy CSV has no header: {path}")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Energy CSV is missing columns: {missing}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("NVT smoke CSV must contain at least initial and final states")

    steps = []
    times = []
    e_tot = []
    e_kin = []
    f_max = []
    torque_max = []
    min_dist = []
    for i, row in enumerate(rows, start=2):
        step_float = _float(row, "Step", i)
        if not step_float.is_integer():
            raise ValueError(f"Non-integral step at CSV row {i}: {step_float}")
        steps.append(int(step_float))
        times.append(_float(row, "Time_ps", i))
        e_tot.append(_float(row, "E_tot", i))
        e_kin.append(_float(row, "E_kin", i))
        f_max.append(_float(row, "f_max", i))
        torque_max.append(_float(row, "torque_max", i))
        min_dist.append(_float(row, "min_dist", i))
        for key in ("E_kin_trans", "E_kin_rot", "E_class", "E_ml"):
            _float(row, key, i)

    if steps[0] != 0:
        raise ValueError(f"NVT smoke log must start at step 0; got {steps[0]}")
    if steps[-1] != int(expected_steps):
        raise ValueError(
            f"NVT smoke did not reach the requested final step: {steps[-1]} != {expected_steps}"
        )
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("NVT smoke steps are not strictly increasing")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("NVT smoke times are not strictly increasing")

    observed_min_dist = min(min_dist)
    observed_max_force = max(f_max)
    observed_max_kinetic = max(e_kin)
    observed_max_torque = max(torque_max)
    failures = []
    if observed_min_dist < min_distance_nm:
        failures.append(
            f"min_dist={observed_min_dist:.6g} nm < {min_distance_nm:.6g} nm"
        )
    if observed_max_force > max_force:
        failures.append(f"max_force={observed_max_force:.6g} > {max_force:.6g}")
    if observed_max_kinetic > max_kinetic:
        failures.append(f"max_Ekin={observed_max_kinetic:.6g} > {max_kinetic:.6g}")
    if failures:
        raise RuntimeError("NVT smoke stability guard failed: " + "; ".join(failures))

    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": "nvt_runtime_smoke",
        "energy_csv": str(path),
        "samples": len(rows),
        "final_step": steps[-1],
        "final_time_ps": times[-1],
        "min_distance_nm": observed_min_dist,
        "max_force": observed_max_force,
        "max_torque": observed_max_torque,
        "max_kinetic_energy": observed_max_kinetic,
        "total_energy_min": min(e_tot),
        "total_energy_max": max(e_tot),
        "total_energy_span": max(e_tot) - min(e_tot),
        "pass": True,
        "energy_conservation_certified": False,
        "note": (
            "NVT stability diagnostic only. Total-energy drift is not a certification metric; "
            "use the dedicated NVE certification workflow for energy-conservation claims."
        ),
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[POST-IBI NVT SMOKE SUMMARY]")
    print(f"samples={len(rows)} final_step={steps[-1]} final_time_ps={times[-1]:.6g}")
    print(f"min_dist={observed_min_dist:.6g} nm max_force={observed_max_force:.6g} max_torque={observed_max_torque:.6g}")
    print(f"max_Ekin={observed_max_kinetic:.6g} E_tot_span={report['total_energy_span']:.6g}")
    print("[PASS] NVT completed with finite energies/forces/torques and without runtime safety-threshold violations.")
    print("[NOTE] This is an NVT stability diagnostic, not an NVE energy-conservation certification.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-csv", required=True)
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument("--min-distance-nm", type=float, default=0.15)
    parser.add_argument("--max-force", type=float, default=10000.0)
    parser.add_argument("--max-kinetic", type=float, default=5000.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    analyze_nvt_smoke(
        args.energy_csv,
        expected_steps=args.expected_steps,
        min_distance_nm=args.min_distance_nm,
        max_force=args.max_force,
        max_kinetic=args.max_kinetic,
        output=args.output,
    )


if __name__ == "__main__":
    main()
