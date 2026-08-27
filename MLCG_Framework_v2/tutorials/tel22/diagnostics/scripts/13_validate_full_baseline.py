#!/usr/bin/env python3
"""Validate a reusable full TEL22 FP32 NVE baseline report and run plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_value(command: list[str], flag: str) -> str | None:
    try:
        idx = command.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(command):
        return None
    return command[idx + 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--dts", nargs="+", type=float, required=True)
    parser.add_argument("--duration-ps", type=float, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--neighbor-search", required=True)
    parser.add_argument("--precision", required=True)
    args = parser.parse_args()

    report_path = args.report.resolve()
    tutorial = args.tutorial.resolve()
    if not report_path.is_file():
        print(f"[FULL BASELINE REUSE] unavailable: {report_path}")
        return 1
    if args.precision != "float32":
        print("[FULL BASELINE REUSE] disabled: this diagnostic only reuses the known FP32 baseline")
        return 1

    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    definition = report.get("definition", {})
    if definition.get("hamiltonian_mode") != "model_active":
        failures.append(f"hamiltonian_mode={definition.get('hamiltonian_mode')!r}, wanted 'model_active'")

    expected_paths = {
        "config": tutorial / "tel22_training_config.json",
        "priors": tutorial / "cg_priors.json",
        "rb_info": tutorial / "rigid_bodies_info.json",
        "dataset": tutorial / "tel22_dataset.bin",
        "checkpoint": tutorial / "equilibrated.npz",
        "model": tutorial / "tel22_model.pt",
        "model_manifest": tutorial / "tel22_model.pt.manifest.json",
    }
    recorded = report.get("inputs_sha256", {})
    for key, path in expected_paths.items():
        if not path.is_file():
            failures.append(f"missing current input {path}")
            continue
        if recorded.get(key) != sha256_file(path):
            failures.append(f"{key} hash mismatch")

    if report.get("device") != args.device:
        failures.append(f"device={report.get('device')!r}, wanted {args.device!r}")
    if report.get("neighbor_search") != args.neighbor_search:
        failures.append(
            f"neighbor_search={report.get('neighbor_search')!r}, wanted {args.neighbor_search!r}"
        )

    runs = report.get("runs", [])
    actual_dts = sorted(float(row["dt_ps"]) for row in runs)
    wanted_dts = sorted(args.dts)
    if actual_dts != wanted_dts:
        failures.append(f"dt grid={actual_dts}, wanted {wanted_dts}")
    for row in runs:
        dt = float(row["dt_ps"])
        wanted_duration = round(args.duration_ps / dt) * dt
        actual_duration = float(row["duration_ps"])
        tol = max(1.0e-10, 1.0e-8 * max(1.0, wanted_duration))
        if abs(actual_duration - wanted_duration) > tol:
            failures.append(
                f"dt={dt:g} duration={actual_duration:.17g}, wanted {wanted_duration:.17g}"
            )

    # The report schema predates recording ml_precision directly. The sibling
    # run_plan.json does retain the exact pypresso command, so use it as the
    # authoritative precision/ML-active check instead of guessing from logs.
    plan_path = report_path.parent / "run_plan.json"
    if not plan_path.is_file():
        failures.append(f"missing sibling run plan {plan_path}")
    else:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_runs = plan.get("runs", [])
        plan_dts = sorted(float(row["dt_ps"]) for row in plan_runs)
        if plan_dts != wanted_dts:
            failures.append(f"run_plan dt grid={plan_dts}, wanted {wanted_dts}")
        for row in plan_runs:
            command = [str(x) for x in row.get("command", [])]
            precision = command_value(command, "--ml_precision")
            if precision != args.precision:
                failures.append(
                    f"run_plan dt={row.get('dt_ps')} precision={precision!r}, wanted {args.precision!r}"
                )
            if "--disable_ml" in command:
                failures.append(f"run_plan dt={row.get('dt_ps')} unexpectedly disables ML")
            model = command_value(command, "--model")
            if model is None:
                failures.append(f"run_plan dt={row.get('dt_ps')} has no --model")

    if failures:
        print("[FULL BASELINE REUSE] rejected:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"[FULL BASELINE REUSE] validated: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
