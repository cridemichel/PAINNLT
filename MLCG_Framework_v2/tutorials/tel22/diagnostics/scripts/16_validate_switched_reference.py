#!/usr/bin/env python3
"""Validate reuse of the existing TEL22 priors-only switched-Morse 5 ps reference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_DTS = [0.002, 0.003, 0.004, 0.005]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command_value(command: list[str], flag: str) -> str | None:
    try:
        idx = command.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(command):
        raise ValueError(f"Flag {flag} has no value in reused command")
    return command[idx + 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--rb-info", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = json.loads(args.report.read_text(encoding="utf-8"))
    plan: dict[str, Any] = json.loads(args.run_plan.read_text(encoding="utf-8"))
    reasons: list[str] = []

    definition = report.get("definition", {})
    if definition.get("hamiltonian_mode") != "conservative_classical_model_provenance_ml_disabled":
        reasons.append("reference is not --disable-ml priors-only")
    if report.get("neighbor_search", definition.get("neighbor_search")) != "link-cell":
        reasons.append("reference neighbor search is not link-cell")
    recorded_mode = report.get("morse_switch_mode", definition.get("morse_switch_mode"))
    if recorded_mode not in (None, "switched"):
        reasons.append(f"reference Morse mode is {recorded_mode!r}, not switched")

    current = {
        "config": args.config,
        "priors": args.priors,
        "rb_info": args.rb_info,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "model": args.model,
    }
    report_hashes = report.get("inputs_sha256", {})
    hash_status: dict[str, Any] = {}
    for name, path in current.items():
        digest = sha256(path)
        recorded = report_hashes.get(name)
        ok = recorded == digest
        hash_status[name] = {"current": digest, "reference": recorded, "match": ok}
        if not ok:
            reasons.append(f"input hash mismatch: {name}")

    runs = sorted(report.get("runs", []), key=lambda row: float(row["dt_ps"]))
    dts = [float(row["dt_ps"]) for row in runs]
    if len(dts) != 4 or any(abs(a - b) > 1.0e-12 for a, b in zip(dts, EXPECTED_DTS)):
        reasons.append(f"reference dt grid is {dts}, expected {EXPECTED_DTS}")
    for row in runs:
        dt = float(row["dt_ps"])
        if abs(float(row["duration_ps"]) - 5.0) > 0.5 * dt + 1.0e-12:
            reasons.append(f"reference dt={dt:g} does not cover 5 ps")

    plan_runs = plan.get("runs", [])
    if len(plan_runs) != 4:
        reasons.append(f"reference run plan has {len(plan_runs)} runs, expected 4")
    for item in plan_runs:
        command = [str(x) for x in item.get("command", [])]
        if "--disable_ml" not in command:
            reasons.append("reference command does not contain --disable_ml")
            continue
        if command_value(command, "--neighbor_search") != "link-cell":
            reasons.append("reference command does not use --neighbor_search link-cell")
        mode = command_value(command, "--morse_switch_mode")
        # Legacy run plans predate the explicit flag; absence means the production default.
        if mode not in (None, "switched"):
            reasons.append(f"reference command uses Morse mode {mode!r}")

    payload = {
        "schema_version": 1,
        "kind": "tel22_morse_switch_reference_validation",
        "reference_report": str(args.report.resolve()),
        "reference_run_plan": str(args.run_plan.resolve()),
        "legacy_absent_morse_flag_means": "switched",
        "runner_hash_intentionally_not_compared": (
            "The diagnostic patch adds an explicit Morse-mode CLI flag, so the current runner hash "
            "must differ from the already completed reference run. Physical input hashes are still exact."
        ),
        "input_hashes": hash_status,
        "pass": not reasons,
        "reasons": reasons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if reasons:
        print("[SWITCHED REFERENCE] rejected:")
        for reason in reasons:
            print(f"  - {reason}")
        return 2
    print(f"[SWITCHED REFERENCE] validated: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
