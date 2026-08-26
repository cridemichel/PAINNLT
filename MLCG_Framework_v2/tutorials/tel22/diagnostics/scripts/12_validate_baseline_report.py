#!/usr/bin/env python3
"""Validate whether an existing TEL22 NVE report can be reused as the FP32 baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--dts", nargs="+", type=float, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--neighbor-search", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument(
        "--allow-unverified-fp32", action="store_true",
        help="Reuse an otherwise fully matched FP32 baseline when old run logs contain no precision banner",
    )
    args = parser.parse_args()

    if args.precision != "float32":
        print("[BASELINE REUSE] disabled: existing baseline is only reused for float32 diagnostics")
        return 1
    if not args.report.is_file():
        print(f"[BASELINE REUSE] unavailable: {args.report}")
        return 1

    report = json.loads(args.report.read_text(encoding="utf-8"))
    expected_paths = {
        "config": args.tutorial / "tel22_training_config.json",
        "priors": args.tutorial / "cg_priors.json",
        "rb_info": args.tutorial / "rigid_bodies_info.json",
        "dataset": args.tutorial / "tel22_dataset.bin",
        "checkpoint": args.tutorial / "equilibrated.npz",
        "model": args.tutorial / "tel22_model.pt",
        "model_manifest": args.tutorial / "tel22_model.pt.manifest.json",
    }
    recorded = report.get("inputs_sha256", {})
    failures: list[str] = []
    for key, path in expected_paths.items():
        if not path.is_file():
            failures.append(f"missing current input {path}")
            continue
        digest = sha256_file(path)
        if recorded.get(key) != digest:
            failures.append(f"{key} hash mismatch")

    if report.get("device") != args.device:
        failures.append(f"device={report.get('device')!r}, wanted {args.device!r}")
    if report.get("neighbor_search") != args.neighbor_search:
        failures.append(
            f"neighbor_search={report.get('neighbor_search')!r}, wanted {args.neighbor_search!r}"
        )
    actual_dts = sorted(float(row["dt_ps"]) for row in report.get("runs", []))
    if actual_dts != sorted(args.dts):
        failures.append(f"dt grid={actual_dts}, wanted {sorted(args.dts)}")

    # certify_nve does not currently store ml_precision in the report. Confirm it
    # from a run log when available. Older/macOS logs can miss the C++ banner
    # because stdout is buffered, so also accept the Cython activation banner.
    run_logs = [Path(str(row.get("run_log", ""))) for row in report.get("runs", [])]
    checked_precision = False
    for log in run_logs:
        if not log.is_file():
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        if (
            "Inference precision: float64" in text
            or "precision=float64" in text
            or '"ml_precision": "float64"' in text
        ):
            failures.append(f"baseline log indicates float64: {log}")
            checked_precision = True
            break
        if (
            "Inference precision: float32" in text
            or "precision=float32" in text
            or '"ml_precision": "float32"' in text
        ):
            checked_precision = True
            break
    if not checked_precision:
        if args.allow_unverified_fp32:
            print(
                "[BASELINE REUSE] WARNING: old run logs contain no precision banner; "
                "accepting the otherwise fully matched baseline because "
                "--allow-unverified-fp32 was explicitly requested"
            )
        else:
            failures.append(
                "could not verify FP32 precision from existing run logs "
                "(use --allow-unverified-fp32 only for a baseline known independently to be FP32)"
            )

    if failures:
        print("[BASELINE REUSE] rejected:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"[BASELINE REUSE] validated: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
