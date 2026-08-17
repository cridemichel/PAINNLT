#!/usr/bin/env python3
"""Validate conservative spline artifacts without running dynamics.

Checks source/output provenance, exact Hermite derivative consistency by finite
differences, and reports tabulated-to-spline fidelity.  Runtime/preprocessing
force parity is a separate ESPResSo diagnostic because it exercises the C++
plugin rather than duplicating it in NumPy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))

from conservative_spline import load_conservative_spline, conservative_spline_value  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(conversion_report: Path, *, fd_rtol: float = 2.0e-6, fd_atol: float = 2.0e-6) -> dict:
    conversion_report = conversion_report.expanduser().resolve()
    report = json.loads(conversion_report.read_text())
    if report.get("kind") != "ibi_conservative_spline_conversion":
        raise ValueError(f"Unsupported conversion report: {conversion_report}")
    source_priors = Path(report["source_priors"])
    output_priors = Path(report["output_priors"])
    if sha256_file(source_priors) != report["source_priors_sha256"]:
        raise ValueError("Source IBI priors changed after conservative conversion")
    if sha256_file(output_priors) != report["output_priors_sha256"]:
        raise ValueError("Converted conservative priors changed after conversion")
    priors = json.loads(output_priors.read_text())

    record_by_output = {r["output_file"]: r for r in report["records"]}
    fd_checks = []
    seen = set()
    for json_key, kind in (("bonds", "bond"), ("angles", "angle"), ("dihedrals", "dihedral")):
        for idx, entry in enumerate(priors.get(json_key, [])):
            if str(entry.get("type", "")).lower() != "conservative_spline":
                continue
            key = (kind, entry["file"])
            if key in seen:
                continue
            seen.add(key)
            if entry["file"] not in record_by_output:
                raise ValueError(f"Converted {json_key}[{idx}] has no conversion record")
            rec = record_by_output[entry["file"]]
            table_path = output_priors.parent / entry["file"]
            if sha256_file(table_path) != rec["output_sha256"]:
                raise ValueError(f"Converted spline changed after conversion: {table_path}")
            table = load_conservative_spline(entry, kind=kind, priors_path=output_priors)
            # Probe interval interiors, away from knots where central FD would
            # straddle two C1 segments and unnecessarily reduce FD accuracy.
            hgrid = (table.maximum - table.minimum) / (len(table.x) - 1)
            indices = np.unique(np.linspace(0, len(table.x) - 2, 41).astype(int))
            max_abs = 0.0
            max_rel = 0.0
            for i in indices:
                q = float(table.x[i] + 0.371 * hgrid)
                eps = min(1.0e-6, 1.0e-3 * hgrid)
                up, _ = conservative_spline_value(table, q + eps)
                um, _ = conservative_spline_value(table, q - eps)
                _u, analytic = conservative_spline_value(table, q)
                fd = (up - um) / (2.0 * eps)
                err = abs(fd - analytic)
                rel = err / max(abs(fd), abs(analytic), 1.0)
                max_abs = max(max_abs, err)
                max_rel = max(max_rel, rel)
                if err > fd_atol + fd_rtol * max(abs(fd), abs(analytic)):
                    raise RuntimeError(
                        f"Finite-difference conservative check failed for {table_path} at q={q}: "
                        f"analytic={analytic}, fd={fd}, abs_err={err}"
                    )
            fd_checks.append({
                "kind": kind, "file": entry["file"],
                "max_abs_dU_dq_error": max_abs, "max_relative_error": max_rel,
            })

    result = {
        "schema_version": 1,
        "framework": "MLCG_Framework_v2",
        "kind": "ibi_conservative_spline_validation",
        "conversion_report": str(conversion_report),
        "source_priors": str(source_priors),
        "source_priors_sha256": sha256_file(source_priors),
        "conservative_priors": str(output_priors),
        "conservative_priors_sha256": sha256_file(output_priors),
        "finite_difference_checks": fd_checks,
        "fidelity": [
            {"kind": r["kind"], "file": r["output_file"], **r["fidelity"]}
            for r in report["records"]
        ],
        "pass": True,
    }
    out = conversion_report.parent / "validation_report.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[CONSERVATIVE SPLINE VALIDATION]")
    for item in fd_checks:
        print(
            f"{item['kind']:5s} {item['file']}: "
            f"FD max_abs={item['max_abs_dU_dq_error']:.3e} "
            f"max_rel={item['max_relative_error']:.3e}"
        )
    print("[PASS] Energy and analytical derivative come from the same Hermite spline to finite-difference tolerance.")
    print(f"report: {out}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-report", required=True, type=Path)
    parser.add_argument("--fd-rtol", type=float, default=2.0e-6)
    parser.add_argument("--fd-atol", type=float, default=2.0e-6)
    args = parser.parse_args()
    validate(args.conversion_report, fd_rtol=args.fd_rtol, fd_atol=args.fd_atol)


if __name__ == "__main__":
    main()
