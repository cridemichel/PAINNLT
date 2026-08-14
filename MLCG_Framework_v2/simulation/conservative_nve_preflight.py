#!/usr/bin/env python3
"""Fail-closed provenance gate for conservative-IBI NVE certification."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from residual_input_provenance import (  # noqa: E402
    referenced_prior_artifacts,
    sha256_file,
    validate_ibi_validation_report,
)

SCHEMA_VERSION = 1
FRAMEWORK = "MLCG_Framework_v2"
KIND = "conservative_ibi_nve_preflight"


def canonical(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def conservative_entries(priors: dict[str, Any]) -> list[str]:
    entries: list[str] = []
    for key in ("bonds", "angles", "dihedrals"):
        for idx, entry in enumerate(priors.get(key, [])):
            if str(entry.get("type", "")).lower() == "conservative_spline":
                entries.append(f"{key}[{idx}]")
    return entries


def forbidden_tabulated_entries(priors: dict[str, Any]) -> list[str]:
    entries: list[str] = []
    for key in ("bonds", "angles", "dihedrals"):
        for idx, entry in enumerate(priors.get(key, [])):
            if str(entry.get("type", "")).lower() == "tabulated":
                entries.append(f"{key}[{idx}]")
    return entries


def validate_conservative_nve_inputs(
    *,
    priors_path: str | Path,
    validation_report: str | Path,
    runtime_parity_report: str | Path,
) -> dict[str, Any]:
    priors_file = canonical(priors_path)
    validation_file = canonical(validation_report)
    parity_file = canonical(runtime_parity_report)

    if not priors_file.is_file():
        raise FileNotFoundError(priors_file)
    priors = json.loads(priors_file.read_text())

    tabulated = forbidden_tabulated_entries(priors)
    if tabulated:
        raise ValueError(
            "Strict conservative-IBI NVE certification forbids legacy tabulated bonded priors: "
            + ", ".join(tabulated)
        )
    conservative = conservative_entries(priors)
    if not conservative:
        raise ValueError(
            "Selected priors contain no conservative_spline bonded interactions; "
            "use the generic NVE certifier for an analytic-only Hamiltonian."
        )
    unsupported_conservative = [item for item in conservative if item.startswith("dihedrals[")]
    if unsupported_conservative:
        raise ValueError(
            "Strict conservative-IBI NVE certification currently certifies bond+angle splines only; "
            "unsupported conservative torsions: " + ", ".join(unsupported_conservative)
        )

    validated = validate_ibi_validation_report(
        validation_file,
        priors_file,
        runtime_parity_report=parity_file,
    )
    if validated.get("mode") != "conservative_spline_validation":
        raise ValueError(
            "Strict conservative-IBI NVE certification requires the Phase-2 conservative validation reports"
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "framework": FRAMEWORK,
        "kind": KIND,
        "mode": "conservative_ibi_only",
        "priors": str(priors_file),
        "priors_sha256": sha256_file(priors_file),
        "prior_artifact_sha256": referenced_prior_artifacts(priors_file),
        "validation_report": str(validation_file),
        "validation_report_sha256": sha256_file(validation_file),
        "runtime_parity_report": str(parity_file),
        "runtime_parity_report_sha256": sha256_file(parity_file),
        "conservative_entries": conservative,
        "max_fd_abs_dU_dq_error": float(validated["max_abs_dU_dq_error"]),
        "runtime_max_force_abs_error": float(validated["runtime_max_force_abs_error"]),
        "runtime_max_energy_abs_error": float(validated["runtime_max_energy_abs_error"]),
        "pass": True,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priors", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--runtime-parity-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = validate_conservative_nve_inputs(
        priors_path=args.priors,
        validation_report=args.validation_report,
        runtime_parity_report=args.runtime_parity_report,
    )
    output = canonical(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[CONSERVATIVE IBI NVE PREFLIGHT]")
    print(f"priors   : {report['priors']}")
    print(f"tables   : {len(report['prior_artifact_sha256']) - 1}")
    print(f"FD max   : {report['max_fd_abs_dU_dq_error']:.3e}")
    print(f"parity F : {report['runtime_max_force_abs_error']:.3e}")
    print(f"parity E : {report['runtime_max_energy_abs_error']:.3e}")
    print(f"report   : {output}")
    print("[PASS] Conservative IBI priors and all referenced spline tables match the persisted Phase-2 validation/parity provenance.")


if __name__ == "__main__":
    main()
