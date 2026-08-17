#!/usr/bin/env python3
"""Summarize the non-promotional periodic conservative-dihedral IBI test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
KIND = "periodic_conservative_dihedral_ibi_test"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path, label: str) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    data = json.loads(path.read_text())
    return data


def nve_metrics(
    report: dict, *, p_min: float, p_max: float, r2_min: float, c2_spread_max: float,
    max_relative_drift: float, required_max_dt: float,
) -> dict:
    strict = report.get("strict_reference") or report.get("certification")
    if not isinstance(strict, dict):
        raise ValueError("NVE report has no strict_reference/certification block")
    scaling = strict.get("scaling")
    if not isinstance(scaling, dict):
        raise ValueError("NVE report has no scaling block")
    runs = list(report.get("runs", []))
    if len(runs) < 3:
        raise ValueError("NVE report has fewer than three timestep runs")
    c2 = []
    for item in runs:
        dt = float(item["dt_ps"])
        sigma = float(item["sigma_E"])
        if dt <= 0.0 or sigma < 0.0 or not np.isfinite(dt + sigma):
            raise ValueError("NVE report contains invalid dt/sigma_E")
        c2.append(sigma / (dt * dt))
    c2 = np.asarray(c2, dtype=float)
    positive = c2[c2 > 0.0]
    spread = float(np.max(positive) / np.min(positive)) if positive.size else float("inf")
    max_drift = max(float(item["relative_block_mean_drift"]) for item in runs)
    max_dt = max(float(item["dt_ps"]) for item in runs)
    p = float(scaling["exponent_p"])
    r2 = float(scaling["loglog_r2"])
    checks = {
        "quadratic_exponent": p_min <= p <= p_max,
        "loglog_r2": r2 >= r2_min,
        "c2_spread": spread <= c2_spread_max,
        "relative_block_drift": max_drift <= max_relative_drift,
        "full_dt_reached": max_dt >= required_max_dt - 1.0e-12,
    }
    return {
        "exponent_p": p,
        "loglog_r2": r2,
        "c2_spread_max_over_min": spread,
        "max_relative_block_drift": max_drift,
        "max_dt_ps": max_dt,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-report", required=True, type=Path)
    parser.add_argument("--ibi-report", required=True, type=Path)
    parser.add_argument("--conversion-report", required=True, type=Path)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--parity-report", required=True, type=Path)
    parser.add_argument("--structure-report", required=True, type=Path)
    parser.add_argument("--nve-report", required=True, type=Path)
    parser.add_argument("--candidate-priors", required=True, type=Path)
    parser.add_argument("--baseline-certification", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nve-p-min", type=float, required=True)
    parser.add_argument("--nve-p-max", type=float, required=True)
    parser.add_argument("--nve-r2-min", type=float, required=True)
    parser.add_argument("--nve-c2-spread-max", type=float, required=True)
    parser.add_argument("--nve-max-relative-drift", type=float, required=True)
    parser.add_argument("--nve-required-max-dt", type=float, required=True)
    parser.add_argument("--model-config-provenance", type=Path, default=None)
    args = parser.parse_args()

    seed = load(args.seed_report, "seed report")
    ibi = load(args.ibi_report, "IBI report")
    conversion = load(args.conversion_report, "conversion report")
    validation = load(args.validation_report, "validation report")
    parity = load(args.parity_report, "runtime parity report")
    structure = load(args.structure_report, "runtime structure report")
    nve = load(args.nve_report, "NVE diagnostic report")
    candidate = args.candidate_priors.expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)

    if seed.get("kind") not in {"tel22_periodic_dihedral_ibi_test_seed", "periodic_dihedral_ibi_test_seed"} or not seed.get("pass", False):
        raise ValueError("Invalid/failed dihedral seed report")
    if int(seed.get("dihedral_occurrences", 0)) <= 0 or int(seed.get("unique_groups", 0)) <= 0:
        raise ValueError("Dihedral test seed contains no derived torsions")
    expected_groups = int(seed["unique_groups"])
    if int(ibi.get("ibi_groups", -1)) != expected_groups or int(ibi.get("dbi_groups", -1)) != 0:
        raise ValueError(
            "Step-35 IBI isolation failed: expected exactly the derived dihedral groups "
            f"({expected_groups} IBI, 0 DBI), got "
            f"{ibi.get('ibi_groups')} IBI and {ibi.get('dbi_groups')} DBI"
        )
    for iteration in ibi.get("metrics", []):
        unexpected = [
            key for key, item in iteration.get("groups", {}).items()
            if item.get("kind") != "dihedral"
        ]
        if unexpected:
            raise ValueError(
                "Step-35 IBI isolation failed: non-dihedral groups were sampled/updated: "
                + ", ".join(unexpected)
            )
    if not validation.get("pass", False):
        raise ValueError("Conservative spline validation did not pass")
    if not parity.get("pass", False):
        raise ValueError("Runtime/preprocessing parity did not pass")
    if not structure.get("pass", False):
        raise ValueError("Runtime structure sampling did not complete")

    converted_dihedrals = [r for r in conversion.get("records", []) if r.get("kind") == "dihedral"]
    if not converted_dihedrals:
        raise ValueError("Conversion report contains no converted dihedral tables")
    if len(converted_dihedrals) != expected_groups:
        raise ValueError(
            f"Expected {expected_groups} converted dihedral tables, got {len(converted_dihedrals)}"
        )
    non_dihedral_converted = [r for r in conversion.get("records", []) if r.get("kind") != "dihedral"]
    if non_dihedral_converted:
        raise ValueError("Test conversion unexpectedly reconverted production bond/angle tables")
    passthrough = list(conversion.get("passthrough_records", []))
    if not passthrough or not all(bool(r.get("byte_identical")) for r in passthrough):
        raise ValueError("Production conservative passthrough tables are not proven byte-identical")

    fd_dihedrals = [
        item for item in validation.get("finite_difference_checks", [])
        if item.get("kind") == "dihedral"
    ]
    if not fd_dihedrals:
        raise ValueError("Validation report has no dihedral finite-difference checks")

    nve_summary = nve_metrics(
        nve, p_min=args.nve_p_min, p_max=args.nve_p_max, r2_min=args.nve_r2_min,
        c2_spread_max=args.nve_c2_spread_max, max_relative_drift=args.nve_max_relative_drift,
        required_max_dt=args.nve_required_max_dt,
    )
    by_kind = structure.get("mean_l1_by_kind", {})
    dihedral_l1 = float(by_kind.get("dihedral", float("nan")))
    if not np.isfinite(dihedral_l1):
        raise ValueError("Runtime structure report has no finite dihedral L1")

    ibi_metrics = list(ibi.get("metrics", []))
    last_ibi_dihedral_l1 = {}
    if ibi_metrics:
        for key, item in ibi_metrics[-1].get("groups", {}).items():
            if item.get("kind") == "dihedral":
                last_ibi_dihedral_l1[key] = float(item["distribution_l1"])

    baseline = None
    if args.baseline_certification is not None and args.baseline_certification.is_file():
        base = load(args.baseline_certification, "baseline certification")
        sigma = base.get("gates", {}).get("fresh_sigma_E_quadratic_scaling", {})
        if sigma:
            baseline = {
                "report": str(args.baseline_certification.expanduser().resolve()),
                "exponent_p": float(sigma.get("exponent_p", float("nan"))),
                "loglog_r2": float(sigma.get("loglog_r2", float("nan"))),
                "c2_spread_max_over_min": float(sigma.get("c2_spread_max_over_min", float("nan"))),
            }

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "scope": "test-only one/few-iteration periodic backbone IBI -> ConservativeSplineDihedral",
        "production_modified": False,
        "promotion_performed": False,
        "candidate_priors": str(candidate),
        "candidate_priors_sha256": sha256_file(candidate),
        "dihedral_seed": {
            "occurrences": int(seed["dihedral_occurrences"]),
            "unique_groups": int(seed["unique_groups"]),
            "groups": seed.get("groups", {}),
        },
        "ibi_iterations_completed": len(ibi_metrics),
        "ibi_last_sampled_dihedral_l1": last_ibi_dihedral_l1,
        "ibi_isolation": {
            "expected_dihedral_groups": expected_groups,
            "ibi_groups": int(ibi["ibi_groups"]),
            "dbi_groups": int(ibi["dbi_groups"]),
            "non_dihedral_groups_updated": 0,
            "pass": True,
        },
        "conversion": {
            "converted_dihedral_tables": len(converted_dihedrals),
            "passthrough_conservative_tables": len(passthrough),
            "passthrough_byte_identical": True,
        },
        "kernel_validation": {
            "max_dihedral_fd_abs_error": max(float(x["max_abs_dU_dq_error"]) for x in fd_dihedrals),
            "runtime_max_force_abs_error": float(parity["worst_force_abs_error"]),
            "runtime_max_energy_abs_error": float(parity["worst_energy_abs_error"]),
            "pass": True,
        },
        "runtime_structure": {
            "mean_l1_by_kind": by_kind,
            "dihedral_mean_l1": dihedral_l1,
            "diagnostic_only": True,
        },
        "candidate_nve": nve_summary,
        "production_baseline_reference": baseline,
        "infrastructure_pass": True,
        "candidate_order2_through_configured_max_dt": bool(nve_summary["pass"]),
        "nve_acceptance_policy": {"p_min": args.nve_p_min, "p_max": args.nve_p_max, "r2_min": args.nve_r2_min, "c2_spread_max": args.nve_c2_spread_max, "max_relative_drift": args.nve_max_relative_drift, "required_max_dt_ps": args.nve_required_max_dt},
        "model_config_provenance": str(args.model_config_provenance.resolve()) if args.model_config_provenance else None,
        "model_config_provenance_sha256": sha256_file(args.model_config_provenance.resolve()) if args.model_config_provenance else None,
        "promotion_ready": False,
        "notes": [
            "This test does not modify or promote production priors.",
            "A short IBI test is not a convergence/structure certification; promotion_ready remains false by design.",
            "candidate_order2_through_configured_max_dt reports whether the short conservative candidate passes the configured sigma_E(dt) acceptance window.",
        ],
    }
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[CONSERVATIVE PERIODIC DIHEDRAL IBI TEST]")
    print(
        f"[SEED] occurrences={report['dihedral_seed']['occurrences']} "
        f"groups={report['dihedral_seed']['unique_groups']} IBI_iterations={report['ibi_iterations_completed']}"
    )
    print(
        f"[KERNEL] dihedral FDmax={report['kernel_validation']['max_dihedral_fd_abs_error']:.3e} "
        f"parityF={report['kernel_validation']['runtime_max_force_abs_error']:.3e} "
        f"parityE={report['kernel_validation']['runtime_max_energy_abs_error']:.3e} pass=True"
    )
    print(f"[STRUCT] dihedral mean L1={dihedral_l1:.6f} (diagnostic only)")
    print(
        f"[NVE] p={nve_summary['exponent_p']:.6f} R2={nve_summary['loglog_r2']:.6f} "
        f"C2spread={nve_summary['c2_spread_max_over_min']:.3f} "
        f"maxdrift={nve_summary['max_relative_block_drift']:.3e} "
        f"maxdt={nve_summary['max_dt_ps']:.3f} pass={nve_summary['pass']}"
    )
    if baseline is not None:
        print(
            f"[REFERENCE production] p={baseline['exponent_p']:.6f} "
            f"R2={baseline['loglog_r2']:.6f} C2spread={baseline['c2_spread_max_over_min']:.3f}"
        )
    print(f"[FINAL] infrastructure_pass=True candidate_order2_through_configured_max_dt={nve_summary['pass']} promotion_ready=False")
    print(f"[DONE] report: {out}")


if __name__ == "__main__":
    main()
