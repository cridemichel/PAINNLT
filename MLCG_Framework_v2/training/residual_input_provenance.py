#!/usr/bin/env python3
"""Record and verify provenance for a residual force-matching training set.

The residual dataset itself is the only prior-dependent input consumed by the
C++ trainer.  Runtime rigid-body metadata and priors are nevertheless part of
the same Hamiltonian definition.  This utility cryptographically binds those
artifacts to the exact atomistic inputs used to rebuild the residual dataset
and can fail closed before training if anything changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
FRAMEWORK = "MLCG_Framework_v2"
KIND = "residual_training_inputs"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def artifact(path: str | Path) -> dict[str, Any]:
    resolved = canonical(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def referenced_prior_artifacts(priors_path: str | Path) -> dict[str, str]:
    priors = canonical(priors_path)
    if not priors.is_file():
        raise FileNotFoundError(priors)
    data = json.loads(priors.read_text())
    hashes = {str(priors): sha256_file(priors)}
    for key in ("bonds", "angles", "dihedrals"):
        for idx, entry in enumerate(data.get(key, [])):
            entry_type = str(entry.get("type", "")).lower()
            if entry_type not in {"tabulated", "conservative_spline"}:
                continue
            if "file" not in entry:
                raise ValueError(f"Referenced {entry_type} {key}[{idx}] is missing 'file'")
            table = Path(str(entry["file"])).expanduser()
            if not table.is_absolute():
                table = priors.parent / table
            table = table.resolve()
            if not table.is_file():
                raise FileNotFoundError(f"Missing referenced table for {key}[{idx}]: {table}")
            hashes[str(table)] = sha256_file(table)
    return dict(sorted(hashes.items()))


def _validate_conservative_conversion_report(
    report: dict[str, Any], report_file: Path, priors: Path
) -> dict[str, Any]:
    if report.get("framework") != FRAMEWORK or report.get("kind") != "ibi_conservative_spline_validation":
        raise ValueError(f"Unsupported conservative IBI validation report: {report_file}")
    if report.get("pass") is not True:
        raise ValueError(f"Conservative IBI validation did not pass: {report_file}")

    reported_priors = canonical(report.get("conservative_priors", ""))
    if reported_priors != priors:
        raise ValueError(
            "Conservative IBI validation report refers to different priors: "
            f"report={reported_priors}, selected={priors}"
        )
    if report.get("conservative_priors_sha256") != sha256_file(priors):
        raise ValueError(f"Conservative priors changed after validation: {priors}")

    conversion_path = canonical(report.get("conversion_report", ""))
    if not conversion_path.is_file():
        raise FileNotFoundError(conversion_path)
    conversion = json.loads(conversion_path.read_text())
    if (
        int(conversion.get("schema_version", -1)) != 1
        or conversion.get("framework") != FRAMEWORK
        or conversion.get("kind") != "ibi_conservative_spline_conversion"
    ):
        raise ValueError(f"Unsupported conservative IBI conversion report: {conversion_path}")
    if conversion.get("source_artifacts_unchanged") is not True:
        raise ValueError(f"Conservative conversion did not preserve source artifacts: {conversion_path}")
    if canonical(conversion.get("output_priors", "")) != priors:
        raise ValueError("Conservative conversion report refers to different output priors")
    if conversion.get("output_priors_sha256") != sha256_file(priors):
        raise ValueError(f"Converted conservative priors changed after conversion: {priors}")

    source_priors = canonical(conversion.get("source_priors", ""))
    if not source_priors.is_file():
        raise FileNotFoundError(source_priors)
    if conversion.get("source_priors_sha256") != sha256_file(source_priors):
        raise ValueError(f"Source IBI priors changed after conservative conversion: {source_priors}")

    converted_paths: set[Path] = set()
    for idx, rec in enumerate(conversion.get("records", [])):
        output_path = canonical(rec.get("output_path", ""))
        if not output_path.is_file():
            raise FileNotFoundError(output_path)
        if rec.get("output_sha256") != sha256_file(output_path):
            raise ValueError(f"Converted spline changed after conversion: {output_path}")
        source_path = canonical(rec.get("source_path", ""))
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if rec.get("source_sha256") != sha256_file(source_path):
            raise ValueError(f"Source IBI table changed after conservative conversion: {source_path}")
        expected_output = (priors.parent / str(rec.get("output_file", ""))).resolve()
        if output_path != expected_output:
            raise ValueError(
                f"Conversion record {idx} output path mismatch: report={output_path}, expected={expected_output}"
            )
        converted_paths.add(output_path)

    selected = json.loads(priors.read_text())
    selected_conservative_paths: set[Path] = set()
    for key in ("bonds", "angles", "dihedrals"):
        for idx, entry in enumerate(selected.get(key, [])):
            if str(entry.get("type", "")).lower() != "conservative_spline":
                continue
            if "file" not in entry:
                raise ValueError(f"Referenced conservative spline {key}[{idx}] is missing 'file'")
            table = Path(str(entry["file"])).expanduser()
            if not table.is_absolute():
                table = priors.parent / table
            selected_conservative_paths.add(table.resolve())
    if selected_conservative_paths != converted_paths:
        missing = sorted(str(p) for p in selected_conservative_paths - converted_paths)
        extra = sorted(str(p) for p in converted_paths - selected_conservative_paths)
        raise ValueError(
            "Conservative conversion records do not exactly match selected spline tables: "
            f"missing={missing}, extra={extra}"
        )
    return conversion


def validate_runtime_parity_report(
    report_path: str | Path, priors_path: str | Path
) -> dict[str, Any]:
    report_file = canonical(report_path)
    if not report_file.is_file():
        raise FileNotFoundError(report_file)
    report = json.loads(report_file.read_text())
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("framework") != FRAMEWORK
        or report.get("kind") != "ibi_conservative_spline_runtime_parity"
        or report.get("pass") is not True
    ):
        raise ValueError(f"Unsupported conservative spline runtime parity report: {report_file}")

    priors = canonical(priors_path)
    if canonical(report.get("priors", "")) != priors:
        raise ValueError("Runtime parity report refers to different conservative priors")
    if report.get("priors_sha256") != sha256_file(priors):
        raise ValueError(f"Conservative priors changed after runtime parity validation: {priors}")

    current = referenced_prior_artifacts(priors)
    recorded = report.get("prior_artifact_sha256")
    if not isinstance(recorded, dict) or current != recorded:
        raise ValueError("Conservative prior artifacts changed after runtime parity validation")

    worst_f = float(report.get("worst_force_abs_error", float("inf")))
    worst_e = float(report.get("worst_energy_abs_error", float("inf")))
    force_atol = float(report.get("force_atol", -1.0))
    energy_atol = float(report.get("energy_atol", -1.0))
    if force_atol < 0.0 or energy_atol < 0.0 or worst_f > force_atol or worst_e > energy_atol:
        raise ValueError(
            "Conservative runtime parity report does not satisfy its recorded tolerances: "
            f"force={worst_f}/{force_atol}, energy={worst_e}/{energy_atol}"
        )
    return report


def validate_ibi_validation_report(
    report_path: str | Path,
    priors_path: str | Path,
    runtime_parity_report: str | Path | None = None,
) -> dict[str, Any]:
    report_file = canonical(report_path)
    if not report_file.is_file():
        raise FileNotFoundError(report_file)
    report = json.loads(report_file.read_text())
    if int(report.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported IBI validation report: {report_file}")

    priors = canonical(priors_path)
    if report.get("kind") == "ibi_conservative_spline_validation":
        conversion = _validate_conservative_conversion_report(report, report_file, priors)
        if runtime_parity_report is None:
            raise ValueError(
                "Conservative IBI provenance requires a persisted ESPResSo/runtime parity report. "
                "Re-run tutorials/tel22_IBI/22_validate_conservative_spline.sh."
            )
        parity = validate_runtime_parity_report(runtime_parity_report, priors)
        fd_checks = report.get("finite_difference_checks", [])
        if not isinstance(fd_checks, list) or not fd_checks:
            raise ValueError(f"Conservative validation report has no finite-difference checks: {report_file}")
        max_fd = max(float(item.get("max_abs_dU_dq_error", float("inf"))) for item in fd_checks)
        return {
            "mode": "conservative_spline_validation",
            "report": report,
            "conversion_report": conversion,
            "runtime_parity_report": parity,
            "max_abs_dU_dq_error": max_fd,
            "runtime_max_force_abs_error": float(parity["worst_force_abs_error"]),
            "runtime_max_energy_abs_error": float(parity["worst_energy_abs_error"]),
        }

    if report.get("mode") != "read_only_validation":
        raise ValueError(f"Unsupported IBI validation report: {report_file}")
    if report.get("source_priors_unchanged") is not True:
        raise ValueError(f"IBI validation did not certify immutable source priors: {report_file}")

    reported_priors = canonical(report.get("priors", ""))
    if reported_priors != priors:
        raise ValueError(
            "IBI validation report refers to different priors: "
            f"report={reported_priors}, selected={priors}"
        )

    current = referenced_prior_artifacts(priors)
    recorded = report.get("source_artifact_sha256")
    if not isinstance(recorded, dict):
        raise ValueError(f"IBI validation report has no source_artifact_sha256 map: {report_file}")
    if current != recorded:
        differing = sorted(set(current) | set(recorded))
        differing = [p for p in differing if current.get(p) != recorded.get(p)]
        raise ValueError(
            "Validated IBI prior artifacts changed after read-only validation: " + ", ".join(differing)
        )
    return {
        "mode": "read_only_validation",
        "report": report,
        "mean_l1": float(report["mean_l1"]),
        "max_l1": float(report["max_l1"]),
    }


def validate_readonly_report(report_path: str | Path, priors_path: str | Path) -> dict[str, Any]:
    """Backward-compatible validator for the legacy read-only IBI report."""
    result = validate_ibi_validation_report(report_path, priors_path)
    if result["mode"] != "read_only_validation":
        raise ValueError(f"Expected legacy read-only IBI validation report: {canonical(report_path)}")
    return result["report"]


def write_manifest(
    *,
    output: str | Path,
    dataset: str | Path,
    rb_info: str | Path,
    priors: str | Path,
    aa_topology: str | Path,
    aa_trajectory: str | Path,
    mapping_config: str | Path,
    validation_report: str | Path,
    runtime_parity_report: str | Path | None = None,
) -> dict[str, Any]:
    validation = validate_ibi_validation_report(
        validation_report, priors, runtime_parity_report=runtime_parity_report
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "framework": FRAMEWORK,
        "kind": KIND,
        "build_inputs": {
            "aa_topology": artifact(aa_topology),
            "aa_trajectory": artifact(aa_trajectory),
            "mapping_config": artifact(mapping_config),
            "priors": artifact(priors),
            "prior_artifact_sha256": referenced_prior_artifacts(priors),
            "ibi_validation_report": artifact(validation_report),
            "ibi_validation_mode": validation["mode"],
        },
        "outputs": {
            "dataset": artifact(dataset),
            "rb_info": artifact(rb_info),
        },
    }
    if validation["mode"] == "read_only_validation":
        manifest["build_inputs"]["ibi_validation_mean_l1"] = validation["mean_l1"]
        manifest["build_inputs"]["ibi_validation_max_l1"] = validation["max_l1"]
    else:
        if runtime_parity_report is None:  # guarded above; keeps type-checkers honest
            raise ValueError("Missing conservative runtime parity report")
        manifest["build_inputs"]["ibi_runtime_parity_report"] = artifact(runtime_parity_report)
        manifest["build_inputs"]["conservative_fd_max_abs_dU_dq_error"] = validation[
            "max_abs_dU_dq_error"
        ]
        manifest["build_inputs"]["conservative_runtime_max_force_abs_error"] = validation[
            "runtime_max_force_abs_error"
        ]
        manifest["build_inputs"]["conservative_runtime_max_energy_abs_error"] = validation[
            "runtime_max_energy_abs_error"
        ]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Residual-training provenance written: {output_path}")
    return manifest


def _check_artifact(label: str, recorded: dict[str, Any], expected_path: str | Path) -> None:
    expected = canonical(expected_path)
    recorded_path = canonical(recorded.get("path", ""))
    if recorded_path != expected:
        raise ValueError(f"{label} path mismatch: manifest={recorded_path}, selected={expected}")
    if not expected.is_file():
        raise FileNotFoundError(expected)
    size = expected.stat().st_size
    if int(recorded.get("size_bytes", -1)) != size:
        raise ValueError(f"{label} size mismatch: manifest={recorded.get('size_bytes')}, current={size}")
    digest = sha256_file(expected)
    if recorded.get("sha256") != digest:
        raise ValueError(f"{label} SHA256 mismatch: {expected}")


def check_manifest(
    *,
    manifest_path: str | Path,
    dataset: str | Path,
    rb_info: str | Path,
    priors: str | Path,
) -> dict[str, Any]:
    path = canonical(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing residual provenance manifest: {path}. Re-run the residual dataset rebuild first."
        )
    manifest = json.loads(path.read_text())
    if (
        int(manifest.get("schema_version", -1)) != SCHEMA_VERSION
        or manifest.get("framework") != FRAMEWORK
        or manifest.get("kind") != KIND
    ):
        raise ValueError(f"Unsupported residual provenance manifest: {path}")

    _check_artifact("dataset", manifest["outputs"]["dataset"], dataset)
    _check_artifact("rb_info", manifest["outputs"]["rb_info"], rb_info)
    _check_artifact("priors", manifest["build_inputs"]["priors"], priors)

    current_prior_hashes = referenced_prior_artifacts(priors)
    recorded_prior_hashes = manifest["build_inputs"].get("prior_artifact_sha256")
    if current_prior_hashes != recorded_prior_hashes:
        differing = sorted(set(current_prior_hashes) | set(recorded_prior_hashes or {}))
        differing = [
            p for p in differing
            if current_prior_hashes.get(p) != (recorded_prior_hashes or {}).get(p)
        ]
        raise ValueError("Prior table provenance mismatch: " + ", ".join(differing))

    validation_record = manifest["build_inputs"]["ibi_validation_report"]
    validation_path = canonical(validation_record["path"])
    _check_artifact("ibi_validation_report", validation_record, validation_path)
    parity_path = None
    if manifest["build_inputs"].get("ibi_validation_mode") == "conservative_spline_validation":
        parity_record = manifest["build_inputs"].get("ibi_runtime_parity_report")
        if not isinstance(parity_record, dict):
            raise ValueError("Conservative residual provenance has no runtime parity report")
        parity_path = canonical(parity_record.get("path", ""))
        _check_artifact("ibi_runtime_parity_report", parity_record, parity_path)
    validate_ibi_validation_report(validation_path, priors, runtime_parity_report=parity_path)

    print("[IBI TRAINING INPUT CHECK]")
    print(f"dataset  : {canonical(dataset)}")
    print(f"rb_info  : {canonical(rb_info)}")
    print(f"priors   : {canonical(priors)}")
    print(f"manifest : {path}")
    print(f"dataset_sha256 = {sha256_file(canonical(dataset))}")
    print(f"rb_info_sha256 = {sha256_file(canonical(rb_info))}")
    print(f"priors_sha256  = {sha256_file(canonical(priors))}")
    print("[PASS] Residual dataset, rigid-body metadata, validated IBI priors and tables match build provenance.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Record provenance immediately after residual dataset rebuild")
    rec.add_argument("--output", required=True)
    rec.add_argument("--dataset", required=True)
    rec.add_argument("--rb-info", required=True)
    rec.add_argument("--priors", required=True)
    rec.add_argument("--aa-topology", required=True)
    rec.add_argument("--aa-trajectory", required=True)
    rec.add_argument("--mapping-config", required=True)
    rec.add_argument("--validation-report", required=True)
    rec.add_argument("--runtime-parity-report", default=None)

    chk = sub.add_parser("check", help="Fail closed if selected training artifacts differ from provenance")
    chk.add_argument("--manifest", required=True)
    chk.add_argument("--dataset", required=True)
    chk.add_argument("--rb-info", required=True)
    chk.add_argument("--priors", required=True)

    args = parser.parse_args()
    if args.command == "record":
        write_manifest(
            output=args.output,
            dataset=args.dataset,
            rb_info=args.rb_info,
            priors=args.priors,
            aa_topology=args.aa_topology,
            aa_trajectory=args.aa_trajectory,
            mapping_config=args.mapping_config,
            validation_report=args.validation_report,
            runtime_parity_report=args.runtime_parity_report,
        )
    else:
        check_manifest(
            manifest_path=args.manifest,
            dataset=args.dataset,
            rb_info=args.rb_info,
            priors=args.priors,
        )


if __name__ == "__main__":
    main()
