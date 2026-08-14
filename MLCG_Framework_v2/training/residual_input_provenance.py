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
            if str(entry.get("type", "")).lower() != "tabulated":
                continue
            if "file" not in entry:
                raise ValueError(f"Tabulated {key}[{idx}] is missing 'file'")
            table = Path(str(entry["file"])).expanduser()
            if not table.is_absolute():
                table = priors.parent / table
            table = table.resolve()
            if not table.is_file():
                raise FileNotFoundError(f"Missing referenced table for {key}[{idx}]: {table}")
            hashes[str(table)] = sha256_file(table)
    return dict(sorted(hashes.items()))


def validate_readonly_report(report_path: str | Path, priors_path: str | Path) -> dict[str, Any]:
    report_file = canonical(report_path)
    if not report_file.is_file():
        raise FileNotFoundError(report_file)
    report = json.loads(report_file.read_text())
    if int(report.get("schema_version", -1)) != 1 or report.get("mode") != "read_only_validation":
        raise ValueError(f"Unsupported IBI validation report: {report_file}")
    if report.get("source_priors_unchanged") is not True:
        raise ValueError(f"IBI validation did not certify immutable source priors: {report_file}")

    priors = canonical(priors_path)
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
    return report


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
) -> dict[str, Any]:
    validation = validate_readonly_report(validation_report, priors)
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
            "ibi_validation_mean_l1": float(validation["mean_l1"]),
            "ibi_validation_max_l1": float(validation["max_l1"]),
        },
        "outputs": {
            "dataset": artifact(dataset),
            "rb_info": artifact(rb_info),
        },
    }
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
    validate_readonly_report(validation_path, priors)

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
