#!/usr/bin/env python3
"""Promote the step-33 validated angle-smoothed IBI prior transactionally.

The candidate is accepted only if its byte-level SHA256 matches the final
validation report and the expected reviewed hash.  The current production
``ibi_conservative`` directory is copied to an immutable backup, the candidate
bonded tables are staged, and production metadata are rewritten to record the
validated promotion.  Existing conservative validation/runtime reports are not
reused: a fresh conversion/validation provenance chain is emitted for the
post-promotion Hamiltonian and must be revalidated by the caller.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "preprocessing", ROOT / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from conservative_spline import conservative_spline_value, load_conservative_spline  # noqa: E402

SCHEMA_VERSION = 1
FRAMEWORK = "MLCG_Framework_v2"
KIND = "validated_ibi_angle_candidate_promotion"


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_unique_file_entries(payload: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    seen: set[tuple[str, str]] = set()
    for key, kind in (("bonds", "bond"), ("angles", "angle"), ("dihedrals", "dihedral")):
        for entry in payload.get(key, []):
            etype = str(entry.get("type", "")).lower()
            if etype not in {"conservative_spline", "tabulated"}:
                continue
            filename = str(entry.get("file", ""))
            if not filename:
                raise ValueError(f"{key} entry is missing file")
            token = (kind, filename)
            if token in seen:
                continue
            seen.add(token)
            yield kind, entry


def _artifact_map(priors_path: Path, payload: dict[str, Any]) -> dict[str, str]:
    out = {str(priors_path): sha256_file(priors_path)}
    for _kind, entry in _iter_unique_file_entries(payload):
        p = Path(str(entry["file"]))
        if not p.is_absolute():
            p = priors_path.parent / p
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        out[str(p)] = sha256_file(p)
    return dict(sorted(out.items()))


def _find_matching_entry(payload: dict[str, Any], kind: str, candidate: dict[str, Any]) -> dict[str, Any]:
    key = {"bond": "bonds", "angle": "angles", "dihedral": "dihedrals"}[kind]
    name = candidate.get("name")
    filename = candidate.get("file")
    for entry in payload.get(key, []):
        if str(entry.get("type", "")).lower() != "conservative_spline":
            continue
        if name and entry.get("name") == name:
            return entry
        if filename and entry.get("file") == filename:
            return entry
    raise KeyError(f"No pre-promotion {kind} entry matches name={name!r} file={filename!r}")


def _fidelity(old_table, new_table, *, points: int) -> dict[str, Any]:
    if not math.isclose(old_table.minimum, new_table.minimum, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
        old_table.maximum, new_table.maximum, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Promotion changed a conservative spline coordinate domain")
    q = np.linspace(old_table.minimum, old_table.maximum, points)
    old_u = np.empty_like(q)
    new_u = np.empty_like(q)
    old_du = np.empty_like(q)
    new_du = np.empty_like(q)
    for i, x in enumerate(q):
        old_u[i], old_du[i] = conservative_spline_value(old_table, float(x))
        new_u[i], new_du[i] = conservative_spline_value(new_table, float(x))
    du = new_u - old_u
    dd = new_du - old_du
    urange = max(float(np.ptp(old_u)), np.finfo(float).eps)
    dscale = max(float(np.sqrt(np.mean(old_du * old_du))), np.finfo(float).eps)
    return {
        "dense_points": int(points),
        "energy_max_abs": float(np.max(np.abs(du))),
        "energy_rms_abs": float(np.sqrt(np.mean(du * du))),
        "energy_rms_relative_to_range": float(np.sqrt(np.mean(du * du)) / urange),
        "force_max_abs": float(np.max(np.abs(dd))),
        "force_p99_abs": float(np.percentile(np.abs(dd), 99)),
        "force_rms_abs": float(np.sqrt(np.mean(dd * dd))),
        "force_rms_relative": float(np.sqrt(np.mean(dd * dd)) / dscale),
    }


def _validated_inputs(
    *,
    current_dir: Path,
    candidate_priors: Path,
    final_report: Path,
    expected_candidate_sha256: str,
    expected_sigma_rad: float,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    current_priors = current_dir / "cg_priors.json"
    if not current_priors.is_file():
        raise FileNotFoundError(current_priors)
    if not candidate_priors.is_file():
        raise FileNotFoundError(candidate_priors)
    if not final_report.is_file():
        raise FileNotFoundError(final_report)

    report = _load(final_report)
    if report.get("kind") != "ibi_angle_final_candidate_validation" or report.get("pass") is not True:
        raise ValueError("Step-33 final candidate validation did not pass")
    if report.get("validated") is not True:
        raise ValueError("Step-33 report does not mark the candidate validated")
    if not math.isclose(float(report.get("candidate_sigma_rad", math.nan)), expected_sigma_rad, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Step-33 candidate sigma mismatch")

    candidate_sha = sha256_file(candidate_priors)
    if candidate_sha != expected_candidate_sha256:
        raise ValueError(
            f"Candidate priors SHA256 mismatch: got {candidate_sha}, expected {expected_candidate_sha256}"
        )
    if str(report.get("candidate_priors_sha256", "")) != expected_candidate_sha256:
        raise ValueError("Step-33 report certifies a different candidate SHA256")
    report_path = _canonical(report.get("candidate_priors", ""))
    if report_path != candidate_priors:
        raise ValueError(f"Step-33 report candidate path mismatch: {report_path} != {candidate_priors}")

    candidate = _load(candidate_priors)
    meta = candidate.get("regularization_candidate", {})
    if not math.isclose(float(meta.get("body_sigma_rad", math.nan)), expected_sigma_rad, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Candidate regularization metadata sigma mismatch")
    current_sha = sha256_file(current_priors)
    if str(meta.get("source_priors_sha256", "")) != current_sha:
        raise ValueError("Candidate was generated from a different current production prior")

    # Ensure every referenced candidate table exists before any mutation.
    for _kind, entry in _iter_unique_file_entries(candidate):
        p = Path(str(entry["file"]))
        if not p.is_absolute():
            p = candidate_priors.parent / p
        if not p.resolve().is_file():
            raise FileNotFoundError(p.resolve())
    return report, candidate, current_sha, candidate_sha


def verify_promoted(
    *, current_dir: Path, expected_candidate_sha256: str, expected_sigma_rad: float
) -> dict[str, Any]:
    promotion_file = current_dir / "promotion_report.json"
    priors = current_dir / "cg_priors.json"
    if not promotion_file.is_file() or not priors.is_file():
        raise FileNotFoundError("Production promotion_report.json/cg_priors.json is missing")
    report = _load(promotion_file)
    if report.get("kind") != KIND or report.get("pass") is not True:
        raise ValueError("Production promotion report is unsupported or failed")
    if report.get("candidate_priors_sha256") != expected_candidate_sha256:
        raise ValueError("Production promotion refers to a different candidate SHA256")
    if not math.isclose(float(report.get("candidate_sigma_rad", math.nan)), expected_sigma_rad, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Production promotion sigma mismatch")
    if report.get("promoted_priors_sha256") != sha256_file(priors):
        raise ValueError("Production cg_priors.json changed after promotion")
    payload = _load(priors)
    current_artifacts = _artifact_map(priors, payload)
    if current_artifacts != report.get("promoted_prior_artifact_sha256"):
        raise ValueError("Production prior table artifacts changed after promotion")
    return report


def promote(
    *,
    current_dir: Path,
    candidate_priors: Path,
    final_report: Path,
    backup_dir: Path,
    expected_candidate_sha256: str,
    expected_sigma_rad: float,
    dataset: Path | None,
    model: Path | None,
) -> dict[str, Any]:
    if (current_dir / "promotion_report.json").is_file():
        return verify_promoted(
            current_dir=current_dir,
            expected_candidate_sha256=expected_candidate_sha256,
            expected_sigma_rad=expected_sigma_rad,
        )

    validation, candidate, old_sha, candidate_sha = _validated_inputs(
        current_dir=current_dir,
        candidate_priors=candidate_priors,
        final_report=final_report,
        expected_candidate_sha256=expected_candidate_sha256,
        expected_sigma_rad=expected_sigma_rad,
    )
    if backup_dir.exists():
        raise FileExistsError(
            f"Backup directory already exists without an active promotion report: {backup_dir}"
        )

    old_payload = _load(current_dir / "cg_priors.json")
    shutil.copytree(current_dir, backup_dir)
    if sha256_file(backup_dir / "cg_priors.json") != old_sha:
        raise RuntimeError("Pre-promotion backup hash mismatch")

    stage = current_dir.parent / f".{current_dir.name}.promotion_tmp"
    displaced = current_dir.parent / f".{current_dir.name}.prepromotion_tmp"
    for p in (stage, displaced):
        if p.exists():
            shutil.rmtree(p)
    stage.mkdir(parents=True)

    # Copy only runtime prior tables from the validated self-contained candidate.
    for _kind, entry in _iter_unique_file_entries(candidate):
        rel = Path(str(entry["file"]))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Candidate table path must be relative and contained: {rel}")
        src = (candidate_priors.parent / rel).resolve()
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    promoted = copy.deepcopy(candidate)
    validation_sha = sha256_file(final_report)
    meta = promoted.setdefault("regularization_candidate", {})
    meta.update({
        "kind": "validated_promoted_ibi_angle_smoothing_candidate",
        "source_priors": str((backup_dir / "cg_priors.json").resolve()),
        "source_priors_sha256": old_sha,
        "candidate_priors_sha256": candidate_sha,
        "final_validation_report": str(final_report),
        "final_validation_report_sha256": validation_sha,
        "validated": True,
        "promoted": True,
    })
    for entry in promoted.get("angles", []):
        reg = entry.get("regularization")
        if isinstance(reg, dict):
            reg["validated"] = True
            reg["promoted"] = True
            reg["final_validation_report_sha256"] = validation_sha
    promoted["promotion"] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "candidate_sigma_rad": expected_sigma_rad,
        "candidate_priors_sha256": candidate_sha,
        "pre_promotion_priors_sha256": old_sha,
        "step33_report_sha256": validation_sha,
        "ml_active_certification_allowed": False,
    }
    stage_priors = stage / "cg_priors.json"
    stage_priors.write_text(json.dumps(promoted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    promoted_sha = sha256_file(stage_priors)

    # Build a conversion-shaped provenance report so the existing conservative
    # validator/preflight can bind the new production spline files.  Here the
    # source is the pre-promotion conservative prior, not the old tabulated IBI.
    records: list[dict[str, Any]] = []
    for kind, new_entry in _iter_unique_file_entries(promoted):
        if str(new_entry.get("type", "")).lower() != "conservative_spline":
            raise ValueError("Promotion certification currently supports conservative_spline bonded files only")
        old_entry = _find_matching_entry(old_payload, kind, new_entry)
        old_table = load_conservative_spline(old_entry, kind=kind, priors_path=backup_dir / "cg_priors.json")
        new_table = load_conservative_spline(new_entry, kind=kind, priors_path=stage_priors)
        rel = Path(str(new_entry["file"]))
        source_path = (backup_dir / str(old_entry["file"])).resolve()
        output_stage = (stage / rel).resolve()
        output_final = (current_dir / rel).resolve()
        records.append({
            "kind": kind,
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "output_file": str(rel),
            "output_path": str(output_final),
            "output_sha256": sha256_file(output_stage),
            "points": int(len(new_table.x)),
            "min": float(new_table.minimum),
            "max": float(new_table.maximum),
            "spline_schema": str(new_entry.get("spline_schema", "pchip_hermite_v1")),
            "transformation": (
                "validated_angle_body_smoothing_promotion" if kind == "angle" else "byte_identical_copy"
            ),
            "fidelity": _fidelity(old_table, new_table, points=max(2001, len(new_table.x))),
        })
    conversion = {
        "schema_version": 1,
        "framework": FRAMEWORK,
        "kind": "ibi_conservative_spline_conversion",
        "promotion_transform": True,
        "source_artifacts_unchanged": True,
        "source_priors": str((backup_dir / "cg_priors.json").resolve()),
        "source_priors_sha256": old_sha,
        "output_priors": str((current_dir / "cg_priors.json").resolve()),
        "output_priors_sha256": promoted_sha,
        "converted_unique_tables": len(records),
        "records": records,
        "note": "Post-validation promotion transform from the pre-smoothing conservative prior; not a new IBI iteration.",
    }
    (stage / "conversion_report.json").write_text(json.dumps(conversion, indent=2, sort_keys=True) + "\n")

    # Candidate-vs-promoted table identity is the Hamiltonian bridge between
    # step 33 and the production path.  JSON metadata may change; table bytes may not.
    candidate_tables: dict[str, str] = {}
    promoted_tables: dict[str, str] = {}
    for _kind, entry in _iter_unique_file_entries(promoted):
        rel = str(entry["file"])
        candidate_tables[rel] = sha256_file(candidate_priors.parent / rel)
        promoted_tables[rel] = sha256_file(stage / rel)
    if candidate_tables != promoted_tables:
        raise RuntimeError("Staged production tables are not byte-identical to the validated candidate")

    stale = {
        "schema_version": 1,
        "kind": "residual_ml_staleness_after_prior_promotion",
        "status": "stale_for_ml_active_use",
        "reason": "Residual labels and PaiNN were built/trained against the pre-promotion conservative priors.",
        "pre_promotion_priors_sha256": old_sha,
        "promoted_priors_sha256": promoted_sha,
        "certified_use": "classical IBI-only Hamiltonian with PaiNN disabled",
        "requires_rebuild_before_ml_active_use": True,
    }
    for label, p in (("residual_dataset", dataset), ("painn_model", model)):
        if p is not None and p.is_file():
            stale[label] = {"path": str(p), "sha256": sha256_file(p)}
    (stage / "residual_ml_status.json").write_text(json.dumps(stale, indent=2, sort_keys=True) + "\n")

    # Prepare report with final-path artifact keys before the atomic swap.
    promoted_artifacts = {str((current_dir / "cg_priors.json").resolve()): promoted_sha}
    for rel, digest in promoted_tables.items():
        promoted_artifacts[str((current_dir / rel).resolve())] = digest
    promotion = {
        "schema_version": SCHEMA_VERSION,
        "framework": FRAMEWORK,
        "kind": KIND,
        "pass": True,
        "candidate_sigma_rad": expected_sigma_rad,
        "candidate_priors": str(candidate_priors),
        "candidate_priors_sha256": candidate_sha,
        "candidate_table_sha256": dict(sorted(candidate_tables.items())),
        "step33_validation_report": str(final_report),
        "step33_validation_report_sha256": validation_sha,
        "step33_validation_pass": bool(validation.get("pass")),
        "pre_promotion_dir": str(backup_dir),
        "pre_promotion_priors_sha256": old_sha,
        "promoted_priors": str((current_dir / "cg_priors.json").resolve()),
        "promoted_priors_sha256": promoted_sha,
        "promoted_table_sha256": dict(sorted(promoted_tables.items())),
        "promoted_prior_artifact_sha256": dict(sorted(promoted_artifacts.items())),
        "runtime_table_identity_with_validated_candidate": True,
        "residual_ml_status": str((current_dir / "residual_ml_status.json").resolve()),
        "note": "Only bonded runtime table bytes define the promoted change; production metadata are rewritten to record validation/promotion.",
    }
    (stage / "promotion_report.json").write_text(json.dumps(promotion, indent=2, sort_keys=True) + "\n")

    try:
        os.replace(current_dir, displaced)
        os.replace(stage, current_dir)
    except Exception:
        if not current_dir.exists() and displaced.exists():
            os.replace(displaced, current_dir)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    if displaced.exists():
        shutil.rmtree(displaced)

    return verify_promoted(
        current_dir=current_dir,
        expected_candidate_sha256=expected_candidate_sha256,
        expected_sigma_rad=expected_sigma_rad,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--current-dir", type=Path, required=True)
    p.add_argument("--candidate-priors", type=Path, required=True)
    p.add_argument("--final-validation-report", type=Path, required=True)
    p.add_argument("--backup-dir", type=Path, required=True)
    p.add_argument("--expected-candidate-sha256", required=True)
    p.add_argument("--expected-sigma-rad", type=float, required=True)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    current_dir = _canonical(args.current_dir)
    candidate = _canonical(args.candidate_priors)
    final_report = _canonical(args.final_validation_report)
    backup = _canonical(args.backup_dir)
    dataset = _canonical(args.dataset) if args.dataset else None
    model = _canonical(args.model) if args.model else None

    if args.verify_only:
        report = verify_promoted(
            current_dir=current_dir,
            expected_candidate_sha256=args.expected_candidate_sha256,
            expected_sigma_rad=args.expected_sigma_rad,
        )
        print(f"[PASS] Existing promoted priors verified: {report['promoted_priors_sha256']}")
        return

    if args.dry_run:
        report, _candidate, old_sha, candidate_sha = _validated_inputs(
            current_dir=current_dir,
            candidate_priors=candidate,
            final_report=final_report,
            expected_candidate_sha256=args.expected_candidate_sha256,
            expected_sigma_rad=args.expected_sigma_rad,
        )
        print("[IBI ANGLE PRIOR PROMOTION DRY-RUN]")
        print(f"current SHA256   : {old_sha}")
        print(f"candidate SHA256 : {candidate_sha}")
        print(f"step33 pass      : {report['pass']}")
        print(f"backup           : {backup}")
        print("[NOTE] No files were modified.")
        return

    report = promote(
        current_dir=current_dir,
        candidate_priors=candidate,
        final_report=final_report,
        backup_dir=backup,
        expected_candidate_sha256=args.expected_candidate_sha256,
        expected_sigma_rad=args.expected_sigma_rad,
        dataset=dataset,
        model=model,
    )
    print("[IBI ANGLE PRIOR PROMOTION]")
    print(f"candidate SHA256 : {report['candidate_priors_sha256']}")
    print(f"promoted SHA256  : {report['promoted_priors_sha256']}")
    print(f"backup           : {report['pre_promotion_dir']}")
    print("[PASS] Validated candidate tables promoted transactionally; residual/PaiNN marked stale.")


if __name__ == "__main__":
    main()
