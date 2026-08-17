#!/usr/bin/env python3
"""Finalize TEL22/TEL22_IBI layout: pipeline roots, diagnostics, reviewed junk.

The tutorial roots are reserved for pipeline/canonical artifacts. Diagnostic and
exploratory evidence is moved below ``diagnostics/``. A small exact allowlist of
unreferenced backups/caches/one-shot repair files is removed. GROMACS products
are never deleted and are fingerprinted before/after an executed migration.

Dry-run is the default. Use ``--run`` only after reviewing the plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = REPO_ROOT / "tutorials"

# Explicitly protected AA/GROMACS inputs and products. The migrator never plans
# any of these for deletion. Representative files are SHA256 checked pre/post.
PROTECTED_GROMACS = (
    ".short_mdp",
    "143D.pdb", "pdb143d.ent.gz", "tel22_clean.pdb", "tel22_processed.gro",
    "box_10.gro", "box_solvated.gro", "box_ions.gro", "ions.tpr", "posre.itp", "topol.top",
    "em.edr", "em.gro", "em.log", "em.tpr", "em.trr",
    "nvt.cpt", "nvt.edr", "nvt.gro", "nvt.log", "nvt.tpr",
    "npt.cpt", "npt.edr", "npt.gro", "npt.log", "npt.tpr",
    "md.cpt", "md.edr", "md.gro", "md.log", "md.tpr", "md.trr", "md_whole.trr", "mdout.mdp",
    ".md_whole.trr_offsets.npz", ".md_whole.trr_offsets.lock",
)

# These entry points are validation/diagnostic workflows, not the production
# pipeline. The patch itself relocates the text scripts; the migration plan is
# retained here for idempotence and compatibility with partially migrated trees.
TEL22_DIAGNOSTIC_SCRIPTS = (
    "06_certify_nve.sh",
    "06b_diagnose_short_range.sh",
    "07_diagnose_morse_breakage.sh",
    "08_diagnose_morse_reversibility.sh",
    "09_diagnose_morse_site_torque.sh",
)

IBI_DIAGNOSTIC_SCRIPTS = TEL22_DIAGNOSTIC_SCRIPTS + (
    "11_build_dbi_preview.sh",
    "17_benchmark_training_multiseed.sh",
    "18_validate_postibi_runtime.sh",
    "19_validate_ibi_ml_ab.sh",
    "23_certify_conservative_ibi_nve.sh",
    "24_diagnose_conservative_ibi_nve_scaling.sh",
    "25_diagnose_conservative_ibi_state_convergence.sh",
    "26_finalize_conservative_ibi_nve_certification.sh",
    "27_diagnose_conservative_ibi_energy_scaling.sh",
    "28_diagnose_sigma_energy_replicas.sh",
    "29_diagnose_ibi_timestep_range.sh",
    "30_diagnose_regularize_ibi_angles.sh",
    "31_validate_ibi_angle_regularization.sh",
    "32_optimize_ibi_angle_smoothing.sh",
    "33_validate_final_ibi_angle_candidate.sh",
    "35_test_conservative_ibi_dihedrals.sh",
    "36_localize_dihedral_ibi_update.sh",
    "37_diagnose_tabulated_dihedral_conservativity.sh",
    "38_test_conservative_in_loop_dihedral_ibi.sh",
    "39_test_conservative_dihedral_ibi_replicas.sh",
)

TEL22_DIAGNOSTIC_PATHS = {
    "nve_certification": "diagnostics/nve/nve_certification",
    "nve_certification_fp32": "diagnostics/nve/nve_certification_fp32",
    "nve_certification_fp32_new_morse": "diagnostics/nve/nve_certification_fp32_new_morse",
    "nve_certification_fp64": "diagnostics/nve/nve_certification_fp64",
    "nve_certification_fp64_1ps": "diagnostics/nve/nve_certification_fp64_1ps",
    "nve_certification_fp64_2ps": "diagnostics/nve/nve_certification_fp64_2ps",
    "nve_certification_fp64_new_morse": "diagnostics/nve/nve_certification_fp64_new_morse",
    "nve_certification_wca_v3_dt10_stress": "diagnostics/nve/nve_certification_wca_v3_dt10_stress",
    "wca12_selective_ab": "diagnostics/nonbonded/wca12_selective_ab",
    "morse_breakage_report.json": "diagnostics/morse/morse_breakage_report.json",
    "morse_reversibility_report.json": "diagnostics/morse/morse_reversibility_report.json",
    "morse_site_torque_report.json": "diagnostics/morse/morse_site_torque_report.json",
    "smoke_equilibrated.npz": "diagnostics/smoke/smoke_equilibrated.npz",
    "md_sanity_100.log": "diagnostics/smoke/md_sanity_100.log",
}

IBI_DIAGNOSTIC_PATHS = {
    **TEL22_DIAGNOSTIC_PATHS,
    # Optional preview/validation/ML checks.
    "ibi_dbi_preview": "diagnostics/ibi/ibi_dbi_preview",
    "ibi_validation_best": "diagnostics/ibi/ibi_validation_best",
    "postibi_runtime_validation": "diagnostics/ml/postibi_runtime_validation",
    "training_multiseed_benchmark": "diagnostics/ml/training_multiseed_benchmark",
    "ibi_ml_ab_validation": "diagnostics/ml/ibi_ml_ab_validation",
    # Conservative NVE diagnostics/certification evidence.
    "conservative_ibi_energy_localization": "diagnostics/nve/conservative_ibi_energy_localization",
    "sigma_energy_replica_window_diagnostic": "diagnostics/nve/sigma_energy_replica_window_diagnostic",
    "nve_certification_conservative_ibi_only": "diagnostics/nve/nve_certification_conservative_ibi_only",
    "nve_certification_conservative_ibi_only_preflight.json": "diagnostics/nve/nve_certification_conservative_ibi_only_preflight.json",
    "nve_diagnostic_conservative_ibi_only": "diagnostics/nve/nve_diagnostic_conservative_ibi_only",
    "nve_diagnostic_conservative_ibi_only_preflight.json": "diagnostics/nve/nve_diagnostic_conservative_ibi_only_preflight.json",
    "nve_equilibration_conservative_ibi_only": "diagnostics/nve/nve_equilibration_conservative_ibi_only",
    "nve_final_certification_conservative_ibi_only": "diagnostics/nve/nve_final_certification_conservative_ibi_only",
    "nve_state_convergence_conservative_ibi_only": "diagnostics/nve/nve_state_convergence_conservative_ibi_only",
    "nve_state_convergence_conservative_ibi_only_preflight.json": "diagnostics/nve/nve_state_convergence_conservative_ibi_only_preflight.json",
    # IBI calibration/angle/dihedral studies.
    "ibi_timestep_range_diagnostic": "diagnostics/ibi/ibi_timestep_range_diagnostic",
    "ibi_angle_regularization_diagnostic": "diagnostics/ibi/ibi_angle_regularization_diagnostic",
    "ibi_angle_regularization_validation": "diagnostics/ibi/ibi_angle_regularization_validation",
    "ibi_angle_smoothing_sweep": "diagnostics/ibi/ibi_angle_smoothing_sweep",
    "ibi_angle_final_candidate_validation": "diagnostics/ibi/ibi_angle_final_candidate_validation",
    "ibi_conservative_pre_smooth_0p0075": "diagnostics/ibi/ibi_conservative_pre_smooth_0p0075",
    "ibi_promoted_final_certification": "diagnostics/ibi/ibi_promoted_final_certification",
    "ibi_dihedral_candidate_test": "diagnostics/ibi/ibi_dihedral_candidate_test",
    "ibi_dihedral_update_localization": "diagnostics/ibi/ibi_dihedral_update_localization",
    "ibi_dihedral_legacy_conservativity_diagnostic": "diagnostics/ibi/ibi_dihedral_legacy_conservativity_diagnostic",
    "ibi_dihedral_conservative_in_loop_test": "diagnostics/ibi/ibi_dihedral_conservative_in_loop_test",
    "ibi_dihedral_conservative_replica_matrix": "diagnostics/ibi/ibi_dihedral_conservative_replica_matrix",
    "ibi_dihedral_test_settings.json": "diagnostics/ibi/ibi_dihedral_test_settings.json",
    # Superseded short IBI lineage. Active ibi_run_16ps* stays at root.
    "ibi_run": "diagnostics/historical/ibi_run",
    # If phase-3 was already executed, keep its evidence but move the archive
    # itself below diagnostics so the tutorial root remains pipeline-only.
    "archive": "diagnostics/historical/phase3_archive",
}

# Exact reviewed junk outside the tutorial data trees. These are backups,
# caches, one-shot repair helpers or retired local test outputs; none is a live
# framework pipeline input. training/build and generic training outputs are NOT
# included because supported tooling still refers to them.
REPO_JUNK = (
    ".pytest_cache",
    "HOWTO.md.orig",
    "HOWTO_EN.md.orig",
    "README.md.orig",
    "repair_certify_nve_inline_sampling_v3_20260813.py",
    "repair_nve_sigma_final_20260813.py",
    "create_zip_per_chatgpt.sh",
    "tests/test_nve_certification.py.orig",
    "simulation/run_cg_md.py.orig",
    "simulation/certify_nve.py.orig",
    "simulation/certify_nve.py.rej",
    "simulation/certify_nve.py.pre_sigma_v2",
    "simulation/certify_nve.py.pre_inline_sampling_repair",
    "training/best_cg_model_old.pt",
    "training/build_test",
    "training/__pycache__",
    "tests/__pycache__",
    "simulation/__pycache__",
    "simulation/espresso_plugin/__pycache__",
    "preprocessing/__pycache__",
    "ibi/__pycache__",
    "tutorials/__pycache__",
    "tutorials/tel22_IBI/TUTORIAL.md.orig",
    "tutorials/tel22_IBI/TUTORIAL.md.rej",
)

# Removed only when still zero-byte. If a user has populated either file, the
# migrator refuses to treat it as junk.
RETIRED_EMPTY_PLACEHOLDERS = (
    "tutorials/tel22/06c_test_selective_wca12.sh",
    "tutorials/tel22_IBI/06c_test_selective_wca12.sh",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_fingerprints(tutorial_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in PROTECTED_GROMACS:
        path = tutorial_dir / name
        if path.is_file():
            out[name] = sha256_file(path)
    return out


def move_plan(tutorial_dir: Path, scripts: tuple[str, ...], artifacts: dict[str, str]):
    plan: list[tuple[Path, Path, str]] = []
    for name in scripts:
        plan.append((tutorial_dir / name, tutorial_dir / "diagnostics" / "scripts" / name, "diagnostic-script"))
    for src, dst in artifacts.items():
        plan.append((tutorial_dir / src, tutorial_dir / dst, "diagnostic-artifact"))
    return plan


def preflight(plan: list[tuple[Path, Path, str]]) -> None:
    collisions = [(s, d) for s, d, _ in plan if s.exists() and d.exists()]
    if collisions:
        details = "\n".join(f"  {s} -> {d}" for s, d in collisions)
        raise RuntimeError("diagnostic-layout destination collision(s):\n" + details)


def apply_plan(plan: list[tuple[Path, Path, str]], run: bool) -> list[dict]:
    records = []
    for src, dst, kind in plan:
        if src.exists():
            status = "moved" if run else "planned"
            print(f"[{'MOVE' if run else 'DRY-RUN'}:{kind}] {src} -> {dst}")
            if run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
        elif dst.exists():
            status = "already-organized"
            print(f"[KEEP:{kind}] already organized: {dst}")
        else:
            status = "absent"
        records.append({"kind": kind, "source": str(src), "destination": str(dst), "status": status})
    return records


def remove_reviewed_junk(run: bool) -> list[dict]:
    records = []
    for rel in REPO_JUNK:
        path = REPO_ROOT / rel
        if not path.exists() and not path.is_symlink():
            records.append({"path": rel, "status": "absent"})
            continue
        print(f"[{'REMOVE' if run else 'DRY-RUN'}:junk] {path}")
        if run:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        records.append({"path": rel, "status": "removed" if run else "planned"})

    for rel in RETIRED_EMPTY_PLACEHOLDERS:
        path = REPO_ROOT / rel
        if not path.exists():
            records.append({"path": rel, "status": "absent"})
            continue
        if not path.is_file() or path.stat().st_size != 0:
            print(f"[KEEP:nonempty-placeholder] {path}")
            records.append({"path": rel, "status": "preserved-nonempty"})
            continue
        print(f"[{'REMOVE' if run else 'DRY-RUN'}:retired-empty-placeholder] {path}")
        if run:
            path.unlink()
        records.append({"path": rel, "status": "removed" if run else "planned"})
    return records


def write_readme(tutorial_dir: Path) -> None:
    path = tutorial_dir / "diagnostics" / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    title = "TEL22_IBI" if tutorial_dir.name == "tel22_IBI" else "TEL22"
    path.write_text(
        f"# {title} diagnostics\n\n"
        "This directory contains validation, certification, convergence and exploratory test evidence.\n"
        "It is deliberately separated from the tutorial root, which contains the pipeline and canonical artifacts.\n\n"
        "- `scripts/`: diagnostic/test entry points.\n"
        "- `nve/`: NVE certification and convergence evidence.\n"
        "- `ibi/`: IBI calibration/angle/dihedral evidence (TEL22_IBI).\n"
        "- `ml/`: residual-ML benchmark/runtime validation evidence (TEL22_IBI).\n"
        "- `historical/`: superseded development evidence kept for provenance.\n"
        "- `morse/`, `nonbonded/`, `smoke/`: focused diagnostics.\n\n"
        "GROMACS-generated trajectory/topology/working files are intentionally kept at the tutorial root and are never deleted by the layout migrator.\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="perform reviewed artifact moves/junk removal; default is dry-run")
    args = ap.parse_args()
    tutorials = [
        (TUTORIALS / "tel22", TEL22_DIAGNOSTIC_SCRIPTS, TEL22_DIAGNOSTIC_PATHS),
        (TUTORIALS / "tel22_IBI", IBI_DIAGNOSTIC_SCRIPTS, IBI_DIAGNOSTIC_PATHS),
    ]
    plans = []
    fingerprints = {}
    for tutorial_dir, scripts, artifacts in tutorials:
        if not tutorial_dir.is_dir():
            raise SystemExit(f"missing tutorial directory: {tutorial_dir}")
        fingerprints[str(tutorial_dir)] = protected_fingerprints(tutorial_dir)
        plan = move_plan(tutorial_dir, scripts, artifacts)
        preflight(plan)
        plans.append((tutorial_dir, plan))

    all_records = []
    for tutorial_dir, plan in plans:
        all_records.extend(apply_plan(plan, args.run))
        if args.run:
            write_readme(tutorial_dir)

    junk_records = remove_reviewed_junk(args.run)

    if args.run:
        for tutorial_dir, _, _ in tutorials:
            after = protected_fingerprints(tutorial_dir)
            before = fingerprints[str(tutorial_dir)]
            if after != before:
                raise RuntimeError(f"GROMACS protection failure in {tutorial_dir}: protected fingerprints changed")
        manifest = {
            "schema_version": 2,
            "kind": "tel22_pipeline_diagnostics_layout_migration",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "gromacs_generated_preserved": True,
            "moves": all_records,
            "junk": junk_records,
        }
        out = TUTORIALS / "tel22_IBI" / "diagnostics" / "layout_migration_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[DONE] diagnostics layout applied; manifest: {out}")
    else:
        print("[NOTE] Dry-run only. Re-run with --run to move diagnostic artifacts and remove reviewed junk.")
    print("[NOTE] GROMACS-generated files are protected and are never moved or removed by this tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
