# TEL22 / TEL22_IBI artifact hygiene

This document describes the final cleanup policy. The scientific outputs are not discarded merely because they are generated: pipeline artifacts are retained, diagnostic evidence is moved below `diagnostics/`, and only reviewed junk is removed.

## Invariants

1. **GROMACS products are retained.** `md.trr`, `md_whole.trr`, `md.gro`, `md.tpr`, `nvt.*`, `npt.*`, `em.*`, topology/box files and related AA working products are never deleted by the final layout migrator.
2. **Pipeline roots remain self-contained.** Canonical datasets, priors, models, manifests, active IBI lineage and production scripts stay at the tutorial root.
3. **Diagnostics are preserved, not discarded.** NVE scans, angle/dihedral studies, ML A/B checks, Morse/WCA tests and their entry points live under `diagnostics/`.
4. **Historical evidence is retained under diagnostics.** Superseded short IBI/previous cleanup evidence is placed under `diagnostics/historical/` rather than mixed with the active pipeline.
5. **Deletion uses an exact allowlist.** No wildcard removes scientific results.

## Final migration command

Dry-run:

```bash
python3 tutorials/organize_tel22_diagnostics.py
```

Execute after reviewing the plan:

```bash
python3 tutorials/organize_tel22_diagnostics.py --run
```

The helper is idempotent and fail-closed on destination collisions. It records a migration manifest in `tutorials/tel22_IBI/diagnostics/layout_migration_manifest.json` and verifies representative retained GROMACS artifacts byte-for-byte with SHA256 before/after the move.

## Reviewed deletion class

The final cleanup may remove only non-pipeline material such as:

```text
*.orig / *.rej / *.pre_* backups
Python __pycache__ and .pytest_cache
one-shot repair_*_20260813.py helpers
create_zip_per_chatgpt.sh
training/build_test/
training/best_cg_model_old.pt
zero-byte 06c_test_selective_wca12.sh placeholders
```

The one-shot repair/packaging helpers have no live framework references; their effects are already incorporated in the maintained source.

## Retained training artifacts

The final cleanup intentionally does **not** remove `training/build/`, `training/cg_dataset.bin`, `training/best_cg_model.pt`, or `training/cg_training_log.csv` when present. They are not used by the TEL22 wrappers as canonical tutorial outputs, but generic supported training tooling still refers to these paths/names. Removing them would therefore fail the stricter "unused by the framework" criterion.

## Generated CG outputs

`cg_trajectory.vtf`, `energy.csv`, `equilibrated.npz` and training logs are pipeline/runtime products and are not part of the final automatic cleanup. Focused diagnostic equivalents, such as `smoke_equilibrated.npz`, are moved below `diagnostics/` rather than deleted.

## Cross-tutorial duplication

Physical deduplication between `tel22` and `tel22_IBI` remains intentionally disabled. The previous SHA256 audit found that the large duplicate byte count is dominated by generated products, while immutable source inputs save too little space to justify coupling the tutorial paths.

See `TEL22_LAYOUT.md` for the resulting directory structure.
