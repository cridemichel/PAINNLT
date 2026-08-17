# TEL22 / TEL22_IBI final layout policy

The tutorial root is reserved for the executable pipeline, canonical model inputs/outputs, and retained GROMACS working products. Validation, certification, convergence studies, calibration experiments and exploratory tests live below `diagnostics/`.

## TEL22

```text
tutorials/tel22/
  01_run_gromacs*.sh
  02_build_dataset.sh
  03_train_model.sh
  04_equilibrate.sh
  05_run_espresso.sh
  md.*, nvt.*, npt.*, em.*, *.gro, *.tpr      retained GROMACS products
  tel22_dataset.bin, tel22_model.pt, ...       canonical CG artifacts
  diagnostics/
    scripts/                                   NVE/Morse/nonbonded entry points
    nve/                                       NVE evidence
    morse/                                     Morse diagnostics
    nonbonded/                                 WCA diagnostics
    smoke/                                     focused smoke evidence
```

## TEL22_IBI

```text
tutorials/tel22_IBI/
  01-05, 10, 12-16, 20-22, 34                pipeline/calibration entry points
  ibi_run_16ps*/                              active IBI iteration lineage
  ibi_conservative/                           promoted conservative production priors
  model_dependent_workflow_config.json        model-dependent policy/configuration
  generated GROMACS products                  retained at tutorial root
  diagnostics/
    scripts/                                   optional validation/test entry points
    nve/                                       NVE certification/convergence evidence
    ibi/                                       DBI preview, angle/dihedral/calibration evidence
    ml/                                        residual-ML benchmark/runtime evidence
    historical/                                superseded development evidence
    morse/, nonbonded/, smoke/                 focused diagnostics
```

`34_promote_and_certify_ibi_angle_prior.sh` remains at the tutorial root because it is the explicit action that changes the canonical `ibi_conservative/` production prior. Its validation inputs and certification outputs live under `diagnostics/`.

`11_build_dbi_preview.sh` is under `diagnostics/scripts/`: the preview is useful for inspecting support/tails but is not an input to `12_run_ibi.sh`.

## Final migration

The patch relocates the diagnostic shell entry points directly. Result directories can contain `.npz`, trajectory fragments or other files that are intentionally absent from source snapshots, so they are migrated by a fail-closed helper.

Preview:

```bash
python3 tutorials/organize_tel22_diagnostics.py
```

Apply:

```bash
python3 tutorials/organize_tel22_diagnostics.py --run
```

The helper fingerprints retained GROMACS files before/after the migration, refuses destination collisions, and writes `tutorials/tel22_IBI/diagnostics/layout_migration_manifest.json`. A partially applied transition may leave a zero-byte script stub at the tutorial root after the complete script has already been installed under `diagnostics/scripts/`; that exact case is treated as a stale placeholder and removed safely. Non-empty source/destination collisions still abort the migration.

## Deliberately retained

- all GROMACS trajectories, checkpoints, topologies and working products;
- canonical priors, datasets, model weights and manifests;
- `ibi_conservative/` and active `ibi_run_16ps*` iteration lineage;
- diagnostic/certification evidence (moved, not deleted);
- `training/build/`, `training/cg_dataset.bin`, `training/best_cg_model.pt` and `training/cg_training_log.csv` when present, because supported generic training tooling still refers to those names.

Only exact reviewed junk with no live pipeline reference is removed: editor/patch backups, Python/test caches, one-shot repair helpers, `training/build_test`, `training/best_cg_model_old.pt`, and zero-byte retired placeholders.
