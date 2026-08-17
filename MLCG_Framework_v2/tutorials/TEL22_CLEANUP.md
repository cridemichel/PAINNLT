# TEL22 / TEL22_IBI artifact hygiene

This note defines the cleanup policy for `tutorials/tel22` and
`tutorials/tel22_IBI`.  Cleanup is deliberately conservative: scientific
provenance and certification evidence are more important than making the
working tree visually small.

## Artifact classes

### CANONICAL -- keep

Keep source/configuration files, topology/mapping inputs, model-dependent
configuration, production priors, manifests, compact reports that establish a
promoted result, and the scripts/documentation required to reproduce them.

For the IBI tutorial this explicitly includes `ibi_conservative/`,
`ibi_promoted_final_certification/`, the step scripts, IBI/model configuration,
and provenance/report JSON needed to explain the promoted angular prior.
Dihedral step 35--39 outputs remain diagnostic evidence; no torsional prior has
been promoted.

### DIAGNOSTIC / DEVELOPMENT -- keep until a separate archive pass

The angle/dihedral/NVE localization, smoothing, replica, benchmark and A/B
results document decisions made during development.  They are not production
priors, but this first cleanup pass does **not** delete or move them because
scripts and reports may still refer to their current paths.

A later archive/refactor pass may group them under `diagnostics/` or `archive/`
after a reference audit and, where appropriate, path migration.

### GENERATED -- regenerable

Large GROMACS working products (`md.trr`, `md_whole.trr`, `*.tpr`, equilibration
outputs, processed structures) and short CG runtime outputs can be regenerated
from the tutorial scripts.  They are ignored by the local `.gitignore` files.
They are only deleted by the cleanup helper when `--generated` is requested.

`md.gro` and `md_whole.trr` are the default inputs to `02_build_dataset.sh`, so
do not delete them until the dataset has been built or unless you are prepared
to rerun step 01.

### JUNK / LOCAL BACKUP -- remove

Editor backups (`#...#`, `*.bak`, `*.orig`, `*.rej`), Python/test caches and the
known local ZIP snapshots (`ibival.zip`, `val.zip`, `ms.zip`) are not part of the
canonical workflow. The retired zero-byte `06c_test_selective_wca12.sh` placeholders
are also removed by the helper, but only while they are still empty; a non-empty local
copy is preserved. Because a local archive can still contain useful material, ZIP
removal is opt-in via `--archives`; inspect it before deletion.

## Cleanup helper

From the repository root, inspect the low-risk cleanup first:

```bash
bash tutorials/cleanup_tel22_artifacts.sh --dry-run
```

Apply only low-risk junk/cache cleanup:

```bash
bash tutorials/cleanup_tel22_artifacts.sh --run
```

Inspect/remove the known local ZIP snapshots separately:

```bash
bash tutorials/cleanup_tel22_artifacts.sh --dry-run --archives
bash tutorials/cleanup_tel22_artifacts.sh --run --archives
```

Inspect the larger cleanup that also includes regenerable AA/CG runtime files:

```bash
bash tutorials/cleanup_tel22_artifacts.sh --dry-run --generated
```

Delete those generated files only when their downstream artifacts are already
safe or you are prepared to regenerate them:

```bash
bash tutorials/cleanup_tel22_artifacts.sh --run --generated
```

The helper never removes the production conservative IBI prior, certification
reports/directories, IBI datasets/models/priors, or the step 23--39 diagnostic
directories.

## Phase 2: SHA256 + reference audit before deduplication

The two tutorials contain many same-name AA inputs and working products. They
must not be deduplicated by filename/size alone. Phase 2 is implemented as a
**non-destructive audit**:

```bash
python3 tutorials/audit_tel22_dedup.py
```

The command writes:

```text
tutorials/.tel22_cleanup_audit/phase2_dedup_audit.json
tutorials/.tel22_cleanup_audit/phase2_dedup_audit.md
```

and does not remove, move, rewrite or symlink anything. The output directory is
ignored by `tutorials/.gitignore`.

For every same-relative-path file, the audit computes SHA256 and classifies the
result conservatively:

- `SHARED_CANDIDATE`: byte-identical immutable source input that may be shared
  only after its references are migrated and reviewed;
- `GENERATED_DUPLICATE`: byte-identical regenerable output; prefer regeneration
  or ignore rules rather than symlinking it;
- `KEEP_SEPARATE_DUPLICATE`: identical bytes today but tutorial-local semantics
  (for example priors/config/model artifacts that may legitimately diverge);
- `KEEP_SEPARATE_DIFFERENT`: same relative path with different bytes.

TEL22_IBI-only material is also labelled `KEEP`, `DIAGNOSTIC` or `HISTORICAL`.
`HISTORICAL` is an **archive candidate**, not deletion authorization. Production
conservative priors/certification remain `KEEP`; angle/dihedral/NVE evidence is
kept as `DIAGNOSTIC` until an explicit archive migration exists.

The JSON report includes textual reference hits for duplicate candidates. A
future phase-3 patch may act only on reviewed `SHARED_CANDIDATE` entries and must
migrate those references atomically. The audit deliberately introduces no
fragile `../tel22/...` dependency into the IBI workflow.

To audit a copy or alternate tree without modifying it:

```bash
python3 tutorials/audit_tel22_dedup.py \
  --repo-root /path/to/MLCG_Framework_v2 \
  --output-dir /tmp/tel22_dedup_audit
```

## Phase 3: conservative archive of terminal TEL22_IBI history

The phase-2 audit showed that most disk-space duplication is generated AA/CG
output, while several `HISTORICAL` TEL22_IBI artifacts are still active inputs
or provenance dependencies of configured workflows. Phase 3 therefore does
**not** move every item labelled `HISTORICAL`.

Use the reviewed archive helper:

```bash
python3 tutorials/archive_tel22_ibi_history.py
```

Dry-run is the default. The current reviewed move plan contains only terminal
historical outputs that are not required by the conservative production/
certification path:

```text
ibi_dbi_preview/                 -> archive/historical_ibi/
ibi_run/                         -> archive/historical_ibi/
training_multiseed_benchmark/    -> archive/ml_residual_experiments/
ibi_ml_ab_validation/            -> archive/ml_residual_experiments/
```

After reviewing the dry-run, execute it with:

```bash
python3 tutorials/archive_tel22_ibi_history.py --run
```

The helper performs a collision preflight before moving anything and writes
`archive/archive_manifest.json` plus `archive/archive_manifest.md`, including
file counts, byte counts and deterministic tree SHA256 digests.

### Historical artifacts intentionally left at the tutorial root

The following material remains in place because current configured workflows or
diagnostics still refer to it:

```text
ibi_run_16ps/
ibi_run_16ps_continue/
ibi_validation_best/
postibi_runtime_validation/
tel22_dataset_ibi_residual.bin
tel22_model_ibi.pt{,.manifest.json}
tel22_model_ibi_conservative.pt{,.manifest.json}
ibi_residual_build_manifest.json
```

Moving these would require a later atomic migration of `model_dependent_workflow_config.json`,
wrapper/documentation references and stored provenance. Phase 3 deliberately
avoids that semantic change.

### GROMACS preservation guarantee

**Phase 3 does not clean, move, deduplicate or archive the files generated by the
GROMACS simulation.** They stay in `tel22` and `tel22_IBI` exactly where they
are. The helper explicitly protects representative AA/GROMACS artifacts such as
`md.trr`, `md_whole.trr`, `md.gro`, all EM/NVT/NPT products, TPR/CPT/EDR/LOG
files, `topol.top`, `posre.itp`, solvated/ionized GRO files and source PDB files.

This is independent of the phase-1 `--generated` option: simply do not run that
option when the generated AA/GROMACS working set is intended to be retained.
