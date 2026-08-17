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

## Deferred deduplication between `tel22` and `tel22_IBI`

The two tutorials currently contain many apparently duplicated AA inputs and
working products.  They must not be deduplicated by filename/size alone.  A
second cleanup phase should:

1. compute SHA256 for every candidate shared file;
2. classify each as immutable shared input, generated output, or
   tutorial-specific artifact;
3. audit every shell/Python/documentation reference;
4. only then replace duplicate immutable inputs with a shared location or a
   documented regeneration path.

This avoids introducing fragile `../tel22/...` dependencies into the IBI
workflow merely to save disk space.
