# TEL22_IBI historical archive

This directory is reserved for **historical development outputs** that are no
longer inputs to the current conservative production/certification path.
Archiving is performed by:

```bash
python3 tutorials/archive_tel22_ibi_history.py          # dry-run
python3 tutorials/archive_tel22_ibi_history.py --run    # move reviewed items
```

The phase-3 archive is intentionally conservative. It currently moves only:

- `ibi_dbi_preview/` and the old short `ibi_run/` into `historical_ibi/`;
- `training_multiseed_benchmark/` and `ibi_ml_ab_validation/` into
  `ml_residual_experiments/`.

Historical artifacts that are still referenced by configured workflows remain
at `tutorials/tel22_IBI/` until a later atomic reference migration is designed.
These include `ibi_run_16ps*`, `ibi_validation_best/`,
`postibi_runtime_validation/`, the residual dataset/model files, and their
provenance manifest.

## GROMACS products are not archived

AA/GROMACS-generated files are deliberately preserved in their existing
locations. The archive helper never moves or removes `md.trr`, `md_whole.trr`,
`md.gro`, `*.tpr`, EM/NVT/NPT outputs, topology/position-restraint files,
solvated/ionized structures, or the source PDB files.

After an executed archive, `archive_manifest.json` and `archive_manifest.md`
record the moved artifacts and deterministic tree SHA256 digests.
