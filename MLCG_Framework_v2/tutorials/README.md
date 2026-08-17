# Tutorials

Tutorial directories are examples only. Core modules must not import files from
this tree. Use them as starting points for system-specific mappings and wrapper
scripts, while keeping reusable logic in `preprocessing/`, `training/` and
`simulation/`.


TEL22 artifact hygiene and the non-destructive cross-tutorial SHA256/reference
audit are documented in `TEL22_CLEANUP.md`. Run
`python3 tutorials/audit_tel22_dedup.py` before any deduplication between
`tel22` and `tel22_IBI`.

Phase 3 can archive only reviewed terminal TEL22_IBI historical outputs while
explicitly preserving all GROMACS-generated files:
`python3 tutorials/archive_tel22_ibi_history.py` (dry-run) followed, after
review, by `python3 tutorials/archive_tel22_ibi_history.py --run`.
