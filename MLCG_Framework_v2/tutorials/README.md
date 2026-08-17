# Tutorials

Tutorial directories are examples only. Core modules must not import files from this tree. Use them as starting points for system-specific mappings and wrapper scripts, while keeping reusable logic in `preprocessing/`, `training/`, `simulation/` and `ibi/`.

TEL22/TEL22_IBI use a strict layout: tutorial roots contain pipeline/canonical artifacts and retained GROMACS products; validation and experimental evidence lives under `diagnostics/`. See `TEL22_LAYOUT.md` and `TEL22_CLEANUP.md`.

For an existing checkout, preview the final migration with:

```bash
python3 tutorials/organize_tel22_diagnostics.py
```

and apply it with `--run` after reviewing the plan.
