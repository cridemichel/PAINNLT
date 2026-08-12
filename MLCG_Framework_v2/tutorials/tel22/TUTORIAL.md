# TEL22 reference example

This directory is an **application example**, not part of the MLCG core API.
No code under `preprocessing/`, `training/`, `simulation/` or `tests/` imports
TEL22 files.

The repository intentionally does not bundle atomistic trajectories, GROMACS
working files, trained weights or diagnostic/ablation outputs. Prepare an AA
trajectory containing forces separately, then run:

```bash
AA_TOPOLOGY=md.gro AA_TRAJECTORY=md_whole.trr bash 02_build_dataset.sh
bash 03_train_model.sh
PYRESSO=/path/to/pypresso DEVICE=auto bash 04_equilibrate.sh
PYRESSO=/path/to/pypresso DEVICE=auto bash 05_run_espresso.sh
```

Inputs kept under version control:

- `tel22_topology.json`: TEL22-specific mapping and priors configuration;
- `tel22_training_config.json`: one TEL22 training profile;
- scripts `02`-`05`: thin wrappers around the generic framework commands.

Generated files such as `tel22_dataset.bin`, `cg_priors.json`,
`rigid_bodies_info.json`, `*.pt`, manifests, checkpoints and trajectories are
runtime artifacts and are intentionally excluded from the source tree.
