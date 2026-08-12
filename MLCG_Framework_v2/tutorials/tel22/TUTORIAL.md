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

## NVE energy-conservation certification

After equilibration, certify the complete conservative Hamiltonian from the same
`equilibrated.npz` checkpoint at several Velocity-Verlet time steps:

```bash
PYRESSO=../../espresso/build/pypresso bash 06_certify_nve.sh
```

The wrapper defaults to CPU and runs the same physical duration at
`dt = 0.002, 0.001, 0.0005, 0.00025 ps`. It writes
`nve_certification/nve_certification_report.json` and returns a non-zero exit
status unless both conditions pass:

1. the RMS total-energy error follows `RMS(dE) ~ dt^p` with the configured
   exponent/r-squared guardrails (defaults: `1.7 <= p <= 2.3`, `R^2 >= 0.97`);
2. the difference between the mean total energy in the final and initial 20%
   blocks is below the configured relative threshold (default `1e-4`).

Useful overrides:

```bash
NVE_DURATION_PS=10 \
NVE_DTS="0.002 0.001 0.0005 0.00025" \
NVE_MAX_RELATIVE_DRIFT=1e-5 \
PYRESSO=../../espresso/build/pypresso \
bash 06_certify_nve.sh --overwrite
```

The certification path refuses explicitly tabulated bonded priors. Analytic
Morse, harmonic, FENE, harmonic-angle and cosine-dihedral priors remain allowed.
