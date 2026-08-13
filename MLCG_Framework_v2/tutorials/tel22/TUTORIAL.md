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

The wrapper defaults to CPU, uses the same physical duration for every
timestep, and samples total energy at **every integration step**. The current
default grid is
`dt = 0.001, 0.002, 0.005, 0.01 ps` with `5 ps` per run. It writes
`nve_certification/nve_certification_report.json` and returns a non-zero exit
status unless both conditions pass:

1. the population standard deviation of total energy follows
   `sigma_E = std(E_tot) ~ dt^p` with the configured exponent/r-squared
   guardrails (defaults: `1.7 <= p <= 2.3`, `R^2 >= 0.97`);
2. the difference between the mean total energy in the final and initial 20%
   blocks is below the configured relative threshold (default `1e-4`).

`RMS(E_tot-E_tot(0))` is retained only as a secondary diagnostic and is not the
quantity fitted for the certification order. The `0.01 ps` point is a useful
stress point and may lie outside the asymptotic regime for a stiff model; steps
much below `0.001 ps` can instead become roundoff-limited.

Useful overrides:

```bash
NVE_DURATION_PS=10 \
NVE_DTS="0.001 0.002 0.005 0.01" \
NVE_MAX_RELATIVE_DRIFT=1e-5 \
PYRESSO=../../espresso/build/pypresso \
bash 06_certify_nve.sh --overwrite
```

The certification path refuses explicitly tabulated bonded priors. Analytic
Morse, harmonic, FENE, harmonic-angle and cosine-dihedral priors remain allowed.

## 6b. Diagnose a short-range contact

If an NVE/production run approaches the short-range safety guardrail, diagnose the
actual minimum-distance PID pair before changing WCA parameters or exclusions:

```bash
bash 06b_diagnose_short_range.sh
```

By default this reads `nve_certification/dt_0p005/energy.csv`, uses a 0.20 nm
short-range threshold, and writes
`nve_certification/short_range_diagnostic_0p005.json`. Override the selected run
or threshold with, for example:

```bash
NVE_DT_TAG=0p002 SHORT_RANGE_THRESHOLD_NM=0.25 bash 06b_diagnose_short_range.sh
```

The diagnostic reconstructs the same runtime particle-ID order as
`simulation/run_cg_md.py`, maps the closest PIDs to molecule/site/type, classifies
the molecule pair as `1-2`, `1-3`, or `nonbonded` using the explicit lists in
`cg_priors.json`, and reports the nominal pair-specific WCA energy and force at
the observed distance. For an excluded pair those WCA values are explicitly
reported as counterfactual: the runtime WCA is not applied to that molecule pair.
This is a post-processing test only and does not modify the Hamiltonian.

## 6c. A/B test the 1-2 WCA exclusion scope

If `06b_diagnose_short_range.sh` shows that the short-range contact is a
`1-2` molecule pair whose WCA is excluded by the production all-sites policy,
run the causal A/B diagnostic:

```bash
WCA12_TEST_OVERWRITE=1 \
PYRESSO=../../espresso/build/pypresso \
bash 06c_test_selective_wca12.sh
```

Defaults are `dt = 0.005 ps` and `1.3 ps`, matching the TEL22 short-range
failure window. This mode deliberately changes **runtime only**:

- intra-rigid-body WCA exclusions are unchanged;
- `1-3` molecule pairs remain excluded for every cross-site pair;
- `1-2` molecule pairs retain WCA for cross-site contacts, except for the
  explicit virtual-site pair(s) directly used by a bonded prior with
  `exclude_wca=true`.

For the current TEL22 priors the 210 direct backbone pairs are all explicit
`site0-site0` bonds, so only those 210 site pairs remain excluded in this A/B
mode. The script writes to the separate `wca12_selective_ab/` directory and
prints the global minimum distance and the PID/type pair that reaches it.

This is **not** a production Hamiltonian and is **not** consistent with the WCA
subtraction used to build the training targets. Its sole purpose is causal: if
the baseline trajectory aborts near 1.2 ps but the selective-1-2 trajectory
completes 1.3 ps without the `B2/B3` collapse, the molecule-level all-sites 1-2
exclusion is identified as the mechanism that permits the short-range contact.
A permanent policy change would then have to be implemented consistently in
both preprocessing and runtime, followed by dataset regeneration and retraining.
