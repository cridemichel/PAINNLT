# Ethanol coarse-graining tutorial

This tutorial exercises one consistent pipeline:

1. map the atomistic trajectory and subtract the analytic priors;
2. train PaiNN on `my_ethanol_dataset.bin`;
3. equilibrate with the same priors, box, exclusions and PaiNN model used in production;
4. run NVE simulations from the same checkpoint at several timesteps;
5. fit the scaling of the total-energy fluctuations.

## Inputs

`01_build_dataset.sh` expects:

- `../../../GROMACS/ethanol.trr`
- `../../../GROMACS/ethanol.gro`

The preprocessing step writes:

- `my_ethanol_dataset.bin`
- `cg_priors.json`
- `rigid_bodies_info.json`

## Training

```bash
./01_build_dataset.sh
./02_train_model.sh
```

The second command writes `my_ethanol_model.pt` and
`fast_training_config.json`.

## ESPResSo equilibration and NVE scaling

Point `PYPRESSO` to an ESPResSo build containing the PaiNN plugin:

```bash
PYPRESSO=/absolute/path/to/espresso/build/pypresso ./03_run_espresso.sh
```

Optional device selection:

```bash
PYPRESSO=/path/to/pypresso DEVICE=cpu ./03_run_espresso.sh
```

The script first creates `equilibrated_ethanol.npz` under the final
analytic-prior + PaiNN Hamiltonian.  It then launches NVE runs with several
values of `dt`, always starting from that same checkpoint.

Outputs:

- `energy_scaling.csv`
- `scaling_plot.png`

For velocity-Verlet in a smooth conservative system, the standard deviation
of the total energy should approach a slope of two on the log-log plot as the
timestep enters the asymptotic regime.

## Complete pipeline

```bash
PYPRESSO=/absolute/path/to/pypresso ./run_full_pipeline.sh
```

The script fails immediately if a required executable or artifact is missing;
it no longer reports success after merely printing a command.
