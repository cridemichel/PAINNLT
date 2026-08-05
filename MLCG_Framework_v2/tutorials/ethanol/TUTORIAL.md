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

The second command writes `my_ethanol_model.pt`,
`my_ethanol_model.pt.manifest.json`, and `fast_training_config.json`. The
manifest binds the weights to the effective architecture, dataset and config.

## ESPResSo equilibration and NVE scaling

Point `PYPRESSO` to an ESPResSo build containing the PaiNN plugin:

```bash
PYPRESSO=/absolute/path/to/espresso/build/pypresso ./03_run_espresso.sh
```

Optional device selection:

```bash
PYPRESSO=/path/to/pypresso DEVICE=cpu ./03_run_espresso.sh
```

The script first runs a periodic-boundary/zero-edge regression against the
compiled PaiNN plugin. It then creates `equilibrated_ethanol.npz` under the
final analytic-prior + PaiNN Hamiltonian and launches NVE runs with several
values of `dt`, always starting from that same checkpoint.

Outputs:

- one preserved `energy_dt_*.csv` series per timestep;
- `energy_scaling.csv` with drift, detrended fluctuations and block lengths;
- `energy_scaling_fit.json` with slope, 95% bootstrap interval, $R^2$, drift,
  adjacent timestep ratios and the individual certification checks;
- `scaling_plot.png`.

The default experiment uses the same checkpoint for 5 ps at each timestep,
discards the initial fraction, detrends each energy series, estimates an
autocorrelation-aware block length and computes moving-block bootstrap
intervals. For velocity-Verlet in a smooth conservative system, the fitted
slope should approach two and halving the timestep should reduce the
fluctuation scale by about four. The tutorial enables strict mode by default:
it fails unless the point slope, confidence interval, $R^2$, drift and adjacent
timestep-ratio criteria all pass. Thresholds are explicit command-line options
of `run_energy_scaling.py`.


## Compatibility after the consistency patch

Regenerate all generated artifacts. In particular, old `rigid_bodies_info.json`
files do not contain principal-axis site coordinates, old model files have no
validated manifest, and old `.npz` checkpoints have no provenance metadata.
Do not use the legacy overrides for the final NVE scaling certificate.

## Complete pipeline

```bash
PYPRESSO=/absolute/path/to/pypresso ./run_full_pipeline.sh
```

The script fails immediately if a required executable or artifact is missing;
it no longer reports success after merely printing a command.
