# Exact synthetic PaiNN: TorchMD vs MLCG/ESPResSo

This benchmark answers one narrow question:

> Does MLCG/ESPResSo reproduce the NVE behavior of the **same synthetic PaiNN Hamiltonian** already used by the TorchMD benchmark?

It is deliberately **not TEL22** and does not touch TEL22 or TEL22_IBI inputs.

## What is held fixed

The shared case exporter recreates the existing synthetic-PaiNN benchmark defaults:

- 820 particles
- 8 species
- PaiNN: 2 layers x 32 hidden channels
- 32 radial basis functions
- fixed directed 24-nearest-neighbor graph (19,680 edges)
- cutoff 12.616 A
- identical deterministic canonical weights
- identical isolated-species energy gauge
- identical harmonic background
- identical initial coordinates, masses and velocities
- PaiNN residual calibrated to 0.5 x the initial harmonic RMS force
- dt = 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005 ps
- 0.60 ps per dt by default

The benchmark evaluates the full synthetic Hamiltonian inside LibTorch in the same numerical units used by the TorchMD test (A and kcal/mol), including the harmonic term. Only the final energy and forces are converted at the ESPResSo boundary to kJ/mol and kJ/mol/nm. This is important for FP32: MLCG does not receive a hidden float64 harmonic contribution.

Before any trajectory is accepted, MLCG must pass a static energy/force/kinetic-energy parity gate against references exported from the Python/Torch implementation.

## Isolation from production

All tracked files live under `benchmarks/`.

`run_mlcg.sh` does temporarily replace the **ESPResSo checkout copy** of `PaiNN_ML_Potential.cpp` with a benchmark-only generated source, rebuilds ESPResSo, runs the benchmark, then restores the production source and rebuilds it again. A shell trap performs restoration even when the NVE certification exits `FAIL`.

The tracked production source under `simulation/espresso_plugin/` is never edited.

## Preflight

From this directory:

```bash
python3 selftest.py
bash run_mlcg.sh --dry-run
```

The environment must contain the normal MLCG/ESPResSo Python dependencies. `ESPRESSO_SRC` and `ESPRESSO_BUILD` can override the default `../../../../espresso` and its `build` directory.

## Run MLCG FP64

```bash
MLCG_PAINN_PARITY_PRECISION=float64 \
  bash run_mlcg.sh --overwrite
```

Output:

```text
../results/mlcg_painn_cpu_float64/mlcg_painn_nve_certification_report.json
```

## Run MLCG FP32

```bash
MLCG_PAINN_PARITY_PRECISION=float32 \
  bash run_mlcg.sh --overwrite
```

Output:

```text
../results/mlcg_painn_cpu_float32/mlcg_painn_nve_certification_report.json
```

A strict overall `FAIL` at `dt=0.005 ps` is not automatically a benchmark failure: the TorchMD synthetic PaiNN itself marginally fails the current `1e-4` block-drift gate there. The diagnostic questions are the fitted exponent, R2, C2 spread, static parity, and FP32-vs-FP64 behavior.

## Compare with the existing TorchMD reports

FP32:

```bash
python3 compare_cross_framework.py \
  ../results/painn_cpu_float32/painn_nve_certification_report.json \
  ../results/mlcg_painn_cpu_float32/mlcg_painn_nve_certification_report.json \
  --output ../results/torchmd_vs_mlcg_painn_float32.json
```

FP64:

```bash
python3 compare_cross_framework.py \
  ../results/painn_cpu_float64/painn_nve_certification_report.json \
  ../results/mlcg_painn_cpu_float64/mlcg_painn_nve_certification_report.json \
  --output ../results/torchmd_vs_mlcg_painn_float64.json
```

The comparator refuses to compare reports if precision, dt grid, particle count, edge count, or the canonical PaiNN fingerprint differ.

It also reports the median `ms/step` ratio. That speed ratio is substantially fairer than comparing TEL22 production against the smaller synthetic benchmark because both sides now execute the same PaiNN graph and Hamiltonian.

## If the process is killed ungracefully

`SIGINT`, `SIGTERM`, normal errors and a strict NVE `FAIL` run through the restoration trap. A hard `SIGKILL` cannot run shell cleanup. If that happens, restore production explicitly from the framework root:

```bash
ESPRESSO_SRC="$PWD/espresso" bash simulation/espresso_plugin/copy_plugin_files.sh
cmake --build espresso/build -j4
```

Then rerun `python3 -m pytest -q` if desired.

## ESPResSo System lifetime

ESPResSo permits only one `System` instance per Python process. The certification
therefore launches one fresh worker process per timestep. This also prevents any
integrator or particle state from leaking between dt runs; model loading and
process startup remain outside the reported integration-loop timing.
