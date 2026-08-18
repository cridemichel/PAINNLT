# TorchMD TEL22-sized NVE certification

This directory is intentionally isolated from `tutorials/tel22` and `tutorials/tel22_IBI`.
It does **not** attempt Hamiltonian parity with TEL22.  Its purpose is narrower: certify the
current TorchMD Velocity-Verlet + PyTorch precision path on a deterministic conservative
system with the same particle-count scale as the profiled TEL22 case (820 particles).

## What is tested

The reference Hamiltonian is a set of independent three-dimensional harmonic modes with a
spread of force constants.  It is smooth, exactly conservative, deterministic, and cheap.
Every timestep starts from the exact same initial positions and velocities.

For the default grid

```text
0.001 0.0015 0.002 0.003 0.004 0.005 ps
```

we run 1.98 ps of NVE per timestep, sample total energy every integration step, and fit

```text
sigma_E = C * dt^p
```

The default certification gates are the same style used by the MLCG NVE checks:

```text
1.7 <= p <= 2.3
log-log R^2 >= 0.97
relative block-mean drift <= 1e-4 for every dt
```

The report also includes `C2 = sigma_E / dt^2` and its max/min spread.

## Units and TorchMD API

TorchMD uses Angstrom for distance, kcal/mol for energy, g/mol for mass, and the Integrator
`timestep` argument is in femtoseconds.  The command-line grid is specified in ps to match
MLCG diagnostics and is multiplied by 1000 before constructing `torchmd.integrator.Integrator`.

## Run

Activate the environment where TorchMD is installed, then:

```bash
cd benchmarks/torchmd_tel22_nve
./run.sh --dry-run
./run.sh --overwrite
```

Default is CPU float64.  For float32:

```bash
TORCHMD_NVE_PRECISION=float32 ./run.sh --overwrite
```

For a future NVIDIA run:

```bash
TORCHMD_NVE_DEVICE=cuda TORCHMD_NVE_PRECISION=float32 ./run.sh --overwrite
TORCHMD_NVE_DEVICE=cuda TORCHMD_NVE_PRECISION=float64 ./run.sh --overwrite
```

You can also call `run_certification.py` directly to change duration, dt grid, particle count,
or thresholds.

## Outputs

Generated files are ignored by Git and written below `results/`:

```text
run_plan.json
initial_state.npz
nve_certification_runs.csv
nve_certification_report.json
dt_*/energy.csv
dt_*/metrics.json
```

The summary printed at the end contains `p`, `R2`, `C2 spread`, maximum relative drift, and
PASS/FAIL.

## Important scope limitation

A PASS means that **this TorchMD numerical integration path** exhibits the expected bounded
second-order NVE energy error on the smooth reference Hamiltonian.  It does not establish
energy/force parity with the MLCG TEL22 Hamiltonian and it is not an accuracy comparison of
PaiNN, SchNet, priors, Morse interactions, rigid bodies, or neighbor-list semantics.

The per-step timing is recorded only as context.  This script synchronizes energies to the
CPU every step to perform the certification, so it must **not** be used as a GPU performance
benchmark.

## Neural-potential / autograd follow-up

`run_neural.sh` keeps the same 820-particle initial-state generator, timestep grid, duration,
TorchMD `Integrator`, and NVE analysis, but replaces the analytic harmonic force calculation
with a frozen deterministic neural potential.  The model is a small 2x64 SiLU MLP that reads
per-particle squared displacement and applies a bounded positive modulation to the harmonic
energy.  Its parameters never require gradients; forces are obtained only from
`F = -dE/dr` using `torch.autograd.grad` with respect to coordinates.

This is deliberately **not** PaiNN, SchNet, or TEL22 Hamiltonian parity.  Its purpose is to
answer the narrower question: does introducing a neural forward pass plus coordinate autograd
cause the FP32 NVE scaling to depart from the nearly ideal analytic-reference result?

Run CPU FP64 and FP32:

```bash
./run_neural.sh --overwrite
TORCHMD_NVE_PRECISION=float32 ./run_neural.sh --overwrite
```

Future NVIDIA runs use the same interface:

```bash
TORCHMD_NVE_DEVICE=cuda TORCHMD_NVE_PRECISION=float32 ./run_neural.sh --overwrite
TORCHMD_NVE_DEVICE=cuda TORCHMD_NVE_PRECISION=float64 ./run_neural.sh --overwrite
```

The reports are written to:

```text
results/neural_cpu_float64/neural_nve_certification_report.json
results/neural_cpu_float32/neural_nve_certification_report.json
```

If the matching analytic reference report is present, compare the two directly:

```bash
python3 compare_reports.py \
  results/cpu_float32/nve_certification_report.json \
  results/neural_cpu_float32/neural_nve_certification_report.json \
  --output results/analytic_vs_neural_cpu_float32.json
```

Interpretation:

- neural `p ~= 2` in FP32: a generic PyTorch neural-forward/autograd path is not sufficient by
  itself to reproduce the TEL22+PaiNN FP32 scaling loss;
- neural FP32 `p` degrades while FP64 stays near 2: numerical precision in the neural force path
  becomes a plausible contributor;
- both precisions degrade: inspect the neural Hamiltonian/timestep regime before attributing the
  effect to floating-point precision.

As with the analytic benchmark, per-step timings are diagnostic only because energy is copied
to the CPU every step for certification.

## Synthetic PaiNN/message-passing follow-up

`run_painn.sh` is the next isolation step after the small MLP test.  It keeps TorchMD's same
Velocity-Verlet integrator and the same deterministic initial-state generator, but replaces the
small per-particle MLP with a frozen **PaiNN-like equivariant message-passing residual**.  The
Python blocks mirror the structure used by MLCG PaiNN: species embedding, scalar/vector message
blocks, vector update blocks, Gaussian RBFs with the Toxvaerd smooth cutoff, and a scalar readout.
Forces are still obtained only from `F = -dE/dr` through `torch.autograd.grad`.

To isolate PaiNN numerics from neighbor-list discontinuities, the graph is a deterministic fixed
k-nearest-neighbor graph built from the equilibrium coordinates.  The default has 820 particles,
24 directed neighbors per particle (19,680 directed edges), 2 layers, 32 hidden channels, 32 RBFs,
and 8 synthetic species.  A harmonic background keeps the trajectory well conditioned; the PaiNN
residual is calibrated once in canonical CPU float64 so that its initial RMS force is 0.5 times the
harmonic RMS force.  The **same scalar calibration and canonical weights** are then used for FP32
and FP64.

Because this test is substantially more expensive than the small MLP diagnostic, its default
physical duration is 0.60 ps per timestep (commensurate with the full dt grid).  For the exact
1.98 ps duration used by the analytic/MLP tests, set `TORCHMD_PAINN_NVE_DURATION_PS=1.98`.

Run FP64 and FP32:

```bash
./run_painn.sh --overwrite
TORCHMD_NVE_PRECISION=float32 ./run_painn.sh --overwrite
```

The reports are:

```text
results/painn_cpu_float64/painn_nve_certification_report.json
results/painn_cpu_float32/painn_nve_certification_report.json
```

Compare precisions:

```bash
python3 compare_painn_reports.py \
  results/painn_cpu_float32/painn_nve_certification_report.json \
  results/painn_cpu_float64/painn_nve_certification_report.json \
  --output results/painn_fp32_vs_fp64.json
```

A future NVIDIA run uses the same interface:

```bash
TORCHMD_NVE_DEVICE=cuda TORCHMD_NVE_PRECISION=float32 ./run_painn.sh --overwrite
TORCHMD_NVE_DEVICE=cuda TORCHMD_NVE_PRECISION=float64 ./run_painn.sh --overwrite
```

Interpretation is deliberately narrow.  If synthetic PaiNN FP32 loses second-order scaling while
FP64 remains near `p=2`, PaiNN-style message passing / accumulation becomes a plausible source of
the TEL22 FP32 floor.  If both stay near `p=2`, the remaining suspects are more specific to the
trained TEL22 PaiNN, its real graph/Hamiltonian, or the LibTorch/ESPResSo coupling.  This benchmark
is **not** TEL22 Hamiltonian parity and is **not** a TorchMD-Net performance benchmark.

## Exact cross-framework synthetic-PaiNN check

`mlcg_espresso_parity/` reuses the synthetic PaiNN case to run the same mathematical Hamiltonian through MLCG/ESPResSo + LibTorch. It performs static energy/force parity before the NVE sweep and compares scaling, drift and runtime against TorchMD. See `mlcg_espresso_parity/README.md`.
