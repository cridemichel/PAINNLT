# Alanine-dipeptide CGnet benchmark

This diagnostic checks PaiNN force matching on the public five-bead alanine
dipeptide example distributed by the original `coarse-graining/cgnet`
repository.  It is deliberately independent of TEL22, rigid bodies, torques,
Morse interactions and MDAnalysis.

The public arrays contain 10,000 frames sampled every 10 ps.  Their bead order
is `C(ACE)-N(ALA)-CA(ALA)-C(ALA)-N(NME)`.  The official CGSchNet example uses
atomic-number embeddings `6,7,6,6,7`, harmonic priors on the four consecutive
bonds and three consecutive angles, and a 5 Angstrom interaction cutoff.  This
benchmark preserves those choices while converting to framework units:

- coordinates: Angstrom to nm (`x 0.1`);
- forces: kcal/(mol Angstrom) to kJ/(mol nm) (`x 41.84`);
- PaiNN cutoff: 5 Angstrom to 0.5 nm;
- torque loss: disabled because every bead is a one-site particle.

The PaiNN configuration also mirrors the published tutorial's graph-model
scale where the architectures have direct analogues: 128 hidden channels,
five interaction layers, 50 radial functions, batch size 512 and learning rate
`3e-4`.  PaiNN is still not architecturally identical to CGSchNet, so the
comparison is a framework diagnostic rather than a claim of model parity.

Priors are fitted only on frames 0--7999.  Frames 8000--9999 form a held-out
tail validation set, matching the trainer configuration and preventing prior
parameter leakage into validation.

## Run

From the framework root, with the Python environment active and the C++
trainer already built:

```bash
PYTHON_BIN="$(command -v python3)" \
TRAINER="$PWD/training/build/train_painn" \
DEVICE=auto \
bash tutorials/ala2_cgnet/diagnostics/scripts/01_test_ala2_cgnet_benchmark.sh
```

`DEVICE` is accepted for consistency with the other diagnostics, but device
selection is performed by `train_painn`.  Set a fresh output directory with
`ALA2_RUN_DIR`.  Existing evidence is never overwritten.

The default run downloads the two official arrays, verifies their SHA-256
digests and shapes, builds a residual-force dataset with CGnet-style harmonic
priors, trains for at most 50 epochs, and writes
`ala2_benchmark_report.json`.

To test direct force matching without harmonic priors, select a fresh run
directory and set:

```bash
ALA2_PRIOR_MODE=none \
ALA2_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/raw_force_50ep" \
bash tutorials/ala2_cgnet/diagnostics/scripts/01_test_ala2_cgnet_benchmark.sh
```

The pipeline test passes when artifacts and metrics are internally consistent;
it does not manufacture a scientific pass threshold.  The report classifies
the validation learning signal from the normalized force-MSE skill relative to
the zero predictor:

- `strong`: at least 10% explained validation residual-force variance;
- `moderate`: 5--10%;
- `weak`: below 5%;
- `negative`: worse than the zero predictor.

This small public subset is a diagnostic, not a state-of-the-art Ala2 model.
The original repository explicitly notes that the paper used one million
frames whereas the tutorial subset contains ten thousand.

Data and reference implementation:

- https://github.com/coarse-graining/cgnet/tree/master/examples/data
- https://github.com/coarse-graining/cgnet/blob/master/examples/CG-Force-Fields-With-SchNet-Embeddings.ipynb
- Wang et al., ACS Central Science (2019), DOI: 10.1021/acscentsci.8b00913
