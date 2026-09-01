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

Harmonic priors are fitted only on frames 0--7999.  Frames 8000--9999 form a held-out
tail validation set, matching the trainer configuration and preventing prior
parameter leakage into validation.  A CGnet-style excluded-volume WCA acts only
between beads separated by more than two bonds.  Its 0.22 nm cutoff is below
the minimum such distance in the official subset (about 0.257 nm), so it is
identically zero on all training/validation targets and only regularizes
out-of-distribution simulation geometries.

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

## Matched free-energy A/B diagnostic

Force-matching loss is dominated by irreducible noise for this mapping, so the
decisive physical diagnostic is whether PaiNN improves the equilibrium
distribution.  After completing the harmonic-prior training run above, run:

```bash
PYTHON_BIN="$(command -v python3)" \
PYRESSO="$PWD/espresso/build/pypresso" \
ALA2_TRAINING_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/cgnet_harmonic_50ep" \
DEVICE=auto \
bash tutorials/ala2_cgnet/diagnostics/scripts/02_test_ala2_free_energy_ab.sh
```

The test selects four evenly spaced reference frames.  For every replica it
creates one common prior-only equilibrated checkpoint, then starts both the
prior-only and prior+PaiNN productions from exactly that mechanical state and
uses the same Langevin seed.  The default is a local diagnostic: four replicas,
25,000 equilibration steps and 250,000 production steps per branch.  Override
these with `ALA2_AB_REPLICAS`, `ALA2_AB_EQUIL_STEPS`,
`ALA2_AB_PRODUCTION_STEPS`, `ALA2_AB_BURNIN_STEPS` and
`ALA2_AB_SAMPLE_INTERVAL`.

Outputs include `ala2_fes_ab_report.json` and `ala2_fes_ab.png`.  The report
contains Jensen-Shannon divergence, reference-support coverage, a paired
replica bootstrap, and the mean squared free-energy-surface difference in
`(kBT)^2` after the optimal additive shift.  The last metric follows the
comparison used in the CGnet paper, restricted to bins sufficiently supported
by the smaller public subset.  Positive `js_improvement` and
`fes_mse_improvement` mean that prior+PaiNN is closer to the atomistic
reference than the prior alone.

This is not numerically identical to the published production protocol.  The
paper trained on one million frames and aggregated 100 independent simulations
of one million steps each; the public repository supplies only a 10,000-frame
subset.  It also used overdamped rather than inertial Langevin dynamics: the
equilibrium distribution is comparable, but the kinetics are not.  To add a
direct fourth surface from a CGnet trajectory with shape
`(frames, 5, 3)`, set `ALA2_CGNET_SAMPLES=/path/to/samples.npy` and optionally
`ALA2_CGNET_UNITS=nm` (the default is Angstrom).

## Controlled comparison with the official dense CGnet

If the matched A/B test does not show a meaningful PaiNN improvement, the next
diagnostic runs the reference implementation rather than a `CGnet-like` model.
It downloads and verifies commit
`a3e0e8ddc06f4b6a9f48f4886b73b4cf372ff481` from the official repository and
leaves its source unchanged. The only runtime compatibility shim restores the
removed NumPy name `np.bool`.

The comparator reproduces the dense Ala2 tutorial architecture: all invariant
distances, backbone angles and sine/cosine dihedrals; five hidden layers of 160
`tanh` units; harmonic bond/angle priors; Adam at `0.003`; and per-batch
Lipschitz projection with strength 4. To keep the comparison controlled, one
intentional change from the notebook is made: feature statistics and priors are
fit on frames 0--7999 only, while frames 8000--9999 are held out exactly as in
the PaiNN diagnostic. The default five-epoch schedule is the one in the
official tutorial.

Run this after the A/B trajectories already exist:

```bash
PYTHON_BIN="$(command -v python3)" \
ALA2_TRAINING_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/cgnet_harmonic_50ep" \
ALA2_AB_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/fes_ab_quick_4x50k" \
ALA2_CGNET_COMPARATOR_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/official_cgnet_quick" \
bash tutorials/ala2_cgnet/diagnostics/scripts/03_compare_official_cgnet.sh
```

The official model is trained on CPU by default because this old reference
code predates MPS. The system is only five beads, so GPU acceleration is not
normally useful; `ALA2_CGNET_DEVICE=mps` or `cuda` remains available for an
explicit experiment. Simulation always uses the official CPU Brownian
integrator. Its retained frame count is inferred from the completed A/B
replicas, preventing a comparison with unequal sample counts.

The decisive outputs are:

- `official_cgnet_training_report.json`, including force MSE relative to the
  harmonic-prior baseline;
- `ala2_painn_vs_official_cgnet_report.json`, including aggregate and paired
  replica bootstrap comparisons against both prior-only and PaiNN;
- `ala2_painn_vs_official_cgnet.png`, containing all four FES surfaces.

A positive
`js_improvement_vs_painn_nats_positive_is_cgnet_better` means that official
CGnet is closer to the atomistic reference. Treat the result as conclusive only
if the corresponding paired 95% bootstrap interval also excludes zero. This
public 10,000-frame subset still cannot reproduce the paper's one-million-frame
production result, but it cleanly tests whether the current failure is specific
to the PaiNN/framework path.
