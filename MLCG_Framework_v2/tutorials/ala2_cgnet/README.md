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
ALA2_CGNET_COMPARATOR_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/official_cgnet_brownian_ab" \
bash tutorials/ala2_cgnet/diagnostics/scripts/03_compare_official_cgnet.sh
```

The official model is trained on CPU by default because this old reference
code predates MPS. The system is only five beads, so GPU acceleration is not
normally useful; `ALA2_CGNET_DEVICE=mps` or `cuda` remains available for an
explicit experiment. Simulation always uses the official CPU Brownian
integrator. The script now runs two matched Brownian branches: harmonic prior
only and harmonic prior plus CGnet. Both start from the same reference frames
and reset the official random generator to the same seed, so `dt`, the full
noise sequence, length and sampling are controlled. Its retained frame count is inferred from
the completed A/B replicas, preventing a comparison with unequal sample
counts.

The decisive outputs are:

- `official_cgnet_training_report.json`, including force MSE relative to the
  harmonic-prior baseline;
- `ala2_painn_vs_official_cgnet_report.json`, including aggregate and paired
  replica bootstrap comparisons against both prior-only and PaiNN, plus the
  matched Brownian prior-only/CGnet A/B result;
- `ala2_painn_vs_official_cgnet.png`, containing all five FES surfaces.

A positive
`js_improvement_vs_painn_nats_positive_is_cgnet_better` means that official
CGnet is closer to the atomistic reference. Treat the result as conclusive only
if the corresponding paired 95% bootstrap interval also excludes zero. This
public 10,000-frame subset still cannot reproduce the paper's one-million-frame
production result, but it cleanly tests whether the current failure is specific
to the PaiNN/framework path.

For the architecture-specific conclusion, use
`cgnet_external.matched_brownian_ab`. Its
`js_improvement_nats_positive_is_cgnet_better` compares CGnet with its own
harmonic prior under the identical official integrator. A positive confidence
interval demonstrates that the learned CGnet correction, rather than the
difference between Brownian and ESPResSo dynamics, improves the sampled FES.

## CGnet-matched PaiNN diagnostic

The controlled official result motivates a PaiNN test that transfers every
CGnet training choice with a direct framework analogue before adding an
Ala2-specific torsional head. This variant keeps the exact same 10,000 frames,
residual targets, harmonic priors and 8000/2000 tail split. It uses:

- the cutoff is increased from 0.5 to 1.0 nm; the validator computes the
  maximum pair distance over all reference frames and refuses the run unless
  every one of the five beads is connected to every other bead;
- after each optimizer step, every dense weight matrix is projected onto a
  spectral-norm ball of strength 4 using deterministic power iteration. The
  embedding table is excluded. This is separate from the legacy
  `lipschitz_lambda`, which is a force-magnitude penalty rather than CGnet's
  dense-layer projection;
- width 160, five interaction layers, batch size 512, AdamW with zero weight
  decay, initial learning rate 0.003, a factor-0.3 decay after every epoch, no
  gradient clipping and five epochs, matching the transferable settings of the
  pinned official tutorial.

This remains PaiNN, not CGnet: its message-passing/RBF representation and SiLU
nonlinearity do not become the official standardized distance-angle-dihedral
features and `tanh` MLP. The report therefore labels the run as a controlled
architecture diagnostic, not as an equivalent reproduction.

Rebuild the trainer after applying the patch, then run the training stage:

```bash
cmake --build training/build -j

PYTHON_BIN="$(command -v python3)" \
TRAINER="$PWD/training/build/train_painn" \
ALA2_ALLTOALL_SOURCE_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/cgnet_harmonic_50ep" \
bash tutorials/ala2_cgnet/diagnostics/scripts/04_test_ala2_painn_alltoall_spectral.sh
```

The script reuses rather than regenerates the baseline dataset and writes a
fresh `painn_cgnetmatched_5ep` run. First inspect
`ala2_benchmark_report.json`. If force skill is not materially better, there is
no reason to spend time on dynamics. If it improves, run the matched quick FES
stage without retraining:

```bash
PYTHON_BIN="$(command -v python3)" \
PYRESSO="$PWD/espresso/build/pypresso" \
ALA2_ALLTOALL_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/painn_cgnetmatched_5ep" \
ALA2_ALLTOALL_AB_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/painn_cgnetmatched_fes_4x50k" \
bash tutorials/ala2_cgnet/diagnostics/scripts/04_test_ala2_painn_alltoall_spectral.sh --fes-only
```

The quick FES stage uses four matched replicas, 25,000 equilibration steps and
50,000 production steps per branch. It is a screening diagnostic, not the
final production estimate. Explicit angle/dihedral features are intentionally
not mixed into that all-to-all diagnostic: its negative result motivates the
separate architectural intervention below.

## Ordered, chirality-aware geometry head

The all-to-all CGnet-matched PaiNN run remained at approximately the same weak
residual-force skill as the original PaiNN model, while its strength-4 spectral
projection never activated. The next diagnostic therefore changes the
representation rather than the optimizer: `painn_ordered_geometry_tanh_v2`
adds a conservative train-normalized head containing all ten five-bead
distances, three consecutive angles and sine/cosine pairs for both ordered
backbone dihedrals.

The initial `v1` trial reused PaiNN's `824.081` force-RMS multiplier for this
standardized head and therefore over-scaled its raw CGnet-like energy by about
197 times relative to the required kcal/mol-to-kJ/mol factor `4.184`. Its
negative force skill is retained as a diagnostic result, not interpreted as a
failure of the ordered representation. Version `v2` gives the ordered branch
an independent `4.184 kJ/mol` energy buffer while leaving the canonical PaiNN
branch and every other matched control unchanged.

The exact feature order, signed-torsion convention, energy expression,
train-only normalization and manifest provenance, fail-closed graph contract,
remaining differences from CGnet and complete commands are documented in
[ORDERED_GEOMETRY_HEAD.md](ORDERED_GEOMETRY_HEAD.md).
Existing base PaiNN models retain the original architecture variant and do not
instantiate this head. The `v1` ordered checkpoint is intentionally rejected
because it lacks the independent energy-scale buffer.

The corrected `v2` trial removed the numerical failure but did not produce a
material architectural gain: its best held-out explained residual-force
variance was `0.691%` and its MAE improvement was `0.359%`, compared with about
`0.65%` for canonical PaiNN and `0.86%` for the pinned official CGnet run.
This motivates one final representation isolation rather than a longer hybrid
training.

## Framework-native CGnet-exact head isolation

The `cgnet_ordered_geometry_tanh_v1` diagnostic removes the PaiNN learned
branch entirely. It retains the identical dataset, residual targets, harmonic
priors, split, optimizer schedule and spectral projection, while matching the
official dense CGnet head in the remaining details:

- features are ordered as all distances, all angles, all dihedral cosines,
  then all dihedral sines;
- five hidden layers of width 160 use `tanh`;
- every dense weight is initialized with Xavier uniform, while the default
  `nn.Linear` bias initialization is retained;
- the raw scalar uses the independent `4.184 kJ/mol` energy scale;
- no embedding, PaiNN message block, update block or PaiNN readout is
  instantiated or optimized.

Rebuild the trainer and run the five-epoch training-only diagnostic:

```bash
cmake --build training/build -j

PYTHON_BIN="$(command -v python3)" \
TRAINER="$PWD/training/build/train_painn" \
ALA2_CGNET_EXACT_SOURCE_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/cgnet_harmonic_50ep" \
ALA2_CGNET_EXACT_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/cgnet_exact_head_5ep" \
bash tutorials/ala2_cgnet/diagnostics/scripts/06_test_ala2_cgnet_exact_head.sh
```

Send `ala2_benchmark_report.json`, `cg_training_log.csv` and
`training_stdout.log` from the new run. Do not launch the optional FES before
examining that report. If the isolated head behaves consistently with the
official CGnet comparator, rebuild ESPResSo and use the same script with
`--fes-only`; its default FES screen is four replicas of 50,000 production
steps.
