# Ordered geometry head for the Ala2 architecture diagnostic

## Motivation and scope

The canonical PaiNN path did not improve when the five-bead Ala2 graph was
made complete and its transferable optimizer controls were matched to the
official CGnet tutorial. Its best held-out residual-force skill remained about
0.63%, versus about 0.65% for the original PaiNN run. The spectral constraint
at strength 4 was inactive because every measured dense-layer norm stayed
below the bound.

The remaining controlled architectural difference is the representation.
Official CGnet consumes an ordered internal-coordinate vector containing
distances, angles and signed dihedrals. Canonical PaiNN produces a scalar energy
from a reflection-even message-passing representation and has no explicit
pseudoscalar that identifies the sign of an ordered torsion. The architecture
variant `painn_ordered_geometry_tanh_v2` adds that missing information while
retaining PaiNN as a parallel learned correction.

This is a diagnostic architecture. It is not enabled for TEL22 or existing
models, and the base variant `painn_canonical_context_silu_v2` is unchanged.

## Ordered feature definition

For an ordered chain of `N` one-site molecules, the head contains
`N(N-1)/2 + (N-2) + 2(N-3)` features. For Ala2, `N=5`, giving 17 features in
this exact order:

1. all ten pair distances `(0,1), (0,2), ..., (3,4)` in lexicographic order;
2. the three consecutive angles `(0,1,2)`, `(1,2,3)`, `(2,3,4)` in radians;
3. `cos(phi), sin(phi)` for `(0,1,2,3)`, followed by `cos(phi), sin(phi)` for
   `(1,2,3,4)`.

Minimum-image displacement vectors are used throughout. For four consecutive
nodes `i,i+1,i+2,i+3`, define:

```text
b0 = x[i+1] - x[i]
b1 = x[i+2] - x[i+1]
b2 = x[i+3] - x[i+2]
n1 = b0 x b1
n2 = b1 x b2
cos(phi) = dot(n1,n2) / (|n1||n2|)
sin(phi) = dot(cross(n1,n2), b1/|b1|) / (|n1||n2|)
```

This convention is implemented identically in the CPU statistics fitter and
the differentiable LibTorch forward pass. Reversing the convention would only
change the learned sign, but mixing conventions between training and runtime
would invalidate the model.

## Train-only normalization and energy

Feature means and population standard deviations are fitted after the exact
8000/2000 split, using only the 8000 training frames. Standard deviations are
floored at `1e-6`. Both vectors are registered model buffers, saved in the
`.pt` file and restored by the ESPResSo plugin; validation and runtime never
refit them. The trainer also writes the 17 means, the 17 standard deviations,
the feature-order identifier, the signed-dihedral convention and the
normalization rule into the model manifest. The finalizer preserves these
trainer-generated fields and refuses to synthesize them from the JSON config.
The independent ordered-head energy scale is also part of the architecture
map, so the invalid `v1` checkpoint cannot be loaded as `v2`.

The standardized 17-vector is passed through five fully connected layers of
width 160 with `tanh`, followed by a scalar output. The head subtracts its
value at the standardized zero vector to fix an additive gauge.

The first `v1` diagnostic incorrectly multiplied both branches by the PaiNN
force scale (`824.081` for this split). This is not appropriate for the ordered
head: it consumes dimensionless standardized internal coordinates and its raw
energy follows the official CGnet kcal/mol convention. After converting the
dataset from Angstrom to nm, the coordinate derivative already supplies the
factor of ten in the force conversion; the head energy therefore needs only
`1 kcal/mol = 4.184 kJ/mol`. The observed branch scale was consequently
`824.081 / 4.184 = 196.96` times too large. Consistently, epoch 1 produced a
training loss of `73.6609` and a maximum gradient norm of `5786.86` even though
no dense spectral norm reached the bound of 4.

Version `v2` separates the two energy buffers. The total learned energy is:

```text
E_learned = force_rms_scale * E_PaiNN_raw
          + 4.184 kJ/mol * (head(z) - head(0))
```

Forces are always obtained by differentiating this single scalar energy with
respect to the same directed minimum-image edge vectors. No direct force head
or nonconservative correction is introduced.

The completed `v2` diagnostic confirmed that this scale correction was
necessary: the best validation force MSE moved from a negative-skill result to
`0.975231` against a zero-predictor baseline of `0.982019`. This corresponds to
`0.691%` explained residual-force variance and a `0.359%` validation-MAE
improvement. The first-epoch blow-up was also greatly reduced. Nevertheless,
the result is not materially better than the approximately `0.65%` canonical
PaiNN baseline, so merely extending the hybrid run is not justified.

## Fail-closed runtime contract

The ordered head requires:

- exactly `ordered_geometry_nodes` contiguous nodes per frame;
- one site per ordered molecule in the dataset used to fit statistics;
- a complete directed graph, because all pair distances are explicit inputs;
- identical node order during preprocessing, training and simulation.

The forward pass throws if node or edge counts violate this contract. For Ala2
the validator independently confirms that the 1.0 nm cutoff exceeds the
maximum reference pair distance. The differentiable head itself requires the
complete 20-edge directed graph on every call. The manifest records the
architecture variant, node count, head depth, head width and fitted feature
statistics so a base PaiNN model—or an ordered model with unidentified
normalization—cannot be loaded accidentally into this runtime.

## Differences that remain relative to official CGnet

The ordered head matches CGnet's explicit geometric information, `tanh` width,
depth, optimizer schedule and spectral bound, but the combined model is still
not official CGnet. It additionally contains PaiNN message passing and its RBF
representation, uses the framework's residual-force normalization and runs in
the framework's conservative ESPResSo integration path. Conclusions must
therefore be based on the matched prior-only versus prior+model FES diagnostic,
not on nominal architecture labels.

## CGnet-exact head-only isolation

The follow-up architecture `cgnet_ordered_geometry_tanh_v1` is deliberately
separate from the hybrid variant. It isolates the official dense
representation inside the framework:

- the PaiNN embedding, message/update blocks and readout are not instantiated;
- torsions follow CGnet's grouped feature order: both cosines precede both
  sines, rather than interleaved cosine/sine pairs;
- each `Linear` layer is first constructed normally and then only its weight is
  overwritten with Xavier-uniform values. This matches
  `cgnet.feature.utils.LinearLayer(weight_init="xavier")`, including retention
  of PyTorch's default bias initialization;
- training and runtime reconstruct the same head-only module, and the manifest
  records both `ordered_geometry_head_only=true` and
  `ordered_geometry_weight_initialization=xavier_uniform_weight_default_bias`.

This is still a controlled reproduction rather than bitwise parity: the C++
trainer uses the converted framework units and its own deterministic batch
implementation. It is, however, the clean test of whether the ordered dense
representation learns the same residual-force signal without interference
from the parallel PaiNN energy.

## Commands

After applying the patch, rebuild the trainer:

```bash
cmake --build training/build -j
```

Run the five-epoch training diagnostic:

```bash
PYTHON_BIN="$(command -v python3)" \
TRAINER="$PWD/training/build/train_painn" \
ALA2_ORDERED_SOURCE_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/cgnet_harmonic_50ep" \
ALA2_ORDERED_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/painn_ordered_geometry_scale4p184_5ep" \
bash tutorials/ala2_cgnet/diagnostics/scripts/05_test_ala2_painn_ordered_geometry.sh
```

The training stage writes `ala2_benchmark_report.json`, `cg_training_log.csv`,
`training_stdout.log`, `ala2_model.pt` and its manifest in the fresh run
directory. The report repeats the fitted feature statistics and checks that
their length, finiteness, floor and conventions agree with the architecture.

Rebuild ESPResSo only before the optional FES stage, because the plugin ABI and
model constructor now carry the ordered-head dimensions:

```bash
cmake --build espresso/build --parallel
```

Then run the quick matched FES without retraining:

```bash
PYRESSO="$PWD/espresso/build/pypresso" \
ALA2_ORDERED_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/painn_ordered_geometry_scale4p184_5ep" \
ALA2_ORDERED_AB_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/painn_ordered_geometry_scale4p184_fes_4x50k" \
bash tutorials/ala2_cgnet/diagnostics/scripts/05_test_ala2_painn_ordered_geometry.sh --fes-only
```

Only proceed to the FES stage if the five-epoch report shows a material gain
over the approximately 0.65% explained residual-force variance of the
canonical PaiNN baseline. This prevents a long simulation from hiding a failed
representation diagnostic.

Run the isolated CGnet-exact head from the framework root with:

```bash
cmake --build training/build -j

PYTHON_BIN="$(command -v python3)" \
TRAINER="$PWD/training/build/train_painn" \
ALA2_CGNET_EXACT_SOURCE_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/cgnet_harmonic_50ep" \
ALA2_CGNET_EXACT_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/cgnet_exact_head_5ep" \
bash tutorials/ala2_cgnet/diagnostics/scripts/06_test_ala2_cgnet_exact_head.sh
```

The default is training-only and takes five epochs. Inspect the three reported
training artifacts before considering `--fes-only`.
