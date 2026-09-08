# TEL22: transfer of the validated CGnet-style head

## Why this experiment exists

The Ala2 benchmark established that the framework can reproduce the official
CGnet force-matching head and can improve the sampled free-energy surface over
the harmonic prior.  The weak TEL22 result therefore cannot be attributed to a
generic failure of training alone.  The canonical PaiNN representation mixes
ten equivalent TEL22 copies in one large graph and does not explicitly expose
the antiparallel tetrad geometry.

This training diagnostic only changes the learned representation while keeping the existing
Morse-free Variant-A residual dataset, split seed, force/torque loss and 15
training epochs fixed.  It is a controlled representation test, not a claim of
thermodynamic validation.

## Fixed physical contract

The architecture identifier is `tel22_shared_geometry_tanh_v1`.  A frame must
contain ten copies in deterministic order, each with 22 residue molecules and
82 physical sites.  Site types must follow the TEL22 mapping:

- A: one site of type 0;
- T: one site of type 1;
- G: six ordered sites of type 2 through 7 (`S`, `B1`, ..., `B5`).

One 131-component vector is built for every copy:

| block | count | definition |
|---|---:|---|
| backbone distances | 21 | consecutive residue anchor sites |
| backbone angles | 20 | consecutive anchor triplets |
| backbone torsions | 38 | 19 cosines followed by 19 sines |
| tetrad orientation | 36 | 18 antiparallel pairs times `B3-B3` plus the local antiparallel `B2-B4` or `B4-B2` endpoint |
| adjacent stacking | 16 | 8 column-neighbour pairs times `B3-B3`, `B5-B5` |

The three tetrads are `(2,10,14,22)`, `(3,9,15,21)` and `(4,8,16,20)` in
one-based residue numbering, matching PDB 143D MODEL 1.  The same MLP is
applied to all ten vectors and the ten scalar energies are summed.  Sharing
weights encodes copy equivalence and fits normalization statistics on
`train_frames x 10` local samples.  Using base-site distances rather than only
residue centres retains orientation dependence and therefore permits nonzero
torques on the rigid guanines.

The physical neighbour cutoff remains 1.2616 nm.  Before training,
`prepare_shared_head_cutoff.py` scans all frames and records the maximum and
the number of descriptor observations outside that cutoff.  The trainer then
adds only the required within-copy topology edges that are absent from the
physical-cutoff graph.  These extra edges feed the head-only descriptor and do
not enlarge a PaiNN message-passing neighbourhood.  Training still fails
closed when the topology, site order, type sequence or mandatory edges
disagree.  TEL22 remains in native
kJ/mol and nm units; the independent energy scale is `kBT = 2.4943387854
kJ/mol`, not Ala2's `4.184` kcal-to-kJ conversion.

Both cross-oriented endpoints are not included for every tetrad pair: on the
available 51-frame diagnostic dataset, some of the nonlocal alternatives reach
1.343 nm and therefore fall outside the established cutoff.  Raising the
cutoff would confound the comparison and inflate the full-system neighbour
list.  The fixed endpoint table selects the local antiparallel alternative;
all selected tetrad descriptors stay below 1.166 nm in that dataset.  The
1001-frame dataset is wider: its maximum is 1.773788 nm for the oriented
G10--G14 descriptor in zero-based copy 1 at frame 719.  The same copy and frame
also contain a 1.672827 nm G10--G22 tetrad distance and a 1.241899 nm G10--G9
stacking distance.  The common G10 endpoint makes this a correlated structural
excursion, not an isolated B2/B4 bookkeeping error.  Increasing the global
cutoff to 1.795 nm would raise the geometric neighbour-volume estimate by about
2.9 times; explicit topology-edge augmentation preserves the sampled open
configuration without that cost.  The complete evidence is retained in
`tel22_shared_cutoff_report.json`.

## Deliberate first-stage model

The first screen is head-only: five 160-unit tanh hidden layers, Xavier weight
initialization and dense-layer spectral projection at 4.0.  This mirrors the
validated CGnet-style learned component.  The PaiNN branch is disabled so that
we can answer one question cleanly: does a topology-aware, copy-shared
representation extract material residual-force signal from this dataset?

Do not start production MD from this model merely because training completes.
First compare its validation skill with the previous Variant-A PaiNN result.
A hybrid shared-head plus PaiNN ablation is justified only if this head-only
screen shows reproducible signal.

## Build and run

The topology-edge augmentation is currently implemented for the trainer.  The
diagnostic therefore requires rebuilding the trainer and remains unsuitable
for production MD.  A future runtime test must first implement and validate the
same explicit-edge policy in the ESPResSo plugin.

```bash
cmake --build training/build -j
```

Reuse the already generated 1001-frame Variant-A dataset:

```bash
PYTHON_BIN="$(command -v python3)" \
TRAINER="$PWD/training/build/train_painn" \
TEL22_SHARED_SOURCE_RUN_DIR="$PWD/tutorials/tel22/diagnostics/smoke/variant_a_long_1001f_15ep" \
TEL22_SHARED_RUN_DIR="$PWD/tutorials/tel22/diagnostics/smoke/shared_head_1001f_15ep" \
bash tutorials/tel22/diagnostics/scripts/28_test_tel22_shared_head.sh
```

The run directory must be new or empty.  No GROMACS trajectory is read and no
dataset is rebuilt.  At completion, retain and share:

- `tel22_shared_head_report.json`;
- `tel22_shared_cutoff_report.json`;
- `cg_training_log.csv`;
- `training_stdout.log`.

The report gives the best validation loss relative to the validation
zero-predictor baseline.  Its 3% and 10% labels are screening bands chosen for
this controlled diagnosis; they are not literature acceptance criteria.  A
production decision still requires replicated CG trajectories, structural
observables, tetrad/contact populations and stability checks.
