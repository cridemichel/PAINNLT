# TEL22 diagnostics

This directory contains non-production validation entry points and their output.

## Antiparallel 143D topology

Validate the source topology against PDB 143D MODEL 1:

```bash
python3 tutorials/tel22/diagnostics/scripts/validate_antiparallel_topology.py \
  --topology tutorials/tel22/tel22_topology.json \
  --pdb tutorials/tel22/143D.pdb \
  --r0-mode auto \
  --require-reference-metadata
```

The source configuration intentionally stores `r0: "auto"`. The corrected
distances must be inferred from the same atomistic trajectory used to rebuild
the residual-force dataset. Existing `cg_priors.json`, datasets, models and
checkpoints generated from the legacy contact graph are stale.

## Local 40-epoch pipeline smoke test

The runner creates a fresh, isolated directory below
`diagnostics/smoke/antiparallel_pipeline_40ep` and executes mapping/prior
inference, residual training, short equilibration and short NVT production:

```bash
AA_TOPOLOGY=/path/to/md.gro \
AA_TRAJECTORY=/path/to/md_whole.trr \
TRAINER=/path/to/training/build/train_painn \
PYRESSO=/path/to/espresso/build/pypresso \
DEVICE=auto \
bash tutorials/tel22/diagnostics/scripts/07_test_antiparallel_pipeline_40ep.sh
```

It refuses to overwrite an existing result directory. Set a fresh
`PIPELINE_TEST_RUN_DIR` for another run. The training profile disables early
stopping within the requested 40 epochs; the final validator requires exactly
40 finite log rows, provenance hashes, finite short-MD output and all expected
artifacts.

This is a functional smoke test. It does not validate thermodynamics, folding
populations, kinetics, production convergence or NVE energy conservation.
# TEL22 diagnostics

This directory contains validation, certification, convergence and exploratory test evidence.
It is deliberately separated from the tutorial root, which contains the pipeline and canonical artifacts.

- `scripts/`: diagnostic/test entry points.
- `nve/`: NVE certification and convergence evidence.
- `ibi/`: IBI calibration/angle/dihedral evidence (TEL22_IBI).
- `ml/`: residual-ML benchmark/runtime validation evidence (TEL22_IBI).
- `historical/`: superseded development evidence kept for provenance.
- `morse/`, `nonbonded/`, `smoke/`: focused diagnostics.

GROMACS-generated trajectory/topology/working files are intentionally kept at the tutorial root and are never deleted by the layout migrator.

## TEL22 Morse / dihedral NVE ablation

`diagnostics/scripts/11_test_nve_without_morse_dihedrals.sh` repeats the standard
TEL22 NVE timestep scan while selectively removing analytic prior terms, without
editing production inputs.  It compares `baseline`, `no_morse`, `no_dihedrals`
and `no_morse_no_dihedrals` using the same trained PaiNN and the same real
mechanical checkpoint state.  When Morse is removed, its technical endpoint
markers are stripped from a derived provenance-bound checkpoint.

The current production `cg_priors.json` contains 180 pair-specific Morse bonds
and zero dihedral priors.  Therefore, for the current TEL22, the no-dihedral
branches are exact aliases and are deliberately not rerun.  This is a numerical
ablation diagnostic: the PaiNN residual is not retrained after removing priors,
so the modified Hamiltonians are not reparameterized production models.

## TEL22 stock-ESPResSo coarse-dt NVE control

`diagnostics/scripts/15_test_nve_stock_espresso_coarse_5ps.sh` isolates the
classical stock-ESPResSo force path used by TEL22. It disables PaiNN and derives
a Morse-free copy of the production priors/checkpoint, removing the technical
Morse marker tail while preserving the physical/runtime checkpoint prefix.
The control retains the production harmonic bonds, harmonic angles and WCA
pair table; WCA is evaluated with stock `espressomd.interactions.LennardJones`.
Production TEL22 has no dihedral or conservative-spline priors. The default scan
uses 5 ps at `dt = 0.002, 0.003, 0.004, 0.005 ps`, with every-step energy
sampling. This is an integration diagnostic, not a reparameterized physical
model.

## TEL22 Morse top-10% a=0.85 long/full-grid robustness

`diagnostics/scripts/21_test_nve_morse_top10_a0p85_robustness.sh` validates the
best numerical-regularity candidate from test 20 on a longer and wider NVE
scan. It compares the production Morse priors against the exact fixed 18-contact
(top-10% checkpoint-local-curvature) candidate with `a` scaled by 0.85. Both
arms disable PaiNN and use the production-like pair-specific marker/non-bonded
switched-Morse runtime, with the identical physical checkpoint state. The
default grid is `0.001, 0.0015, 0.002, 0.003, 0.004, 0.005 ps`, 10 ps per
point, every-step energy sampling. The summary also compares the shared coarse
grid against the 5 ps test-20 evidence to detect window-specific improvements.
For TEL22 these Morse terms are treated as structural/numerical stabilizers,
not as physically inferred Morse parameters; any full PaiNN+prior production
promotion still requires rebuilding/retraining the residual against the changed
priors.

## TEL22 Morse stabilizer A/B/C: selective vs uniform a=0.85

`diagnostics/scripts/22_test_nve_morse_uniform_abc.sh` follows test 21 by
comparing three empirical Morse-stabilizer policies on the same 10 ps full-grid
priors-only NVE protocol. Arm A is production (`a=0.30` on all 180 Morse), arm
B is the test-21 selective candidate (`a=0.255` only on the fixed top-18
checkpoint-local-curvature contacts), and arm C applies `a=0.255` uniformly to
all 180 Morse contacts. A and B are validated and reused from test 21, so only
C requires new MD. All arms use the production-like marker/non-bonded switched
Morse runtime with PaiNN disabled and the identical physical checkpoint state.
The uniform candidate preserves every Morse D/r0/cutoff/endpoint and changes
only `a`; the derived checkpoint changes provenance metadata only. Because the
Morse terms are empirical TEL22 structural/numerical stabilizers rather than
contact-specific physical fits, the test asks whether one uniform stabilizer
parameter is cleaner and more robust than the snapshot-ranked selective rule.
Any chosen changed-prior model must still go through the PaiNN closure and then
residual regeneration/retraining before production validation.

## Variant A: WCA plus harmonic bonds and angles, without Morse

This ablation changes only the prior topology: it derives a temporary topology
from the corrected PDB-143D source, removes all 180 pair-specific Morse
contacts, and retains WCA, 210 harmonic backbone bonds and 200 harmonic
backbone angles. The source topology and normal TEL22 pipeline are not edited.

The diagnostic uses the same architecture, optimizer, random split and seed as
the 40-epoch Morse run, but stops after 15 epochs because that run reached its
best validation loss at epoch 6. Run it in a new isolated directory:

```bash
AA_TOPOLOGY="$PWD/tutorials/tel22/md.gro" \
AA_TRAJECTORY="$PWD/tutorials/tel22/long_run/md_whole.trr" \
TRAINER="$PWD/training/build/train_painn" \
PYRESSO="$PWD/espresso/build/pypresso" \
VARIANT_A_RUN_DIR="$PWD/tutorials/tel22/diagnostics/smoke/variant_a_long_1001f_15ep" \
DEVICE=auto \
bash tutorials/tel22/diagnostics/scripts/08_test_variant_a_pipeline_15ep.sh
```

The generated `pipeline_test_report.json` records the best validation epoch
and loss in addition to the initial/final values. Compare raw validation MAE
and best validation loss with the Morse run; normalized losses from datasets
with different residual priors are not sufficient on their own.

## Variant A-R: reduced-capacity, regularized PaiNN

This test keeps the Variant-A priors and the same dataset split/seed, but uses
32 instead of 64 hidden channels and Adam weight decay `1e-4`. It allows up to
30 epochs, reduces the learning rate after four validation plateaus and stops
after eight epochs without improvement. The runner and report accept a valid
early-stopped training log.

```bash
AA_TOPOLOGY="$PWD/tutorials/tel22/md.gro" \
AA_TRAJECTORY="$PWD/tutorials/tel22/long_run/md_whole.trr" \
TRAINER="$PWD/training/build/train_painn" \
PYRESSO="$PWD/espresso/build/pypresso" \
VARIANT_AR_RUN_DIR="$PWD/tutorials/tel22/diagnostics/smoke/variant_ar_long_1001f" \
VARIANT_A_REUSE_DATASET_DIR="$PWD/tutorials/tel22/diagnostics/smoke/variant_a_long_1001f_15ep" \
DEVICE=auto \
bash tutorials/tel22/diagnostics/scripts/09_test_variant_ar_regularized_pipeline.sh
```

Use a fresh run directory. This experiment intentionally changes model
capacity and weight regularization together; if it improves validation, a
later one-factor ablation can determine which change is responsible. The
optional `VARIANT_A_REUSE_DATASET_DIR` avoids rebuilding the identical
Variant-A dataset; its priors are revalidated as Morse-free before training.
