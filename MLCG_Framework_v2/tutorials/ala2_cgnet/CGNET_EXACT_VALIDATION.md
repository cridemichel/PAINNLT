# Ala2 CGnet-exact validation and TEL22 transfer gate

This document records the controlled Ala2 result obtained before transferring
the ordered-geometry architecture to TEL22. It distinguishes force-field
implementation parity from finite-sampling and integrator effects.

## Controlled training result

Both implementations used frames 0--7999 for training and frames 8000--9999
for validation. Geometry statistics and harmonic bond/angle priors were fitted
on the training split only. The dense architecture was 17 invariant features,
five hidden `tanh` layers of width 160 and one scalar energy output. Both used
Adam at 0.003, a factor-0.3 epoch schedule, batch size 512, Xavier weights and
a strength-4 spectral projection.

| implementation | validation MSE | validation MAE | explained residual variance |
| --- | ---: | ---: | ---: |
| official CGnet | 377.6794 kcal2/(mol2 Angstrom2) | 14.9851 kcal/(mol Angstrom) | 0.8605% |
| framework CGnet-exact head | 377.929 (converted) | 14.9920 (converted) | 0.7949% |

The framework differs by about 0.066% in MSE and 0.046% in MAE. This is the
primary implementation-parity result: residual-force skill is weak for both
models on the public 10,000-frame subset, but the framework reproduces the
official dense CGnet optimization to substantially better than 0.1%.

## Matched short FES screen

The screen used four replicas with 801 retained frames each and a 48 by 48
histogram. The atomistic reference contained 10,000 public frames.

| model | FES MSE (kBT2) | FES RMSE (kBT) | JS divergence (nats) | reference mass covered |
| --- | ---: | ---: | ---: | ---: |
| harmonic prior, ESPResSo | 1.5595 | 1.2488 | 0.3425 | 78.73% |
| prior + framework CGnet-exact head | 1.3810 | 1.1752 | 0.3000 | 86.27% |
| prior + official CGnet | 1.3549 | 1.1640 | 0.2760 | 85.76% |

The framework head improves JS divergence over its prior by 0.04255 nats
(12.4%). Its paired-replica bootstrap interval is [0.03507, 0.04604] nats.
Official CGnet is better than the framework head by 0.02398 nats in aggregate
JS and 0.02605 (kBT)2 in FES MSE. The corresponding paired JS interval is
[0.00554, 0.03593] nats at this sampling length.

That cross-engine difference is not a pure architecture measurement. The
framework trajectories use inertial Langevin dynamics in ESPResSo, while the
official implementation uses overdamped Brownian dynamics. The official
prior-only/CGnet branches control coordinates, integrator, step, length and
Brownian noise internally; the framework/official comparison does not share
an integrator. The 3,204 sampled frames are also sparse at 48-bin resolution.
Consequently, the force parity is the implementation test and the short FES is
a thermodynamic screen, not a claim that either short trajectory has converged
to the published million-frame/million-step result.

## Report schema correction

The `cgnet_ordered_geometry_tanh_v1` model has
`painn_branch_enabled=false`. Schema-v1 reports nevertheless called every
learned branch `prior_plus_painn`; this was only a historical filename/key and
was scientifically misleading for the isolated head.

Schema v2 records `model_identity`, uses `prior_plus_model`, model-neutral
comparison keys and model-neutral verdicts. The analyzer infers the label from
the training report, while `--model-key` and `--model-label` permit an explicit
identity for future diagnostics. The official comparator accepts old
`prior_plus_painn_samples.npz` trajectories as read-only legacy input, but new
runs write `prior_plus_model_samples.npz`.

Existing schema-v1 JSON and trajectories remain valid evidence and must not be
renamed or edited. Rerunning a simulation is not required solely to adopt the
new reporting vocabulary.

## Gate for TEL22

The Ala2 result supports transferring the representation, not the numerical
energy scale or a claim of expected TEL22 accuracy. The TEL22 implementation
will therefore be a separate patch and diagnostic with these constraints:

1. one ordered local feature vector per 22-bead TEL22 copy;
2. one set of head weights shared across all ten copies;
3. total learned energy equal to the sum of the ten per-copy energies, so
   forces remain conservative;
4. train-split-only feature normalization and explicit manifest provenance;
5. a TEL22-native energy scale in kJ/mol, never Ala2's 4.184 conversion factor;
6. controlled head-only, PaiNN-only and hybrid ablations on the identical
   frames, targets, priors and split;
7. topology-aware features derived from the validated antiparallel 143D
   mapping, without silently reintroducing Morse contacts as learned features.

Before implementation, the exact TEL22 feature contract and copy-to-particle
mapping must be frozen in a machine-validated configuration. The transfer is
therefore intentionally not included in the report-label correction patch.
