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
