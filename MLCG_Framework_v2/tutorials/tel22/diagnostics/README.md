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
