# TEL22_IBI diagnostics

This directory contains validation, certification, convergence and exploratory test evidence.
It is deliberately separated from the tutorial root, which contains the pipeline and canonical artifacts.

- `scripts/`: diagnostic/test entry points.
- `nve/`: NVE certification and convergence evidence.
- `ibi/`: IBI calibration/angle/dihedral evidence (TEL22_IBI).
- `ml/`: residual-ML benchmark/runtime validation evidence (TEL22_IBI).
- `historical/`: superseded development evidence kept for provenance.
- `morse/`, `nonbonded/`, `smoke/`: focused diagnostics.

GROMACS-generated trajectory/topology/working files are intentionally kept at the tutorial root and are never deleted by the layout migrator.
