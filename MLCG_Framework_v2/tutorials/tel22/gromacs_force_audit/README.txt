GROMACS force audit outputs
===========================

Primary report:
  gromacs_force_audit.json

Per-frame numerical diagnostics:
  raw_vs_whole_frames.csv
  stored_vs_rerun_frames.csv

Native GROMACS diagnostics:
  md_tpr.dump
  md_from_tpr.mdp
  gmx_check_md_trr.txt
  gmx_check_md_whole_trr.txt
  gmx_check_stored_subset.txt
  gmx_check_rerun.txt
  trjconv_subset.log
  mdrun_rerun.log

Interpretation:
- raw -> whole should preserve forces to numerical precision.
- stored -> rerun compares original TRR forces with forces recomputed from the
  exact saved coordinates using md.tpr. The default relative-RMS acceptance
  tolerance is 1e-3; override with GMX_AUDIT_RERUN_REL_TOL.
- nstxout != nstfout is reported as WARN, but the decisive structural check is
  that every TRR frame actually consumed contains both positions and forces.
