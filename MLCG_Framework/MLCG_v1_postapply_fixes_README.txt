MLCG Framework v1 - post-application fixes
===========================================

Base archive
------------
MLCG_framework_v1(2).zip uploaded on 2026-08-04.

Apply from the repository root
------------------------------
  cd ~/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework
  git status
  git switch -c fix-ibi-postapply
  git apply --check ~/Downloads/MLCG_v1_postapply_fixes.patch
  git apply ~/Downloads/MLCG_v1_postapply_fixes.patch

Changed files
-------------
- ibi/run_ibi_loop.py
- simulation/run_cg_md.py
- simulation/equilibrate.py
- tutorials/tel22_IBI/06_run_espresso.sh
- tutorials/tel22_IBI/run_ibi_scaling.py

Corrections
-----------
- compatible histogram/table grids for DBI and iterative IBI;
- correct update_ibi_potential arguments and return values;
- raw-observable probability ratio for IBI updates;
- conservative angle-wall signs reapplied after every update;
- uniform high-precision tabulated files;
- site-aware trajectory archive with COM, virtual sites, lookup metadata and box;
- site_i/site_j/site_k/site_l-aware trajectory analysis;
- correct first site type when reading the binary dataset;
- removal of the duplicate +1-offset WCA setup;
- use of ibi_priors/cg_priors_final.json in production and scaling;
- fail-fast scaling runs and removal of stale energy.csv files.

Validation performed
--------------------
- git apply --check: PASS
- Python py_compile: PASS
- synthetic bond DBI/IBI update (299 histogram points -> 2001 table points): PASS
- ESPResSo angle-wall sign checks: PASS
- uniform/high-precision table save check: PASS
- static site-aware trajectory and tutorial path checks: PASS

After applying
--------------
Regenerate all derived artifacts from scratch:
  DBI/IBI tables -> cg_priors_final.json -> residual dataset -> PaiNN model
  -> equilibrated checkpoint -> priors-only NVE -> PaiNN+priors NVE.

Known limitation not addressed by this Python patch
----------------------------------------------------
ESPResSo's standard tabulated interaction interpolates energy and force
independently. Exact mathematical conservativity between table nodes requires
a shared interpolant in the ESPResSo core (for example Hermite interpolation)
or forces derived directly from the energy interpolant.
