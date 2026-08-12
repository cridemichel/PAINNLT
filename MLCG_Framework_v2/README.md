# MLCG Framework v2

A molecule-agnostic force-matching framework for coarse-grained molecular dynamics.
The core pipeline is intentionally independent of any particular chemistry:
residue names, atom-to-site mappings, CG site types, bonded priors, rigid-body
layouts, network size and simulation inputs are provided at runtime through
JSON/files rather than being compiled into the code.

## Pipeline

1. `preprocessing/build_cg_dataset.py`
   - maps an atomistic trajectory to configurable CG sites;
   - aggregates reference forces and torques;
   - optionally infers/subtracts analytic priors;
   - writes a binary force-matching dataset plus `cg_priors.json` and
     `rigid_bodies_info.json`.
2. `training/train_painn.cpp`
   - trains the canonical PaiNN residual potential with LibTorch;
   - configuration is supplied by JSON;
   - writes the trained model and a provenance manifest.
3. `simulation/equilibrate.py` and `simulation/run_cg_md.py`
   - reconstruct the same priors/rigid bodies in ESPResSo;
   - load the PaiNN plugin and run CG dynamics.

`tutorials/tel22/` is only a reference example. Nothing under `preprocessing/`,
`training/`, `simulation/` or `tests/` depends on that tutorial.

See `HOWTO.md` or `HOWTO_EN.md` for usage.

## ESPResSo extension

Bonded Morse priors are evaluated by the conservative analytic `MorseBond` extension in `simulation/espresso_plugin/`; run `copy_plugin_files.sh` and rebuild ESPResSo before simulations that use `type: "morse"`.
