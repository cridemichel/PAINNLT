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
   - optionally infers/subtracts analytic or tabulated priors;
   - writes a binary force-matching dataset plus `cg_priors.json` and
     `rigid_bodies_info.json`.
2. `ibi/build_dbi_priors.py` and `ibi/run_ibi_loop.py` (optional)
   - build site-addressable bonded distance/angle/dihedral DBI tables;
   - iterate selected bonded groups in priors-only NVT simulations;
   - write self-contained final tabulated priors;
   - optionally convert bond/angle/dihedral IBI priors to the single-source conservative
     Hermite representation;
   - for periodic dihedral IBI, optionally run the IBI loop directly with
     `ConservativeSplineDihedral`, so the sampled and promoted Hamiltonians use
     the same energy/force representation;
   - optionally regularize **conservative angular IBI** by smoothing only the
     de-walled angle-potential body, then re-adding the unchanged endpoint wall
     and exporting C2 nodal derivatives.
   Regularization candidates are always unvalidated artifacts: they must pass
   matched structural and NVE timestep-scaling validation before promotion.
   After any final prior change, rebuild the force-matching dataset before
   training or re-enabling a residual ML model.
3. `training/train_painn.cpp`
   - trains the canonical PaiNN residual potential with LibTorch;
   - configuration is supplied by JSON;
   - writes the trained model and a provenance manifest.
4. `simulation/equilibrate.py` and `simulation/run_cg_md.py`
   - reconstruct the same priors/rigid bodies in ESPResSo;
   - load the PaiNN plugin and run CG dynamics.

`tutorials/tel22/` is only a reference example. Nothing under `preprocessing/`,
`training/`, `simulation/` or `tests/` depends on that tutorial.

See `HOWTO.md` or `HOWTO_EN.md` for usage. The TEL22 IBI tutorial documents
the validated regularized-angle path (`smooth_0p0075`) as a system-specific
example; the smoothing bandwidth is **not** a universal default.

## Model-dependent configuration policy

The framework distinguishes three classes of workflow information:

- **CORE_INVARIANT** — a universal physical/methodological rule, implemented and
  tested in generic code (for example `F = -grad(U)`, periodic dihedral splines,
  runtime/preprocessing parity, or the IBI update formula);
- **MODEL_PARAMETER** — a choice that can change with molecule, mapping, dataset,
  temperature, Hamiltonian or sampling protocol and therefore must come from an
  external model configuration;
- **CALIBRATED_PARAMETER** — a model parameter selected by a diagnostic or
  validation study; it is stored in model configuration together with provenance.

For the TEL22 IBI workflow, the external configuration is
`tutorials/tel22_IBI/model_dependent_workflow_config.json`. Steps 11–39 that
contain model-dependent choices load this file through `model_config.sh`; step 20
is only the model-independent ESPResSo kernel installer. The config includes,
among other things, IBI grouping/mixing/sampling, regularization sweeps, replica
counts, timestep grids and certification thresholds. Values such as the TEL22
angle smoothing width `0.0075 rad`, a dihedral replica count, or an accepted NVE
window are not framework defaults.

Validate a workflow config before use:

```bash
python3 simulation/model_dependent_config.py validate \
  --config tutorials/tel22_IBI/model_dependent_workflow_config.json
```

To use a different model configuration with a tutorial wrapper:

```bash
IBI_MODEL_DEPENDENT_CONFIG=/path/to/my_model_workflow_config.json \
  bash tutorials/tel22_IBI/38_test_conservative_in_loop_dihedral_ibi.sh --run
```

Explicit environment overrides remain supported and are recorded as overrides in
a model-config provenance sidecar (`model_config_provenance*.json`). Explicit `ibi_settings.json` files are
authoritative: the production workflow does not silently fill missing
model-dependent fields from internal defaults.

## Tests

Framework tests are intentionally scoped to `tests/`. The repository can also
contain a bundled ESPResSo source tree with its own upstream test suite and build
dependencies; those tests are not part of the MLCG framework test run.

Install the test dependency and run the framework suite from the repository root:

```bash
python3 -m pip install -r requirements-test.txt
python3 -m pytest -q
```

`pytest.ini` sets `testpaths = tests`, so the no-argument command above is
equivalent to `python3 -m pytest -q tests` and does not recursively collect
`espresso/testsuite`, `espresso/samples`, or other bundled ESPResSo tests.

## ESPResSo extension

Bonded Morse priors are evaluated by the conservative analytic `MorseBond` extension in `simulation/espresso_plugin/`; run `copy_plugin_files.sh` and rebuild ESPResSo before simulations that use `type: "morse"`.
