# MLCG Framework v2 - generic usage

The framework implements coarse-grained force matching with analytic priors and
a residual PaiNN potential. The **core is independent of TEL22**: residue names,
atom-to-CG mappings, site types, bonded topology and rigid-body definitions are
runtime inputs.

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Preprocessing requires an MDAnalysis-compatible topology and a trajectory that
contains atomistic forces.

## 2. Mapping configuration

`preprocessing/topology_config.json` is a neutral template. Replace `MOL`,
`CG_A`, `CG_B` and atom names with those of the target system. Important keys:

- `mapping.mapping_method`: `COM`, `COG` or `ATOM`;
- `mapping.residues`: mapping by residue name;
- `mapping.site_types`: non-negative integer PaiNN species IDs;
- `bonds`, `angles`, `dihedrals`: molecular topology/priors;
- optional `rigid_bodies` definitions;
- WCA and temperature parameters.

`cg_priors.json` and `rigid_bodies_info.json` are generated artifacts and should
not be treated as source configuration files.

## 3. Dataset construction

```bash
python3 preprocessing/build_cg_dataset.py \
  --topology /path/system.tpr \
  --trajectory /path/trajectory.trr \
  --config /path/topology_config.json \
  --output work/cg_dataset.bin
```

By default metadata outputs are placed next to the dataset:

```text
work/cg_dataset.bin
work/cg_priors.json
work/rigid_bodies_info.json
```

Use `--priors-output` and `--rb-info-output` to override those paths.

## 4. Training

```bash
cd training
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . -j
```

Adapt `training/cg_model_config.json`; `num_species` must exceed the largest
site type in the dataset and model/cutoff hyperparameters are system dependent.
Then run:

```bash
training/build/train_painn work/cg_dataset.bin work/cg_model.pt /path/cg_model_config.json
```

## 5. ESPResSo

After integrating `simulation/espresso_plugin/`, pass all runtime artifacts
explicitly to `simulation/equilibrate.py` and `simulation/run_cg_md.py`.

## 6. Core tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

The Python core tests are chemistry agnostic and do not depend on TEL22.
LibTorch/ESPResSo-specific regressions require their corresponding runtimes.

## 7. TEL22 tutorial

`tutorials/tel22/` is an optional minimal reference application only. It is not
imported by or required by the core framework.
