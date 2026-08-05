# MLCG Framework - User Manual

Welcome to the **Machine Learning Coarse-Graining (MLCG) Framework**!
This pipeline is designed to extract Coarse-Grained (CG) interactions from All-Atom (AA) molecular trajectories using *Direct Boltzmann Inversion* to calculate thermodynamic priors, and Graph Neural Networks (PaiNN) in C++ (PyTorch) to fit the residual forces.

The structure is divided into three main phases, each contained in its respective folder:

1. **Preprocessing:** Dataset construction and physical prior subtraction.
2. **Training:** C++ Neural Network training.
3. **Simulation:** Coarse-Grained MD execution in ESPResSo.

---

## Phase 0: Environment Setup (Self-Contained)

To ensure maximum reproducibility, it is highly recommended to create a dedicated Python virtual environment for this framework and install the necessary packages, such as `MDAnalysis` and `numpy`.

```bash
# Enter the framework folder
cd MLCG_Framework_v2

# Create a virtual environment named "mlcg_venv"
python3 -m venv mlcg_venv

# Activate the virtual environment
source mlcg_venv/bin/activate

# Install the necessary packages
pip install -r requirements.txt
```
*Note: remember to activate the virtual environment (`source mlcg_venv/bin/activate`) every time you open a new terminal before running the Python scripts of the framework.*

---

## Phase 1: Preprocessing and Boltzmann Inversion

All the code required to generate the training dataset is located in `preprocessing/`.

### 1.1 Configure the Topology
Before running the script, you need to define how the All-Atom atoms are mapped onto the Coarse-Grained sites and which CG sites are covalently bonded.
Open the `preprocessing/topology_config.json` file and define:
- `temperature`: The temperature (in Kelvin) of your All-Atom MD (needed to calculate $k_B T$).
- `mapping`: Lists of AA atom indices (0-indexed) or atom names that make up each CG site.
- `bonds`: Index pairs or FENE/Morse configurations for prior insertion.

This script supports flexible mapping based on JSON files. You can generate `.bin` files containing positions, forces, and torques on individual sites or centers of mass.

Execution example:
```bash
python build_cg_dataset.py \
    --trajectory ../GROMACS/ethanol.trr \
    --topology ../GROMACS/ethanol.gro \
    --config topology_config.json \
    --dbi \
    --output cg_dataset.bin
```

**Options supported by `build_cg_dataset.py`:**
- `-c`, `--topology`: Topology file (e.g., `.tpr` or `.gro`).
- `-f`, `--trajectory`: Trajectory file (e.g., `.trr` or `.xtc`).
- `-j`, `--config`: JSON file with CG topology and mapping rules (default: `topology_config.json`).
- `--dbi`: Enable global Direct Boltzmann Inversion (DBI) to extract bond force constants directly from thermal distributions.
- `-p`, `--priors`: JSON file with priors (e.g., `cg_priors.json`). If provided, the script skips the statistical calculation and applies the given priors directly to the dataset. Necessary during IBI.
- `-o`, `--output`: Output binary file name (default: `../training/cg_dataset.bin`).

#### Example of `topology_config.json`
The configuration file controls temperatures, WCA potentials, spring bonds (priors), and mapping rules (Multi-Bead, COM, ATOM, COG):

```json
{
    "temperature": 300.0,
    "wca_sigma": 0.0,
    "wca_epsilon": 0.0,
    "wca_overrides": [
        {"type_i": 2, "type_j": 2, "sigma": 0.8, "epsilon": 2.5}
    ],
    "bonds": [
        [0, 1]
    ],
    "angles": [
        {"mol_i": 0, "mol_j": 1, "mol_k": 2, "site_i": 0, "site_j": 0, "site_k": 0, "theta0": "auto", "k": "auto"}
    ],
    "mapping": {
        "mapping_method": "COM",
        "residues": {
            "ETH": {
                "CG_CH3": ["C1", "H1", "H2", "H3"],
                "CG_CH2": ["C2", "H4", "H5"],
                "CG_OH":  ["O1", "H6"]
            }
        },
        "site_types": {
            "CG_CH3": 0, "CG_CH2": 1, "CG_OH": 2
        }
    },
    "rigid_bodies": {
        "ETH": {
            "auto_align_sites": true,
            "sites": {
                "CG_CH3": {"type": 0, "relative_pos_nm": [0.0, 0.0, 0.0]},
                "CG_CH2": {"type": 1, "relative_pos_nm": [0.15, 0.0, 0.0]},
                "CG_OH":  {"type": 2, "relative_pos_nm": [0.25, 0.1, 0.0]}
            }
        }
    }
}
```

> [!IMPORTANT]
> **Rigid Bodies and Kabsch Auto-Alignment**
> If you define `"rigid_bodies"` in your topology, the script will map those molecules into ESPResSo Rigid Bodies (composed of a central particle with real mass/inertia and virtual sites).
> 
> By default (`"auto_align_sites": true`), the `build_cg_dataset.py` script computes the ideal "average" geometry of these sites by extracting all snapshots from the GROMACS trajectory and aligning them with the **Kabsch algorithm**. The resulting geometries are saved in `rigid_bodies_info.json`.
> During dataset generation (Pass 2), the script uses this exact Kabsch rotation to mathematically reconstruct the ideal rigid sites on every frame before evaluating the prior forces (WCA/Harmonic). This guarantees **strict physical consistency** with ESPResSo (which does not deform rigid bodies), preventing the ML model from learning massive, unphysical counter-forces caused by instantaneous thermal vibrations.
> 
> If you prefer to manually provide the perfect ideal coordinates (e.g. from a PDB) and don't want the script to overwrite them with the trajectory average, set `"auto_align_sites": false`. The script will exactly use the `relative_pos_nm` you typed in the JSON!

> [!WARNING]
> **Regeneration required after this patch**
> Legacy `rigid_bodies_info.json` files do not declare the principal-axis frame, and legacy checkpoints contain no provenance. Regenerate the dataset/rigid-body metadata, model/manifest and checkpoint in that order. Legacy overrides are for controlled diagnostics, not NVE certification.

#### 1.2 Thermodynamic architecture (analytical priors + residual ML)

The v2 framework uses an explicit Hamiltonian decomposition:

```text
U_tot = U_priors + U_PaiNN
```

Prior forces are subtracted from the targets during preprocessing and the same priors are recreated in ESPResSo. The PaiNN plugin applies the exact gradient of the network energy, without hidden energy or force clipping. `toxvaerd_alpha` is part of the architecture and must match between training and runtime.

#### Rigid bodies and principal frame

`rigid_bodies_info.json` stores principal moments and virtual-site coordinates in the same principal-axis frame. At startup the COM quaternion is reconstructed by aligning the body-frame template to the initial configuration. Numerical virtual-site masses are `1e-5`; physical mass and inertia remain on the COM.

#### WCA and PBC

WCA mixing uses Lorentz-Berthelot. When `wca_sigma` is `"auto"`, minimum distances use the minimum-image convention, including contacts across periodic faces.

#### PaiNN cutoff

The radial basis uses the Toxvaerd cutoff implemented in `PaiNN_Architecture.hpp`. There are no runtime `use_bias` or `apply_envelope` switches: training, parity and plugin share the same header and parametrization.

#### Advanced Physics: Site-Dependent Priors
By default, Harmonic bonds, Angles, and Dihedrals act on the Centers of Mass. However, you can map them to specific Virtual Sites using the `site_i`, `site_j`, `site_k`, `site_l` parameters (0-indexed referring to the molecule's mapping definition).
When applied to Virtual Sites, the forces are geometrically exact, and the framework automatically computes the **torque** $\tau = \vec{r}_{site} \times \vec{F}_{site}$ to transfer the rotational momentum back to the main Center of Mass.



> [!TIP]
> **Equilibrium Distance Auto-calculation (`r0`)**
> For explicit bonds (FENE, Morse, etc.), if you omit the numerical parameter and set `"r0": "auto"`, the script will automatically extract the exact mean distance for that pair of atoms directly from the molecular trajectory! This prevents thermodynamic explosions and elegantly resolves scale mismatches between all-atom and coarse-grained representations.

> [!NOTE]
> **Morse and tabulated bonds in NVE**
> Explicit Morse bonds are represented as `TabulatedDistance`. Production applies no automatic force cap. Because separately interpolated energy and force tables are not accepted as a conservative NVE certificate, `run_cg_md.py --nve` rejects Morse/tabulated priors by default; the override is diagnostic only.

Examples of explicit definition:
```json
"bonds": [
    {
        "mol_i": 0, "mol_j": 1,
        "site_i": 2, "site_j": 0,
        "type": "fene",
        "k": 1000.0,
        "r0": "auto",
        "r_max": 0.3
    },
    {
        "mol_i": 2, "mol_j": 3,
        "type": "morse",
        "D": 20.0,
        "a": 3.0,
        "r0": "auto"
    }
]
```

The script will calculate the resulting forces (and their related **torques**) and subtract them from the target values before saving the binary dataset, exporting the exact parameters to the `cg_priors.json` file.
It will also analytically subtract these harmonic/FENE/Morse forces (and WCA) from the mapped CG forces, generating the residual dataset `cg_dataset.bin` in `training/`.
Finally, it will calculate the masses and inertia tensors for the CG sites/molecules and save them in `rigid_bodies_info.json`.

---

## Phase 2: Neural Network Training (C++)

Once the `cg_dataset.bin` dataset has been generated, the training on the *residuals* takes place entirely in C++.

### 2.1 Compilation
Make sure you have downloaded LibTorch and compiled the binary:
```bash
cd training
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j4
```

### 2.2 Configure Network Parameters
In the `training/` folder you will find the `cg_model_config.json` file. This file acts as the **control hub** for the network: before starting the training, you can modify parameters such as `hidden_channels`, `n_layers`, `cutoff`, `learning_rate`, and `epochs` here. The C++ code will read this file at runtime without any need to recompile!

> [!TIP]
> **Model manifest**
> `train_painn` writes `<model>.manifest.json` with the effective architecture, split and input sizes. To add SHA-256 values for model, dataset and config, run `python3 training/create_model_manifest.py --model MODEL.pt --config CONFIG.json --dataset DATASET.bin`. Equilibration, production and parity validate the manifest before loading weights.

> [!TIP]
> **Lipschitz Regularization**
> In the `.json` file you can add or modify the `"lipschitz_lambda": 0.001` parameter. This introduces an L2 penalty on the force magnitude during training (inspired by CGnet). By enabling it, the model learns to predict smoother energy surfaces, preventing massive gradients and explosions during ESPResSo simulations. If set to `0.0`, the additional overhead is completely bypassed ensuring absolute backward compatibility.

### 2.3 Running the Training
Start the training. By default, the program will look for `cg_dataset.bin`, `best_cg_model.pt` (for saving), and `cg_model_config.json` (for configuration).
```bash
cd training
./train_painn
```
*Note: You can pass custom paths via command-line arguments:*
`./train_painn <dataset.bin> <output_model.pt> <config.json> [--resume]`

Training saves a LibTorch archive of the PaiNN weights, loaded by the same C++ architecture used in the ESPResSo plugin.

The trainer never resumes an existing output implicitly: choose a new path or remove the old model for a clean run. `--resume` is explicit and requires a compatible manifest; rerun `create_model_manifest.py` after every training run to refresh hashes.

---

#### Using the Morse Potential for Stacking Interactions (TEL22 Example)
Graph Neural Networks sometimes struggle to natively model non-linear and "fragile" long-range forces like Guanine tetrad stacking or intra-chain Van der Waals forces, especially with limited training data. An elegant and fast way to solve this is to introduce an explicit **Morse Potential** as a prior.

The Morse potential perfectly models the energetic "well" of biological stacking:
- It provides deep stability at equilibrium (regulated by the parameter `D`, well depth).
- It allows the interaction to smoothly "break" at larger distances (regulated by the parameter `a` or $\alpha$, well width), unlike harmonic bonds which would generate infinite forces and physically prevent phenomena like thermal melting or unfolding.

**Use Case (TEL22 Tutorial):**
In the case of G-Quadruplexes (TEL22), planar stacking between guanines is essential for structural compactness. Rather than forcing the Machine Learning Model to learn this complex force entirely from scratch, we explicitly inject "scaffold" Morse bonds between stacked guanines:
```json
{
    "mol_i": 2, "mol_j": 8,
    "type": "morse",
    "D": 50.0,
    "a": 3.0,
    "r0": "auto"
}
```
In this setup:
- `D` at `50.0` kJ/mol ensures the structure remains stably folded at physiological temperatures (300K). Lower values (e.g., `20.0`) would facilitate visible thermal unfolding.
- `"r0": "auto"` allows the framework to read the exact stacking distance directly from the atomistic trajectory (preventing thermodynamic explosions caused by a manually entered `r0` that misaligns with CG dimensions).
- Morse is implemented as a tabulated bond and receives no automatic production force cap. `run_cg_md.py --nve` rejects it by default; use the override only for diagnostics, not for an energy-conservation certificate.

## Phase 3: Integration and Simulation in ESPResSo

The final phase is the utilization of the model and priors for MD production. To do this, you must first install the PaiNN C++ plugin within the ESPResSo source code.

### 3.1 Installing the Plugin in ESPResSo
In the `simulation/espresso_plugin/` folder you will find the 3 files needed for integration (and a symlink to the architecture file).
Copy these files into the ESPResSo source code:

```bash
# Replace "/path/to/espresso" with your ESPResSo source directory
cp simulation/espresso_plugin/PaiNN_Architecture.hpp /path/to/espresso/src/core/nonbonded_interactions/
cp simulation/espresso_plugin/PaiNN_ML_Potential.cpp /path/to/espresso/src/core/nonbonded_interactions/
cp simulation/espresso_plugin/PaiNN_ML_Potential.hpp /path/to/espresso/src/core/nonbonded_interactions/
cp simulation/espresso_plugin/painn.pyx /path/to/espresso/src/python/espressomd/
```

Afterward, recompile ESPResSo making sure you have PyTorch linked in your Cmake toolchain:
```bash
cd /path/to/espresso/build
make -j4
```

### 3.2 Equilibration
Before starting the production simulation, it is essential to relax the system to remove any steric clashes (overlaps between atoms/beads) originating from the initial topology, especially when using hard repulsive potentials.

For equilibration, use the `equilibrate.py` script. The script performs a multi-phase procedure:
1. **Classical steepest descent** with WCA and analytic priors.
2. **Capped classical NVT** Langevin warm-up.
3. **Capped ML NVT** after activating PaiNN.
4. **Uncapped ML NVT** under exactly the production Hamiltonian; this final phase defines the saved checkpoint.

```bash
python equilibrate.py \
    --model best_cg_model.pt \
    --config best_cg_model_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset cg_dataset.bin \
    --out_checkpoint equilibrated.npz \
    --dt 0.002 \
    --kT 2.49 \
    --steps_sd 5000 \
    --steps_md 2000 \
    --device auto
```
**Options supported by `equilibrate.py`:**
- `--model`, `--config`, `--priors`, `--rb_info`, `--dataset`: Required input files.
- `--out_checkpoint`: Versioned checkpoint containing dynamic state, box, particle identity and SHA-256 provenance for all inputs.
- `--dt`: Time-step for the MD phase (default: 0.002 ps).
- `--kT`: Temperature in kJ/mol (default: 2.49 for 300K).
- `--steps_sd`: Number of steps for Phase 1 Steepest Descent (default: 5000).
- `--steps_md`: Number of capped classical NVT steps.
- `--steps_ml_capped`: Number of capped PaiNN NVT steps.
- `--steps_ml_uncapped`: Final uncapped NVT steps used to define the production checkpoint.
- `--warmup_chunk`: Progress-reporting interval.
- `--allow_missing_model_manifest`: Explicit legacy-model override.
- `--allow_unsafe_mpi`: Enables uncertified MPI experiments only.
- `--device`: PyTorch device (`cpu`, `cuda`, `mps`, `auto`).

### 3.3 Running the Production Dynamics
The integration of ML + Priors is elegantly handled within the framework. The Neural Network (C++ Plugin) deals **exclusively** with the complex prediction. The Priors (WCA, Harmonic, FENE, Morse) are natively added inside ESPResSo's MD engine.

To simulate, use the `run_cg_md.py` script which will load the equilibrated coordinates from the checkpoint and start the production:

```bash
python run_cg_md.py \
    --model best_cg_model.pt \
    --config best_cg_model_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset cg_dataset.bin \
    --checkpoint equilibrated.npz \
    --steps 10000 \
    --dt 0.002 \
    --kT 2.49 \
    --device auto
```

**Options supported by `run_cg_md.py`:**
- `--model`, `--config`, `--priors`, `--rb_info`, `--dataset`: Required input files.
- `--checkpoint`: `.npz` file produced by `equilibrate.py` containing starting coordinates and velocities. If omitted, it will start from the dataset's frame 0 coordinates.
- `--steps`: Number of simulation steps (default: 10000).
- `--dt`: Time-step in picoseconds (default: 0.002 ps).
- `--kT`: Temperature in kJ/mol (default: 2.49).
- `--device`: PyTorch device.
- `--nve`: Runs the simulation in the NVE ensemble (no thermostat).
- `--allow_missing_model_manifest`: explicit override for legacy models without a manifest.
- `--allow_legacy_checkpoint` / `--allow_checkpoint_mismatch`: explicit overrides for legacy or inconsistent checkpoints.
- `--allow_unsafe_mpi`: enables experimental multi-rank tests only; PaiNN MPI is not certified.
- `--allow_nonconservative_tables`: allows Morse/tables in NVE for diagnostics only.

> [!TIP]
> **Checkpoint provenance**
> Patched checkpoints contain SHA-256 values for dataset, model, configuration, priors and rigid-body information, plus box and particle identity. A mismatch stops the run before integration unless explicitly overridden.

### 3.4 Rigid Body Dynamics and Particle Filtering
In the framework, simulating multi-site molecules (Multi-Bead) leverages ESPResSo's **Virtual Sites**:
1. **The Real Particle (Center of Mass)**: For each rigid molecule, ESPResSo requires a single central particle endowed with mass and an inertia tensor. This is the only particle that the integrator physically moves in space.
2. **The Virtual Sites (The CG Beads)**: The interaction sites (e.g., `CH3`, `OH`) are instantiated as massless particles, whose position is rigidly anchored to the real particle. Any force applied to a virtual site is automatically translated by ESPResSo into a net force and **torque** on the real particle.

**The Neural Network Problem:**
The PaiNN neural network is trained *exclusively* on sites (e.g., types `0, 1, 2`). It knows nothing about a "Center of Mass". If we passed the Center of Mass to the ML Model, the latter would crash or produce random noise trying to interpret an unknown chemical species.

**The Solution (The `num_species` Filter):**
The C++ plugin (`PaiNN_ML_Potential.cpp`) includes an elegant filter: it accepts the `num_species` parameter (the total number of types known to the neural network). 
During the ESPResSo simulation:
- Virtual sites are assigned standard types (e.g., `0`, `1`, `2`). The Neural Network sees them, calculates distances, and predicts forces.
- The "Real" particle (the Center of Mass) is intentionally assigned a "ghost" type, which is an ID greater than or equal to `num_species` (e.g., `type = 100`).
- The C++ plugin **actively ignores** all particles with `type >= num_species`. 

The Center of Mass thus becomes "invisible" to Machine Learning, but remains perfectly active for ESPResSo's mechanics!
If you wish to make the Center of Mass particle interact:
- **Classically (e.g., WCA between molecules)**: Just define the interaction in ESPResSo for `type 100`.
- **With the Neural Network**: Simply include the Center of Mass as an explicit site in the `topology_config.json` during preprocessing, train the model including it (it will have its own `type` like `3`), and assign it that type in the simulation.

> [!NOTE]
> **"Real" vs "Virtual" Particles in ESPResSo**
> ESPResSo does not use the `type` to determine if a particle is Real or Virtual. The `type` is just a label for the Neural Network or Lennard-Jones parameters. The true nature of the particle is decided by the `virtual=True` or `virtual=False` flag during its creation in Python.
> 
> * Example: `system.part.add(pos=..., type=100, virtual=False)` creates the real particle. Newton will move it, but the neural network (stopping at types < 100) will ignore it.
> * Example: `system.part.add(pos=..., type=0, virtual=True)` creates a ghost site attached to the rigid body. The Neural Network will see it, apply force to it, and ESPResSo will leverage it, transferring force and torque to the real rigid body it is bound to.

### 3.5 Conservativity and C++ plugin limits

The plugin applies no hidden clipping: PaiNN forces are the gradient of the same reported energy. PyTorch checkpoint loading is fail-fast. The validated path is single-rank; scripts reject multi-rank PaiNN runs by default because many-body halo communication and MPI energy accounting still require a dedicated 1/2/4-rank parity test.

### 4. Energy Validation (Quadratic Scaling)
To ensure that the integration of PyTorch and the Priors within ESPResSo conserves energy (symplectic NVE simulation), you can use the dedicated test script:
```bash
cd simulation
/path/to/espresso/build/pypresso verify_energy_scaling.py
```
The tutorial script preserves every energy series, removes linear drift, estimates autocorrelation, applies a moving-block bootstrap and reports the slope, confidence interval, $R^2$, drift and successive timestep ratios. Velocity-Verlet should give a slope close to 2 and approximately four times smaller deviations when `dt` is halved.
