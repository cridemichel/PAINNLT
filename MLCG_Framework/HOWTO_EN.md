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
cd MLCG_Framework

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
    --traj ../GROMACS/ethanol.trr \
    --topol ../GROMACS/ethanol.gro \
    --config topology_config.json \
    --output cg_dataset.bin
```

#### Example of `topology_config.json`
The configuration file controls temperatures, WCA potentials, spring bonds (priors), and mapping rules (Multi-Bead, COM, ATOM, COG):

```json
{
    "temperature": 300.0,
    "wca_sigma": 0.0,
    "wca_epsilon": 0.0,
    "bonds": [
        [0, 1]
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
    }
}
```

#### Priors and Boltzmann Inversion
If you include the `"bonds"` array as a list of indices (e.g., `[[0, 1]]`), the script will perform statistical **Boltzmann Inversion** on the trajectory distances to automatically derive the harmonic constant $k$ and the equilibrium distance $r_0$.

Alternatively, you can completely disable the priors by leaving `bonds: []`, or **explicitly** define the desired parameters by providing dictionaries. You can use **Harmonic**, **FENE** or **Morse** potentials, with native support for bonds between specific sites rather than just Centers of Mass (using the optional `site_i` and `site_j` keys).

Examples of explicit definition:
```json
"bonds": [
    {
        "mol_i": 0, "mol_j": 1,
        "site_i": 2, "site_j": 0,
        "type": "fene",
        "k": 1000.0,
        "r0": 0.2,
        "r_max": 0.3
    },
    {
        "mol_i": 1, "mol_j": 2,
        "type": "harmonic",
        "k": 5000.0,
        "r0": 0.15
    },
    {
        "mol_i": 2, "mol_j": 3,
        "type": "morse",
        "D": 50.0,
        "a": 20.0,
        "r0": 0.2
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

### 2.3 Running the Training
Start the training. By default, the program will look for `cg_dataset.bin`, `best_cg_model.pt` (for saving), and `cg_model_config.json` (for configuration).
```bash
cd training
./train_painn
```
*Note: You can pass custom paths via command-line arguments:*
`./train_painn <dataset.bin> <output_model.pt> <config.json>`

The training will save the compiled PyTorch JIT model optimized for ESPResSo.

---

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

### 3.2 Running the Dynamics
The integration of ML + Priors is elegantly handled within the framework. The Neural Network (C++ Plugin) deals **exclusively** with the complex prediction. The Priors (WCA, Harmonic, FENE, Morse) are natively added inside ESPResSo's MD engine.

To simulate, use the `run_cg_md.py` script which:
1. Instantiates the molecules and Virtual Sites.
2. Reads `cg_priors.json` and applies ESPResSo bonds to the sites/particles.
3. Activates the PaiNN C++ potential which will inject its predicted force on each site.

```bash
python run_cg_md.py \
    --model best_model.pt \
    --config best_cg_model_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset cg_dataset.bin \
    --steps 10000 \
    --dt 0.002
```

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

### 4. Energy Validation (Quadratic Scaling)
To ensure that the integration of PyTorch and the Priors within ESPResSo conserves energy (symplectic NVE simulation), you can use the dedicated test script:
```bash
cd simulation
/path/to/espresso/build/pypresso verify_energy_scaling.py
```
The script will iteratively reduce the time-step `dt` and calculate the standard deviation of the total energy ($E_{kin} + E_{bonded} + E_{ML}$). Since the integration algorithm is *Velocity Verlet*, the error must scale with $O(dt^2)$, which means that halving the time-step will reduce the fluctuation exactly by a factor of $\sim 0.25$!
