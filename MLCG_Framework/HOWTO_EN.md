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
- `mapping`: Lists of AA atom indices (0-indexed) that make up each CG site.
- `bonds`: Pairs of CG site indices that are connected by a harmonic prior.
- `wca_sigma` / `wca_epsilon`: Global parameters (optional) for the excluded volume prior (WCA).

### 1.2 Generate the Dataset
Once configured, run the script providing your GROMACS/All-Atom trajectory and topology as input:
```bash
cd preprocessing
python3 build_cg_dataset.py --traj /path/to/traj.xtc --top /path/to/conf.gro
```
The script will do three things:
1. It will calculate the optimal spring constants ($k, r_0$) using Boltzmann Inversion on the CG bond distances observed in the trajectory and save them in `cg_priors.json`.
2. It will analytically subtract these harmonic forces (and the WCA) from the mapped CG forces, generating the residual dataset `cg_dataset.bin` in `training/`.
3. It will calculate the masses and inertia tensors for the CG sites/molecules and save them in `rigid_bodies_info.json`.

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
In the `simulation/espresso_plugin/` folder you will find the 3 files needed for integration.
Copy these files into the ESPResSo source code:

```bash
# Replace "/path/to/espresso" with your ESPResSo source directory
cp simulation/espresso_plugin/PaiNN_ML_Potential.cpp /path/to/espresso/src/core/machine_learning/
cp simulation/espresso_plugin/PaiNN_ML_Potential.hpp /path/to/espresso/src/core/machine_learning/
cp simulation/espresso_plugin/painn.pyx /path/to/espresso/src/python/espressomd/
```

Afterward, recompile ESPResSo making sure you have PyTorch enabled in your Cmake toolchain:
```bash
cd /path/to/espresso/build
make -j4
```

### 3.2 Running the Dynamics
You will find the template script `run_cg_md.py` in the `simulation/` folder.

```bash
cd simulation
/path/to/espresso/build/pypresso run_cg_md.py
```
The script will handle the following:
1. Load the masses and inertia tensors from the `rigid_bodies_info.json` file (generated in Step 1) to configure the particles in ESPResSo.
2. Load the topological priors from the `cg_priors.json` file and configure the native harmonic bonds of ESPResSo.
3. Load the WCA in ESPResSo.
4. Initialize the `best_cg_model.pt` neural network in ESPResSo via the C++ ML Potential plugin.
5. Launch the Molecular Dynamics NVE or NVT (Langevin) combining the analytical forces with the neural predictions!

### 3.3 Energy Validation (Quadratic Scaling)
To ensure that the integration of PyTorch and the Priors within ESPResSo conserves energy (symplectic NVE simulation), you can use the dedicated test script:
```bash
cd simulation
/path/to/espresso/build/pypresso verify_energy_scaling.py
```
The script will iteratively reduce the time-step `dt` and calculate the standard deviation of the total energy ($E_{kin} + E_{bonded} + E_{ML}$). Since the integration algorithm is *Velocity Verlet*, the error must scale with $O(dt^2)$, which means that halving the time-step will reduce the fluctuation exactly by a factor of $\sim 0.25$!
