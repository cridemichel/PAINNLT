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
    }
}
```

#### Advanced Physics: Virtual Sites, Mass Scaling, and WCA Mixing
The framework introduces a rigid-body structure to accurately map complex molecules.
- **Mass Scaling for Virtual Sites**: The primary Center of Mass (COM) retains the true mass and inertia of the rigid body. Virtual sites have their mass and inertia artificially scaled by $10^{-5}$ to prevent them from absorbing kinetic energy from the ESPResSo Langevin thermostat, preserving exact thermodynamic temperature.
- **Lorentz-Berthelot WCA**: WCA interactions between distinct sites are blended using arithmetic mean for $\sigma_{ij} = (\sigma_i + \sigma_j)/2$ and geometric mean for $\epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}$.
- **WCA Overrides**: You can define specific LJ properties for peripheral sites (e.g. bulky bases like Guanine) using the `wca_overrides` array.

#### Advanced Physics: Site-Dependent Priors
By default, Harmonic bonds, Angles, and Dihedrals act on the Centers of Mass. However, you can map them to specific Virtual Sites using the `site_i`, `site_j`, `site_k`, `site_l` parameters (0-indexed referring to the molecule's mapping definition).
When applied to Virtual Sites, the forces are geometrically exact, and the framework automatically computes the **torque** $\tau = \vec{r}_{site} \times \vec{F}_{site}$ to transfer the rotational momentum back to the main Center of Mass.

#### Priors and Boltzmann Inversion (DBI vs IBI)

The framework supports two fundamental philosophies to extract prior energies from the All-Atom trajectory: **Direct Boltzmann Inversion (DBI)** and **Iterative Boltzmann Inversion (IBI)**.
*(Note on Jacobians: For exact analytical probability matching, the DBI phase corrects for the Phase Space Volume by dividing the raw histogram by the mathematical Jacobian: $1/r^2$ for bonds, and $1/\sin(\theta)$ for angles, preventing geometric entropy bias).*

**1. Analytical Functions (DBI, FENE, Morse, Angles, Dihedrals)**
If you include the `"bonds"` array as index lists (e.g., `[[0, 1]]`), the preprocessing script will perform basic statistics (classical DBI) to derive the harmonic constant $k$ and the equilibrium distance $r_0$.
Alternatively, you can disable automatic inference and explicitly define much more complex analytical parameters for various degrees of freedom. You can use:
- **Harmonic Bond** (`"type": "harmonic"`): the classic Hooke's spring.
- **FENE Bond** (`"type": "fene"`): very useful for polymer chains where monomers must not drift beyond a certain $R_{max}$.
- **Morse Bond** (`"type": "morse"`): essential for non-linear bonds that must be able to break (like tetrad stacking or hydrogen bonds).
- **Harmonic Angles** (in the `"angles"` array): to stabilize the angle between three sites.
- **Dihedrals** (in the `"dihedrals"` array): to stabilize the torsional conformation between four sites.
This parametric approach is ultra-fast to evaluate but relies on ideal closed equations.

**2. Aggregated Statistics (Typed Topology)**
If you want multiple bonds (or angles, or dihedrals) to share the **exact same statistics**, you can group them by assigning a `"name"` attribute.
- *Without name (Bond-by-Bond)*: Each bond receives a $k, r_0$ or an IBI curve calculated exclusively using the frames of its specific atomic pair. Great for unique and exact geometries (e.g. G-Quadruplexes).
- *With name (Aggregated)*: All bonds with the same `"name"` merge their trajectories into a single large data pool. The framework will extract a global mean/variance (for "auto" springs) or a global IBI curve. Perfect for transferable models or solvents (e.g. assigning `"name": "water_OH"` to all water OH bonds).

```json
"bonds": [
    {"mol_i": 0, "mol_j": 1, "type": "ibi", "name": "PO_bond"},
    {"mol_i": 1, "mol_j": 2, "type": "ibi", "name": "PO_bond"}
]
```

**2. Iterative Boltzmann Inversion (IBI) [Exact Tabulated Curves]**
If your system is highly anharmonic or suffers from cross-interferences (e.g., steric repulsion modifies bond distances), the harmonic approximation of DBI is not sufficient. In this case, you can use the powerful integrated IBI pipeline with the real ESPResSo engine:
- Use the script in the `ibi/` folder to mathematically extract the exact potentials. The `run_ibi_loop.py` script natively reads the `_dataset.bin` file and performs **real Molecular Dynamics simulations in ESPResSo**, calculating the Kullback-Leibler divergence and correcting the curves (splines) iteratively via the Henderson equation until the simulated distribution perfectly matches the All-Atom target.
- The user has total control over the inversion types via the command line. For example, you can request IBI only for bonds, leaving DBI (more stable and efficient) for angles and dihedrals:
```bash
uv run ibi/run_ibi_loop.py \
    --dataset preprocessing/tel22_dataset.bin \
    --priors preprocessing/cg_priors.json \
    --iterations 5
```
- Once convergence is achieved, the optimal curves are saved as `.dat` files.
- Next, use `build_cg_dataset.py` again by passing the `--priors` flag to create the final dataset. This way the script will know not to recalculate the statistical priors, but will directly read the exact IBI tables and subtract them to extract the true residuals:
```bash
uv run preprocessing/build_cg_dataset.py \
    --topology md.gro \
    --trajectory md_whole.trr \
    --config topology_config.json \
    --priors cg_priors.json \
    --output dataset_ibi.bin
```
- To simulate, in your automatically updated `cg_priors.json` file, the setting will be converted to `"type": "tabulated"`, indicating the path to the generated spline:
```json
{
    "mol_i": 0, "mol_j": 1,
    "type": "tabulated",
    "file": "ibi_priors/bond_ibi_spline_0.dat",
    "min": 0.01, "max": 3.0
}
```
ESPResSo will read the numerical table (both for `TabulatedDistance` bonds, `TabulatedAngle` angles, and dihedrals) injecting the perfect IBI potential. This choice guarantees native backward compatibility and allows freely mixing DBI springs and IBI tables for different degrees of freedom!

### Practical Guide to the IBI Workflow (The Three Scripts)
The architecture logically separates the statistics extraction from the force subtraction. In the tutorials (e.g., `tel22_IBI`), you will find this workflow divided into 3 scripts:

1. **`01_build_dataset.sh` (Statistics Extraction):**
   Runs `build_cg_dataset.py` on the topology that contains the priors with `"type": "ibi"`. At this stage, the script does *not* subtract the IBI forces from the target forces (because the tables don't exist yet!). It only saves a `tel22_dataset.bin` file containing the fragment distributions and the mapped original atomistic forces.
2. **`02_run_ibi.sh` (Table Generation):**
   Reads the intermediate dataset and executes the IBI loop. It uses the target distribution to compute the DBI (iteration 0) and then iteratively launches ESPResSo to correct the potential. Upon convergence, it exports the optimal `.dat` potentials to the `ibi_priors/` folder and updates `cg_priors.json` changing their type to `"tabulated"`.
3. **`03_subtract_ibi.sh` (Force Subtraction):**
   Reruns `build_cg_dataset.py`, but this time passing the `--priors cg_priors.json` flag generated by the previous step. In this way, the framework skips the statistical inference, sees the bonds as `"tabulated"`, loads the definitive `.dat` tables, and performs the interpolation to rigorously subtract the exact force (IBI) from the residual forces of the dataset. The final output `tel22_dataset_ibi.bin` is ready to train the Machine Learning model!

> [!TIP]
> **Equilibrium Distance Auto-calculation (`r0`)**
> For explicit bonds (FENE, Morse, etc.), if you omit the numerical parameter and set `"r0": "auto"`, the script will automatically extract the exact mean distance for that pair of atoms directly from the molecular trajectory! This prevents thermodynamic explosions and elegantly resolves scale mismatches between all-atom and coarse-grained representations.

> [!NOTE]
> **Morse Potential and Force Capping**
> In ESPResSo, explicit Morse bonds are injected under the hood as `TabulatedDistance` (extended beyond the box size). The framework automatically applies a **Force Capping** (hard limit) to prevent integration explosions caused by the exponentially steep repulsive wall when monomers get too close, ensuring a perfect balance between bond breakability and thermodynamic stability.

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
> **Lipschitz Regularization**
> In the `.json` file you can add or modify the `"lipschitz_lambda": 0.001` parameter. This introduces an L2 penalty on the force magnitude during training (inspired by CGnet). By enabling it, the model learns to predict smoother energy surfaces, preventing massive gradients and explosions during ESPResSo simulations. If set to `0.0`, the additional overhead is completely bypassed ensuring absolute backward compatibility.

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
- ESPResSo will automatically apply "Force Capping" on these tabulated bonds to prevent integration explosions if monomers suffer severe thermal collisions at very short distances.

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
