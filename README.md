# PaiNN-ESPResSo Integration Framework

This framework provides a complete pipeline to train and deploy the **PaiNN (Polarizable Atom Interaction Neural Network)** model natively in **ESPResSo**, supporting both **All-Atom** and **Coarse-Grained (CG)** Molecular Dynamics simulations.

---

## 🚀 Current Capabilities & How-To

### 1. All-Atom Simulations (e.g., Ethanol MD17)
You can train and simulate standard all-atom systems where every atom is a separate entity interacting through the PaiNN force field.
* **Training:** Run the legacy `painn.cpp` training script to train the model on all-atom datasets.
* **Simulation in ESPResSo:** 
  Use the Python script `espresso_integration/test_espresso_painn.py`. 
  - Ensure you map the ESPResSo particle `type` to the actual atomic numbers (e.g., `1` for H, `6` for C).
  - Call `espressomd.painn.activate_painn_potential(model_path="best_painn_etanolo.pt", ...)` with the correct `num_species` from the configuration JSON.
  - The C++ plugin will seamlessly collect all particles, compute interactions within the cutoff, and evaluate forces natively.

### 2. Coarse-Grained (CG) Simulations with Virtual Sites & Rigid Bodies
You can project All-Atom GROMACS trajectories into Coarse-Grained rigid bodies, train PaiNN to predict the effective CG forces and torques, and run the simulation using ESPResSo Virtual Sites.
* **Dataset Generation (GROMACS -> Binary):**
  Use `python_scripts/convert_gro2bin.py` alongside the `cg_mapping.json` file. This script groups atoms into virtual sites, correctly handles Periodic Boundary Conditions (MIC unwrapping), and outputs a `.bin` dataset containing centers of geometry, target forces, and target torques.
* **Dataset Inspection:**
  Use `python_scripts/inspect_bin.py` to inspect the generated dataset, verify box dimensions, molecule compositions, and site coordinates.
* **Training:**
  Run the new `build/cg_painn_train` C++ executable. It automatically excludes intra-molecular interactions from the message-passing graph (as virtual sites belonging to the same rigid body do not interact in ESPResSo) and trains the network to predict the total force and torque on each rigid body.
* **Simulation in ESPResSo:**
  Define the Rigid Bodies in ESPResSo using the real particles and virtual sites. Pass the trained `.pt` CG model to the Python interface. The C++ integration automatically handles the evaluation.

### 3. Deployment on Supercomputers (e.g., CINECA Leonardo)
A complete containerized environment is provided to run everything seamlessly on HPC clusters.
* **Docker/Apptainer:** 
  A `Dockerfile` is included to build an isolated PyTorch + CUDA 11.8 environment containing all ESPResSo dependencies (MPI, FFTW, Boost).
* **SLURM Submission:** 
  The `leonardo_submit.slurm` template shows how to deploy the container via Apptainer and bind the working directories to ensure data persistence on the cluster. Check `HOWTO_LEONARDO.md` for step-by-step deployment instructions.

---

## 🛠️ Developer Notes & TODOs

### TODO LIST
- `[ ]` Energy normalization (add energy shift and scale). *Note: Better not to implement this for coarse-grained systems.*
  - SchNetPack addresses the issue of energy magnitudes with two built-in mechanisms: `schnetpack.atomistic.Atomref` (baseline shift) and `schnetpack.transform.Standardize` (mean/std scaling).
- `[X]` Implement cosine cutoff.
  - Done as in SchNetPack `CosineCutoff` to ensure spatial derivatives smoothly drop to zero at the edge of the Neighbor List.
- `[X]` Gradient Clipping.
  - Implemented gradient clipping to prevent a single unlucky short-range repulsion in the batch from permanently ruining the weights of the AdamW optimizer.
- `[ ]` Mixed precision (GPU float, rest double) if convergence issues arise.

### Coarse-Graining Strategy & Notes
1. **Virtual Sites:** Given a group of atoms, they are replaced by a cluster of virtual sites and a real particle, or a single real particle. Each virtual site has a type (e.g., atomic number Z) and a `mol_id`.
2. **Graph Construction:** In PaiNN, atoms with the same `mol_id` do NOT interact. This mirrors ESPResSo where virtual sites belonging to the same rigid body do not interact.
3. **Loss Function:** The strategy is to calculate the total force and torque acting on a GROMACS CG group, and minimize both using PyTorch's `autograd`.
4. **Priors:** It might be necessary to include priors in the loss function to prevent overlaps (WCA) or harmonic potentials for bonded atoms.
5. **Periodic Boundary Conditions (PBC):** The `convert_gro2bin.py` script applies the Minimum Image Convention to unwrap broken molecules across box boundaries before computing Centers of Mass.

### Mapping Methods (`convert_gro2bin.py`)
1. **COM (Center of Mass):** Weighs coordinates by mass. Best for force-matching models.
2. **COG (Center of Geometry):** Arithmetic mean of coordinates.
3. **ATOM (Atom-centered):** Selects a specific reference atom (e.g., C_alpha for proteins).
