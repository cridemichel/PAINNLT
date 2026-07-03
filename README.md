ENG
===

TODO LIST

[ ] Energy normalization (add energy shift and scale)

SchNetPack addresses the issue of energy magnitudes with two built-in mechanisms that correspond to our "Scale and Shift":

schnetpack.atomistic.Atomref: This module appends the reference energy of the individual isolated atoms (the baseline shift) to the end of the network.

schnetpack.transform.Standardize: During preprocessing, SchNetPack statistically computes the mean and standard deviation of the remaining energies or forces and standardizes the output. This ensures that the neural network always works with numbers close to zero, letting the framework re-multiply and re-add the real values only at the moment of the final output.

[ ] Implement cosine cutoff

If you look in the schnetpack/nn/cutoff.py folder, you will find an entire class called CosineCutoff.
When you declare the PaiNN or SchNet model in SchNetPack, the cutoff_network parameter is initialized by default with this exact class. They use exactly the scaled cosine function that I suggested to ensure that the spatial derivatives smoothly drop to zero at the edge of the Neighbor List, preventing discontinuous jumps in the forces.

[X] In version 2.0, SchNetPack delegated the entire training loop to PyTorch Lightning.

If you check their training configuration files (managed via the Hydra system in the configs/trainer folder), you will find the gradient_clip_val parameter. It is common practice in their tutorials to set this parameter right around 0.5 or 1.0. As we discussed, since the forces are the derivative of the energy, without this "leash" on the gradient, a single unlucky short-range repulsion in the batch would permanently ruin the weights of the AdamW optimizer.

[ ] Mixed precision

GROMACS TO BIN

1) Given a group of atoms, a script must be prepared that replaces this group of atoms with a cluster of virtual sites and a real particle, or with a single real particle.
Each virtual site must be associated with a type (we can use the atomic number Z) and a mol_id (the molecule it belongs to) (I need to discuss this with Laura).
If there are no virtual sites, the single real particle will still have its own mol_id.

2) In the construction of the interactions in PaiNN, atoms with the same mol_id must not interact.

3) In ESPResSo, virtual sites do not interact with each other, so nothing will need to be done; just import the model and use it as I already do now.

4) Most likely, priors will also need to be included in the loss function to prevent overlaps (using WCA) or harmonic (or FENE) potentials for bonded atoms.
The forces in the loss will be those predicted by the network + those derived from the priors.
Everything will be implemented as follows: in ESPResSo, the various virtual sites will still have a WCA interaction, and some pairs of atoms will have a harmonic or FENE interaction. In the calculation of the loss during training, we should include these interactions as explained earlier.

5) Best strategy is to calculate the total force and
torque acting a GROMACS CG group, and consider
total force and torque acting on real particle and its N virtual sites,
in suche a way that loss function minimize botth total force and torque
(autograd of pytorch can do this)

6) our first real implementation step in Python will be a Data Parsing operation. You will need to
take the trajectories and forces from your original All-Atom simulation (GROMACS/BIN), apply your
Coarse-Graining logic (aggregating masses, defining virtual sites, and projecting total forces and
torques), and save each resulting frame into a new dataset_cg.db file using the ASE library.  Once
that .db file is created, the training script I showed you earlier will automatically "digest" it.
