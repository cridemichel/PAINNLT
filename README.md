Ecco la traduzione del testo in un inglese tecnico, fluido e appropriato per l'ambito del Machine Learning applicato alla dinamica molecolare.

TODO LIST
Energy normalization (add energy shift and scale)
SchNetPack addresses the issue of energy magnitudes using two built-in mechanisms that correspond to our "Scale and Shift" approach:

schnetpack.atomistic.Atomref: This module appends the reference energy of the individual isolated atoms (the baseline shift) to the network's final output.

schnetpack.transform.Standardize: During preprocessing, SchNetPack statistically computes the mean and standard deviation of the residual energies or forces and standardizes the output. This ensures that the neural network always operates on values close to zero, leaving the framework to re-multiply and re-add the real values only at the final output stage.

Implement cosine cutoff
If you look inside the schnetpack/nn/cutoff.py directory, you will find an entire class named CosineCutoff. When you declare a PaiNN or SchNet model in SchNetPack, the cutoff_network parameter is initialized with this exact class by default. They use the precise scaled cosine function I suggested to ensure that spatial derivatives smoothly vanish at the boundary of the Neighbor List, thereby preventing discontinuous jumps in the forces.

In version 2.0, SchNetPack delegated the entire training loop to PyTorch Lightning. If you check their training configuration files (managed via the Hydra system in the configs/trainer directory), you will find the gradient_clip_val parameter. It is common practice in their tutorials to set this parameter around 0.5 or 1.0. As we discussed, since forces are the derivative of the energy, without this "leash" on the gradient, a single unfortunate short-range repulsion within the batch would permanently ruin the weights of the AdamW optimizer.

Mixed precision

GROMACS TO BIN

Given a group of atoms, a script must be prepared to replace this group of atoms with a cluster of virtual sites and one real particle, or with a single real particle.

Each virtual site must be associated with a type (we can use the atomic number Z) and a mol_id (the molecule it belongs to). (I need to discuss this with Laura).
If there are no virtual sites, the single real particle will still have its own mol_id.
When constructing interactions in PaiNN, atoms with the same mol_id must not interact.
In ESPResSo, virtual sites do not interact with one another, so no action is required there; we just need to import the model and use it as I currently do.
Most likely, priors will need to be included in the loss function to prevent overlaps (using WCA) or harmonic (or FENE) potentials for bonded atoms.
The forces in the loss function will be the sum of those predicted by the network and those originating from the priors.
The overall implementation will be as follows: in ESPResSo, the various virtual sites will still feature a WCA interaction, and certain atom pairs will have a harmonic or FENE interaction. During training, we will need to include these interactions in the loss calculation as explained above.


