import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from ase import Atoms
from ase.neighborlist import neighbor_list

dataset = ASEAtomsData('cg_dataset.db')
inputs = dataset[0]

print("Keys:", inputs.keys())
cell = inputs["_cell"].view(3, 3).detach().numpy()
pbc = inputs["_pbc"].view(3).detach().numpy()
R = inputs["_positions"].detach().numpy()
Z = inputs["_atomic_numbers"].detach().numpy()

atoms = Atoms(numbers=Z, positions=R, cell=cell, pbc=pbc)
print("Calling neighbor_list...", flush=True)
idx_i, idx_j, S, D = neighbor_list("ijSD", atoms, 1.0, self_interaction=False)
print("Neighbor list generated", flush=True)
