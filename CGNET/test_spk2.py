import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from ase.neighborlist import neighbor_list
import torch

dataset = ASEAtomsData('cg_dataset.db')
atoms_dict = dataset[0]
at = dataset.get_atoms(0)
print(at)
print(at.cell)
print(at.pbc)
i, j, D, d = neighbor_list('ijDd', at, 1.0)
print(f"NL done: {len(i)} edges")
