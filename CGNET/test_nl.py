import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ase import Atoms
from ase.neighborlist import neighbor_list
import numpy as np

atoms = Atoms('O3', positions=[[0,0,0], [0,1,0], [1,0,0]], cell=[10,10,10], pbc=True)
print("Calling neighbor_list", flush=True)
i, j, D, d = neighbor_list('ijDd', atoms, 1.5)
print("Done", flush=True)
