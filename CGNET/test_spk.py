import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData

print("Loading", flush=True)
dataset = ASEAtomsData('cg_dataset.db')
atoms = dataset[0]
print("Atoms keys:", atoms.keys(), flush=True)

transform = trn.ASENeighborList(cutoff=1.0)
print("Applying transform", flush=True)
res = transform(atoms)
print("Transform done", flush=True)
print(res.keys())
