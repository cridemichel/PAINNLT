import ase.io
from ase.neighborlist import neighbor_list
import time

print("Reading DB...", flush=True)
db = ase.io.read('cg_dataset.db', index=':')
print(f"Read {len(db)} atoms. Computing neighbors...", flush=True)

for i in range(200):
    idx_i, idx_j, S = neighbor_list("ijS", db[i], 1.0, self_interaction=False)
    if (i+1) % 10 == 0:
        print(f"Computed {i+1}", flush=True)
print("Done!", flush=True)
