import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn

print("Loading dataset...", flush=True)
ds = ASEAtomsData('cg_dataset.db')
nl = trn.ASENeighborList(cutoff=1.0)
for i in range(200):
    item = ds[i]
    item = nl(item)
    if (i+1) % 10 == 0:
        print(f"Processed {i+1}", flush=True)
print("Done!", flush=True)
