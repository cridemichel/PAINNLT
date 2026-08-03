import numpy as np
import json

data = np.load('equilibrated.npz')
v = data['v']
omega = data['omega']

with open('rigid_bodies_info.json', 'r') as f:
    rb_info = json.load(f)

e_kin = 0.0
dofs = 0

for mol_idx, mol_data in rb_info.items():
    mol_idx = int(mol_idx)
    # The COM is the first particle for each molecule, or we can just iterate over the length
    # But wait, rb_info only has the mass/inertia of the COM.
    # The id of COM particles might be 0, 10, 20... let's just find them.
    # We know in v2 that there are 10 molecules (if it's tel22 box of 10) + ions + water maybe? No, tel22_dataset.bin is just 10 DNA molecules + ions.
    # Let's just do it directly from the checkpoint since mass and rinertia should be in `rb_info`
    pass

# Actually, the simplest way is to load the system in espressomd, load checkpoint, and compute!
import espressomd

system = espressomd.System(box_l=[1.0, 1.0, 1.0])
# We just need to load the checkpoint and the script run_cg_md.py will do the rest.
