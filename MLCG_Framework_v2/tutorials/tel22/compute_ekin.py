import numpy as np
import json

data = np.load('equilibrated.npz')
v = data['v']
omega = data['omega']

with open('rigid_bodies_info.json', 'r') as f:
    rb_info = json.load(f)

e_kin_trans = 0.0
e_kin_rot = 0.0
dofs_trans = 0
dofs_rot = 0

for mol_idx_str, mol_data in rb_info.items():
    mol_idx = int(mol_idx_str)
    mass = mol_data['mass']
    I = np.array(mol_data['rinertia'])
    
    # We need to find the particle ID for this COM
    # Usually in the checkpoint, COM particles have the mass
    pass

# A better way is just to load them and look for the COM particles which have mass > 0.1
e_kin_trans = 0.0
e_kin_rot = 0.0
dofs_trans = 0
dofs_rot = 0

# From rigid_bodies_info, we know the masses.
# The particles in the checkpoint are exactly the ones in the system.
# We can just use espresso to load the priors and tell us the masses.
import sys
import os

with open('calculate_T.py', 'w') as f:
    f.write('''import espressomd
import numpy as np
import json

# Setup system
system = espressomd.System(box_l=[11.0, 11.0, 11.0])

data = np.load('equilibrated.npz')
v = data['v']
omega = data['omega']

with open('cg_priors.json', 'r') as f:
    priors = json.load(f)

with open('rigid_bodies_info.json', 'r') as f:
    rb_info = json.load(f)
    
num_com = len(rb_info)
masses = []
for k, v_data in rb_info.items():
    masses.append(v_data['mass'])

# Just assume the first `num_com` particles are the COMs or we can match their masses
# Actually we can just compute: sum 0.5 * m * v^2 over all particles where we assign mass.
# But wait, in the checkpoint the order is what run_cg_md sets.
# Let's just approximate by looking at the nonzero velocities!
non_zero_v = [vv for vv in v if np.sum(vv**2) > 1e-6]
print("Number of particles with velocity > 0:", len(non_zero_v))
''')
