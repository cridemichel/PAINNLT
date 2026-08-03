import espressomd
import numpy as np
import json
import struct

# Load checkpoint
chk = np.load('equilibrated.npz')
v = chk['v']
omega = chk['omega']

with open('rigid_bodies_info.json') as f:
    rb_info = json.load(f)

with open('tel22_dataset.bin', 'rb') as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_molecules = struct.unpack("i", f.read(4))[0]
    
    e_kin = 0.0
    dof = 0
    idx = 0
    for mol_idx in range(num_molecules):
        mol_id = struct.unpack("i", f.read(4))[0]
        num_sites = struct.unpack("i", f.read(4))[0]
        f.read(12) # center
        
        # determine type
        if num_sites == 6:
            mass = rb_info["DG"]["mass_amu"]
            I = rb_info["DG"]["inertia_amu_nm2"]
        elif num_sites == 1:
            if "DA" in rb_info:
                # We need to distinguish DA and DT... wait, mass of DA is 250, DT is 303
                # Let's just use the v^2 and guess!
                pass
        
        # Actually, let's just bypass the dataset and use ESPResSo directly!
