import numpy as np

# Load checkpoint
data = np.load('equilibrated.npz')
v = data['v']
omega = data['omega']

# We know the first 220 particles are the COMs?
# Wait! The order in `equilibrate.py` is:
# For each molecule: COM, then VS1, VS2...
# So COM is at index 0, then VS, VS... then next COM.
# That's why v[:220] is NOT just COMs! It's the first molecule and some of the second!
# That explains why the sum of v^2 was weird.

import json
with open('cg_priors.json') as f:
    priors = json.load(f)
with open('rigid_bodies_info.json') as f:
    rb_info = json.load(f)

# Reconstruct the mass array
import struct

with open('md.gro', 'r') as f:
    lines = f.readlines()
    num_atoms = int(lines[1])
    
with open('tel22_dataset.bin', 'rb') as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_molecules = struct.unpack("i", f.read(4))[0]
    
    com_indices = []
    masses = []
    inertias = []
    
    idx = 0
    for mol_idx in range(num_molecules):
        mol_id = struct.unpack("i", f.read(4))[0]
        num_sites = struct.unpack("i", f.read(4))[0]
        f.read(12) # center
        
        mol_type = "DA"
        if num_sites == 6: mol_type = "DG"
        elif num_sites == 1:
            if "DA" in rb_info and np.isclose(rb_info["DA"]["mass_amu"], 250.238):
                mol_type = "DA" if mol_idx % 2 == 0 else "DT" # Just rough check, actually let's just use the fact that tel22 has sequence
        
        # tel22 sequence: AGG GTU ... wait, it's 22 nucleotides.
        # we can just run run_cg_md.py and inject a print!
