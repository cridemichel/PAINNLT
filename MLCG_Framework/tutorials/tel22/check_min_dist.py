import struct
import numpy as np
import math

with open("tel22_dataset.bin", "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_mols = struct.unpack("i", f.read(4))[0]
    num_sites = struct.unpack("i", f.read(4))[0]
    box = struct.unpack("3f", f.read(12))
    
    positions = []
    mol_ids = []
    
    for mol in range(num_mols):
        mol_id = struct.unpack("i", f.read(4))[0]
        n_s = struct.unpack("i", f.read(4))[0]
        center = struct.unpack("3f", f.read(12))
        force = struct.unpack("3f", f.read(12))
        torque = struct.unpack("3f", f.read(12))
        
        for _ in range(n_s):
            stype = struct.unpack("i", f.read(4))[0]
            spos = struct.unpack("3f", f.read(12))
            positions.append(spos)
            mol_ids.append(mol_id)

positions = np.array(positions)
mol_ids = np.array(mol_ids)

min_dist = 999.0
for i in range(len(positions)):
    for j in range(i+1, len(positions)):
        if mol_ids[i] == mol_ids[j]: continue
        dist = np.linalg.norm(positions[i] - positions[j])
        if dist < min_dist:
            min_dist = dist

print("Min inter-molecular distance:", min_dist)
