import struct
import numpy as np
import sys

dataset_path = 'tel22_dataset.bin'

pos = []
with open(dataset_path, "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_molecules = struct.unpack("i", f.read(4))[0]
    num_total_sites = struct.unpack("i", f.read(4))[0]
    box_dim = struct.unpack("3f", f.read(12))
    
    for mol_idx in range(num_molecules):
        mol_id = struct.unpack("i", f.read(4))[0]
        num_sites = struct.unpack("i", f.read(4))[0]
        center = struct.unpack("3f", f.read(12))
        force = struct.unpack("3f", f.read(12))
        torque = struct.unpack("3f", f.read(12))
        
        pos.append((mol_id, -1, center)) # COM
        
        for s in range(num_sites):
            stype = struct.unpack("i", f.read(4))[0]
            spos = struct.unpack("3f", f.read(12))
            pos.append((mol_id, s, spos))

n = len(pos)
print(f"Checking {n} particles in frame 0 for topological intersections...")

intersections = []
for i in range(n):
    mol_i, site_i, p_i = pos[i]
    pi = np.array(p_i)
    for j in range(i + 1, n):
        mol_j, site_j, p_j = pos[j]
        pj = np.array(p_j)
        
        if mol_i == mol_j:
            continue # Ignore particles in the same rigid body/nucleotide
            
        d = np.linalg.norm(pi - pj)
        if d < 0.15: # 0.15 nm
            intersections.append((mol_i, site_i, mol_j, site_j, d))

intersections.sort(key=lambda x: x[4])

min_d = 1000.0
min_pair = None
for i in range(n):
    mol_i, site_i, p_i = pos[i]
    pi = np.array(p_i)
    for j in range(i + 1, n):
        mol_j, site_j, p_j = pos[j]
        pj = np.array(p_j)
        
        if mol_i == mol_j:
            continue
            
        d = np.linalg.norm(pi - pj)
        if d < min_d:
            min_d = d
            min_pair = (mol_i, site_i, mol_j, site_j)

print(f"La distanza minima assoluta tra molecole diverse è: {min_d:.4f} nm")
if min_pair:
    print(f"Tra Mol {min_pair[0]} Site {min_pair[1]} e Mol {min_pair[2]} Site {min_pair[3]}")

