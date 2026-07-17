import json
import numpy as np
import struct

with open("tel22_topology.json", "r") as f:
    config = json.load(f)

exclusions = set()
for b in config.get("bonds", []):
    m1, m2 = min(b["mol_i"], b["mol_j"]), max(b["mol_i"], b["mol_j"])
    exclusions.add((m1, m2))
for a in config.get("angles", []):
    m1, m2 = min(a["mol_i"], a["mol_k"]), max(a["mol_i"], a["mol_k"])
    exclusions.add((m1, m2))

with open("tel22_dataset.bin", "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_mols = struct.unpack("i", f.read(4))[0]
    num_sites = struct.unpack("i", f.read(4))[0]
    box = struct.unpack("3f", f.read(12))
    box_dim = np.array(box)

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

sigma = 0.45
epsilon = 1.0

max_wca = 0
for i in range(len(positions)):
    for j in range(i+1, len(positions)):
        m1, m2 = min(mol_ids[i], mol_ids[j]), max(mol_ids[i], mol_ids[j])
        if m1 == m2: continue
        if (m1, m2) in exclusions: continue
        
        diff = positions[i] - positions[j]
        diff -= box_dim * np.round(diff / box_dim)
        dist = np.linalg.norm(diff)
        
        if dist < sigma * (2**(1/6)):
            sr6 = (sigma/dist)**6
            f = 24 * epsilon * (2 * sr6**2 - sr6) / dist
            if f > max_wca:
                max_wca = f
                print(f"New max WCA {f} at dist {dist} between {m1} and {m2}")

print("Max WCA force remaining:", max_wca)
