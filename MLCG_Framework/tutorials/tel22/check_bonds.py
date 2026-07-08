import struct
import json
import math

with open("rigid_bodies_info.json", "r") as f:
    rb_info = json.load(f)

with open("cg_priors.json", "r") as f:
    priors = json.load(f)

with open("tel22_dataset.bin", "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_molecules = struct.unpack("i", f.read(4))[0]
    num_total_sites = struct.unpack("i", f.read(4))[0]
    box_dim = struct.unpack("3f", f.read(12))
    
    mol_coms = {}
    mol_vss = {}
    
    for mol_idx in range(num_molecules):
        mol_id = struct.unpack("i", f.read(4))[0]
        num_sites = struct.unpack("i", f.read(4))[0]
        center = struct.unpack("3f", f.read(12))
        force = struct.unpack("3f", f.read(12))
        torque = struct.unpack("3f", f.read(12))
        
        mol_coms[mol_idx] = center
        
        sites = []
        for s in range(num_sites):
            stype = struct.unpack("i", f.read(4))[0]
            spos = struct.unpack("3f", f.read(12))
            sites.append(spos)
        mol_vss[mol_idx] = sites

for b in priors.get("bonds", []):
    mol_i, mol_j = b["mol_i"], b["mol_j"]
    site_i, site_j = b.get("site_i", -1), b.get("site_j", -1)
    
    p1 = mol_coms[mol_i] if site_i == -1 else mol_vss[mol_i][site_i]
    p2 = mol_coms[mol_j] if site_j == -1 else mol_vss[mol_j][site_j]
    
    dist = math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))
    if dist > 1.0:
        print(f"Bond {mol_i}:{site_i} - {mol_j}:{site_j} | dist={dist:.2f} nm, r0={b['r0']:.2f} nm")

