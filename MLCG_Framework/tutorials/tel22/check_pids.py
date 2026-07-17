import struct
import json

with open("tel22_topology.json", "r") as f:
    config = json.load(f)

with open("tel22_dataset.bin", "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_mols = struct.unpack("i", f.read(4))[0]
    num_sites = struct.unpack("i", f.read(4))[0]
    box = struct.unpack("3f", f.read(12))

    mol_com_parts = {}
    mol_vs_parts = {}
    part_id = 0
    
    # Same logic as equilibrate.py
    for mol_idx in range(num_mols):
        mol_com_parts[mol_idx] = part_id
        part_id += 1
        
        mol_id_read = struct.unpack("i", f.read(4))[0]
        n_s = struct.unpack("i", f.read(4))[0]
        f.read(24) # center, force
        f.read(12) # torque
        
        for site_idx in range(n_s):
            f.read(16)
            mol_vs_parts[(mol_idx, site_idx)] = part_id
            part_id += 1

def identify(pid):
    for m, p in mol_com_parts.items():
        if p == pid: return f"COM of Mol {m}"
    for (m, s), p in mol_vs_parts.items():
        if p == pid: return f"Site {s} of Mol {m}"
    return "Unknown"

print("473:", identify(473))
print("480:", identify(480))
print("487:", identify(487))
print("500:", identify(500))
print("507:", identify(507))
print("498:", identify(498))
