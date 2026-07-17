import struct
import json

mol_com_parts = {}
mol_vs_parts = {}
pid = 0

with open("tel22_dataset.bin", "rb") as f:
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
        
        mol_com_parts[mol_idx] = pid
        pid += 1
        
        for s in range(num_sites):
            stype = struct.unpack("i", f.read(4))[0]
            spos = struct.unpack("3f", f.read(12))
            mol_vs_parts[(mol_idx, s)] = pid
            pid += 1

target_ids = [730, 756, 757, 758, 765, 784, 811]
for tid in target_ids:
    found = False
    for m, p in mol_com_parts.items():
        if p == tid:
            print(f"ID {tid} is COM of Mol {m}")
            found = True
    for (m, s), p in mol_vs_parts.items():
        if p == tid:
            print(f"ID {tid} is Virtual Site {s} of Mol {m}")
            found = True
    if not found:
        print(f"ID {tid} not found!")
