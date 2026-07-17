import struct
mol_com_parts = {}
mol_vs_parts = {}
pid = 0
with open("tel22_dataset.bin", "rb") as f:
    f.read(12)
    box_dim = struct.unpack("3f", f.read(12))
    for mol_idx in range(220):
        f.read(4)
        num_sites = struct.unpack("i", f.read(4))[0]
        f.read(36)
        mol_com_parts[mol_idx] = pid
        pid += 1
        for s in range(num_sites):
            f.read(16)
            mol_vs_parts[(mol_idx, s)] = pid
            pid += 1

for tid in [862, 869, 876]:
    for m, p in mol_com_parts.items():
        if p == tid: print(f"ID {tid} is COM of Mol {m}")
    for (m, s), p in mol_vs_parts.items():
        if p == tid: print(f"ID {tid} is Virtual Site {s} of Mol {m}")
