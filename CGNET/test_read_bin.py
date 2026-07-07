import struct
import numpy as np
with open("../GROMACS/cg_dataset.bin", "rb") as f:
    f.read(4)
    num_molecules, num_total_sites = struct.unpack("ii", f.read(8))
    box = np.array(struct.unpack("3f", f.read(12)))
    print("Molecules:", num_molecules, "Sites:", num_total_sites, "Box:", box)
    for mol_idx in range(min(num_molecules, 10)):
        mol_id, num_sites = struct.unpack("ii", f.read(8))
        cx, cy, cz, fx, fy, fz, tx, ty, tz = struct.unpack("9f", f.read(36))
        print(f"Mol {mol_idx}: sites={num_sites} center=({cx}, {cy}, {cz})")
        for s in range(num_sites):
            f.read(16)
