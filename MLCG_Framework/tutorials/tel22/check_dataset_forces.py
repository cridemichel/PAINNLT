import struct
import numpy as np

with open("tel22_dataset.bin", "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    
    # Read frame 0
    num_mols = struct.unpack("i", f.read(4))[0]
    num_sites = struct.unpack("i", f.read(4))[0]
    box = struct.unpack("3f", f.read(12))
    
    max_f = 0
    forces_list = []
    for mol in range(num_mols):
        mol_id = struct.unpack("i", f.read(4))[0]
        n_s = struct.unpack("i", f.read(4))[0]
        center = struct.unpack("3f", f.read(12))
        force = struct.unpack("3f", f.read(12))
        f_norm = np.linalg.norm(force)
        max_f = max(max_f, f_norm)
        forces_list.append(f_norm)
        torque = struct.unpack("3f", f.read(12))
        
        for _ in range(n_s):
            f.read(16)

print("Max force in frame 0 of dataset:", max_f)
forces_list = np.array(forces_list)
print("Mean force:", np.mean(forces_list))
print("Percentiles 90, 95, 99:", np.percentile(forces_list, [90, 95, 99]))
