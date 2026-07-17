import struct
import numpy as np

with open("tel22_dataset.bin", "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    
    # Read frame 0
    num_mols = struct.unpack("i", f.read(4))[0]
    num_sites = struct.unpack("i", f.read(4))[0]
    box = struct.unpack("3f", f.read(12))
    
    max_f = 0
    max_t = 0
    for mol in range(num_mols):
        mol_id = struct.unpack("i", f.read(4))[0]
        n_s = struct.unpack("i", f.read(4))[0]
        center = struct.unpack("3f", f.read(12))
        force = struct.unpack("3f", f.read(12))
        torque = struct.unpack("3f", f.read(12))
        
        max_f = max(max_f, np.linalg.norm(force))
        max_t = max(max_t, np.linalg.norm(torque))
        
        for _ in range(n_s):
            f.read(16)

print("Max Force:", max_f)
print("Max Torque:", max_t)
