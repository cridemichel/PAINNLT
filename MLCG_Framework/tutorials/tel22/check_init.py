import struct
import numpy as np

with open("tel22_dataset.bin", "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_molecules = struct.unpack("i", f.read(4))[0]
    num_total_sites = struct.unpack("i", f.read(4))[0]
    box_dim = struct.unpack("3f", f.read(12))
    
    com_positions = []
    for _ in range(num_molecules):
        com_positions.append(struct.unpack("3f", f.read(12)))
        
    vs_positions = []
    for _ in range(num_molecules):
        num_vs = struct.unpack("i", f.read(4))[0]
        f.read(4 * num_vs) # types
        pos = []
        for _ in range(num_vs):
            pos.append(struct.unpack("3f", f.read(12)))
        vs_positions.append(pos)

vs_count = num_molecules
for i in range(num_molecules):
    for j in range(len(vs_positions[i])):
        if vs_count == 835: pos_835 = vs_positions[i][j]
        if vs_count == 842: pos_842 = vs_positions[i][j]
        if vs_count == 279: pos_279 = vs_positions[i][j]
        if vs_count == 286: pos_286 = vs_positions[i][j]
        if vs_count == 460: pos_460 = vs_positions[i][j]
        if vs_count == 467: pos_467 = vs_positions[i][j]
        vs_count += 1
        
print(f"Initial 835-842: {np.linalg.norm(np.array(pos_835) - np.array(pos_842)):.4f} nm")
print(f"Initial 279-286: {np.linalg.norm(np.array(pos_279) - np.array(pos_286)):.4f} nm")
print(f"Initial 460-467: {np.linalg.norm(np.array(pos_460) - np.array(pos_467)):.4f} nm")
