import numpy as np
import struct
import sys

def get_std(filename):
    try:
        with open(filename, "rb") as f:
            num_frames = struct.unpack("i", f.read(4))[0]
            all_forces = []
            for _ in range(num_frames):
                num_mols = struct.unpack("i", f.read(4))[0]
                num_sites = struct.unpack("i", f.read(4))[0]
                box = struct.unpack("3f", f.read(12))
                for m in range(num_mols):
                    mol_id = struct.unpack("i", f.read(4))[0]
                    ns = struct.unpack("i", f.read(4))[0]
                    f.read(12) # center
                    fx, fy, fz = struct.unpack("3f", f.read(12)) # force
                    all_forces.append([fx, fy, fz])
                    f.read(12) # torque
                    for s in range(ns):
                        f.read(4) # type
                        f.read(12) # pos
                        
        all_forces = np.array(all_forces)
        std = np.std(all_forces)
        return std
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    std_orig = get_std("tel22_dataset.bin")
    std_ibi = get_std("tel22_dataset_ibi.bin")
    print(f"Original Force STD: {std_orig}")
    print(f"Residual Force STD: {std_ibi}")
