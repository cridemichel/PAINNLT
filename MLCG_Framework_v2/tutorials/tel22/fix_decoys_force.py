import sys

filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/preprocessing/build_cg_dataset.py"
with open(filepath, "r") as f:
    content = f.read()

# Replace the flattening loop for forces
old_forces_loop = """    idx = 0
    for m_idx, sites in enumerate(d_sites):
        for _ in sites:
            flat_forces.append(d_forces[idx])
            idx += 1
            
    flat_pos = np.array(flat_pos, dtype=np.float32)
    flat_forces = np.array(flat_forces, dtype=np.float32)"""
new_forces_loop = """    flat_pos = np.array(flat_pos, dtype=np.float32)
    flat_forces = np.zeros((len(flat_pos), 3), dtype=np.float32)"""

content = content.replace(old_forces_loop, new_forces_loop)

with open(filepath, "w") as f:
    f.write(content)
print("Fixed decoys force output logic!")
