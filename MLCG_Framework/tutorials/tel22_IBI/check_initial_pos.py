import numpy as np

pos = np.load("_tmp_initial_pos.npy")
dist = np.linalg.norm(pos[164] - pos[165])
print(f"Distance between 164 and 165: {dist:.6f} nm")

# Let's also check all bonds
import json
with open("cg_priors.json") as f:
    priors = json.load(f)

for b in priors.get("bonds", []):
    i = b["mol_i"]
    j = b["mol_j"]
    d = np.linalg.norm(pos[i] - pos[j])
    if d > 0.5 or d < 0.1:
        print(f"Abnormal bond {i}-{j}: {d:.6f} nm")
