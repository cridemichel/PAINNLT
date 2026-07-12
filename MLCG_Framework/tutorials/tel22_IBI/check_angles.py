import numpy as np
import json

pos = np.load("_tmp_initial_pos.npy")
with open("cg_priors.json") as f:
    priors = json.load(f)

# Hardcode the valid regions found earlier
valid_ranges = {
    "ang_T_T_A": (0.63, 3.22),
    "ang_G_G_T": (1.81, 3.22),
    "ang_G_G_G": (1.71, 3.22),
    "ang_G_T_T": (0.42, 2.11),
    "ang_A_G_G": (1.71, 3.22),
    "ang_T_A_G": (1.00, 3.22)
}

for a in priors.get("angles", []):
    i, j, k = a["mol_i"], a["mol_j"], a["mol_k"]
    name = a["name"]
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    vmin, vmax = valid_ranges.get(name, (0.0, 3.14))
    if theta < vmin or theta > vmax:
        print(f"Angle {name} {i}-{j}-{k}: {theta:.3f} is OUT OF BOUNDS [{vmin}, {vmax}]")
