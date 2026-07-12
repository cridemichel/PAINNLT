import numpy as np
import json
import glob

pos = np.load("_tmp_initial_pos.npy")
with open("cg_priors.json") as f:
    priors = json.load(f)

def get_valid_bounds(pattern):
    valid_ranges = {}
    for file in glob.glob(pattern):
        data = np.loadtxt(file)
        x = data[:, 0]
        f = data[:, 2]
        valid_idx = np.where(np.abs(f) < 1000)[0]
        if len(valid_idx) > 0:
            first, last = valid_idx[0], valid_idx[-1]
            name = file.split("tabulated_")[-1].replace(".dat", "")
            valid_ranges[name] = (x[first], x[last])
    return valid_ranges

bond_bounds = get_valid_bounds("ibi_priors/bond_tabulated_*.dat")
angle_bounds = get_valid_bounds("ibi_priors/angle_tabulated_*.dat")

print("Checking Bonds:")
for b in priors.get("bonds", []):
    if b["type"] != "tabulated": continue
    i, j = b["mol_i"], b["mol_j"]
    name = b["name"]
    d = np.linalg.norm(pos[i] - pos[j])
    vmin, vmax = bond_bounds.get(name, (0.0, 3.0))
    if d < vmin or d > vmax:
        print(f"Bond {name} {i}-{j}: {d:.3f} is OUT OF BOUNDS [{vmin:.3f}, {vmax:.3f}]")

print("Checking Angles:")
for a in priors.get("angles", []):
    if a.get("type", "harmonic") != "tabulated": continue
    i, j, k = a["mol_i"], a["mol_j"], a["mol_k"]
    name = a["name"]
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    theta = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    vmin, vmax = angle_bounds.get(name, (0.0, 3.14))
    if theta < vmin or theta > vmax:
        print(f"Angle {name} {i}-{j}-{k}: {theta:.3f} is OUT OF BOUNDS [{vmin:.3f}, {vmax:.3f}]")

