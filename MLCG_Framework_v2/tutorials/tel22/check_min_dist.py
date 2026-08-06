import torch
import numpy as np
import json

data = torch.load("tel22_dataset.bin", map_location='cpu', weights_only=False)
positions = data["positions"].numpy() # (frames, num_atoms, 3)

with open("cg_priors.json") as f:
    priors = json.load(f)

wca = priors.get("wca", {})
num_species = positions.shape[1]

# Calculate all pairwise distances for all frames
from scipy.spatial.distance import pdist

min_dists = np.ones((num_species, num_species)) * np.inf

for frame in positions:
    dist_matrix = np.linalg.norm(frame[:, None, :] - frame[None, :, :], axis=-1)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Exclude bonded particles if they are excluded by intra-molecular?
    # Actually just get the minimum over all pairs
    for i in range(num_species):
        for j in range(num_species):
            if dist_matrix[i, j] < min_dists[i, j]:
                min_dists[i, j] = dist_matrix[i, j]

for i in range(5):
    print(f"min_dist(0, {i}) = {min_dists[0, i]:.4f}")
    
# Let's compare with sigma in cg_priors
for k, v in wca.get("overrides", {}).items():
    i = int(k)
    sigma = v.get("sigma", wca["sigma"])
    cutoff = sigma * (2.0**(1/6))
    print(f"Type {i}: sigma = {sigma:.4f}, cutoff = {cutoff:.4f}, min_dist = {min_dists[i].min():.4f}")
    if i >= 5: break
