import numpy as np
from scipy.spatial.distance import pdist

data = np.load("equilibrated.npz")
pos = data["positions"]

# pos shape is likely (num_atoms, 3)
dists = pdist(pos)
print("Min distance in equilibrated.npz:", dists.min())

# Check how many pairs are below 0.35 nm
below = np.sum(dists < 0.35)
print("Pairs below 0.35 nm:", below)

# Check pairs below 0.5 nm
below = np.sum(dists < 0.5)
print("Pairs below 0.5 nm:", below)
