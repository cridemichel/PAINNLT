import numpy as np

# Load checkpoint
chk = np.load("tel22_system.npz")
pos = chk["pos"]

# Wait, the checkpoint only has `pos`, but no `mol_id`. 
# We don't have the topology here.
