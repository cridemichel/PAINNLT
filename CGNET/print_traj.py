import numpy as np
data = np.load("nve_dt_0.0010.npz")
print("E_tots:", data['e_tots'][:10])
