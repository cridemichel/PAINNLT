import numpy as np
data = np.load("nve_dt_0.0010.npz")
for i in range(10):
    print(f"Step {i}: e_tot={data['e_tots'][i]:.4f} e_kin={data['e_kins'][i]:.4f} e_pot={data['e_pots'][i]:.4f}")
