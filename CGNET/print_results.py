import numpy as np

dt_list = [0.001, 0.002, 0.004, 0.006, 0.008, 0.010]
for dt in dt_list:
    data = np.load(f"nve_dt_{dt:.4f}.npz")
    print(f"dt={dt:.4f}: dE = {data['dE']:.6e}")
