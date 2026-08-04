import numpy as np
data = np.loadtxt('ibi_priors/angle_tabulated_ang_A_G_G.dat')
print("min V:", np.min(data[:, 1]))
print("min V idx:", np.argmin(data[:, 1]), "x:", data[np.argmin(data[:, 1]), 0])
print("F at min:", data[np.argmin(data[:, 1]), 2])
print("F at 0:", data[0, 2])
print("F at end:", data[-1, 2])
