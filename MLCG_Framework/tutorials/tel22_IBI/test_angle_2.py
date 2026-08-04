import numpy as np

data = np.loadtxt('ibi_priors/angle_tabulated_ang_A_G_G.dat')
print("F:", data[:5, 2])
