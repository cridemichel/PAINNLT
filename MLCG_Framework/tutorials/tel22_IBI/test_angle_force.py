import numpy as np
import sys

try:
    data = np.loadtxt('ibi_priors/angle_tabulated_ang_A_G_G.dat')
    print("x:", data[:5, 0])
    print("V:", data[:5, 1])
    print("F:", data[:5, 2])
except Exception as e:
    print(e)
