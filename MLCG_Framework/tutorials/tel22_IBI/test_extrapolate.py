import numpy as np
from scipy.interpolate import CubicSpline

x = np.linspace(0.0, 3.14, 600)
V = np.zeros_like(x)
x_left = x[20]
V_left = 50.0

for i in range(20 - 1, -1, -1):
    dx = x[i] - x_left
    V[i] = V_left - 10.0 * dx

spline = CubicSpline(x[:20], V[:20])
force_deriv = spline(x[:20], 1)
print(force_deriv)
