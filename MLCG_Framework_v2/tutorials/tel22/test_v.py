import numpy as np
data = np.load('equilibrated.npz')
v = data['v']
omega = data['omega']
print("Sum of |v| for COM:", np.sum(np.abs(v[0::2])))
print("Sum of |v| for VS:", np.sum(np.abs(v[1::2])))
