import numpy as np
import json

data = np.load('equilibrated.npz')
v = data['v']
omega = data['omega']

with open('rigid_bodies_info.json', 'r') as f:
    rb_info = json.load(f)

# E_kin = sum 1/2 m v^2 + sum 1/2 I omega^2

# We need the masses and inertias.
# From the file size it's 219 particles? Let's just print the max velocity.
print("Max velocity:", np.max(np.abs(v)))
print("Max omega:", np.max(np.abs(omega)))
