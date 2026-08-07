import numpy as np
chk = np.load('equilibrated.npz')
vel = chk['v']
e_kin = 0
for i in range(len(vel)):
    has_omega = sum(chk['omega'][i]**2) > 0
    if has_omega:
        e_kin += 0.5 * 250 * sum(vel[i]**2)
print("Calculated E_kin:", e_kin)
