import numpy as np
import espressomd

system = espressomd.System(box_l=[10,10,10])
system.time_step = 0.002
system.cell_system.skin = 0.4

chk = np.load('equilibrated.npz')
v_np = chk['v']

p = system.part.add(pos=[0,0,0], v=v_np[0])
print("ESPResSo p.v before:", p.v)
system.integrator.run(1)
print("ESPResSo p.v after:", p.v)
