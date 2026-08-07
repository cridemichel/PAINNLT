import numpy as np
import espressomd

system = espressomd.System(box_l=[10,10,10])
system.time_step = 0.002
system.cell_system.skin = 0.4

# just 1 particle with mass 250
p = system.part.add(pos=[0,0,0], v=[1, 1, 1], mass=250.0)

# Add a force
p.ext_force = [1000000, 1000000, 1000000]

print("v before:", p.v)
e_kin = 0.5 * 250 * sum(p.v**2)
print("E_kin before:", e_kin)

system.integrator.run(1)

print("v after 1:", p.v)
e_kin = 0.5 * 250 * sum(p.v**2)
print("E_kin after 1:", e_kin)

system.integrator.run(1)

print("v after 2:", p.v)
e_kin = 0.5 * 250 * sum(p.v**2)
print("E_kin after 2:", e_kin)

