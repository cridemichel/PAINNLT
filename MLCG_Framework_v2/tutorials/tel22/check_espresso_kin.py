import espressomd
import numpy as np

system = espressomd.System(box_l=[11.0, 11.0, 11.0])
system.time_step = 0.002
system.cell_system.skin = 0.4

# Create 1 particle to test
system.part.add(pos=[0,0,0], v=[1.0, 1.0, 1.0], mass=2.0)

print("Before run:")
print("v:", system.part.by_id(0).v)
print("E_kin:", system.analysis.energy()["kinetic"])

system.integrator.run(1)

print("After run:")
print("v:", system.part.by_id(0).v)
print("E_kin:", system.analysis.energy()["kinetic"])

