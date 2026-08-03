import espressomd
import numpy as np

system = espressomd.System(box_l=[11.0, 11.0, 11.0])
system.time_step = 0.002
system.cell_system.skin = 0.4
p0 = system.part.add(pos=[0,0,0], mass=2.0)
p1 = system.part.add(pos=[1,0,0], mass=1.0)
p1.vs_auto_relate_to(p0)

p0.v = [1,1,1]
print("p0.v:", p0.v)
print("p1.v:", p1.v)
system.integrator.run(0)
print("After run(0) p0.v:", p0.v)

