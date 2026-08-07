import espressomd
import numpy as np

# Inizializza il sistema nudo crudo
system = espressomd.System(box_l=[10, 10, 10])
system.time_step = 0.002
system.cell_system.skin = 0.4

# Crea una particella e guarda la forza prima e dopo il cap
p = system.part.add(pos=[5,5,5], v=[1,1,1])
print("F before run(0):", p.f)
system.integrator.run(0)
print("F after run(0):", p.f)
