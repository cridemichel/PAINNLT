import espressomd
import numpy as np

system = espressomd.System(box_l=[10.0, 10.0, 10.0])
system.time_step = 0.01
system.cell_system.skin = 0.4

system.part.add(pos=[5.0, 5.0, 5.0])
system.part.add(pos=[5.0 + 1.12246, 5.0, 5.0]) # Exactly at cutoff 2^(1/6)

system.non_bonded_inter[0, 0].lennard_jones.set_params(
    epsilon=1.0, sigma=1.0, cutoff=1.122462, shift=0.0
)
print("Shift=0.0 Energy at cutoff:", system.analysis.energy()['total'])

system.non_bonded_inter[0, 0].lennard_jones.set_params(
    epsilon=1.0, sigma=1.0, cutoff=1.122462, shift="auto"
)
print("Shift='auto' Energy at cutoff:", system.analysis.energy()['total'])
