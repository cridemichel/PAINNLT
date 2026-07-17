import espressomd
import espressomd.interactions
system = espressomd.System(box_l=[10, 10, 10])
system.time_step = 0.01
system.cell_system.skin = 0.4
p0 = system.part.add(pos=[5,5,5])
p1 = system.part.add(pos=[5,5,6])
hb = espressomd.interactions.HarmonicBond(k=1000000, r_0=2.0)
system.bonded_inter.add(hb)
p0.add_bond((hb, p1))
system.force_cap = 10.0
system.integrator.run(0)
print(f"Force on p0: {p0.f}")
