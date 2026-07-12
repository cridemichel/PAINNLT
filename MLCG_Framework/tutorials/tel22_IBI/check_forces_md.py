import espressomd
import espressomd.interactions
import json
import numpy as np

# Load priors
with open('cg_priors_tmp_ibi.json', 'r') as f:
    priors = json.load(f)

# Set up system
system = espressomd.System(box_l=[10.0, 10.0, 10.0]) # Generic box for isolated molecule
system.time_step = 0.002
system.cell_system.skin = 0.4

# Load initial positions
initial_pos = np.load('_tmp_initial_pos.npy')
num_particles = len(initial_pos)

# Setup particles with realistic initial coordinates
for i in range(num_particles):
    system.part.add(id=i, pos=initial_pos[i], type=0)

# WCA Non-Bonded interactions
wca = priors.get("wca", {})
if wca.get("epsilon", 0.0) > 0 and wca.get("sigma", 0.0) > 0:
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            system.non_bonded_inter[0, 0].lennard_jones.set_params(
                epsilon=wca["epsilon"], sigma=wca["sigma"],
                cutoff=wca["sigma"] * (2.0**(1/6)), shift="auto"
            )

# Apply interactions
system.force_cap = 0.0

for b in priors.get("bonds", []):
    if b["type"] == "tabulated":
        data = np.loadtxt(b["file"])
        tb = espressomd.interactions.TabulatedDistance(
            min=data[0, 0], max=data[-1, 0], 
            energy=data[:, 1], force=data[:, 2]
        )
        system.bonded_inter.add(tb)
        system.part.by_id(b["mol_i"]).add_bond((tb, b["mol_j"]))
    elif b["type"] == "harmonic":
        hb = espressomd.interactions.HarmonicBond(k=b["k"], r_0=b["r0"])
        system.bonded_inter.add(hb)
        system.part.by_id(b["mol_i"]).add_bond((hb, b["mol_j"]))

for a in priors.get("angles", []):
    if a.get("type", "harmonic") == "harmonic":
        ha = espressomd.interactions.AngleHarmonic(bend=a["k"], phi0=a["theta0"])
        system.bonded_inter.add(ha)
        system.part.by_id(a["mol_j"]).add_bond((ha, a["mol_i"], a["mol_k"]))
    elif a.get("type") == "tabulated":
        data = np.loadtxt(a["file"])
        ta = espressomd.interactions.TabulatedAngle(
            min=data[0, 0], max=data[-1, 0],
            energy=data[:, 1], force=data[:, 2]
        )
        system.bonded_inter.add(ta)
        system.part.by_id(a["mol_j"]).add_bond((ta, a["mol_i"], a["mol_k"]))

for d in priors.get("dihedrals", []):
    if d.get("type", "cosine") == "cosine":
        cd = espressomd.interactions.Dihedral(bend=d["k"], mult=d.get("n", 1), phase=d["phi0"])
        system.bonded_inter.add(cd)
        system.part.by_id(d["mol_j"]).add_bond((cd, d["mol_i"], d["mol_k"], d["mol_l"]))
    elif d.get("type") == "tabulated":
        data = np.loadtxt(d["file"])
        td = espressomd.interactions.TabulatedDihedral(
            min=data[0, 0], max=data[-1, 0],
            energy=data[:, 1], force=data[:, 2]
        )
        system.bonded_inter.add(td)
        system.part.by_id(d["mol_j"]).add_bond((td, d["mol_i"], d["mol_k"], d["mol_l"]))

# Minimize energy
print("Minimizing energy...")
system.integrator.set_steepest_descent(f_max=10.0, gamma=10.0, max_displacement=0.01)

system.integrator.set_vv()

# Thermostat (must be set after steepest descent)
system.thermostat.set_langevin(kT=2.49, gamma=1.0, seed=42)

# Run MD and save trajectory
import builtins
print("Running MD...")
# Burn-in

# system.force_cap = 0  # Leave it capped just in case


positions = []
for _ in range(5000):
    system.integrator.run(10)
    pos = []
    for p in system.part:
        pos.append(p.pos)
    positions.append(pos)

np.save('_tmp_traj.npy', np.array(positions))

system.integrator.run(0)
forces = system.part.all().f
for i, f in enumerate(forces):
    mag = np.linalg.norm(f)
    if mag > 500:
        print(f"Particle {i} has HUGE force: {mag:.2f}  vector: {f}")

f164 = system.part.by_id(164).f
f165 = system.part.by_id(165).f
print(f"Force on 164: {np.linalg.norm(f164):.2f}")
print(f"Force on 165: {np.linalg.norm(f165):.2f}")
