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
types = [0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7, 1, 1, 0, 7, 7, 7]

# Setup particles with realistic initial coordinates
for i in range(num_particles):
    system.part.add(id=i, pos=initial_pos[i], type=int(types[i]))

# WCA Exclusions (1-2 and 1-3)
wca_exclusions = set()
for b in priors.get("bonds", []):
    m1, m2 = min(b["mol_i"], b["mol_j"]), max(b["mol_i"], b["mol_j"])
    wca_exclusions.add((m1, m2))
for a in priors.get("angles", []):
    m1, m2 = min(a["mol_i"], a["mol_k"]), max(a["mol_i"], a["mol_k"])
    wca_exclusions.add((m1, m2))
for (m1, m2) in wca_exclusions:
    system.part.by_id(m1).add_exclusion(m2)

# WCA Non-Bonded interactions
wca = priors.get("wca", {})
has_wca = wca.get("sigma", 0.0) > 0 or len(wca.get("overrides", {})) > 0
if wca.get("epsilon", 0.0) > 0 and has_wca:
    wca_sigma = wca.get("sigma", 0.3)
    wca_eps = wca.get("epsilon", 1.0)
    overrides = wca.get("overrides", {})
    unique_types = set(int(t) for t in types)
    for t_i in unique_types:
        sigma_i = overrides.get(str(t_i), {}).get("sigma", wca_sigma)
        eps_i = overrides.get(str(t_i), {}).get("epsilon", wca_eps)
        for t_j in unique_types:
            sigma_j = overrides.get(str(t_j), {}).get("sigma", wca_sigma)
            eps_j = overrides.get(str(t_j), {}).get("epsilon", wca_eps)
            
            sig = 0.5 * (sigma_i + sigma_j)
            eps = np.sqrt(eps_i * eps_j)
            system.non_bonded_inter[t_i, t_j].lennard_jones.set_params(
                epsilon=eps, sigma=sig,
                cutoff=sig * (2.0**(1/6)), shift="auto"
            )

# Apply interactions
system.force_cap = 2000.0

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
            energy=data[:, 1], force=data[:, 2]
        )
        system.bonded_inter.add(td)
        system.part.by_id(d["mol_j"]).add_bond((td, d["mol_i"], d["mol_k"], d["mol_l"]))

# Debug prints
print("Distance 164-165 START:", np.linalg.norm(system.part.by_id(164).pos - system.part.by_id(165).pos))

# Check initial forces
system.integrator.run(0)
print("--- INITIAL ENERGY ---")
print(system.analysis.energy())
forces = system.part.all().f
print("--- INITIAL FORCES (BEFORE SD) ---")
for i, f in enumerate(forces):
    if np.linalg.norm(f) > 500:
        print(f"HIGH FORCE START: Particle {i} f={f} mag={np.linalg.norm(f):.2f}")

# Minimize energy / Burn-in
print("Gentle MD burn-in (Steepest Descent)...")
system.integrator.set_steepest_descent(f_max=1000.0, gamma=50.0, max_displacement=0.001)
system.integrator.run(3000)

print("--- FORCES AFTER SD ---")
forces = system.part.all().f
for i, f in enumerate(forces):
    if np.linalg.norm(f) > 500:
        print(f"HIGH FORCE AFTER SD: Particle {i} f={f} mag={np.linalg.norm(f):.2f}")

print("Phase 2: Warm-up MD with small timestep and high friction...")
system.integrator.set_vv()
system.thermostat.set_langevin(kT=2.49, gamma=50.0, seed=42)
system.force_cap = 0.0
system.time_step = 0.0001

for _ in range(50):
    system.integrator.run(100)

print("Phase 3: Production MD...")
system.force_cap = 0.0
system.thermostat.set_langevin(kT=2.49, gamma=50.0, seed=42)
system.time_step = 0.002
system.time_step = 0.002

system.integrator.run(100000)

print("Distance 164-165 AFTER SD:", np.linalg.norm(system.part.by_id(164).pos - system.part.by_id(165).pos), flush=True)

# Run MD and save trajectory
import builtins
print("Running MD...")
# system.force_cap = 0  # Leave it capped just in case


positions = []
for _ in range(5000):
    system.integrator.run(10)
    pos = []
    for p in system.part:
        pos.append(p.pos)
    positions.append(pos)

np.save('_tmp_traj.npy', np.array(positions))
