import espressomd
import espressomd.painn
import json
import argparse
import numpy as np
import struct
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Trained ML potential (.pt)")
parser.add_argument("--config", type=str, required=True, help="NN config JSON")
parser.add_argument("--priors", type=str, required=True, help="cg_priors.json")
parser.add_argument("--rb_info", type=str, required=True, help="rigid_bodies_info.json")
parser.add_argument("--dataset", type=str, required=True, help="Dataset to get initial frame from (e.g. cg_dataset.bin)")
parser.add_argument("--dt", type=float, default=0.002, help="Time step (ps)")
parser.add_argument("--out_checkpoint", type=str, default="equilibrated.npz", help="Output checkpoint file")
parser.add_argument("--device", type=str, default="auto", help="Device for ML (cpu/mps/cuda)")
parser.add_argument("--kT", type=float, default=2.49, help="Simulation temperature in kJ/mol (default 2.49 for 300K)")
parser.add_argument("--steps_sd", type=int, default=5000, help="Number of steps for Phase 1 Steepest Descent (default 5000)")
parser.add_argument("--steps_md", type=int, default=2000, help="Number of steps for Phase 2 Classical MD Warmup (default 2000)")

parser.add_argument("--toxvaerd_alpha", type=float, default=0.1, help="Toxvaerd smoothing dimensionless parameter")
args = parser.parse_args()
print("[INFO] Loading configurations...")
with open(args.config, "r") as f:
    nn_config = json.load(f)
with open(args.priors, "r") as f:
    priors = json.load(f)
with open(args.rb_info, "r") as f:
    rb_info = json.load(f)

# The dummy particle type for COMs should be higher than the max ML species type
DUMMY_COM_TYPE = nn_config["num_species"] + 1

# Setup ESPResSo System
print("[INFO] Initializing ESPResSo system...")
# For a real run, box_l should be read from the first frame of the dataset or config.
# Here we just set a large box and will resize it if needed.
system = espressomd.System(box_l=[10.0, 10.0, 10.0])
system.time_step = args.dt
system.cell_system.skin = 0.4
# Set temperature using the provided kT argument
# Thermostat is OFF initially because Steepest Descent does not support it

def get_rb_data_by_sites(site_types, rb_info):
    for resname, data in rb_info.items():
        expected_types = [site["type"] for site in data["sites"].values()]
        if sorted(expected_types) == sorted(site_types):
            return data
    raise ValueError(f"Unknown site types {site_types}")

print("[INFO] Reading initial frame from dataset...")
mol_com_parts = {}
mol_vs_parts = {}

with open(args.dataset, "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_molecules = struct.unpack("i", f.read(4))[0]
    num_total_sites = struct.unpack("i", f.read(4))[0]
    box_dim = struct.unpack("3f", f.read(12))
    
    # Ensure box is large enough for cutoff=5.0 + skin=0.4
    min_box = 11.0
    system.box_l = [max(b, min_box) for b in box_dim]
    
    for mol_idx in range(num_molecules):
        mol_id = struct.unpack("i", f.read(4))[0]
        num_sites = struct.unpack("i", f.read(4))[0]
        center = struct.unpack("3f", f.read(12))
        force = struct.unpack("3f", f.read(12))
        torque = struct.unpack("3f", f.read(12))
        
        site_types = []
        site_positions = []
        for s in range(num_sites):
            stype = struct.unpack("i", f.read(4))[0]
            spos = struct.unpack("3f", f.read(12))
            site_types.append(stype)
            site_positions.append(spos)
            
        rb_data = get_rb_data_by_sites(site_types, rb_info)
        mass = rb_data["mass_amu"]
        inertia = rb_data["inertia_amu_nm2"]
        
        p_com = system.part.add(
            pos=center, type=DUMMY_COM_TYPE,
            mass=mass, rinertia=inertia,
            rotation=[True, True, True]
        )
        mol_com_parts[mol_idx] = p_com.id
        
        for site_idx, (stype, spos) in enumerate(zip(site_types, site_positions)):
            # Virtual sites must have near-zero mass/inertia to not inflate the total system mass.
            # ESPResSo requires mass > 0, so we use 1e-5.
            p_vs = system.part.add(pos=spos, type=stype, mass=1e-5, rinertia=[1e-5, 1e-5, 1e-5])
            p_vs.virtual = True
            p_vs.vs_auto_relate_to(p_com.id)
            mol_vs_parts[(mol_idx, site_idx)] = p_vs.id

print("[INFO] Setting up WCA exclusions (1-2 and 1-3)...")
wca_exclusions = set()
for b in priors.get("bonds", []):
    m1, m2 = min(b["mol_i"], b["mol_j"]), max(b["mol_i"], b["mol_j"])
    wca_exclusions.add((m1, m2))
for a in priors.get("angles", []):
    m1, m2 = min(a["mol_i"], a["mol_k"]), max(a["mol_i"], a["mol_k"])
    wca_exclusions.add((m1, m2))

for (m1, m2) in wca_exclusions:
    m1_parts = [mol_com_parts[m1]] + [pid for (m, s), pid in mol_vs_parts.items() if m == m1]
    m2_parts = [mol_com_parts[m2]] + [pid for (m, s), pid in mol_vs_parts.items() if m == m2]
    for p1 in m1_parts:
        for p2 in m2_parts:
            system.part.by_id(p1).add_exclusion(p2)

print("[INFO] Adding priors...")
# WCA
import math
wca = priors.get("wca", {})
has_wca = wca.get("sigma", 0.0) > 0 or len(wca.get("overrides", {})) > 0
if wca.get("epsilon", 0.0) > 0 and has_wca:
    for i in range(nn_config["num_species"]):
        sigma_i = wca.get("overrides", {}).get(str(i), {}).get("sigma", wca["sigma"])
        eps_i = wca.get("overrides", {}).get(str(i), {}).get("epsilon", wca["epsilon"])
        for j in range(i, nn_config["num_species"]):
            sigma_j = wca.get("overrides", {}).get(str(j), {}).get("sigma", wca["sigma"])
            eps_j = wca.get("overrides", {}).get(str(j), {}).get("epsilon", wca["epsilon"])
            
            sigma_mix = (sigma_i + sigma_j) / 2.0
            eps_mix = math.sqrt(eps_i * eps_j)
            
            system.non_bonded_inter[i, j].lennard_jones.set_params(
                epsilon=eps_mix, sigma=sigma_mix,
                cutoff=sigma_mix * (2.0**(1/6)), shift="auto"
            )

# Add safety hard-core WCA between all COMs to prevent collapse (r < 0.112 nm)
system.non_bonded_inter[DUMMY_COM_TYPE, DUMMY_COM_TYPE].lennard_jones.set_params(
    epsilon=100.0, sigma=0.1,
    cutoff=0.1 * (2.0**(1/6)), shift="auto"
)

# Bonds (Harmonic, FENE, Morse)
for idx, b in enumerate(priors.get("bonds", [])):
    b_type = b.get("type", "harmonic")
    if b_type == "harmonic":
        bond = espressomd.interactions.HarmonicBond(k=b["k"], r_0=b["r0"])
    elif b_type == "fene":
        bond = espressomd.interactions.FeneBond(k=b["k"], d_r_max=b["r_max"], r_0=b["r0"])
    elif b_type == "morse":
        # We model Morse as tabulated to allow large r without breaking FENE limits
        rmin_tab = 0.001
        rmax_tab = 15.0 # Extend up to 15 nm (larger than the box) so it never crashes!
        r_vals = np.linspace(rmin_tab, rmax_tab, 5000)
        
        diff = r_vals - b["r0"]
        exp_term = np.exp(-b["a"] * diff)
        
        energy = b["D"] * (1.0 - exp_term)**2
        force = -2.0 * b["a"] * b["D"] * (1.0 - exp_term) * exp_term
        
        bond = espressomd.interactions.TabulatedDistance(
            min=rmin_tab, max=rmax_tab, energy=energy, force=force
        )
    elif b_type == "tabulated":
        data = np.loadtxt(b["file"])
        rmin_tab = float(b["min"])
        rmax_tab = float(b["max"])
        r_vals = data[:, 0]
        energy = data[:, 1]
        force = data[:, 2]
        
        bond = espressomd.interactions.TabulatedDistance(
            min=rmin_tab, max=rmax_tab, energy=energy, force=force
        )
    else:
        print(f"[WARNING] Unknown bond type: {b_type}")
        continue
    
    system.bonded_inter.add(bond)
    print(f"[INFO] Added {b_type} bond {idx}: mol {b['mol_i']} <-> mol {b['mol_j']}")
    
    mol_i, mol_j = b["mol_i"], b["mol_j"]
    site_i, site_j = b.get("site_i", -1), b.get("site_j", -1)
    
    p1 = mol_com_parts[mol_i] if site_i == -1 else mol_vs_parts[(mol_i, site_i)]
    p2 = mol_com_parts[mol_j] if site_j == -1 else mol_vs_parts[(mol_j, site_j)]
    
    system.part.by_id(p1).add_bond((bond, p2))

# Angles
for idx, a in enumerate(priors.get("angles", [])):
    a_type = a.get("type", "harmonic")
    if a_type == "harmonic":
        k_bend = a["k"]
        phi0 = a["theta0"]
        angle = espressomd.interactions.AngleHarmonic(bend=k_bend, phi0=phi0)
    elif a_type == "tabulated":
        import numpy as np
        data = np.loadtxt(a["file"])
        min_tab = float(a["min"]) # Typically 0.0 radians
        max_tab = float(a["max"]) # Typically pi radians
        angle = espressomd.interactions.TabulatedAngle(
            min=min_tab, max=max_tab, energy=data[:, 1], force=data[:, 2]
        )
    else:
        print(f"[WARNING] Unknown angle type: {a_type}")
        continue
        
    system.bonded_inter.add(angle)
    
    mol_i, mol_j, mol_k = a["mol_i"], a["mol_j"], a["mol_k"]
    site_i, site_j, site_k = a.get("site_i", -1), a.get("site_j", -1), a.get("site_k", -1)
    
    p1 = mol_com_parts[mol_i] if site_i == -1 else mol_vs_parts[(mol_i, site_i)]
    p2 = mol_com_parts[mol_j] if site_j == -1 else mol_vs_parts[(mol_j, site_j)]
    p3 = mol_com_parts[mol_k] if site_k == -1 else mol_vs_parts[(mol_k, site_k)]
    
    # In ESPResSo, l'angolo si applica alla particella CENTRALE
    system.part.by_id(p2).add_bond((angle, p1, p3))
    print(f"[INFO] Added Angle bond {idx}: {mol_i}:{site_i} - {mol_j}:{site_j} - {mol_k}:{site_k}")

# Dihedrals
for idx, d in enumerate(priors.get("dihedrals", [])):
    d_type = d.get("type", "cosine")
    if d_type == "cosine":
        k_dih = d["k"]
        mult = d.get("n", 1)
        phase = d["phi0"]
        dihedral = espressomd.interactions.Dihedral(bend=k_dih, mult=mult, phase=phase)
    elif d_type == "tabulated":
        import numpy as np
        data = np.loadtxt(d["file"])
        min_tab = float(d.get("min", -np.pi))
        max_tab = float(d.get("max", np.pi))
        dihedral = espressomd.interactions.TabulatedDihedral(
            min=min_tab, max=max_tab, energy=data[:, 1], force=data[:, 2]
        )
    else:
        print(f"[WARNING] Unknown dihedral type: {d_type}")
        continue

    system.bonded_inter.add(dihedral)
    
    mol_i, mol_j, mol_k, mol_l = d["mol_i"], d["mol_j"], d["mol_k"], d["mol_l"]
    site_i, site_j, site_k, site_l = d.get("site_i", -1), d.get("site_j", -1), d.get("site_k", -1), d.get("site_l", -1)
    
    p1 = mol_com_parts[mol_i] if site_i == -1 else mol_vs_parts[(mol_i, site_i)]
    p2 = mol_com_parts[mol_j] if site_j == -1 else mol_vs_parts[(mol_j, site_j)]
    p3 = mol_com_parts[mol_k] if site_k == -1 else mol_vs_parts[(mol_k, site_k)]
    p4 = mol_com_parts[mol_l] if site_l == -1 else mol_vs_parts[(mol_l, site_l)]
    
    # In ESPResSo, il diedro si applica alla SECONDA particella
    system.part.by_id(p2).add_bond((dihedral, p1, p3, p4))
    print(f"[INFO] Added Dihedral bond {idx}: {mol_i}:{site_i} - {mol_j}:{site_j} - {mol_k}:{site_k} - {mol_l}:{site_l}")

print("[INFO] Setting up WCA non-bonded interactions to prevent particle collapse...")
import math
wca = priors.get("wca", {})
wca_sigma = wca.get("sigma", 0.3)
wca_eps = wca.get("epsilon", 1.0)
overrides = wca.get("overrides", {})

has_wca = wca_sigma > 0 or len(overrides) > 0
if wca_eps > 0 and has_wca:
    for i in range(nn_config["num_species"]):
        sigma_i = overrides.get(str(i), {}).get("sigma", wca_sigma)
        eps_i = overrides.get(str(i), {}).get("epsilon", wca_eps)
        for j in range(i, nn_config["num_species"]):
            sigma_j = overrides.get(str(j), {}).get("sigma", wca_sigma)
            eps_j = overrides.get(str(j), {}).get("epsilon", wca_eps)
            sigma_mix = (sigma_i + sigma_j) / 2.0
            eps_mix = math.sqrt(eps_i * eps_j)
            
            system.non_bonded_inter[1+i, 1+j].lennard_jones.set_params(
                epsilon=eps_mix, sigma=sigma_mix,
                cutoff=sigma_mix * (2.0**(1/6)), shift="auto"
            )

# Add dummy soft_sphere to ensure Verlet lists include all other types
for i in range(nn_config["num_species"] + 2):
    for j in range(i, nn_config["num_species"] + 2):
        if not (1 <= i <= nn_config["num_species"] and 1 <= j <= nn_config["num_species"]):
            system.non_bonded_inter[i, j].soft_sphere.set_params(
                a=0.0, n=1, cutoff=5.0, offset=0.0)

print("[INFO] Phase 1: Warmup with Steepest Descent (Classical Potentials Only)...")
system.integrator.set_steepest_descent(f_max=10000.0, gamma=50.0, max_displacement=0.001)
sd_loops = max(1, args.steps_sd // 100)
for step in range(sd_loops):
    system.integrator.run(100)
    print(f"\r[INFO] Phase 1 Progress: {(step+1)*100}/{args.steps_sd} steps", end="", flush=True)
print(flush=True)

system.integrator.set_vv()



print("[INFO] Phase 2: Warm-up con force-capping per rilassare i gradi di libertà rotazionali (Classical)...", flush=True)
system.force_cap = 500.0
system.thermostat.set_langevin(kT=args.kT, gamma=50.0, gamma_rot=50.0, seed=42)
system.time_step = 0.0001

md_loops = max(1, args.steps_md // 100)
for warmup_step in range(md_loops):
    system.integrator.run(100)
    system.force_cap = 500.0 + warmup_step * 2.5
    print(f"\r[INFO] Phase 2 Progress: {(warmup_step+1)*100}/{args.steps_md} steps", end="", flush=True)
print(flush=True)

system.force_cap = 0
system.thermostat.set_langevin(kT=args.kT, gamma=1.0, gamma_rot=1.0, seed=42)

print("[INFO] Activating ML Potential now that the system is physically relaxed...")
espressomd.painn.activate_painn_potential(
    model_path=args.model,
    num_species=nn_config["num_species"],
    hidden_channels=nn_config["hidden_channels"],
    n_layers=nn_config["n_layers"],
    num_rbf=nn_config["num_rbf"],
    cutoff=nn_config["cutoff"],

    toxvaerd_alpha=args.toxvaerd_alpha,
    device=args.device
)

# Skip Steepest Descent with ML potential to avoid unphysical rigid body rotations
system.integrator.set_vv()

print("[INFO] Phase 4: Final Warm-up with ML Potential to relax orientations...", flush=True)
system.force_cap = 500.0
system.thermostat.set_langevin(kT=args.kT, gamma=50.0, gamma_rot=50.0, seed=42)
system.time_step = 0.0001

for warmup_step in range(40):
    system.integrator.run(50)
    system.force_cap = 500.0 + warmup_step * 25.0
    print(f"\r[INFO] Phase 4 Progress: {(warmup_step+1)*50}/2000 steps", end="", flush=True)
print(flush=True)

system.force_cap = 0
system.thermostat.set_langevin(kT=args.kT, gamma=1.0, gamma_rot=1.0, seed=42)

print("[INFO] Warm-up terminato. Azzeramento delle velocità residue prima del salvataggio...")
for p in system.part:
    if p.mass > 1e-4:
        p.v = [0.0, 0.0, 0.0]
        p.omega_body = [0.0, 0.0, 0.0]

print(f"[INFO] Saving equilibrated state to {args.out_checkpoint}...")
pos = []
vel = []
quat = []
omega = []
# Ensure particles are saved in exact ID order
for i in range(len(system.part)):
    p = system.part.by_id(i)
    pos.append(p.pos)
    vel.append(p.v)
    quat.append(p.quat)
    # Virtual sites might not have omega_body properly exposed in all ESPResSo versions,
    # but COM particles definitely do. We save them for all and filter on load.
    try:
        omega.append(p.omega_body)
    except:
        omega.append([0.0, 0.0, 0.0])
np.savez(args.out_checkpoint, pos=np.array(pos), v=np.array(vel), quat=np.array(quat), omega=np.array(omega))

print("[INFO] Equilibration finished successfully.")

# Force immediate exit to bypass PyTorch/MPI teardown crashes on macOS
os._exit(0)
