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
parser.add_argument("--steps", type=int, default=10000, help="Simulation steps")
parser.add_argument("--device", type=str, default="auto", help="Device for ML (cpu/mps/cuda)")
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
# Set room temperature: 300 K * 0.008314 kJ/(mol*K) = 2.49 kJ/mol
system.thermostat.set_langevin(kT=2.49, gamma=10.0, seed=42)

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
    
    system.box_l = box_dim
    
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
            p_vs = system.part.add(pos=spos, type=stype)
            p_vs.virtual = True
            p_vs.vs_auto_relate_to(p_com.id)
            mol_vs_parts[(mol_idx, site_idx)] = p_vs.id

print("[INFO] Adding priors...")
# WCA
wca = priors.get("wca", {})
if wca.get("epsilon", 0.0) > 0 and wca.get("sigma", 0.0) > 0:
    for i in range(nn_config["num_species"]):
        for j in range(i, nn_config["num_species"]):
            system.non_bonded_inter[i, j].lennard_jones.set_params(
                epsilon=wca["epsilon"], sigma=wca["sigma"],
                cutoff=wca["sigma"] * (2.0**(1/6)), shift="auto"
            )

# Bonds (Harmonic, FENE, Morse)
for idx, b in enumerate(priors.get("bonds", [])):
    b_type = b.get("type", "harmonic")
    if b_type == "harmonic":
        bond = espressomd.interactions.HarmonicBond(k=b["k"], r_0=b["r0"])
    elif b_type == "fene":
        bond = espressomd.interactions.FeneBond(k=b["k"], d_r_max=b["r_max"], r_0=b["r0"])
    elif b_type == "morse":
        # ESPResSo does not have a native MorseBond, so we create a tabulated bond
        import numpy as np
        rmin_tab = 0.01
        rmax_tab = 15.0 # Extend up to 15 nm (larger than the box) so it never crashes!
        r_vals = np.linspace(rmin_tab, rmax_tab, 5000)
        
        diff = r_vals - b["r0"]
        exp_term = np.exp(-b["a"] * diff)
        
        energy = b["D"] * (1.0 - exp_term)**2
        force = -2.0 * b["a"] * b["D"] * (1.0 - exp_term) * exp_term
        
        # Cap forces to avoid integrator explosions near the steep repulsive wall
        force = np.clip(force, -10000.0, 10000.0)
        
        bond = espressomd.interactions.TabulatedDistance(
            min=rmin_tab, max=rmax_tab, energy=energy, force=force
        )
    elif b_type == "tabulated":
        data = np.loadtxt(b["file"])
        rmin_tab = float(b["min"])
        rmax_tab = float(b["max"])
        bond = espressomd.interactions.TabulatedDistance(
            min=rmin_tab, max=rmax_tab, energy=data[:, 1], force=data[:, 2]
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
    
    # Assumiamo che angoli e diedri agiscano sui Centri di Massa (COM) per ora
    p1 = mol_com_parts[mol_i]
    p2 = mol_com_parts[mol_j]
    p3 = mol_com_parts[mol_k]
    
    # In ESPResSo, l'angolo si applica alla particella CENTRALE
    system.part.by_id(p2).add_bond((angle, p1, p3))
    print(f"[INFO] Added Angle bond {idx}: mol {mol_i} - {mol_j} - {mol_k}")

# Dihedrals
for idx, d in enumerate(priors.get("dihedrals", [])):
    k_dih = d["k"]
    mult = d.get("n", 1)
    phase = d["phi0"]
    dihedral = espressomd.interactions.Dihedral(bend=k_dih, mult=mult, phase=phase)
    system.bonded_inter.add(dihedral)
    
    mol_i, mol_j, mol_k, mol_l = d["mol_i"], d["mol_j"], d["mol_k"], d["mol_l"]
    
    p1 = mol_com_parts[mol_i]
    p2 = mol_com_parts[mol_j]
    p3 = mol_com_parts[mol_k]
    p4 = mol_com_parts[mol_l]
    
    # In ESPResSo, il diedro si applica alla SECONDA particella
    system.part.by_id(p2).add_bond((dihedral, p1, p3, p4))
    print(f"[INFO] Added Dihedral bond {idx}: mol {mol_i} - {mol_j} - {mol_k} - {mol_l}")

print("[INFO] Activating ML Potential...")
espressomd.painn.activate_painn_potential(
    model_path=args.model,
    num_species=nn_config["num_species"],
    hidden_channels=nn_config["hidden_channels"],
    n_layers=nn_config["n_layers"],
    num_rbf=nn_config["num_rbf"],
    cutoff=nn_config["cutoff"],
    device=args.device
)

import espressomd.io.writer.vtf

print(f"[INFO] Running {args.steps} integration steps...")
with open("energy.csv", "w") as f_out:
    f_out.write("Step,E_tot,E_kin\n")
vtf_filename = "cg_trajectory.vtf"
with open(vtf_filename, "w") as vtf_file:
    espressomd.io.writer.vtf.writevsf(system, vtf_file)
    
    # Inject fake visual bonds connecting COM to its Virtual Sites
    # This allows VMD `pbc unwrap` to treat the whole nucleotide as a single fragment
    for mol_idx, com_id in mol_com_parts.items():
        for (m_idx, s_idx), vs_id in mol_vs_parts.items():
            if m_idx == mol_idx:
                vtf_file.write(f"bond {com_id}:{vs_id}\n")
    
    chunk_size = 100
    num_chunks = args.steps // chunk_size
    for step in range(num_chunks):
        system.integrator.run(chunk_size)
        espressomd.io.writer.vtf.writevcf(system, vtf_file)
        
        energy = system.analysis.energy()
        step_val = step * chunk_size
        e_tot = energy['total']
        e_kin = energy['kinetic']
        print(f"\r[INFO] Step {step_val}/{args.steps} | E_tot: {e_tot:.2f} | E_kin: {e_kin:.2f}", end="")
        
        # Salviamo le energie in un file CSV per poterle graficare e controllare il surriscaldamento
        with open("energy.csv", "a") as f_out:
            f_out.write(f"{step_val},{e_tot:.4f},{e_kin:.4f}\n")

print("\n[INFO] Simulation finished successfully.")

# Force immediate exit to bypass PyTorch/MPI teardown crashes on macOS
os._exit(0)
