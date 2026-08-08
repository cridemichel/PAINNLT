import espressomd
import espressomd.painn
import json
import argparse
import numpy as np
import struct
import os

from framework_utils import (
    ensure_single_rank,
    get_rb_data_by_sites,
    input_hashes,
    rigid_body_quaternion,
    save_checkpoint,
    validate_model_manifest,
    validate_wca_exclusion_policy,
    wca_topology_exclusion_pairs,
)

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
parser.add_argument("--steps_md", type=int, default=2000, help="Number of steps for Phase 2 classical MD warmup")
parser.add_argument("--steps_ml_capped", type=int, default=2000, help="ML warmup steps with an ESPResSo force cap")
parser.add_argument("--steps_ml_uncapped", type=int, default=2000, help="Final NVT steps with the production Hamiltonian and no force cap")
parser.add_argument("--warmup_chunk", type=int, default=100, help="Progress-reporting chunk size")

parser.add_argument("--toxvaerd_alpha", type=float, default=None, help="Override the value stored in the model config")
parser.add_argument("--allow_missing_model_manifest", action="store_true", help="Allow legacy .pt files without the patched training manifest")
parser.add_argument("--allow_unsafe_mpi", action="store_true", help="Allow the uncertified multi-rank PaiNN path")
args = parser.parse_args()
print("[INFO] Loading configurations...")
with open(args.config, "r") as f:
    nn_config = json.load(f)
with open(args.priors, "r") as f:
    priors = json.load(f)
with open(args.rb_info, "r") as f:
    rb_info = json.load(f)

if args.toxvaerd_alpha is None:
    args.toxvaerd_alpha = float(nn_config.get("toxvaerd_alpha", 0.1))
runtime_nn_config = dict(nn_config)
runtime_nn_config["toxvaerd_alpha"] = float(args.toxvaerd_alpha)
validate_model_manifest(
    args.model,
    runtime_nn_config,
    allow_missing=args.allow_missing_model_manifest,
)

if args.dt <= 0:
    raise ValueError("--dt must be positive")
if min(args.steps_sd, args.steps_md, args.steps_ml_capped, args.steps_ml_uncapped) < 0:
    raise ValueError("equilibration step counts must be non-negative")
if args.warmup_chunk <= 0:
    raise ValueError("--warmup_chunk must be positive")

# The dummy particle type for COMs should be higher than the max ML species type
DUMMY_COM_TYPE = nn_config["num_species"] + 1

# Setup ESPResSo System
print("[INFO] Initializing ESPResSo system...")
# ESPResSo requires an initial box; it is replaced immediately by the dataset box.
system = espressomd.System(box_l=[10.0, 10.0, 10.0])
system.cell_system.skin = 0.4
system.time_step = float(args.dt)
ensure_single_rank(system, allow_unsafe_mpi=args.allow_unsafe_mpi)
# Set temperature using the provided kT argument
# Thermostat is OFF initially because Steepest Descent does not support it

print("[INFO] Reading initial frame from dataset...")
mol_com_parts = {}
mol_vs_parts = {}

with open(args.dataset, "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_molecules = struct.unpack("i", f.read(4))[0]
    num_total_sites = struct.unpack("i", f.read(4))[0]
    box_dim = struct.unpack("3f", f.read(12))
    
    # Use exactly the same periodic box as preprocessing and production.
    system.box_l = [float(b) for b in box_dim]
    required_length = 2.0 * (float(nn_config.get("cutoff", 0.0)) + system.cell_system.skin)
    if min(system.box_l) <= required_length:
        raise ValueError(
            f"Dataset box {list(system.box_l)} is too small for cutoff+skin; "
            f"each dimension must exceed {required_length:.6g} nm"
        )
    
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
            
        resname, rb_data = get_rb_data_by_sites(site_types, rb_info)
        mass = rb_data["mass_amu"]
        inertia = rb_data["inertia_amu_nm2"]
        body_quat = rigid_body_quaternion(center, site_positions, box_dim, rb_data)
        
        p_com = system.part.add(
            pos=center, type=DUMMY_COM_TYPE,
            mass=mass, rinertia=inertia, quat=body_quat,
            rotation=[True, True, True] if num_sites > 1 else [False, False, False],
            mol_id=mol_idx
        )
        mol_com_parts[mol_idx] = p_com.id
        
        for site_idx, (stype, spos) in enumerate(zip(site_types, site_positions)):
            # Virtual sites must have near-zero mass/inertia to not inflate the total system mass.
            # ESPResSo requires mass > 0, so we use 1e-5.
            p_vs = system.part.add(pos=spos, type=stype, mass=1e-5, rinertia=[1e-5, 1e-5, 1e-5], mol_id=mol_idx)
            p_vs.virtual = True
            p_vs.vs_auto_relate_to(p_com.id)
            p_vs.gamma = 0.0
            p_vs.gamma_rot = 0.0
            mol_vs_parts[(mol_idx, site_idx)] = p_vs.id

print("[INFO] Setting up WCA exclusions (intra-rigid-body + 1-2/1-3)...")
mol_to_vs = {}
for (m_idx, _site_idx), pid in mol_vs_parts.items():
    mol_to_vs.setdefault(m_idx, []).append(pid)

for pids in mol_to_vs.values():
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            system.part.by_id(pids[i]).add_exclusion(pids[j])

validate_wca_exclusion_policy(priors)
wca_direct_pairs, wca_one_three_pairs = wca_topology_exclusion_pairs(priors, num_molecules)
for mol_i, mol_j in sorted(wca_direct_pairs | wca_one_three_pairs):
    for pid_i in mol_to_vs.get(mol_i, []):
        for pid_j in mol_to_vs.get(mol_j, []):
            system.part.by_id(pid_i).add_exclusion(pid_j)
print(
    f"[INFO] WCA topology exclusions active: {len(wca_direct_pairs)} 1-2 pairs, "
    f"{len(wca_one_three_pairs)} 1-3 pairs."
)


print("[INFO] Adding priors...")
# WCA from cg_priors.json (Unified truth)
import json
import os

cg_priors_path = args.priors
if os.path.exists(cg_priors_path):
    print(f"[INFO] Loading unified WCA priors from {cg_priors_path}")
    with open(cg_priors_path, "r") as f:
        cg_priors = json.load(f)
        
    wca_dict = cg_priors.get("wca_pairs", {})
    for pair_key, wca_info in wca_dict.items():
        type_i = wca_info["type_i"]
        type_j = wca_info["type_j"]
        sig = wca_info["sigma_nm"]
        eps = wca_info["epsilon_kjmol"]
        cut = wca_info["cutoff_nm"]
        
        system.non_bonded_inter[type_i, type_j].lennard_jones.set_params(
            epsilon=eps, sigma=sig,
            cutoff=cut, shift="auto"
        )
else:
    print(f"[WARNING] {cg_priors_path} not found! No WCA will be applied.")

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

# Zero-strength interactions make the ML cutoff visible to ESPResSo's
# neighbor-list machinery for every particle-type pair.
ml_cutoff = float(nn_config.get("cutoff", 5.0))
for i in range(nn_config["num_species"] + 2):
    for j in range(i, nn_config["num_species"] + 2):
        system.non_bonded_inter[i, j].soft_sphere.set_params(
            a=0.0, n=1, cutoff=ml_cutoff, offset=0.0
        )


def run_chunks(total_steps, chunk_size, phase_name, after_chunk=None):
    completed = 0
    while completed < total_steps:
        current = min(chunk_size, total_steps - completed)
        system.integrator.run(current)
        completed += current
        if after_chunk is not None:
            after_chunk(completed, total_steps)
        print(
            f"\r[INFO] {phase_name} Progress: {completed}/{total_steps} steps",
            end="",
            flush=True,
        )
    if total_steps:
        print(flush=True)


if args.steps_sd > 0:
    print("[INFO] Phase 1: Steepest Descent with classical potentials...")
    system.integrator.set_steepest_descent(
        f_max=10000.0, gamma=50.0, max_displacement=0.001
    )
    run_chunks(args.steps_sd, args.warmup_chunk, "Phase 1")
else:
    print("[INFO] Phase 1 skipped (--steps_sd 0).")

system.integrator.set_vv()
system.time_step = args.dt

if args.steps_md > 0:
    print("[INFO] Phase 2: Classical NVT warmup with force cap...", flush=True)
    system.force_cap = 500.0
    system.thermostat.set_langevin(kT=args.kT, gamma=50.0, gamma_rot=50.0, seed=42)

    def update_classical_cap(completed, total):
        fraction = completed / max(total, 1)
        system.force_cap = 500.0 + 500.0 * fraction

    run_chunks(
        args.steps_md,
        args.warmup_chunk,
        "Phase 2",
        after_chunk=update_classical_cap,
    )
else:
    print("[INFO] Phase 2 skipped (--steps_md 0).")

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
    device=args.device,
)

system.integrator.set_vv()
system.time_step = args.dt

if args.steps_ml_capped > 0:
    print("[INFO] Phase 3: ML NVT warmup with force cap...", flush=True)
    system.force_cap = 500.0
    system.thermostat.set_langevin(kT=args.kT, gamma=50.0, gamma_rot=50.0, seed=42)

    def update_ml_cap(completed, total):
        fraction = completed / max(total, 1)
        system.force_cap = 500.0 + 1000.0 * fraction

    run_chunks(
        args.steps_ml_capped,
        args.warmup_chunk,
        "Phase 3",
        after_chunk=update_ml_cap,
    )
else:
    print("[INFO] Phase 3 skipped (--steps_ml_capped 0).")

system.force_cap = 0
system.thermostat.set_langevin(kT=args.kT, gamma=1.0, gamma_rot=1.0, seed=42)

if args.steps_ml_uncapped > 0:
    print("[INFO] Phase 4: Final uncapped ML NVT equilibration...", flush=True)
    run_chunks(args.steps_ml_uncapped, args.warmup_chunk, "Phase 4")
else:
    print("[INFO] Phase 4 skipped (--steps_ml_uncapped 0).")

print("[INFO] Warm-up terminato. Preparazione del salvataggio...")

print(f"[INFO] Saving equilibrated state to {args.out_checkpoint}...")
system.integrator.run(0, recalc_forces=True)
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
hashes = input_hashes(
    dataset=args.dataset,
    config=args.config,
    priors=args.priors,
    rb_info=args.rb_info,
    model=args.model,
)
save_checkpoint(
    args.out_checkpoint,
    system=system,
    pos=np.array(pos),
    vel=np.array(vel),
    quat=np.array(quat),
    omega=np.array(omega),
    hashes=hashes,
    config=runtime_nn_config,
    dt=args.dt,
    kT=args.kT,
)

print("[INFO] Equilibration finished successfully.")

# Force immediate exit to bypass PyTorch/MPI teardown crashes on macOS
os._exit(0)
