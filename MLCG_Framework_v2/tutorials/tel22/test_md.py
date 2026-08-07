import espressomd
import espressomd.painn
import json
import argparse
import numpy as np
import struct
import os
from contextlib import ExitStack

from framework_utils import (
    ensure_single_rank,
    get_rb_data_by_sites,
    input_hashes,
    nonconservative_prior_entries,
    rigid_body_quaternion,
    validate_checkpoint,
    validate_model_manifest,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=False, default=None, help="Trained ML potential (.pt)")
parser.add_argument("--config", type=str, required=True, help="NN config JSON")
parser.add_argument("--priors", type=str, required=True, help="cg_priors.json")
parser.add_argument("--rb_info", type=str, required=True, help="rigid_bodies_info.json")
parser.add_argument("--dataset", type=str, required=True, help="Dataset to get initial frame from (e.g. cg_dataset.bin)")
parser.add_argument("--checkpoint", type=str, default=None, help="NPZ file with pos and v to load instead of dataset positions")
parser.add_argument("--dt", type=float, default=0.002, help="Time step (ps)")
parser.add_argument("--steps", type=int, default=10000, help="Simulation steps")
parser.add_argument("--no_log", action="store_true", help="Disable wandb logging")
parser.add_argument("--log_interval", type=int, default=10, help="Interval for writing trajectory (default: 10)")
parser.add_argument("--device", type=str, default="auto", help="Device for ML (cpu, mps, cuda, auto)")
parser.add_argument("--kT", type=float, default=2.49, help="Simulation temperature in kJ/mol (default 2.49 for 300K)")
parser.add_argument("--init_kT", type=float, default=None, help="Initialize velocities from Maxwell-Boltzmann at this kT")
parser.add_argument("--nve", action="store_true", help="Run NVE simulation (no thermostat)")

parser.add_argument("--toxvaerd_alpha", type=float, default=None, help="Override the value stored in the model config")
parser.add_argument("--allow_missing_model_manifest", action="store_true", help="Allow legacy .pt files without the patched training manifest")
parser.add_argument("--allow_legacy_checkpoint", action="store_true", help="Allow checkpoints without provenance metadata")
parser.add_argument("--allow_checkpoint_mismatch", action="store_true", help="Continue despite checkpoint hash or particle-identity mismatches")
parser.add_argument("--allow_unsafe_mpi", action="store_true", help="Allow the uncertified multi-rank PaiNN path")
parser.add_argument("--allow_nonconservative_tables", action="store_true", help="Allow Morse/tabulated priors during NVE despite separate energy/force interpolation")
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
if args.model:
    validate_model_manifest(
        args.model,
        runtime_nn_config,
        allow_missing=args.allow_missing_model_manifest,
    )

unsafe_tables = nonconservative_prior_entries(priors)
if args.nve and unsafe_tables and not args.allow_nonconservative_tables:
    raise RuntimeError(
        "NVE certification is disabled for Morse/tabulated priors because ESPResSo "
        "interpolates energy and force separately. Offending entries: " + ", ".join(unsafe_tables)
        + ". Pass --allow_nonconservative_tables only for a deliberate diagnostic run."
    )

if args.dt <= 0:
    raise ValueError("--dt must be positive")
if args.steps < 0:
    raise ValueError("--steps must be non-negative")
if args.log_interval <= 0:
    raise ValueError("--log_interval must be positive")

# The dummy particle type for COMs should be higher than the max ML species type
DUMMY_COM_TYPE = nn_config["num_species"] + 1

# Setup ESPResSo System
print("[INFO] Initializing ESPResSo system...")
# ESPResSo requires an initial box; it is replaced immediately by the dataset box.
system = espressomd.System(box_l=[10.0, 10.0, 10.0])
system.time_step = args.dt
system.cell_system.skin = 0.4
if args.model:
    ensure_single_rank(system, allow_unsafe_mpi=args.allow_unsafe_mpi)
# Production uses no global force cap; PaiNN forces remain the exact energy gradient.
if not args.nve:
    system.thermostat.set_langevin(kT=args.kT, gamma=1.0, gamma_rot=1.0, seed=42)
else:
    system.thermostat.turn_off()


print(f"[INFO] Running {args.steps} integration steps...")

print("[INFO] Reading initial frame from dataset...")
mol_com_parts = {}
mol_vs_parts = {}

with open(args.dataset, "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_molecules = struct.unpack("i", f.read(4))[0]
    num_total_sites = struct.unpack("i", f.read(4))[0]
    box_dim = struct.unpack("3f", f.read(12))
    
    system.box_l = [b for b in box_dim]
    
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


if args.checkpoint:
    print(f"[INFO] Overriding coordinates, velocities, and orientations from checkpoint {args.checkpoint}...")
    expected_hashes = input_hashes(
        dataset=args.dataset,
        config=args.config,
        priors=args.priors,
        rb_info=args.rb_info,
        model=args.model,
    )
    with np.load(args.checkpoint, allow_pickle=False) as chk:
        validate_checkpoint(
            chk,
            system=system,
            expected_hashes=expected_hashes,
            expected_config=runtime_nn_config,
            allow_legacy=args.allow_legacy_checkpoint,
            allow_mismatch=args.allow_checkpoint_mismatch,
        )
        pos = np.asarray(chk["pos"], dtype=float)
        vel = np.asarray(chk["v"], dtype=float)
        quat = np.asarray(chk["quat"], dtype=float) if "quat" in chk.files else None
        omega = np.asarray(chk["omega"], dtype=float) if "omega" in chk.files else None
    
    if len(pos) != len(system.part):
        raise ValueError(f"Checkpoint particle count ({len(pos)}) does not match system ({len(system.part)})")
        
    for i in range(len(system.part)):
        p = system.part.by_id(i)
        # Virtual sites positions/velocities are strictly tied to COM.
        # We only set the properties of the real (COM) particles, and the
        # virtual sites will follow automatically based on their auto-relation.
        if not p.is_virtual:
            p.pos = pos[i]
            p.v = vel[i]
            if quat is not None:
                p.quat = quat[i]
            if omega is not None:
                p.omega_body = omega[i]

if args.init_kT is not None:
    print(f"[INFO] Initializing velocities to kT={args.init_kT}...")
    for p in system.part:
        if not p.is_virtual:
            mass = p.mass
            # Translational velocity
            p.v = np.sqrt(args.init_kT / mass) * np.random.randn(3)
            # Rotational velocity
            if any(p.rotation):
                I = p.rinertia
                # Only apply to axes that are allowed to rotate
                omega = np.zeros(3)
                for axis in range(3):
                    if p.rotation[axis]:
                        omega[axis] = np.sqrt(args.init_kT / I[axis]) * np.random.randn()
                p.omega_body = omega


print("[INFO] Setting up WCA exclusions (Intra-molecular only)...")

# 1. Intra-molecular exclusions
mol_to_vs = {}
for (m_idx, s_idx), pid in mol_vs_parts.items():
    if isinstance(m_idx, int): # Ignore the absolute index mapping keys added previously
        if m_idx not in mol_to_vs:
            mol_to_vs[m_idx] = []
        mol_to_vs[m_idx].append(pid)

for m_idx, pids in mol_to_vs.items():
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            p1 = system.part.by_id(pids[i])
            p2 = system.part.by_id(pids[j])
            try:
                p1.add_exclusion(p2)
            except Exception:
                pass





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

# No additional COM-COM hard core is added: runtime interactions must match
# the priors subtracted during preprocessing.
# system.non_bonded_inter[DUMMY_COM_TYPE, DUMMY_COM_TYPE].lennard_jones.set_params(
#     epsilon=1.0, sigma=0.35,
#     cutoff=0.35 * (2.0**(1/6)), shift="auto"
# )

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

print("[INFO] Setting up dummy interactions for Verlet lists...")
for i in range(nn_config["num_species"] + 2):
    for j in range(i, nn_config["num_species"] + 2):
        ml_cutoff = nn_config["cutoff"] if "cutoff" in nn_config else 5.0
        system.non_bonded_inter[i, j].soft_sphere.set_params(
            a=0.0, n=1, cutoff=ml_cutoff, offset=0.0)

if args.model:
    print("[INFO] Activating ML Potential...")
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
else:
    print("[INFO] No --model provided. Running PURELY CLASSICAL Coarse-Grained MD.")

if not args.nve:
    system.thermostat.set_langevin(kT=args.kT, gamma=1.0, gamma_rot=1.0, seed=42)

import sys
import espressomd.io.writer.vtf
print(f"[INFO] Running {args.steps} integration steps...")


def measure_energies():
    energies = system.analysis.energy()
    e_class = energies["total"]
    e_ml = 0.0
    if args.model:
        e_ml = espressomd.painn.get_painn_energy()
    
    e_tot = e_class + e_ml

    e_kin = energies["kinetic"]
    e_kin_trans = 0.0
    e_kin_rot = 0.0
    for p in system.part:
        if p.mass < 1e-4:
            continue
        v_sq = sum(v**2 for v in p.v)
        e_kin_trans += 0.5 * p.mass * v_sq
        e_kin_rot += 0.5 * sum(
            I * w**2 for I, w in zip(p.rinertia, p.omega_body)
        )
    return e_tot, e_kin, e_kin_trans, e_kin_rot, e_class, e_ml


with ExitStack() as stack:
    energy_file = None
    vtf_file = None
    if not args.no_log:
        energy_file = stack.enter_context(open("energy.csv", "w"))
        energy_file.write("Step,E_tot,E_kin,E_kin_trans,E_kin_rot,E_class,E_ml\n")
        vtf_file = stack.enter_context(open("cg_trajectory.vtf", "w"))
        espressomd.io.writer.vtf.writevsf(system, vtf_file)
        for mol_idx, com_id in mol_com_parts.items():
            for (m_idx, _s_idx), vs_id in mol_vs_parts.items():
                if m_idx == mol_idx:
                    vtf_file.write(f"bond {com_id}:{vs_id}\n")

    completed = 0
    while completed <= args.steps:
        current = min(args.log_interval, args.steps - completed)
        system.integrator.run(current)
        completed += current

        e_tot, e_kin, e_kin_trans, e_kin_rot, e_class, e_ml = measure_energies()
        max_f = max(
            sum(f_c**2 for f_c in p.f) ** 0.5 for p in system.part
        )
        real_particles = [p for p in system.part if p.mass > 1e-4]
        max_t = max(
            (sum(t_c**2 for t_c in p.torque_lab) ** 0.5 for p in real_particles),
            default=0.0,
        )

        print(
            f"[INFO] Step {completed}/{args.steps} | E_tot: {e_tot:.6f} | "
            f"E_kin: {e_kin:.2f} | E_class: {e_class:.2f} | E_ML: {e_ml:.2f} | max_f: {max_f:.2f}"
        )

        if energy_file is not None:
            energy_file.write(
                f"{completed},{e_tot},{e_kin},{e_kin_trans},{e_kin_rot},{e_class},{e_ml}\n"
            )
            energy_file.flush()

        if vtf_file is not None:
            vtf_file.write(f"\ntimestep {completed}\n")
            espressomd.io.writer.vtf.writevcf(system, vtf_file)

print("\n[INFO] Simulation finished successfully.")

# Force immediate exit to bypass PyTorch/MPI teardown crashes on macOS
import sys
sys.stdout.flush()
os._exit(0)
