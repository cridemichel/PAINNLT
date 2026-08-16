import espressomd
import espressomd.interactions
import espressomd.io.writer.vtf
import espressomd.painn
import json
import csv
import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform
import struct
import os
from contextlib import ExitStack

from framework_utils import (
    configure_neighbor_search,
    ensure_single_rank,
    get_rb_data_by_sites,
    input_hashes,
    mask_excluded_particle_distances,
    particle_is_virtual,
    resolve_referenced_path,
    nonconservative_prior_entries,
    rigid_body_quaternion,
    save_checkpoint,
    sha256_file,
    validate_checkpoint,
    validate_model_manifest,
    validate_wca_exclusion_policy,
    wca_topology_exclusion_pairs,
    wca_direct_bonded_site_exclusions,
)

from conservative_spline_runtime import create_conservative_spline_interaction

from espresso_interactions import (
    configure_pair_specific_morse,
    create_pair_specific_morse_markers,
    configure_type_pair_morse,
    max_type_pair_morse_cutoff,
    prepare_pair_specific_morse,
    prepare_type_pair_morse,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=False, default=None, help="Trained ML potential (.pt)")
parser.add_argument("--disable_ml", action="store_true", help="Validate --model provenance but do not activate PaiNN; useful for matched classical/ML A/B runs")
parser.add_argument("--config", type=str, required=True, help="NN config JSON")
parser.add_argument("--priors", type=str, required=True, help="cg_priors.json")
parser.add_argument("--rb_info", type=str, required=True, help="rigid_bodies_info.json")
parser.add_argument("--dataset", type=str, required=True, help="Dataset to get initial frame from (e.g. cg_dataset.bin)")
parser.add_argument("--checkpoint", type=str, default=None, help="NPZ file with pos and v to load instead of dataset positions")
parser.add_argument("--dt", type=float, default=0.002, help="Time step (ps)")
parser.add_argument("--steps", type=int, default=10000, help="Simulation steps")
parser.add_argument("--no_log", action="store_true", help="Disable energy and trajectory logging")
parser.add_argument("--no_vtf", action="store_true", help="Disable VTF trajectory output while keeping the energy log")
parser.add_argument("--energy_file", type=str, default="energy.csv", help="Energy CSV output path")
parser.add_argument("--trajectory_file", type=str, default="cg_trajectory.vtf", help="VTF trajectory output path")
parser.add_argument("--sample_npz", type=str, default=None, help="Structured COM/site trajectory for analysis/IBI")
parser.add_argument("--state_sample_npz", type=str, default=None, help="Structured real-particle mechanical-state trajectory for convergence diagnostics")
parser.add_argument("--out_checkpoint", type=str, default=None, help="Save the final mechanical state as a provenance-bound checkpoint")
parser.add_argument("--sample_start_step", type=int, default=0, help="First logged step included in --sample_npz")
parser.add_argument("--log_interval", type=int, default=10, help="Interval for energy/trajectory logging (default: 10 steps)")
parser.add_argument("--device", type=str, default="auto", help="Device for ML (cpu, mps, cuda, auto)")
parser.add_argument("--ml_precision", choices=("float32", "float64"), default="float32", help="PaiNN inference precision; float64 is a CPU diagnostic mode")
parser.add_argument("--neighbor_search", choices=("verlet", "link-cell"), default="verlet", help="Pair traversal in ESPResSo regular decomposition")
parser.add_argument("--kT", type=float, default=2.49, help="Simulation temperature in kJ/mol (default 2.49 for 300K)")
parser.add_argument("--init_kT", type=float, default=None, help="Initialize velocities from Maxwell-Boltzmann at this kT")
parser.add_argument("--velocity_seed", type=int, default=314159, help="Seed used by --init_kT")
parser.add_argument("--thermostat_seed", type=int, default=42, help="Langevin thermostat seed")
parser.add_argument("--nve", action="store_true", help="Run NVE simulation (no thermostat)")
parser.add_argument("--toxvaerd_alpha", type=float, default=None, help="Override the value stored in the model config")
parser.add_argument("--allow_missing_model_manifest", action="store_true", help="Allow legacy .pt files without the patched training manifest")
parser.add_argument("--allow_legacy_checkpoint", action="store_true", help="Allow checkpoints without provenance metadata")
parser.add_argument("--allow_checkpoint_mismatch", action="store_true", help="Continue despite checkpoint hash or particle-identity mismatches")
parser.add_argument("--allow_unsafe_mpi", action="store_true", help="Allow the uncertified multi-rank PaiNN path")
parser.add_argument("--allow_nonconservative_tables", action="store_true", help="Allow explicitly tabulated priors during NVE despite separate energy/force interpolation")
args = parser.parse_args()

if args.disable_ml and not args.model:
    raise ValueError("--disable_ml requires --model so the disabled branch remains bound to the same model provenance")
ml_active = bool(args.model and not args.disable_ml)

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
        "NVE certification is disabled for explicitly tabulated priors because ESPResSo "
        "interpolates energy and force separately. Reversible analytic Morse priors are conservative. "
        "Offending entries: " + ", ".join(unsafe_tables)
        + ". Pass --allow_nonconservative_tables only for a deliberate diagnostic run."
    )

if args.dt <= 0:
    raise ValueError("--dt must be positive")
if args.steps < 0:
    raise ValueError("--steps must be non-negative")
if args.log_interval <= 0:
    raise ValueError("--log_interval must be positive")
if args.sample_start_step < 0 or args.sample_start_step > args.steps:
    raise ValueError("--sample_start_step must lie between 0 and --steps")
if args.sample_npz and args.sample_start_step % args.log_interval != 0:
    raise ValueError("--sample_start_step must be a multiple of --log_interval")
if args.init_kT is not None and args.init_kT <= 0.0:
    raise ValueError("--init_kT must be positive")

# Plan pair-specific reversible Morse contacts before creating particles.
# Physical CG-site types remain untouched; explicit contacts are carried by
# coincident technical virtual markers created after the physical sites.
morse_marker_types, morse_contacts = prepare_pair_specific_morse(
    priors, nn_config["num_species"]
)
morse_type_pairs = prepare_type_pair_morse(priors, nn_config["num_species"])
if morse_contacts and morse_type_pairs:
    print(
        "[WARNING] Both pair-specific Morse contacts and site type-pair Morse "
        "interactions are active. Their contributions are additive; verify that this "
        "is intentional and not prior double counting."
    )
DUMMY_COM_TYPE = nn_config["num_species"] + 1

# Setup ESPResSo System
print("[INFO] Initializing ESPResSo system...")
# ESPResSo requires an initial box; it is replaced immediately by the dataset box.
system = espressomd.System(box_l=[10.0, 10.0, 10.0])
system.time_step = args.dt
system.cell_system.skin = 0.4
if args.model:
    ensure_single_rank(system, allow_unsafe_mpi=args.allow_unsafe_mpi)
# Certification/production invariant: no force capping and explicit Velocity Verlet.
# Force capping changes forces without changing the reported energy and therefore
# invalidates an NVE conservation test.
system.force_cap = 0.0
system.integrator.set_vv()
if args.nve:
    system.thermostat.turn_off()
else:
    system.thermostat.set_langevin(kT=args.kT, gamma=1.0, gamma_rot=1.0, seed=args.thermostat_seed)


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


morse_marker_parts = create_pair_specific_morse_markers(
    system, morse_marker_types, mol_com_parts, mol_vs_parts
)
if morse_marker_parts:
    print(
        f"[INFO] Created {len(morse_marker_parts)} technical virtual markers "
        "for pair-specific Morse endpoints."
    )

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
        if not particle_is_virtual(p):
            p.pos = pos[i]
            p.v = vel[i]
            if quat is not None:
                p.quat = quat[i]
            if omega is not None:
                p.omega_body = omega[i]

if args.init_kT is not None:
    print(f"[INFO] Initializing velocities to kT={args.init_kT} with seed={args.velocity_seed}...")
    rng = np.random.default_rng(args.velocity_seed)
    real_particles = [p for p in system.part if not particle_is_virtual(p)]
    for p in real_particles:
        mass = float(p.mass)
        p.v = np.sqrt(args.init_kT / mass) * rng.standard_normal(3)
        if any(p.rotation):
            inertia = np.asarray(p.rinertia, dtype=float)
            omega = np.zeros(3, dtype=float)
            for axis in range(3):
                if p.rotation[axis]:
                    omega[axis] = np.sqrt(args.init_kT / inertia[axis]) * rng.standard_normal()
            p.omega_body = omega

    # Remove the global translational drift without changing internal thermal motion.
    total_mass = sum(float(p.mass) for p in real_particles)
    if total_mass > 0.0:
        com_velocity = sum(
            (float(p.mass) * np.asarray(p.v, dtype=float) for p in real_particles),
            start=np.zeros(3, dtype=float),
        ) / total_mass
        for p in real_particles:
            p.v = np.asarray(p.v, dtype=float) - com_velocity


print("[INFO] Setting up WCA exclusions (intra-rigid-body + 1-2/1-3)...")

# Keep the safety-distance diagnostic aligned with the actual ESPResSo
# nonbonded topology.  Without this mask, close topologically excluded pairs
# (especially all-site 1-3 exclusions) can falsely trigger min_dist < 0.15 nm.
diagnostic_nonbonded_excluded_pid_pairs = set()

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

validate_wca_exclusion_policy(priors)
wca_direct_pairs, wca_one_three_pairs = wca_topology_exclusion_pairs(priors, num_molecules)
direct_site_exclusions = wca_direct_bonded_site_exclusions(priors, num_molecules)

# Policy v3: 1-3 remains an all-sites exclusion.  For 1-2 pairs WCA stays
# active across the two rigid bodies except for explicitly bonded site pairs.
for mol_i, mol_j in sorted(wca_one_three_pairs):
    for pid_i in mol_to_vs.get(mol_i, []):
        for pid_j in mol_to_vs.get(mol_j, []):
            try:
                system.part.by_id(pid_i).add_exclusion(system.part.by_id(pid_j))
                diagnostic_nonbonded_excluded_pid_pairs.add(
                    (min(int(pid_i), int(pid_j)), max(int(pid_i), int(pid_j)))
                )
            except Exception:
                pass

applied_direct_site_exclusions = 0
for (mol_i, mol_j), site_pairs in sorted(direct_site_exclusions.items()):
    for site_i, site_j in sorted(site_pairs):
        pid_i = mol_vs_parts.get((mol_i, site_i))
        pid_j = mol_vs_parts.get((mol_j, site_j))
        if pid_i is None or pid_j is None:
            raise RuntimeError(
                "WCA policy v3 references a missing bonded virtual site: "
                f"mol/site {mol_i}:{site_i} <-> {mol_j}:{site_j}"
            )
        try:
            system.part.by_id(pid_i).add_exclusion(system.part.by_id(pid_j))
            diagnostic_nonbonded_excluded_pid_pairs.add(
                (min(int(pid_i), int(pid_j)), max(int(pid_i), int(pid_j)))
            )
        except Exception:
            pass
        applied_direct_site_exclusions += 1

print(
    f"[INFO] Non-bonded topology exclusions active (WCA/type-pair potentials): {len(wca_direct_pairs)} 1-2 molecule pairs "
    f"with {applied_direct_site_exclusions} bonded site-pair exclusions; "
    f"{len(wca_one_three_pairs)} 1-3 all-sites exclusions (policy v3)."
)
print(
    f"[INFO] Safety min-distance diagnostic masks "
    f"{len(diagnostic_nonbonded_excluded_pid_pairs)} excluded physical-site pairs."
)


print("[INFO] Adding priors...")
# WCA from cg_priors.json
import json
import os
import math

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

# No additional COM-COM hard core is added: runtime interactions must match
# the priors subtracted during preprocessing.

# Structural bonds (Morse contacts were configured above as reversible non-bonded priors)
for idx, b in enumerate(priors.get("bonds", [])):
    b_type = b.get("type", "harmonic")
    if b_type == "harmonic":
        bond = espressomd.interactions.HarmonicBond(k=b["k"], r_0=b["r0"])
    elif b_type == "fene":
        bond = espressomd.interactions.FeneBond(k=b["k"], d_r_max=b["r_max"], r_0=b["r0"])
    elif b_type == "morse":
        continue
    elif b_type == "tabulated":
        data = np.loadtxt(resolve_referenced_path(b["file"], args.priors))
        rmin_tab = float(b["min"])
        rmax_tab = float(b["max"])
        r_vals = data[:, 0]
        energy = data[:, 1]
        force = data[:, 2]
        
        bond = espressomd.interactions.TabulatedDistance(
            min=rmin_tab, max=rmax_tab, energy=energy, force=force
        )
    elif b_type == "conservative_spline":
        bond = create_conservative_spline_interaction(
            espressomd.interactions, b, kind="bond", priors_path=args.priors
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
        data = np.loadtxt(resolve_referenced_path(a["file"], args.priors))
        min_tab = float(a["min"]) # Typically 0.0 radians
        max_tab = float(a["max"]) # Typically pi radians
        angle = espressomd.interactions.TabulatedAngle(
            min=min_tab, max=max_tab, energy=data[:, 1], force=data[:, 2]
        )
    elif a_type == "conservative_spline":
        angle = create_conservative_spline_interaction(
            espressomd.interactions, a, kind="angle", priors_path=args.priors
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
        data = np.loadtxt(resolve_referenced_path(d["file"], args.priors))
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

print("[INFO] Setting up dummy interactions for neighbor search...")
# Only ML site types participate in PaiNN. Do not activate the dummy
# zero-strength SoftSphere for COM particle types: single-site molecules place
# their virtual ML site exactly at the COM, so a SoftSphere pair at r=0 would
# evaluate the singular power-law form and contaminate the reported energy
# with NaN even when a=0.
for i in range(nn_config["num_species"]):
    for j in range(i, nn_config["num_species"]):
        ml_cutoff = nn_config["cutoff"] if "cutoff" in nn_config else 5.0
        system.non_bonded_inter[i, j].soft_sphere.set_params(
            a=0.0, n=1, cutoff=ml_cutoff, offset=0.0)

regular_cutoff = max(
    float(nn_config.get("cutoff", 0.0)),
    max((float(item.get("cutoff_nm", 0.0)) for item in priors.get("wca_pairs", {}).values()), default=0.0),
    max_type_pair_morse_cutoff(morse_type_pairs),
)
# Pair-specific Morse contacts use dedicated technical marker types on the
# N-square side of the hybrid decomposition. Type-pair Morse acts on ordinary
# physical CG site types and therefore contributes to the regular cutoff above.
if morse_type_pairs:
    type_pair_cutoff = max_type_pair_morse_cutoff(morse_type_pairs)
    required_length = 2.0 * (type_pair_cutoff + float(system.cell_system.skin))
    if min(system.box_l) <= required_length:
        raise ValueError(
            "Morse type-pair cutoff is too large for the periodic regular decomposition: "
            f"box={list(system.box_l)}, max Morse r_cut={type_pair_cutoff:.6g}, "
            f"skin={float(system.cell_system.skin):.6g}; each box dimension must exceed "
            f"{required_length:.6g} nm. Pair-specific Morse marker contacts do not have "
            "this regular-cell constraint because they use the N-square side of the hybrid decomposition."
        )
morse_n_square_types = {DUMMY_COM_TYPE, *morse_marker_types.values()} if morse_contacts else None
configure_neighbor_search(
    system, args.neighbor_search,
    n_square_types=morse_n_square_types,
    cutoff_regular=regular_cutoff if morse_contacts else None,
)

# Register the long-cutoff pair-specific Morse interactions only after the
# hybrid decomposition is active. ESPResSo validates a newly configured
# non-bonded cutoff against the current cell system; configuring the 15 nm
# marker interaction while the default regular decomposition is still active
# incorrectly subjects it to the regular-cell range limit.
# Type-pair Morse acts on physical CG site types in the regular side of the
# already configured hybrid decomposition. Configuring it here also makes the
# explicit regular-cutoff validation below authoritative instead of letting the
# default cell system reject the interaction first.
configure_type_pair_morse(system, morse_type_pairs)
for item in morse_type_pairs:
    print(
        "[INFO] Added type-pair reversible Morse interaction "
        f"{item['index']}: site type {item['type_i']} <-> {item['type_j']} "
        f"(r_switch={item['r_switch']:.6g}, r_cut={item['r_cut']:.6g})"
    )

configure_pair_specific_morse(system, morse_contacts, morse_marker_types)
for contact in morse_contacts:
    print(
        "[INFO] Added pair-specific reversible Morse contact "
        f"{contact['index']}: {contact['mol_i']}:{contact['site_i']} <-> "
        f"{contact['mol_j']}:{contact['site_j']} "
        f"(site=-1 means COM; r_switch={contact['r_switch']:.6g}, "
        f"r_cut={contact['r_cut']:.6g})"
    )

if ml_active:
    print("[INFO] Activating ML Potential...")
    espressomd.painn.activate_painn_potential(
        model_path=args.model,
        num_species=nn_config["num_species"],
        hidden_channels=nn_config["hidden_channels"],
        n_layers=nn_config["n_layers"],
        num_rbf=nn_config["num_rbf"],
        cutoff=nn_config["cutoff"],
        toxvaerd_alpha=args.toxvaerd_alpha,
        device=args.device,
        precision=args.ml_precision
    )
elif args.disable_ml:
    print("[INFO] PaiNN disabled by --disable_ml; --model is retained only for provenance/checkpoint validation.")
else:
    print("[INFO] No --model provided. Running PURELY CLASSICAL Coarse-Grained MD.")

# Re-assert the integration invariants after all interactions and the ML plugin
# have been configured.
system.force_cap = 0.0
system.integrator.set_vv()
if args.nve:
    system.thermostat.turn_off()

import sys
import espressomd.io.writer.vtf
print(f"[INFO] Running {args.steps} integration steps...")



def log_diagnostics(step):
    pos = []
    types = []
    mol_ids = []
    forces = []
    pids = []
    for p in system.part:
        if particle_is_virtual(p):
            pos.append(p.pos)
            types.append(p.type)
            mol_ids.append(p.mol_id)
            forces.append(p.f)
            pids.append(p.id)
            
    pos = np.array(pos)
    types = np.array(types)
    mol_ids = np.array(mol_ids)
    forces = np.array(forces)
    pids = np.array(pids)
    
    dist_matrix = squareform(pdist(pos))
    mask = mol_ids[:, None] == mol_ids[None, :]
    dist_matrix[mask] = np.inf
    np.fill_diagonal(dist_matrix, np.inf)
    mask_excluded_particle_distances(
        dist_matrix, pids, diagnostic_nonbonded_excluded_pid_pairs
    )
    
    num_species = nn_config["num_species"]
    min_dists = {}
    
    global_min_dist = np.inf
    global_min_pair = None
    global_min_pids = None
    
    for i in range(num_species):
        for j in range(i, num_species):
            mask_types = (types[:, None] == i) & (types[None, :] == j)
            # Make symmetric
            mask_types = mask_types | ((types[:, None] == j) & (types[None, :] == i))
            
            valid_dists = dist_matrix[mask_types]
            if len(valid_dists) > 0:
                m_d = np.min(valid_dists)
                min_dists[(i, j)] = m_d
                if m_d < global_min_dist:
                    global_min_dist = m_d
                    global_min_pair = (i, j)
                    # Find the exact particles
                    # Get indices where mask_types and dist_matrix == m_d
                    idx1, idx2 = np.where((dist_matrix == m_d) & mask_types)
                    if len(idx1) > 0:
                        global_min_pids = (pids[idx1[0]], pids[idx2[0]])
            else:
                min_dists[(i, j)] = np.inf
                
    f_max = np.max(np.linalg.norm(forces, axis=1))
    
    return global_min_dist, global_min_pair, global_min_pids, f_max


def measure_energies():
    energies = system.analysis.energy()

    bad_energy_terms = []
    for key, value in energies.items():
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(scalar):
            bad_energy_terms.append((key, scalar))

    if bad_energy_terms:
        print("[CRITICAL] Non-finite ESPResSo energy terms:")
        for key, value in bad_energy_terms:
            print(f"    {key!r}: {value!r}")

        print("[INFO] Finite top-level ESPResSo energies:")
        for key in ("kinetic", "kinetic_lin", "kinetic_rot",
                    "bonded", "non_bonded",
                    "coulomb", "external_fields"):
            if key in energies:
                print(f"    {key!r}: {energies[key]!r}")

        raise RuntimeError(
            "Non-finite ESPResSo energy at the current state"
        )

    e_class = energies["total"]
    e_ml = 0.0
    if ml_active:
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


def stringify_pair(value):
    if value is None:
        return ""
    return ":".join(str(int(v)) for v in value)


sample_steps = []
sample_com = []
sample_sites = []
sample_site_keys = sorted(mol_vs_parts)
state_sample_steps = []
state_sample_positions = []
state_sample_velocities = []
state_sample_quaternions = []
state_sample_omegas = []
state_sample_particle_ids = sorted(
    int(p.id) for p in system.part if float(p.mass) > 1.0e-4
)
state_sample_rotation_flags = np.asarray([
    [bool(v) for v in system.part.by_id(pid).rotation]
    for pid in state_sample_particle_ids
], dtype=bool)
if sorted(mol_com_parts) != list(range(num_molecules)):
    raise RuntimeError("COM particle mapping is not contiguous in molecule-index order")


def record_structured_sample(step):
    if args.sample_npz is None or step < args.sample_start_step:
        return
    sample_steps.append(int(step))
    sample_com.append(np.asarray([
        system.part.by_id(mol_com_parts[mol]).pos for mol in range(num_molecules)
    ], dtype=float))
    sample_sites.append(np.asarray([
        system.part.by_id(mol_vs_parts[key]).pos for key in sample_site_keys
    ], dtype=float))


def record_state_sample(step):
    if args.state_sample_npz is None:
        return
    state_sample_steps.append(int(step))
    particles = [system.part.by_id(pid) for pid in state_sample_particle_ids]
    state_sample_positions.append(np.asarray([p.pos for p in particles], dtype=float))
    state_sample_velocities.append(np.asarray([p.v for p in particles], dtype=float))
    state_sample_quaternions.append(np.asarray([p.quat for p in particles], dtype=float))
    omega = []
    for particle in particles:
        try:
            omega.append(particle.omega_body)
        except Exception:
            omega.append([0.0, 0.0, 0.0])
    state_sample_omegas.append(np.asarray(omega, dtype=float))


def record_state(step, energy_writer, energy_handle, vtf_handle):
    e_tot, e_kin, e_kin_trans, e_kin_rot, e_class, e_ml = measure_energies()
    g_dist, g_pair, g_pids, max_f = log_diagnostics(step)
    real_particles = [p for p in system.part if p.mass > 1e-4]
    max_t = max(
        (sum(t_c**2 for t_c in p.torque_lab) ** 0.5 for p in real_particles),
        default=0.0,
    )
    time_ps = float(step) * float(args.dt)

    print(
        f"[INFO] Step {step}/{args.steps} | t={time_ps:.6f} ps | E_tot: {e_tot:.9f} | "
        f"E_kin: {e_kin:.6f} | E_ML: {e_ml:.6f} | max_f: {max_f:.2f} | "
        f"min_dist: {g_dist:.3f} nm (types {g_pair})"
    )

    if energy_writer is not None:
        energy_writer.writerow([
            step,
            time_ps,
            e_tot,
            e_kin,
            e_kin_trans,
            e_kin_rot,
            e_class,
            e_ml,
            g_dist,
            stringify_pair(g_pair),
            stringify_pair(g_pids),
            max_f,
            max_t,
        ])
        energy_handle.flush()

    if vtf_handle is not None:
        vtf_handle.write(f"\ntimestep {step}\n")
        espressomd.io.writer.vtf.writevcf(system, vtf_handle)

    record_structured_sample(step)
    record_state_sample(step)
    unsafe = max_f > 10000.0 or e_kin > 5000.0 or g_dist < 0.15
    return unsafe


simulation_ok = True
with ExitStack() as stack:
    energy_handle = None
    energy_writer = None
    vtf_handle = None
    if not args.no_log:
        energy_handle = stack.enter_context(open(args.energy_file, "w", newline=""))
        energy_writer = csv.writer(energy_handle)
        energy_writer.writerow([
            "Step",
            "Time_ps",
            "E_tot",
            "E_kin",
            "E_kin_trans",
            "E_kin_rot",
            "E_class",
            "E_ml",
            "min_dist",
            "min_pair",
            "min_pids",
            "f_max",
            "torque_max",
        ])
        energy_handle.flush()

        if not args.no_vtf:
            vtf_handle = stack.enter_context(open(args.trajectory_file, "w"))
            espressomd.io.writer.vtf.writevsf(system, vtf_handle)
            for mol_idx, com_id in mol_com_parts.items():
                for (m_idx, _s_idx), vs_id in mol_vs_parts.items():
                    if m_idx == mol_idx:
                        vtf_handle.write(f"bond {com_id}:{vs_id}\n")

    # Initialize the force-dependent PaiNN energy at the exact initial state.
    # This is required for a meaningful E(t=0) in NVE certification.
    system.integrator.run(0, recalc_forces=True)
    if record_state(0, energy_writer, energy_handle, vtf_handle):
        print("[CRITICAL] Safety abort triggered at the initial state.")
        simulation_ok = False

    completed = 0
    while simulation_ok and completed < args.steps:
        current = min(args.log_interval, args.steps - completed)
        system.integrator.run(current)
        completed += current

        if record_state(completed, energy_writer, energy_handle, vtf_handle):
            print("[CRITICAL] Safety abort triggered! max_f > 10000, E_kin > 5000, or min_dist < 0.15")
            system.integrator.run(0, recalc_forces=True)
            with open("crash.vtf", "w") as crash_vtf:
                espressomd.io.writer.vtf.writevsf(system, crash_vtf)
                espressomd.io.writer.vtf.writevcf(system, crash_vtf)
            np.savez(
                "crash_checkpoint.npz",
                positions=system.part.all().pos,
                velocities=system.part.all().v,
                forces=system.part.all().f,
                quaternions=system.part.all().quat,
                omega_body=system.part.all().omega_body,
            )
            print("[CRITICAL] Crash checkpoint saved. Exiting with non-zero status.")
            simulation_ok = False
            break

if simulation_ok:
    if args.sample_npz is not None:
        if not sample_steps:
            raise RuntimeError("Structured sampling produced no frames")
        sample_path = os.path.abspath(args.sample_npz)
        sample_dir = os.path.dirname(sample_path)
        if sample_dir:
            os.makedirs(sample_dir, exist_ok=True)
        np.savez_compressed(
            sample_path,
            schema_version=np.asarray(1, dtype=np.int32),
            complete=np.asarray(1, dtype=np.int8),
            steps=np.asarray(sample_steps, dtype=np.int64),
            time_ps=np.asarray(sample_steps, dtype=float) * float(args.dt),
            com=np.asarray(sample_com, dtype=float),
            sites=np.asarray(sample_sites, dtype=float),
            site_molecule=np.asarray([key[0] for key in sample_site_keys], dtype=np.int32),
            site_index=np.asarray([key[1] for key in sample_site_keys], dtype=np.int32),
            box=np.asarray(system.box_l, dtype=float),
        )
        print(
            f"[INFO] Structured sampling: {len(sample_steps)} frames written to {sample_path} "
            f"(start step {sample_steps[0]}, end step {sample_steps[-1]})"
        )
    if args.state_sample_npz is not None:
        if not state_sample_steps:
            raise RuntimeError("Mechanical-state sampling produced no frames")
        state_path = os.path.abspath(args.state_sample_npz)
        state_dir = os.path.dirname(state_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
        if args.disable_ml:
            state_hamiltonian_mode = "conservative_classical_model_provenance_ml_disabled"
        elif ml_active:
            state_hamiltonian_mode = "painn_active"
        else:
            state_hamiltonian_mode = "classical_only"
        state_metadata = {
            "schema_version": 1,
            "kind": "mlcg_real_particle_state_trajectory",
            "dt_ps": float(args.dt),
            "log_interval_steps": int(args.log_interval),
            "hamiltonian_mode": state_hamiltonian_mode,
            "sampling_ensemble": "NVE" if args.nve else "NVT_Langevin",
            "input_hashes": input_hashes(
                dataset=args.dataset,
                config=args.config,
                priors=args.priors,
                rb_info=args.rb_info,
                model=args.model,
            ),
            "source_checkpoint_sha256": (
                sha256_file(args.checkpoint) if args.checkpoint is not None else None
            ),
            "ml_active": bool(ml_active),
            "ml_disabled_by_flag": bool(args.disable_ml),
        }
        np.savez_compressed(
            state_path,
            schema_version=np.asarray(1, dtype=np.int32),
            complete=np.asarray(1, dtype=np.int8),
            steps=np.asarray(state_sample_steps, dtype=np.int64),
            time_ps=np.asarray(state_sample_steps, dtype=float) * float(args.dt),
            particle_ids=np.asarray(state_sample_particle_ids, dtype=np.int64),
            rotation_flags=state_sample_rotation_flags,
            positions=np.asarray(state_sample_positions, dtype=float),
            velocities=np.asarray(state_sample_velocities, dtype=float),
            quaternions=np.asarray(state_sample_quaternions, dtype=float),
            omega_body=np.asarray(state_sample_omegas, dtype=float),
            box=np.asarray(system.box_l, dtype=float),
            metadata_json=np.asarray(json.dumps(state_metadata, sort_keys=True)),
        )
        print(
            f"[INFO] Mechanical-state sampling: {len(state_sample_steps)} frames written to {state_path} "
            f"(start step {state_sample_steps[0]}, end step {state_sample_steps[-1]})"
        )
    if args.out_checkpoint is not None:
        checkpoint_path = os.path.abspath(args.out_checkpoint)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        # A saved checkpoint is a pure mechanical state.  Turn off the thermostat
        # before the final force refresh so NVE consumers inherit only positions,
        # orientations and finite translational/rotational velocities.
        system.thermostat.turn_off()
        system.integrator.run(0, recalc_forces=True)
        positions = []
        velocities = []
        quaternions = []
        omegas = []
        for i in range(len(system.part)):
            particle = system.part.by_id(i)
            positions.append(particle.pos)
            velocities.append(particle.v)
            quaternions.append(particle.quat)
            try:
                omegas.append(particle.omega_body)
            except Exception:
                omegas.append([0.0, 0.0, 0.0])
        hashes = input_hashes(
            dataset=args.dataset,
            config=args.config,
            priors=args.priors,
            rb_info=args.rb_info,
            model=args.model,
        )
        if args.disable_ml:
            hamiltonian_mode = "conservative_classical_model_provenance_ml_disabled"
        elif ml_active:
            hamiltonian_mode = "painn_active"
        else:
            hamiltonian_mode = "classical_only"
        source_checkpoint_sha256 = (
            sha256_file(args.checkpoint) if args.checkpoint is not None else None
        )
        save_checkpoint(
            checkpoint_path,
            system=system,
            pos=np.asarray(positions, dtype=float),
            vel=np.asarray(velocities, dtype=float),
            quat=np.asarray(quaternions, dtype=float),
            omega=np.asarray(omegas, dtype=float),
            hashes=hashes,
            config=runtime_nn_config,
            dt=args.dt,
            kT=args.kT,
            extra_metadata={
                "checkpoint_origin": "run_cg_md_final_state",
                "hamiltonian_mode": hamiltonian_mode,
                "sampling_ensemble": "NVE" if args.nve else "NVT_Langevin",
                "completed_steps": int(completed),
                "source_checkpoint_sha256": source_checkpoint_sha256,
                "neighbor_search": args.neighbor_search,
                "thermostat_seed": None if args.nve else int(args.thermostat_seed),
                "ml_active": bool(ml_active),
                "ml_disabled_by_flag": bool(args.disable_ml),
            },
        )
        print(
            f"[INFO] Final checkpoint saved: {checkpoint_path} "
            f"(hamiltonian_mode={hamiltonian_mode})"
        )
    print("\n[INFO] Simulation finished successfully.")
else:
    print("\n[ERROR] Simulation terminated by a safety guardrail.")

# Force immediate exit to bypass PyTorch/MPI teardown crashes on macOS
import sys
sys.stdout.flush()
os._exit(0 if simulation_ok else 2)
