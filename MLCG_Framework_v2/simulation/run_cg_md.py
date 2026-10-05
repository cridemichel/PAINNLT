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
import re
import os
import time
_WALL_T0 = time.perf_counter()   # bilancio dei tempi a fine corsa ([TIMING])
from contextlib import ExitStack

from framework_utils import (
    require_cbt_dihedral,
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
    pair_contact_summary,
    configure_pair_specific_morse_bonds,
    create_pair_specific_morse_markers,
    configure_debye_huckel,
    configure_type_pair_morse,
    max_type_pair_morse_cutoff,
    prepare_pair_specific_morse,
    prepare_debye_huckel,
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
parser.add_argument("--dump_initial_forces", type=str, default=None, help="Scrive forza e coppia di ogni corpo (COM) sulla configurazione iniziale, senza termostato, ed esce: confronto runtime <-> dataset (compare_prior_parity.py)")
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
parser.add_argument("--painn_profile_report", type=str, default=None, help="Write opt-in PaiNN C++ stage timing JSON (--device cpu or cuda; CUDA stages are synchronised only while profiling)")
parser.add_argument("--energy_interval", type=int, default=1, help="Calcola la decomposizione completa delle energie (system.analysis.energy(), ~1 s) solo ogni K registrazioni; nelle altre E_kin viene dalle velocita' ed E_tot/E_class/E_bonded/E_non_bonded sono NaN nel CSV (default 1: sempre)")
parser.add_argument("--legacy_diagnostics", action="store_true", help="Log per particella (lento, ~1.5 s per registrazione) invece di quello vettoriale verificato al passo 0")
parser.add_argument("--painn_profile_warmup_calls", type=int, default=20, help="PaiNN force calls excluded before profiling accumulation")
parser.add_argument("--neighbor_search", choices=("verlet", "link-cell", "nsquare"), default="verlet", help="Pair traversal in ESPResSo; nsquare is an all-pairs diagnostic mode")
parser.add_argument("--morse_switch_mode", choices=("switched", "stock-shifted"), default="switched", help="Pair-specific/type-pair Morse runtime branch; stock-shifted is a diagnostic control that keeps markers/cutoff but disables the C2 tail switch")
parser.add_argument(
    "--pair_specific_morse_runtime",
    choices=("marker-nonbonded", "bonded-analytic"),
    default="marker-nonbonded",
    help=(
        "Runtime realization for explicit pair-specific Morse contacts. "
        "marker-nonbonded is production behavior; bonded-analytic is a diagnostic "
        "control that applies the same D/a/r0/r_cut to the physical endpoints via MorseBond."
    ),
)
parser.add_argument("--kT", type=float, default=2.49, help="Simulation temperature in kJ/mol (default 2.49 for 300K)")
parser.add_argument("--init_kT", type=float, default=None, help="Initialize velocities from Maxwell-Boltzmann at this kT")
parser.add_argument("--velocity_seed", type=int, default=314159, help="Seed used by --init_kT")
parser.add_argument("--thermostat_seed", type=int, default=42, help="Langevin thermostat seed")
parser.add_argument("--nve", action="store_true", help="Run NVE simulation (no thermostat)")
# In ESPResSo gamma e' un coefficiente d'ATTRITO (m dv/dt = F - gamma v + rumore),
# non un tasso: il tempo di rilassamento della velocita' e' m/gamma.  Con masse
# di 250-330 amu nelle unita' amu-nm-ps, gamma = 1 da' ~300 ps -- su corse di
# pochi ps il termostato non agisce e la dinamica e' di fatto microcanonica.
# Per termalizzare davvero (per esempio dopo aver acceso il potenziale ML, che
# rilascia energia potenziale) servono valori dell'ordine di 10-50.  L'ensemble
# di equilibrio non dipende da gamma; la cinetica si'.
parser.add_argument("--gamma", type=float, default=1.0,
                    help="Langevin friction (mass/time; relaxation time = m/gamma). Default 1.0")
parser.add_argument("--gamma_rot", type=float, default=None,
                    help="Rotational Langevin friction; default: same as --gamma")
parser.add_argument("--toxvaerd_alpha", type=float, default=None, help="Override the value stored in the model config")
parser.add_argument("--allow_missing_model_manifest", action="store_true", help="Allow legacy .pt files without the patched training manifest")
parser.add_argument("--allow_legacy_checkpoint", action="store_true", help="Allow checkpoints without provenance metadata")
parser.add_argument("--allow_checkpoint_mismatch", action="store_true", help="Continue despite checkpoint hash or particle-identity mismatches")
parser.add_argument("--allow_unsafe_mpi", action="store_true", help="Allow the uncertified multi-rank PaiNN path")
parser.add_argument("--allow_nonconservative_tables", action="store_true", help="Allow explicitly tabulated priors during NVE despite separate energy/force interpolation")
parser.add_argument("--generalized_fd_report", type=str, default=None, help="Write a zero-step finite-difference force/torque consistency report for real particles")
parser.add_argument("--generalized_fd_eps_pos", type=float, default=1.0e-6, help="Cartesian displacement for --generalized_fd_report (nm)")
parser.add_argument("--generalized_fd_eps_rot", type=float, default=1.0e-6, help="Lab-frame rotation angle for --generalized_fd_report (rad)")
parser.add_argument("--generalized_fd_max_bodies", type=int, default=8, help="Maximum number of real bodies sampled by --generalized_fd_report; 0 means all")
args = parser.parse_args()

if args.disable_ml and not args.model:
    raise ValueError("--disable_ml requires --model so the disabled branch remains bound to the same model provenance")
if args.pair_specific_morse_runtime == "bonded-analytic" and args.morse_switch_mode != "switched":
    raise ValueError(
        "--pair_specific_morse_runtime bonded-analytic requires --morse_switch_mode switched; "
        "the bonded diagnostic has no switch branch"
    )
ml_active = bool(args.model and not args.disable_ml)
if args.painn_profile_warmup_calls < 0:
    raise ValueError("--painn_profile_warmup_calls must be non-negative")
if args.painn_profile_report is not None:
    if not ml_active:
        raise ValueError("--painn_profile_report requires an active PaiNN model")
    if args.device not in ("cpu", "cuda"):
        raise ValueError(
            "PaiNN stage profiling synchronises CPU and CUDA stage boundaries only; "
            "use --device cpu or --device cuda so wall-clock stage timings are meaningful."
        )

# Plugin graph path (read by the C++ constructor): "legacy" (default) or
# "device" (pair search + node forces on the model device).  Validated here so
# a typo fails before the system is built.
painn_graph = os.environ.get("MLCG_PAINN_GRAPH", "legacy")
if painn_graph not in ("legacy", "device"):
    raise ValueError(f"MLCG_PAINN_GRAPH must be 'legacy' or 'device', got {painn_graph!r}")
if ml_active:
    print(f"[INFO] PaiNN graph path: {painn_graph}")

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
if args.generalized_fd_eps_pos <= 0.0 or args.generalized_fd_eps_rot <= 0.0:
    raise ValueError("generalized finite-difference epsilons must be positive")
if args.generalized_fd_max_bodies < 0:
    raise ValueError("--generalized_fd_max_bodies must be non-negative")

# Plan pair-specific reversible Morse contacts before creating particles.
# Physical CG-site types remain untouched; explicit contacts are carried by
# coincident technical virtual markers created after the physical sites.
morse_marker_types, morse_contacts = prepare_pair_specific_morse(
    priors, nn_config["num_species"]
)
morse_type_pairs = prepare_type_pair_morse(priors, nn_config["num_species"])
# Debye-Hueckel fra siti carichi (ioni impliciti): cariche sui siti fisici,
# nessuna esclusione (ESPResSo non le applica all'elettrostatica).
debye_huckel = prepare_debye_huckel(priors)
if debye_huckel:
    print(
        f"[INFO] Debye-Hueckel: cariche per tipo {debye_huckel['charges']}, "
        f"prefactor={debye_huckel['prefactor']:.4f} kJ/mol nm, "
        f"lambda_D={1.0 / debye_huckel['kappa']:.3f} nm, r_cut={debye_huckel['r_cut']:.3f} nm"
    )
if args.pair_specific_morse_runtime == "bonded-analytic" and morse_type_pairs:
    raise ValueError(
        "bonded-analytic diagnostic supports explicit pair-specific Morse contacts only; "
        "morse_type_pairs must be empty"
    )
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
    _gamma_rot = args.gamma if args.gamma_rot is None else args.gamma_rot
    system.thermostat.set_langevin(kT=args.kT, gamma=args.gamma, gamma_rot=_gamma_rot,
                                   seed=args.thermostat_seed)
    print(f"[INFO] Langevin: gamma={args.gamma}, gamma_rot={_gamma_rot} "
          f"(rilassamento ~m/gamma)")


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
            if debye_huckel and int(stype) in debye_huckel["charges"]:
                p_vs.q = debye_huckel["charges"][int(stype)]
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
        chk_virtual = (np.asarray(chk["particle_is_virtual"], dtype=bool)
                       if "particle_is_virtual" in chk.files else None)
        chk_mol = (np.asarray(chk["particle_mol_ids"], dtype=np.int64)
                   if "particle_mol_ids" in chk.files else None)

    if len(pos) == len(system.part):
        source_of = {i: i for i in range(len(system.part))}
    else:
        # Numero di particelle diverso: succede quando l'insieme di prior ha
        # contatti pair-specific con estremi nuovi (piu' marker virtuali, per
        # esempio un insieme derivato con add_state_contacts.py).  Lo stato
        # fisico sono solo i corpi rigidi (particelle reali): i siti CG e i
        # marker sono virtuali e seguono il corpo.  Con --allow_checkpoint_mismatch
        # si accoppiano le particelle reali nell'ordine degli id, controllando
        # che siano tante quante e appartengano alle stesse molecole.
        if not args.allow_checkpoint_mismatch or chk_virtual is None:
            raise ValueError(
                f"Checkpoint particle count ({len(pos)}) does not match system ({len(system.part)}); "
                "con un insieme di prior con piu' marker usa --allow_checkpoint_mismatch")
        chk_real = np.flatnonzero(~chk_virtual)
        run_real = [i for i in range(len(system.part)) if not particle_is_virtual(system.part.by_id(i))]
        if len(chk_real) != len(run_real):
            raise ValueError(f"Checkpoint real-particle count ({len(chk_real)}) does not match system ({len(run_real)})")
        if chk_mol is not None:
            run_mol = np.array([int(system.part.by_id(i).mol_id) for i in run_real])
            if not np.array_equal(chk_mol[chk_real], run_mol):
                raise ValueError("Checkpoint real particles belong to different molecules than the runtime ones")
        source_of = {i: int(j) for i, j in zip(run_real, chk_real)}
        print(f"[WARNING] Checkpoint con {len(pos)} particelle, sistema con {len(system.part)}: "
              f"ripristinati i {len(run_real)} corpi rigidi per id (siti e marker sono virtuali).")

    for i in range(len(system.part)):
        p = system.part.by_id(i)
        # Virtual sites positions/velocities are strictly tied to COM.
        # We only set the properties of the real (COM) particles, and the
        # virtual sites will follow automatically based on their auto-relation.
        if not particle_is_virtual(p):
            j = source_of[i]
            p.pos = pos[j]
            p.v = vel[j]
            if quat is not None:
                p.quat = quat[j]
            if omega is not None:
                p.omega_body = omega[j]

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
    elif b_type in ("morse", "lj"):
        # contatti pair-specific: configurati sui marker, non come legami
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
if any(d.get("cbt", False) for d in priors.get("dihedrals", [])):
    require_cbt_dihedral(espressomd)
for idx, d in enumerate(priors.get("dihedrals", [])):
    d_type = d.get("type", "cosine")
    if d_type == "cosine":
        k_dih = d["k"]
        mult = d.get("n", 1)
        phase = d["phi0"]
        if d.get("cbt", False):
            # diedro con smorzamento angolare (install_dihedral_cbt.py): mult < 0
            # lo spegne in modo C1 quando un angolo di legame va verso 180 gradi,
            # cosi' la forza resta finita quando tre siti si allineano
            mult = -abs(int(mult))
        dihedral = espressomd.interactions.Dihedral(bend=k_dih, mult=mult, phase=phase)
    elif d_type == "tabulated":
        import numpy as np
        data = np.loadtxt(resolve_referenced_path(d["file"], args.priors))
        min_tab = float(d.get("min", -np.pi))
        max_tab = float(d.get("max", np.pi))
        dihedral = espressomd.interactions.TabulatedDihedral(
            min=min_tab, max=max_tab, energy=data[:, 1], force=data[:, 2]
        )
    elif d_type == "conservative_spline":
        dihedral = create_conservative_spline_interaction(
            espressomd.interactions, d, kind="dihedral", priors_path=args.priors
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
    # Il DH agisce sui siti fisici, lato regolare della decomposizione ibrida:
    # con un cutoff_regular piu' corto le coppie oltre verrebbero perse in silenzio.
    float(debye_huckel["r_cut"]) if debye_huckel else 0.0,
)
if debye_huckel:
    _dh_required = 2.0 * (float(debye_huckel["r_cut"]) + float(system.cell_system.skin))
    if min(system.box_l) <= _dh_required:
        raise ValueError(
            f"Debye-Hueckel r_cut={debye_huckel['r_cut']:.4g} nm troppo lungo per la scatola "
            f"{list(system.box_l)}: serve box > {_dh_required:.4g} nm"
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
marker_nonbonded_morse = bool(
    morse_contacts and args.pair_specific_morse_runtime == "marker-nonbonded"
)
morse_n_square_types = (
    {DUMMY_COM_TYPE, *morse_marker_types.values()} if marker_nonbonded_morse else None
)
configure_neighbor_search(
    system, args.neighbor_search,
    n_square_types=morse_n_square_types,
    cutoff_regular=regular_cutoff if marker_nonbonded_morse else None,
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
configure_type_pair_morse(system, morse_type_pairs, switch_mode=args.morse_switch_mode)
configure_debye_huckel(system, debye_huckel)
if debye_huckel:
    _n_q = sum(1 for p in system.part if abs(float(p.q)) > 0.0)
    print(f"[INFO] Debye-Hueckel attivo su {_n_q} siti carichi (carica totale "
          f"{sum(float(p.q) for p in system.part):.1f})")
for item in morse_type_pairs:
    print(
        "[INFO] Added type-pair reversible Morse interaction "
        f"{item['index']}: site type {item['type_i']} <-> {item['type_j']} "
        f"(mode={args.morse_switch_mode}, r_switch={item['r_switch']:.6g}, "
        f"r_cut={item['r_cut']:.6g})"
    )

if args.pair_specific_morse_runtime == "marker-nonbonded":
    configure_pair_specific_morse(
        system, morse_contacts, morse_marker_types, switch_mode=args.morse_switch_mode
    )
    for contact in morse_contacts:
        print(
            "[INFO] Added pair-specific reversible contact "
            f"{contact['index']}: {contact['mol_i']}:{contact['site_i']} <-> "
            f"{contact['mol_j']}:{contact['site_j']} "
            f"(site=-1 means COM; runtime=marker-nonbonded, mode={args.morse_switch_mode}, "
            f"{pair_contact_summary(contact)})"
        )
else:
    configure_pair_specific_morse_bonds(
        system, morse_contacts, mol_com_parts, mol_vs_parts, espressomd.interactions
    )
    if morse_contacts:
        print(
            f"[INFO] Added {len(morse_contacts)} pair-specific analytic Morse bonds "
            "directly on physical endpoints; technical Morse markers remain inert "
            "for checkpoint/particle-set parity."
        )

if ml_active:
    print("[INFO] Activating ML Potential...")
    _painn_kwargs = dict(
        model_path=args.model,
        num_species=nn_config["num_species"],
        hidden_channels=nn_config["hidden_channels"],
        n_layers=nn_config["n_layers"],
        num_rbf=nn_config["num_rbf"],
        cutoff=nn_config["cutoff"],
        toxvaerd_alpha=args.toxvaerd_alpha,
        ordered_geometry_nodes=int(nn_config.get("ordered_geometry_nodes", 0)),
        ordered_geometry_head_layers=int(nn_config.get("ordered_geometry_head_layers", 0)),
        ordered_geometry_head_width=int(nn_config.get("ordered_geometry_head_width", 0)),
        ordered_geometry_energy_scale_kj_mol=float(
            nn_config.get("ordered_geometry_energy_scale_kj_mol", 0.0)
        ),
        ordered_geometry_head_only=bool(
            nn_config.get("ordered_geometry_head_only", False)
        ),
        ordered_geometry_copies=int(nn_config.get("ordered_geometry_copies", 1)),
        tel22_shared_geometry=bool(
            nn_config.get("architecture_variant") == "tel22_shared_geometry_tanh_v1"
        ),
        device=args.device,
        precision=args.ml_precision,
    )
    # Compatibilita' driver/plugin: il lato plugin di
    # PATCH_TEL22_SHARED_CGNET_HEAD.md puo' non essere applicato, quindi
    # espressomd.painn puo' non conoscere alcuni kwargs. Vengono scartati SOLO
    # se il loro valore e' inerte; se portano un valore significativo il run si
    # ferma, perche' ignorarli cambierebbe silenziosamente il modello simulato.
    _painn_inert_defaults = {
        "ordered_geometry_nodes": 0,
        "ordered_geometry_head_layers": 0,
        "ordered_geometry_head_width": 0,
        "ordered_geometry_energy_scale_kj_mol": 0.0,
        "ordered_geometry_head_only": False,
        "ordered_geometry_copies": 1,
        "tel22_shared_geometry": False,
    }
    for _ in range(len(_painn_inert_defaults) + 1):
        try:
            espressomd.painn.activate_painn_potential(**_painn_kwargs)
            break
        except TypeError as _exc:
            _match = re.search(r"unexpected keyword argument '([^']+)'", str(_exc))
            if _match is None:
                raise
            _bad = _match.group(1)
            if _bad not in _painn_kwargs:
                raise
            _val = _painn_kwargs.pop(_bad)
            if (
                _bad not in _painn_inert_defaults
                or _val != _painn_inert_defaults[_bad]
            ):
                raise SystemExit(
                    f"[FATAL] espressomd.painn non supporta '{_bad}', ma la "
                    f"configurazione lo richiede (valore={_val!r}). Applica il "
                    "lato plugin di PATCH_TEL22_SHARED_CGNET_HEAD.md e ricompila "
                    "ESPResSo: ignorarlo cambierebbe il modello simulato."
                )
            print(
                f"[WARN] espressomd.painn non supporta '{_bad}' (plugin piu' "
                f"vecchio del driver). Valore inerte {_val!r}: proseguo senza."
            )
    else:
        raise SystemExit(
            "[FATAL] Impossibile attivare il potenziale PaiNN: kwargs non "
            "supportati oltre il limite di tentativi."
        )
    # Libera il potenziale (tensori, grafi CUDA) all'uscita di Python, mentre
    # il runtime CUDA e' ancora attivo; i distruttori statici arrivano dopo.
    if hasattr(espressomd.painn, "deactivate_painn_potential"):
        import atexit
        atexit.register(espressomd.painn.deactivate_painn_potential)
    if args.painn_profile_report is not None:
        espressomd.painn.configure_painn_profiling(
            True, warmup_calls=args.painn_profile_warmup_calls
        )
        print(
            "[PROFILE] PaiNN C++ stage profiling enabled "
            f"(device={args.device}, graph={painn_graph}, "
            f"warmup_calls={args.painn_profile_warmup_calls})"
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



def _minimum_image_box():
    """Lati della box e loro inversi (0 sugli assi non periodici)."""
    box = np.asarray(system.box_l, dtype=float)
    periodic = np.asarray(system.periodicity, dtype=bool)
    inverse = np.where(periodic, 1.0 / box, 0.0)
    return box, inverse


def _minimum_image(diff, box, inverse):
    """Spostamenti (..., 3) ridotti all'immagine minima."""
    return diff - box * np.round(diff * inverse)


def _minimum_image_distance_matrix(pos):
    """Matrice delle distanze sito-sito con l'immagine minima.

    Le posizioni lette da ESPResSo sono non ripiegate: senza immagine minima
    due siti di copie diverse a contatto attraverso il bordo della box
    risultano lontani, e il controllo su min_dist non li vede.
    """
    box, inverse = _minimum_image_box()
    squared = np.zeros((len(pos), len(pos)), dtype=float)
    for axis in range(3):
        d = pos[:, axis][:, None] - pos[:, axis][None, :]
        d = d - box[axis] * np.round(d * inverse[axis])
        squared += d * d
    return np.sqrt(squared)


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
    
    dist_matrix = _minimum_image_distance_matrix(pos)
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


def measure_energies(fast=None):
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
    e_bonded = float(energies.get("bonded", 0.0))
    e_non_bonded = float(energies.get("non_bonded", 0.0))

    e_kin = energies["kinetic"]
    if fast is not None:
        e_kin_trans, e_kin_rot = _fast_kinetic(fast)
        return e_tot, e_kin, e_kin_trans, e_kin_rot, e_class, e_ml, e_bonded, e_non_bonded
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
    return e_tot, e_kin, e_kin_trans, e_kin_rot, e_class, e_ml, e_bonded, e_non_bonded


def run_generalized_fd_probe(report_path):
    """Check full-system generalized gradients against force and torque.

    Translation uses central Cartesian finite differences. Rotation uses
    ``ParticleHandle.rotate(axis, angle)`` with lab-frame Cartesian axes and is
    compared with ``torque_lab``. The probe restores the exact state before
    returning and never advances physical time.
    """
    system.integrator.run(0, recalc_forces=True)
    real_particles = [p for p in system.part if float(p.mass) > 1.0e-4]
    if not real_particles:
        raise RuntimeError("No real particles available for generalized FD probe")
    rotational = [p for p in real_particles if any(bool(v) for v in p.rotation)]
    rotational_ids = {int(p.id) for p in rotational}
    nonrot = [p for p in real_particles if int(p.id) not in rotational_ids]
    ordered = rotational + nonrot
    if args.generalized_fd_max_bodies:
        ordered = ordered[:args.generalized_fd_max_bodies]

    base = {}
    for p in ordered:
        base[int(p.id)] = {
            "pos": np.asarray(p.pos, dtype=float).copy(),
            "quat": np.asarray(p.quat, dtype=float).copy(),
            "force": np.asarray(p.f, dtype=float).copy(),
            "torque_lab": np.asarray(p.torque_lab, dtype=float).copy(),
            "rotation": [bool(v) for v in p.rotation],
        }

    def total_energy():
        return float(measure_energies()[0])

    axes = np.eye(3, dtype=float)
    rows = []
    worst_force = 0.0
    worst_torque = 0.0
    for p in ordered:
        pid = int(p.id)
        info = base[pid]
        translation = []
        for axis_index, axis in enumerate(axes):
            p.pos = info["pos"] + args.generalized_fd_eps_pos * axis
            system.integrator.run(0, recalc_forces=True)
            e_plus = total_energy()
            p.pos = info["pos"] - args.generalized_fd_eps_pos * axis
            system.integrator.run(0, recalc_forces=True)
            e_minus = total_energy()
            p.pos = info["pos"]
            system.integrator.run(0, recalc_forces=True)
            fd = -(e_plus - e_minus) / (2.0 * args.generalized_fd_eps_pos)
            actual = float(info["force"][axis_index])
            error = abs(fd - actual)
            worst_force = max(worst_force, error)
            translation.append({
                "axis": int(axis_index), "actual_force": actual,
                "fd_force": float(fd), "abs_error": float(error),
            })

        rotation = []
        if any(info["rotation"]):
            for axis_index, axis in enumerate(axes):
                p.quat = info["quat"]
                p.rotate(axis, float(args.generalized_fd_eps_rot))
                system.integrator.run(0, recalc_forces=True)
                e_plus = total_energy()
                p.quat = info["quat"]
                p.rotate(axis, -float(args.generalized_fd_eps_rot))
                system.integrator.run(0, recalc_forces=True)
                e_minus = total_energy()
                p.quat = info["quat"]
                system.integrator.run(0, recalc_forces=True)
                fd = -(e_plus - e_minus) / (2.0 * args.generalized_fd_eps_rot)
                actual = float(info["torque_lab"][axis_index])
                error = abs(fd - actual)
                worst_torque = max(worst_torque, error)
                rotation.append({
                    "axis": int(axis_index), "actual_torque_lab": actual,
                    "fd_torque": float(fd), "abs_error": float(error),
                })
        rows.append({
            "particle_id": pid,
            "rotation_flags": info["rotation"],
            "translation": translation,
            "rotation": rotation,
        })

    # Restore every sampled particle and refresh all dependent virtual sites.
    for p in ordered:
        info = base[int(p.id)]
        p.pos = info["pos"]
        p.quat = info["quat"]
    system.integrator.run(0, recalc_forces=True)
    report = {
        "schema_version": 1,
        "kind": "full_system_generalized_energy_gradient",
        "hamiltonian_mode": (
            "conservative_classical_model_provenance_ml_disabled" if args.disable_ml
            else "painn_active" if ml_active else "classical_only"
        ),
        "eps_position_nm": float(args.generalized_fd_eps_pos),
        "eps_rotation_rad": float(args.generalized_fd_eps_rot),
        "n_bodies": int(len(rows)),
        "worst_force_abs_error": float(worst_force),
        "worst_torque_abs_error": float(worst_torque),
        "particles": rows,
    }
    path = os.path.abspath(report_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        "[GENERALIZED FD] "
        f"bodies={len(rows)} max|dF|={worst_force:.3e} max|dTau|={worst_torque:.3e} "
        f"report={path}"
    )
    return report


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


def record_structured_sample(step, fast=None):
    if args.sample_npz is None or step < args.sample_start_step:
        return
    sample_steps.append(int(step))
    if fast is not None:
        sample_com.append(fast["com"].get("pos"))
        sample_sites.append(fast["sites"].get("pos"))
        return
    sample_com.append(np.asarray([
        system.part.by_id(mol_com_parts[mol]).pos for mol in range(num_molecules)
    ], dtype=float))
    sample_sites.append(np.asarray([
        system.part.by_id(mol_vs_parts[key]).pos for key in sample_site_keys
    ], dtype=float))


def record_state_sample(step, fast=None):
    if args.state_sample_npz is None:
        return
    state_sample_steps.append(int(step))
    if fast is not None:
        try:
            omega_fast = fast["state"].get("omega_body")
        except Exception:  # noqa: BLE001 - come il percorso per particella
            omega_fast = None
        if omega_fast is not None:
            state_sample_positions.append(fast["state"].get("pos"))
            state_sample_velocities.append(fast["state"].get("v"))
            state_sample_quaternions.append(fast["state"].get("quat", width=4))
            state_sample_omegas.append(omega_fast)
            return
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



# ---------------------------------------------------------------------------
# Log vettoriale.  Il log per particella (system.part / by_id, una chiamata
# all'interfaccia di ESPResSo per ogni proprieta' di ogni particella) costava
# ~1.5-1.8 s per registrazione: a dt 4 fs con un log ogni ps sono ~6-7 ms per
# passo, quasi quanto i prior.  Qui le proprieta' statiche (tipi, mol_id,
# masse, inerzie, coppie ammesse per min_dist) si leggono una volta e quelle
# dinamiche con una sola lettura per slice.  Al primo uso (passo 0) il
# risultato e' confrontato con quello per particella; se differisce si torna
# al percorso per particella con un avviso.  --legacy_diagnostics lo forza.
# ---------------------------------------------------------------------------
class _OrderedSlice:
    """Accesso vettoriale a una lista fissa di particelle, nell'ordine dato.

    ESPResSo restituisce le proprieta' ottimizzate (pos, type, q) ordinate per
    id e le altre nell'ordine della selezione: la slice si costruisce sugli id
    ordinati e ogni array e' riportato all'ordine richiesto.
    """

    def __init__(self, ids):
        ids = np.asarray([int(i) for i in ids], dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("_OrderedSlice: id di particella duplicati")
        order = np.argsort(ids, kind="stable")
        self.ids = ids
        self._inverse = np.empty_like(order)
        self._inverse[order] = np.arange(len(ids))
        self._slice = system.part.by_ids([int(i) for i in ids[order]]) if len(ids) else None

    def get(self, attribute, width=3):
        if self._slice is None:
            return np.zeros((0, width), dtype=float)
        values = np.asarray(getattr(self._slice, attribute), dtype=float)
        return values.reshape(len(self.ids), -1)[self._inverse]


_fast_log_state = {"cache": None, "disabled": bool(args.legacy_diagnostics), "records": 0}
if args.energy_interval < 1:
    raise ValueError("--energy_interval must be >= 1")


def _build_fast_log_cache():
    all_ids, vs_ids, vs_types, vs_mol = [], [], [], []
    kin_ids, kin_mass, kin_inertia, torque_ids = [], [], [], []
    for p in system.part:
        pid = int(p.id)
        all_ids.append(pid)
        if particle_is_virtual(p):
            vs_ids.append(pid)
            vs_types.append(int(p.type))
            vs_mol.append(int(p.mol_id))
        mass = float(p.mass)
        if not mass < 1e-4:          # stesso criterio di measure_energies
            kin_ids.append(pid)
            kin_mass.append(mass)
            kin_inertia.append(np.asarray(p.rinertia, dtype=float))
        if mass > 1e-4:              # stesso criterio di record_state
            torque_ids.append(pid)
    vs_types = np.asarray(vs_types, dtype=np.int64)
    vs_mol = np.asarray(vs_mol, dtype=np.int64)
    n_vs = len(vs_ids)
    blocked = np.zeros((n_vs, n_vs), dtype=float)
    blocked[vs_mol[:, None] == vs_mol[None, :]] = np.inf
    np.fill_diagonal(blocked, np.inf)
    mask_excluded_particle_distances(
        blocked, vs_ids, diagnostic_nonbonded_excluded_pid_pairs
    )
    num_species = int(nn_config["num_species"])
    in_range = (vs_types >= 0) & (vs_types < num_species)
    allowed = np.isfinite(blocked) & in_range[:, None] & in_range[None, :]
    upper_i, upper_j = np.triu_indices(n_vs, 1)
    keep = allowed[upper_i, upper_j]
    return {
        "all": _OrderedSlice(all_ids),
        "vs": _OrderedSlice(vs_ids),
        "vs_ids": np.asarray(vs_ids, dtype=np.int64),
        "vs_types": vs_types,
        "pair_i": upper_i[keep],
        "pair_j": upper_j[keep],
        "kin": _OrderedSlice(kin_ids),
        "kin_mass": np.asarray(kin_mass, dtype=float),
        "kin_inertia": np.asarray(kin_inertia, dtype=float).reshape(-1, 3),
        "torque": _OrderedSlice(torque_ids),
        "com": _OrderedSlice([mol_com_parts[mol] for mol in range(num_molecules)]),
        "sites": _OrderedSlice([mol_vs_parts[key] for key in sample_site_keys]),
        "state": _OrderedSlice(state_sample_particle_ids),
    }


def _fast_diagnostics(cache):
    pos = cache["vs"].get("pos")
    forces = cache["vs"].get("f")
    f_max = np.max(np.linalg.norm(forces, axis=1))
    pair_i, pair_j = cache["pair_i"], cache["pair_j"]
    if len(pair_i) == 0:
        return np.inf, None, None, f_max
    box, inverse = _minimum_image_box()
    diff = _minimum_image(pos[pair_i] - pos[pair_j], box, inverse)
    dist = np.sqrt(diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2])
    k = int(np.argmin(dist))
    i, j = pair_i[k], pair_j[k]
    t_i, t_j = int(cache["vs_types"][i]), int(cache["vs_types"][j])
    pair = (min(t_i, t_j), max(t_i, t_j))
    pids = (cache["vs_ids"][i], cache["vs_ids"][j])
    return float(dist[k]), pair, pids, f_max


def _fast_kinetic(cache):
    v = cache["kin"].get("v")
    omega = cache["kin"].get("omega_body")
    e_trans = 0.5 * float(np.sum(cache["kin_mass"] * np.sum(v * v, axis=1)))
    e_rot = 0.5 * float(np.sum(cache["kin_inertia"] * omega * omega))
    return e_trans, e_rot


def _fast_max_torque(cache):
    torque = cache["torque"].get("torque_lab")
    if len(torque) == 0:
        return 0.0
    return float(np.max(np.sqrt(np.sum(torque * torque, axis=1))))


def _fast_vcf_text(cache):
    pos = cache["all"].get("pos")
    lines = ["\ntimestep indexed\n"]
    for vtf_id, row in enumerate(pos):
        lines.append(f"{vtf_id} {' '.join(map(str, row))}\n")
    return "".join(lines)


def _legacy_max_torque():
    real_particles = [p for p in system.part if p.mass > 1e-4]
    return max(
        (sum(t_c**2 for t_c in p.torque_lab) ** 0.5 for p in real_particles),
        default=0.0,
    )


def _verify_fast_log(cache, step):
    """Confronta log vettoriale e per particella sullo stato corrente."""
    import io
    problems = []

    def close(a, b, rtol=1e-10, atol=1e-12):
        return (a == b) or abs(a - b) <= atol + rtol * max(abs(a), abs(b))

    t0 = time.perf_counter()
    legacy_e = measure_energies()
    legacy_d = log_diagnostics(step)
    legacy_t = _legacy_max_torque()
    legacy_vcf = io.StringIO()
    espressomd.io.writer.vtf.writevcf(system, legacy_vcf)
    legacy_com = np.asarray([
        system.part.by_id(mol_com_parts[mol]).pos for mol in range(num_molecules)
    ], dtype=float)
    legacy_sites = np.asarray([
        system.part.by_id(mol_vs_parts[key]).pos for key in sample_site_keys
    ], dtype=float)
    t1 = time.perf_counter()
    fast_e = measure_energies(cache)
    fast_d = _fast_diagnostics(cache)
    fast_t = _fast_max_torque(cache)
    fast_vcf = _fast_vcf_text(cache)
    fast_com = cache["com"].get("pos")
    fast_sites = cache["sites"].get("pos")
    t2 = time.perf_counter()

    for name, a, b in (("E_kin_trans", legacy_e[2], fast_e[2]),
                       ("E_kin_rot", legacy_e[3], fast_e[3]),
                       ("max_f", legacy_d[3], fast_d[3]),
                       ("max_t", legacy_t, fast_t)):
        if not close(float(a), float(b)):
            problems.append(f"{name}: {a!r} contro {b!r}")
    if not (close(float(legacy_d[0]), float(fast_d[0]), rtol=0.0, atol=1e-12)
            or (np.isinf(legacy_d[0]) and np.isinf(fast_d[0]))):
        problems.append(f"min_dist: {legacy_d[0]!r} contro {fast_d[0]!r}")
    if stringify_pair(legacy_d[1]) != stringify_pair(fast_d[1]):
        problems.append(f"min_pair: {legacy_d[1]!r} contro {fast_d[1]!r}")
    if stringify_pair(legacy_d[2]) != stringify_pair(fast_d[2]):
        problems.append(f"min_pids: {legacy_d[2]!r} contro {fast_d[2]!r}")
    if legacy_vcf.getvalue() != fast_vcf:
        problems.append("testo VCF diverso")
    if not (np.array_equal(legacy_com, fast_com) and np.array_equal(legacy_sites, fast_sites)):
        problems.append("posizioni COM/siti dei campioni diverse")

    # Tempi per voce del percorso vettoriale, e controllo dell'energia
    # cinetica dalle velocita' (usata nelle registrazioni senza energy()).
    voices = {}
    tic = time.perf_counter()
    energies = system.analysis.energy()
    voices["energy()"] = time.perf_counter() - tic
    for name, call in (("cinetica", lambda: _fast_kinetic(cache)),
                       ("min_dist/max_f", lambda: _fast_diagnostics(cache)),
                       ("max_t", lambda: _fast_max_torque(cache)),
                       ("VCF", lambda: _fast_vcf_text(cache)),
                       ("campioni", lambda: (cache["com"].get("pos"), cache["sites"].get("pos")))):
        tic = time.perf_counter()
        call()
        voices[name] = time.perf_counter() - tic
    e_trans, e_rot = _fast_kinetic(cache)
    kinetic_espresso = float(energies["kinetic"])
    kinetic_rel = abs(e_trans + e_rot - kinetic_espresso) / max(abs(kinetic_espresso), 1e-12)
    return problems, t1 - t0, t2 - t1, voices, kinetic_rel


def _fast_log_cache(step):
    """Cache del log vettoriale, o None per il percorso per particella."""
    state = _fast_log_state
    if state["disabled"]:
        return None
    if state["cache"] is None:
        try:
            cache = _build_fast_log_cache()
            problems, t_legacy, t_fast, voices, kinetic_rel = _verify_fast_log(cache, step)
        except Exception as exc:  # noqa: BLE001 - mai fermare la produzione per il log
            problems, t_legacy, t_fast, cache = [f"eccezione: {exc!r}"], 0.0, 0.0, None
            voices, kinetic_rel = {}, float("nan")
        if problems:
            print("[WARN] Log vettoriale NON coerente con quello per particella; "
                  "uso il percorso per particella. Differenze:")
            for problem in problems:
                print(f"    {problem}")
            state["disabled"] = True
            return None
        state["cache"] = cache
        print(f"[INFO] Log vettoriale verificato al passo {step}: "
              f"per particella {t_legacy:.3f} s, vettoriale {t_fast:.3f} s per registrazione")
        print("[INFO] Log vettoriale, tempi per voce: " + ", ".join(
            f"{name} {1000.0 * value:.1f} ms" for name, value in voices.items()))
        if args.energy_interval > 1:
            state["kinetic_ok"] = kinetic_rel <= 1e-6
            print(f"[INFO] Energia completa ogni {args.energy_interval} registrazioni; "
                  f"E_kin dalle velocita' contro ESPResSo: scarto relativo {kinetic_rel:.2e}"
                  + ("" if state["kinetic_ok"] else
                     " -> NON coincide: energia completa a ogni registrazione"))
    return state["cache"]

def _light_energies(fast):
    """Registrazione senza system.analysis.energy(): cinetica dalle velocita',
    E_ML dal plugin (gia' calcolata con le forze), il resto NaN."""
    e_trans, e_rot = _fast_kinetic(fast)
    e_ml = espressomd.painn.get_painn_energy() if ml_active else 0.0
    nan = float("nan")
    return nan, e_trans + e_rot, e_trans, e_rot, nan, e_ml, nan, nan


def record_state(step, energy_writer, energy_handle, vtf_handle):
    fast = _fast_log_cache(step)
    state = _fast_log_state
    full_energy = (
        fast is None
        or args.energy_interval <= 1
        or not state.get("kinetic_ok", False)
        or state["records"] % args.energy_interval == 0
    )
    state["records"] += 1
    if full_energy:
        e_tot, e_kin, e_kin_trans, e_kin_rot, e_class, e_ml, e_bonded, e_non_bonded = measure_energies(fast)
    else:
        e_tot, e_kin, e_kin_trans, e_kin_rot, e_class, e_ml, e_bonded, e_non_bonded = _light_energies(fast)
    if fast is None:
        g_dist, g_pair, g_pids, max_f = log_diagnostics(step)
        max_t = _legacy_max_torque()
    else:
        g_dist, g_pair, g_pids, max_f = _fast_diagnostics(fast)
        max_t = _fast_max_torque(fast)
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
            e_bonded,
            e_non_bonded,
            g_dist,
            stringify_pair(g_pair),
            stringify_pair(g_pids),
            max_f,
            max_t,
        ])
        energy_handle.flush()

    if vtf_handle is not None:
        vtf_handle.write(f"\ntimestep {step}\n")
        if fast is None:
            espressomd.io.writer.vtf.writevcf(system, vtf_handle)
        else:
            vtf_handle.write(_fast_vcf_text(fast))

    record_structured_sample(step, fast)
    record_state_sample(step, fast)
    unsafe = max_f > 10000.0 or e_kin > 5000.0 or g_dist < 0.15
    return unsafe


if args.generalized_fd_report is not None:
    run_generalized_fd_probe(args.generalized_fd_report)

simulation_ok = True
if args.dump_initial_forces:
    # Parita' dei prior runtime <-> dataset: forze e coppie di ogni corpo sulla
    # configurazione iniziale (il frame 0 del dataset se non c'e' --checkpoint),
    # senza termostato, cosi' da confrontarle con quelle che build_cg_dataset.py
    # ha sottratto (--dump-prior-forces).
    system.thermostat.turn_off()
    system.integrator.run(0, recalc_forces=True)
    _ids = [mol_com_parts[m] for m in sorted(mol_com_parts)]
    np.savez(
        args.dump_initial_forces,
        mol=np.asarray(sorted(mol_com_parts), dtype=np.int64),
        com=np.asarray([system.part.by_id(i).pos for i in _ids], dtype=float),
        force=np.asarray([system.part.by_id(i).f for i in _ids], dtype=float),
        torque_lab=np.asarray([system.part.by_id(i).torque_lab for i in _ids], dtype=float),
        ml_active=np.asarray(ml_active),
        e_ml=np.asarray(espressomd.painn.get_painn_energy() if ml_active else 0.0, dtype=float),
        painn_graph=np.asarray(painn_graph),
    )
    print(f"[DONE] forze e coppie iniziali di {len(_ids)} corpi -> {args.dump_initial_forces}")
    sys.exit(0)

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
            "E_bonded",
            "E_non_bonded",
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
    integration_wall_seconds = 0.0
    record_wall_seconds = 0.0
    record_count = 0
    _WALL_LOOP_START = time.perf_counter()
    while simulation_ok and completed < args.steps:
        current = min(args.log_interval, args.steps - completed)
        integration_start = time.perf_counter()
        system.integrator.run(current)
        integration_wall_seconds += time.perf_counter() - integration_start
        completed += current

        record_start = time.perf_counter()
        unsafe_state = record_state(completed, energy_writer, energy_handle, vtf_handle)
        record_wall_seconds += time.perf_counter() - record_start
        record_count += 1
        if unsafe_state:
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

_WALL_LOOP_END = time.perf_counter()

if simulation_ok:
    if args.painn_profile_report is not None:
        profile = espressomd.painn.get_painn_profile()
        timings = profile.get("timings_ms", {})
        total_mean_ms = float(timings.get("total_mean", 0.0))
        stage_keys = (
            "node_index_mean",
            "neighbor_traversal_mean",
            "edge_pack_mean",
            "tensor_inputs_mean",
            "forward_mean",
            "energy_scalar_mean",
            "autograd_mean",
            "force_to_cpu_mean",
            "force_scatter_mean",
            "unattributed_cleanup_mean",
        )
        profile["stage_fraction_of_painn_total"] = {
            key.removesuffix("_mean"): (
                float(timings.get(key, 0.0)) / total_mean_ms if total_mean_ms > 0.0 else 0.0
            )
            for key in stage_keys
        }
        profile["integration"] = {
            "requested_steps": int(args.steps),
            "completed_steps": int(completed),
            "wall_seconds_integrator_only": float(integration_wall_seconds),
            "wall_ms_per_step": (
                1000.0 * float(integration_wall_seconds) / completed if completed else 0.0
            ),
            "painn_total_ms_per_force_call": total_mean_ms,
            "painn_force_calls_per_requested_step": (
                float(profile.get("total_calls", 0)) / completed if completed else 0.0
            ),
        }
        profile["runtime_config"] = {
            "device": args.device,
            "painn_graph": painn_graph,
            "ml_precision": args.ml_precision,
            "neighbor_search": args.neighbor_search,
            "dt_ps": float(args.dt),
            "num_species": int(nn_config["num_species"]),
            "hidden_channels": int(nn_config["hidden_channels"]),
            "n_layers": int(nn_config["n_layers"]),
            "num_rbf": int(nn_config["num_rbf"]),
            "cutoff_nm": float(nn_config["cutoff"]),
            "architecture_variant": str(nn_config["architecture_variant"]),
            "ordered_geometry_nodes": int(nn_config.get("ordered_geometry_nodes", 0)),
            "ordered_geometry_head_layers": int(nn_config.get("ordered_geometry_head_layers", 0)),
            "ordered_geometry_head_width": int(nn_config.get("ordered_geometry_head_width", 0)),
            "ordered_geometry_energy_scale_kj_mol": float(
                nn_config.get("ordered_geometry_energy_scale_kj_mol", 0.0)
            ),
            "ordered_geometry_head_only": bool(
                nn_config.get("ordered_geometry_head_only", False)
            ),
            "ordered_geometry_copies": int(
                nn_config.get("ordered_geometry_copies", 1)
            ),
        }
        report_path = os.path.abspath(args.painn_profile_report)
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(
            "[PROFILE] "
            f"wall={profile['integration']['wall_ms_per_step']:.6g} ms/step "
            f"PaiNN={total_mean_ms:.6g} ms/force-call "
            f"calls={profile.get('measured_calls', 0)} measured "
            f"N={profile.get('graph', {}).get('particles_mean', 0):.1f} "
            f"E={profile.get('graph', {}).get('directed_edges_mean', 0):.1f}"
        )
        print(f"[PROFILE] report: {report_path}")

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
            "morse_switch_mode": args.morse_switch_mode,
            "pair_specific_morse_runtime": args.pair_specific_morse_runtime,
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
                "morse_switch_mode": args.morse_switch_mode,
            "pair_specific_morse_runtime": args.pair_specific_morse_runtime,
            },
        )
        print(
            f"[INFO] Final checkpoint saved: {checkpoint_path} "
            f"(hamiltonian_mode={hamiltonian_mode})"
        )
    print("\n[INFO] Simulation finished successfully.")
else:
    print("\n[ERROR] Simulation terminated by a safety guardrail.")

# Bilancio dei tempi: dove va il tempo di una produzione (avvio, integrazione,
# registrazioni, scritture finali) e ns/giorno effettivi.
try:
    _wall_end = time.perf_counter()
    _total = _wall_end - _WALL_T0
    _startup = _WALL_LOOP_START - _WALL_T0
    _loop = _WALL_LOOP_END - _WALL_LOOP_START
    _final = _wall_end - _WALL_LOOP_END
    _other = _loop - integration_wall_seconds - record_wall_seconds
    _ns = completed * float(args.dt) / 1000.0
    print(
        f"[TIMING] totale {_total:.1f} s | avvio {_startup:.1f} s | "
        f"integrazione {integration_wall_seconds:.1f} s "
        f"({1000.0 * integration_wall_seconds / max(1, completed):.2f} ms/passo, "
        f"{completed} passi in {record_count} chiamate) | "
        f"registrazioni {record_wall_seconds:.1f} s "
        f"({1000.0 * record_wall_seconds / max(1, record_count):.0f} ms ciascuna) | "
        f"resto del ciclo {_other:.1f} s | scritture finali {_final:.1f} s"
    )
    if _total > 0 and _ns > 0:
        print(
            f"[TIMING] {_ns:.4g} ns simulati: {_ns / _total * 86400:.2f} ns/giorno effettivi, "
            f"{_ns / max(integration_wall_seconds, 1e-9) * 86400:.2f} ns/giorno di sola integrazione"
        )
except NameError:
    pass  # uscita prima del ciclo principale

# Force immediate exit to bypass PyTorch/MPI teardown crashes on macOS
import sys
sys.stdout.flush()
os._exit(0 if simulation_ok else 2)
