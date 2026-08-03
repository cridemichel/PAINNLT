import MDAnalysis as mda
import numpy as np
import struct
import json
import argparse
import sys
import os
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist



# =====================================================================
# 1. PARSING ARGOMENTI E CARICAMENTO TOPOLOGIA
# =====================================================================
parser = argparse.ArgumentParser(description="Coarse-Graining Dataset Builder with Direct Boltzmann Inversion")
parser.add_argument("-c", "--topology", type=str, required=True, help="File di topologia (es. .tpr o .gro)")
parser.add_argument("-f", "--trajectory", type=str, required=True, help="File di traiettoria (es. .trr o .xtc)")
parser.add_argument("-j", "--config", type=str, default="topology_config.json", help="File JSON con topologia CG e mapping")

parser.add_argument("-p", "--priors", type=str, default=None, help="File JSON con prior pre-esistenti (salta calcolo statistico DBI e carica questo)")
parser.add_argument("-o", "--output", type=str, default="../training/cg_dataset.bin", help="Nome del file binario di output")
parser.add_argument("--clip_forces", type=float, default=None, help="Valore massimo per il modulo delle forze residue. Se non specificato, nessun clip viene applicato (raccomandato per priors analitici dolci).")
args = parser.parse_args()

try:
    with open(args.config, "r") as mf:
        config_data = json.load(mf)
except FileNotFoundError:
    print(f"[ERRORE] File di configurazione '{args.config}' non trovato!")
    sys.exit(1)

TEMPERATURE = config_data.get("temperature", 300.0)
KBOLTZMANN = 1.38064852e-23
AVOGADRO = 6.022140857e23
JPERKCAL = 4184
# Convert kJ/mol/K
# 1 kcal = 4.184 kJ.
# R = N_A * k_B = 8.314 J / (mol K) = 0.008314 kJ / (mol K)
R_KJ_MOL_K = 0.008314462618
BETA = 1.0 / (R_KJ_MOL_K * TEMPERATURE)

MAPPING_DATA = config_data.get("mapping", {})
MAPPING_METHOD = MAPPING_DATA.get("mapping_method", "COM")
mapping_by_resname = MAPPING_DATA.get("residues", {})
site_types = MAPPING_DATA.get("site_types", {})

BONDS = config_data.get("bonds", [])
WCA_SIGMA = config_data.get("wca_sigma", 0.0)
WCA_EPSILON = config_data.get("wca_epsilon", 0.0)
WCA_OVERRIDES = config_data.get("wca_overrides", {})



# ANGLES will be loaded below

ANGLES = config_data.get("angles", [])
DIHEDRALS = config_data.get("dihedrals", [])

print(f"[INFO] Caricamento MDAnalysis: {args.topology}, {args.trajectory}...")
u = mda.Universe(args.topology, args.trajectory)

# Mapping dictionary for atomic masses
ATOMIC_MASSES = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'P': 30.974, 'S': 32.065, 'K': 39.098}
def get_mass(atom_name):
    alpha_chars = ''.join([c for c in atom_name if c.isalpha()]).upper()
    return ATOMIC_MASSES.get(alpha_chars[0], 12.0) if alpha_chars else 12.0



def get_unwrapped_positions(positions, box_dim):
    unwrapped = np.copy(positions)
    ref = unwrapped[0]
    for i in range(1, len(unwrapped)):
        dvec = unwrapped[i] - ref
        dvec -= box_dim * np.round(dvec / box_dim)
        unwrapped[i] = ref + dvec
    return unwrapped

def compute_com(positions, masses):
    total_mass = np.sum(masses)
    return np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass

def compute_inertia_tensor(positions, masses, center):
    rel_pos = positions - center
    I = np.zeros((3, 3))
    for r, m in zip(rel_pos, masses):
        I += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return I

def kabsch_align(P, Q):
    """Finds rotation matrix R that aligns P to Q using SVD"""
    H = np.dot(P.T, Q)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R


def mic_vector(pos1, pos2, box_dim):
    """Vettore da pos1 a pos2 con Minimum Image Convention"""
    dvec = pos2 - pos1
    return dvec - box_dim * np.round(dvec / box_dim)

def get_angle(pos_i, pos_j, pos_k, box_dim):
    r_ji = mic_vector(pos_j, pos_i, box_dim)
    r_jk = mic_vector(pos_j, pos_k, box_dim)
    d_ji = np.linalg.norm(r_ji)
    d_jk = np.linalg.norm(r_jk)
    if d_ji < 1e-6 or d_jk < 1e-6: return 0.0
    cos_theta = np.clip(np.dot(r_ji, r_jk) / (d_ji * d_jk), -1.0, 1.0)
    return np.arccos(cos_theta)

def angle_forces(pos_i, pos_j, pos_k, box_dim, k, theta0):
    r_ji = mic_vector(pos_j, pos_i, box_dim)
    r_jk = mic_vector(pos_j, pos_k, box_dim)
    d_ji = np.linalg.norm(r_ji)
    d_jk = np.linalg.norm(r_jk)
    if d_ji < 1e-6 or d_jk < 1e-6: return np.zeros(3), np.zeros(3), np.zeros(3)
    cos_theta = np.clip(np.dot(r_ji, r_jk) / (d_ji * d_jk), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    if sin_theta < 1e-6: return np.zeros(3), np.zeros(3), np.zeros(3)
    dV_dtheta = k * (theta - theta0)
    grad_i_cos = r_jk / (d_ji * d_jk) - cos_theta * r_ji / (d_ji**2)
    grad_k_cos = r_ji / (d_ji * d_jk) - cos_theta * r_jk / (d_jk**2)
    force_i = (dV_dtheta / sin_theta) * grad_i_cos
    force_k = (dV_dtheta / sin_theta) * grad_k_cos
    return force_i, -(force_i + force_k), force_k

def get_dihedral(pos_i, pos_j, pos_k, pos_l, box_dim):
    b1 = mic_vector(pos_i, pos_j, box_dim)
    b2 = mic_vector(pos_j, pos_k, box_dim)
    b3 = mic_vector(pos_k, pos_l, box_dim)
    m1 = np.cross(b1, b2)
    m2 = np.cross(b2, b3)
    m1_sq = np.dot(m1, m1)
    m2_sq = np.dot(m2, m2)
    if m1_sq < 1e-12 or m2_sq < 1e-12: return 0.0
    b2_norm = np.linalg.norm(b2)
    cos_phi = np.clip(np.dot(m1, m2) / np.sqrt(m1_sq * m2_sq), -1.0, 1.0)
    sin_phi = np.dot(b2, np.cross(m1, m2)) / (b2_norm * np.sqrt(m1_sq * m2_sq))
    return np.arctan2(sin_phi, cos_phi)

def dihedral_forces(pos_i, pos_j, pos_k, pos_l, box_dim, K, n, phi0):
    b1 = mic_vector(pos_i, pos_j, box_dim)
    b2 = mic_vector(pos_j, pos_k, box_dim)
    b3 = mic_vector(pos_k, pos_l, box_dim)
    m1 = np.cross(b1, b2)
    m2 = np.cross(b2, b3)
    m1_sq = np.dot(m1, m1)
    m2_sq = np.dot(m2, m2)
    if m1_sq < 1e-12 or m2_sq < 1e-12:
        return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    b2_norm = np.linalg.norm(b2)
    cos_phi = np.clip(np.dot(m1, m2) / np.sqrt(m1_sq * m2_sq), -1.0, 1.0)
    sin_phi = np.dot(b2, np.cross(m1, m2)) / (b2_norm * np.sqrt(m1_sq * m2_sq))
    phi = np.arctan2(sin_phi, cos_phi)
    dV_dphi = K * n * np.sin(n * phi - phi0)
    grad_i = (b2_norm / m1_sq) * m1
    grad_l = -(b2_norm / m2_sq) * m2
    dot12 = np.dot(b1, b2)
    dot23 = np.dot(b2, b3)
    grad_j = ((dot12 / (b2_norm**2)) - 1.0) * grad_i - (dot23 / (b2_norm**2)) * grad_l
    grad_k = -(grad_i + grad_j + grad_l)
    return -dV_dphi * grad_i, -dV_dphi * grad_j, -dV_dphi * grad_k, -dV_dphi * grad_l

# =====================================================================
# 2. PASS 1: DIRECT BOLTZMANN INVERSION
# =====================================================================
print(f"[INFO] Pass 1: Direct Boltzmann Inversion su {len(u.trajectory)} frame a T={TEMPERATURE} K...")

bond_distances = {tuple(b): [] for b in BONDS if isinstance(b, list)}
for b_idx, b in enumerate(BONDS):
    if isinstance(b, dict): bond_distances[f"dict_{b_idx}"] = []
angle_values = {f"dict_{idx}": [] for idx in range(len(ANGLES))}
dihedral_values = {f"dict_{idx}": [] for idx in range(len(DIHEDRALS))}
rigid_bodies_info = {}

min_pairwise_distances = {} # Per auto-WCA: (type_i, type_j) -> min_distance

# Helper to map entire trajectory frames into memory for CG centers
# Assuming each molecule is a CG site for the non-bonded/bonded priors as per convert_gro2bin
# Wait, if mapping is per molecule (rigid body):
cg_centers_history = []
cg_forces_history = []
cg_torques_history = []
box_dim_history = []
sites_data_history = []
mol_site_indices = {}

for ts_idx, ts in enumerate(u.trajectory):
    if ts_idx % 100 == 0:
        print(f"\r[INFO] Inversione Boltzmann: Frame {ts_idx}/{len(u.trajectory)}", end="")
    
    box_dim = ts.dimensions[:3] / 10.0 # nm
    box_dim_history.append(box_dim)
    
    frame_centers = []
    frame_forces = []
    frame_torques = []
    frame_sites = []
    
    valid_residues = [res for res in u.residues if res.resname in mapping_by_resname]
    if ts_idx == 0:
        mol_resnames = [res.resname for res in valid_residues]
        
    for mol_id, residue in enumerate(valid_residues):
        resname = residue.resname
        current_mapping = mapping_by_resname[resname]
        
        atoms = residue.atoms
        positions_nm = atoms.positions / 10.0
        # Check if forces exist
        if hasattr(atoms, 'forces'):
            forces_nm = atoms.forces * 10.0
        else:
            forces_nm = np.zeros_like(positions_nm)
            
        try:
            masses = atoms.masses
        except:
            masses = np.array([get_mass(name) for name in atoms.names])
            
        unwrapped_pos = get_unwrapped_positions(positions_nm, box_dim)
        center = compute_com(unwrapped_pos, masses)
        
        total_force = np.sum(forces_nm, axis=0)
        r_vec = unwrapped_pos - center
        total_torque = np.sum(np.cross(r_vec, forces_nm), axis=0)
        
        if ts_idx == 0:
            total_mass = float(np.sum(masses))
            I_tensor = compute_inertia_tensor(unwrapped_pos, masses, center)
            eigvals = np.linalg.eigvalsh(I_tensor)
            if resname not in rigid_bodies_info:
                rigid_bodies_info[resname] = {
                    "mass_amu": round(total_mass, 4),
                    "inertia_amu_nm2": [round(v, 4) for v in eigvals],
                    "sites": {}
                }
        
        frame_centers.append(center)
        frame_forces.append(total_force)
        frame_torques.append(total_torque)
        
        if ts_idx == 0:
            mol_site_indices[mol_id] = {}
            for site_name, atom_names in current_mapping.items():
                if atom_names == ["*"]:
                    mol_site_indices[mol_id][site_name] = list(range(len(atoms)))
                else:
                    sel = "name " + " ".join(atom_names)
                    site_atoms = atoms.select_atoms(sel)
                    mol_site_indices[mol_id][site_name] = [list(atoms.names).index(n) for n in site_atoms.names]
        
        sites_for_mol = []
        for site_name, atom_names in current_mapping.items():
            indices = mol_site_indices[mol_id][site_name]
            if len(indices) == 0: continue
            
            if MAPPING_METHOD == "COM":
                site_pos = compute_com(unwrapped_pos[indices], masses[indices])
            elif MAPPING_METHOD == "COG":
                site_pos = np.mean(unwrapped_pos[indices], axis=0)
            elif MAPPING_METHOD == "ATOM":
                site_pos = unwrapped_pos[indices[0]]
            
            site_type = site_types[site_name]
            if ts_idx == 0:
                if resname in rigid_bodies_info and site_name not in rigid_bodies_info[resname].get("sites", {}):
                    relative_pos_nm = site_pos - center
                    rigid_bodies_info[resname]["sites"][site_name] = {
                        "type": site_type, "relative_pos_nm": [round(v, 4) for v in relative_pos_nm]
                    }
            sites_for_mol.append((site_type, site_pos))
            
        frame_sites.append(sites_for_mol)
        
    cg_centers_history.append(frame_centers)
    cg_forces_history.append(frame_forces)
    cg_torques_history.append(frame_torques)
    sites_data_history.append(frame_sites)
    
    # Collect bond distances for Boltzmann Inversion (only if specified as a pair [i, j] or [i, j, site_i, site_j])
    for b_idx, b in enumerate(BONDS):
        if isinstance(b, list) and len(b) >= 2:
            i, j = b[0], b[1]
            site_i = b[2] if len(b) > 2 else -1
            site_j = b[3] if len(b) > 3 else -1
            b_key = tuple(b)
        elif isinstance(b, dict) and b.get("r0") == "auto":
            i, j = b["mol_i"], b["mol_j"]
            site_i = b.get("site_i", -1)
            site_j = b.get("site_j", -1)
            b_key = f"dict_{b_idx}"
        else:
            continue
            
        if i < len(frame_centers) and j < len(frame_centers):
            pos_i = frame_centers[i] if site_i == -1 else frame_sites[i][site_i][1]
            pos_j = frame_centers[j] if site_j == -1 else frame_sites[j][site_j][1]
            
            r_vec = mic_vector(pos_i, pos_j, box_dim)
            r = np.linalg.norm(r_vec)
            bond_distances[b_key].append(r)
            
    for idx, a in enumerate(ANGLES):
        i, j, k = a["mol_i"], a["mol_j"], a["mol_k"]
        if i < len(frame_centers) and j < len(frame_centers) and k < len(frame_centers):
            pos_i, pos_j, pos_k = frame_centers[i], frame_centers[j], frame_centers[k]
            angle_values[f"dict_{idx}"].append(get_angle(pos_i, pos_j, pos_k, box_dim))
            
    for idx, d in enumerate(DIHEDRALS):
        i, j, k, l = d["mol_i"], d["mol_j"], d["mol_k"], d["mol_l"]
        if i < len(frame_centers) and j < len(frame_centers) and k < len(frame_centers) and l < len(frame_centers):
            pos_i, pos_j, pos_k, pos_l = frame_centers[i], frame_centers[j], frame_centers[k], frame_centers[l]
            dihedral_values[f"dict_{idx}"].append(get_dihedral(pos_i, pos_j, pos_k, pos_l, box_dim))
            
    if WCA_SIGMA == "auto":
        flat_pos = []
        flat_types = []
        flat_mols = []
        for mol_idx, sites in enumerate(frame_sites):
            # The COM might not be in sites, but if it's interacting it should be.
            # Usually only sites defined in mapping interact via WCA in our model.
            for stype, spos in sites:
                flat_pos.append(spos)
                flat_types.append(stype)
                flat_mols.append(mol_idx)
        
        if len(flat_pos) > 0:
            flat_pos = np.array(flat_pos)
            flat_types = np.array(flat_types)
            flat_mols = np.array(flat_mols)
            
            # Compute full pairwise distance matrix (very fast for ~1000 particles)
            dist_matrix = cdist(flat_pos, flat_pos)
            
            # Mask out particles in the same molecule
            same_mol_mask = flat_mols[:, None] == flat_mols[None, :]
            dist_matrix[same_mol_mask] = np.inf
            
            # Vectorized minimum distance computation
            i_idx, j_idx = np.tril_indices(len(flat_types), k=-1)
            dist_flat = dist_matrix[i_idx, j_idx]
            valid_mask = dist_flat < np.inf
            
            if np.any(valid_mask):
                valid_dist = dist_flat[valid_mask]
                types_i = flat_types[i_idx[valid_mask]]
                types_j = flat_types[j_idx[valid_mask]]
                
                t1 = np.minimum(types_i, types_j)
                t2 = np.maximum(types_i, types_j)
                pair_ids = t1 * 10000 + t2
                
                unique_pairs = np.unique(pair_ids)
                for pid in unique_pairs:
                    min_d = np.min(valid_dist[pair_ids == pid])
                    pair = (int(pid // 10000), int(pid % 10000))
                    if pair not in min_pairwise_distances or min_d < min_pairwise_distances[pair]:
                        min_pairwise_distances[pair] = min_d

print("\n[INFO] Esecuzione allineamento Kabsch per mediare le geometrie dei corpi rigidi...")
for resname, info in rigid_bodies_info.items():
    if not info.get("auto_align_sites", True):
        print(f"[INFO] {resname}: auto_align_sites=False. Salto la media di Kabsch e uso le posizioni manuali.")
        continue
        
    if "sites" not in info or len(info["sites"]) < 2:
        continue
        
    site_names = list(info["sites"].keys())
    N_sites = len(site_names)
    
    # Raccogli tutti gli snapshot delle coordinate relative
    snapshots = []
    for frame_idx, frame_sites in enumerate(sites_data_history):
        frame_centers = cg_centers_history[frame_idx]
        for mol_id, rname in enumerate(mol_resnames):
            if rname == resname:
                center = frame_centers[mol_id]
                site_positions = [site[1] for site in frame_sites[mol_id]]
                if len(site_positions) == N_sites:
                    rel_pos = np.array(site_positions) - center
                    snapshots.append(rel_pos)
                    
    snapshots = np.array(snapshots)
    if len(snapshots) == 0: continue
    
    # Allineamento iterativo (3 iterazioni)
    ref = snapshots[0].copy()
    for iteration in range(3):
        aligned_snapshots = []
        for snap in snapshots:
            R = kabsch_align(snap, ref)
            aligned_snapshots.append((R @ snap.T).T)
        ref = np.mean(aligned_snapshots, axis=0)
        
    # Aggiorna il JSON con le posizioni medie ideali
    for i, sname in enumerate(site_names):
        info["sites"][sname]["relative_pos_nm"] = [round(v, 4) for v in ref[i]]
    print(f"  -> {resname}: mediati {len(snapshots)} snapshot per ottenere la geometria ideale.")

print("\n[INFO] Pass 1 completato. Calcolo parametri Harmonic Priors...")

if args.priors:
    print(f"[INFO] Trovato flag --priors. Salto inferenza statistica e carico i prior esatti da: {args.priors}")
    with open(args.priors, "r") as f:
        derived_priors = json.load(f)
        wca_info = derived_priors.get("wca", {})
        WCA_SIGMA_VAL = float(wca_info.get("sigma", 0.0))
        # Override solo se epsilon è specificato nei priors, altrimenti tieni args
        if "epsilon" in wca_info:
            WCA_EPSILON = float(wca_info["epsilon"])
        WCA_OVERRIDES = wca_info.get("overrides", {})
else:
    derived_priors = None

if derived_priors is None:
    # Calcolo k e r0 statistico (Boltzmann Inversion)
    if WCA_SIGMA == "auto":
        print("\n[INFO] WCA_SIGMA è 'auto'. Tuning automatico dei diametri (sigma = min_dist / 1.122)...")
        
        overrides_dict = {}
        unique_types = np.unique(flat_types) if 'flat_types' in locals() else []
        for t in unique_types:
            min_d_t = np.inf
            for (t1, t2), min_d in min_pairwise_distances.items():
                if t1 == t or t2 == t:
                    if min_d < min_d_t:
                        min_d_t = min_d
                        
            if min_d_t < np.inf:
                optimal_sigma = min_d_t / (2**(1/6))
                # Usa np.floor per garantire che il cutoff (sigma * 1.122) sia strettamente MINORE di min_d_t!
                # In questo modo F_wca = 0 su tutto il dataset di training.
                optimal_sigma = np.floor(optimal_sigma * 1000) / 1000.0
                overrides_dict[str(int(t))] = {
                    "sigma": optimal_sigma,
                    "epsilon": WCA_EPSILON if WCA_EPSILON > 0 else 1000.0
                }
                print(f"  -> Tipo {int(t)}: min_dist = {min_d_t:.4f} nm -> optimal_sigma = {optimal_sigma:.3f} nm")
                
        WCA_OVERRIDES = overrides_dict
        WCA_SIGMA_VAL = 0.25 # Valore di fallback sicuro
    else:
        WCA_SIGMA_VAL = float(WCA_SIGMA)

    derived_priors = {"bonds": [], "wca": {"sigma": WCA_SIGMA_VAL, "epsilon": WCA_EPSILON if WCA_EPSILON > 0 else 1000.0, "overrides": WCA_OVERRIDES}}
    
    for b, dists in bond_distances.items():
        if isinstance(b, str): continue
        if len(dists) == 0: continue
        r_arr = np.array(dists)
        mean_r = np.mean(r_arr)
        var_r = np.var(r_arr)
        
        k_stat = 1.0 / (BETA * var_r) if var_r > 1e-12 else 0.0
        
        derived_priors["bonds"].append({
            "mol_i": b[0],
            "mol_j": b[1],
            "site_i": b[2] if len(b) > 2 else -1,
            "site_j": b[3] if len(b) > 3 else -1,
            "type": "harmonic",
            "k": float(k_stat),
            "r0": float(mean_r)
        })
        print(f"  -> Legame (Boltzmann) mol {b[0]} - mol {b[1]}: r0 = {mean_r:.4f} nm, k = {k_stat:.2f} kJ/mol/nm^2")

    # Pool statistics for named groups
    shared_bond_distances = {}
    for b_idx, b in enumerate(BONDS):
        if isinstance(b, dict) and "name" in b:
            b_name = b["name"]
            if b_name not in shared_bond_distances: shared_bond_distances[b_name] = []
            shared_bond_distances[b_name].extend(bond_distances[f"dict_{b_idx}"])

    # Aggiungi eventuali legami espliciti (FENE, Morse, o Harmonic custom) forniti come dizionari
    for b_idx, b in enumerate(BONDS):
        if isinstance(b, dict):
            b_type = b.get("type", "unknown")
            if b.get("r0") == "auto":
                vals = shared_bond_distances[b["name"]] if "name" in b else bond_distances[f"dict_{b_idx}"]
                if len(vals) > 0:
                    b["r0"] = float(np.mean(vals))
                    if b.get("k") == "auto":
                        var_r = np.var(vals)
                        b["k"] = float(1.0 / (BETA * var_r)) if var_r > 1e-12 else 0.0
                    print(f"  -> Auto-calcolato r0 per legame esplicito ({b_type}) {b['mol_i']}-{b['mol_j']}: r0 = {b['r0']:.4f} nm")
            
            derived_priors["bonds"].append(b)
            print(f"  -> Legame esplicito ({b.get('type', b_type)}) aggiunto: {b['mol_i']}-{b['mol_j']}")

    shared_angle_values = {}
    for idx, a in enumerate(ANGLES):
        if "name" in a:
            a_name = a["name"]
            if a_name not in shared_angle_values: shared_angle_values[a_name] = []
            shared_angle_values[a_name].extend(angle_values[f"dict_{idx}"])

    derived_priors["angles"] = []
    for idx, a in enumerate(ANGLES):
        vals = shared_angle_values[a["name"]] if "name" in a else angle_values[f"dict_{idx}"]
        a_type = a.get("type", "harmonic")
            
        if len(vals) > 0:
            if a.get("theta0") == "auto":
                a["theta0"] = float(np.mean(vals))
            if a.get("k") == "auto":
                var_theta = np.var(vals)
                a["k"] = float(1.0 / (BETA * var_theta)) if var_theta > 1e-12 else 0.0
        
        derived_priors["angles"].append(a)
        print(f"  -> Angle aggiunto: {a['mol_i']}-{a['mol_j']}-{a['mol_k']} (theta0={a.get('theta0', 0):.4f} rad, k={a.get('k', 0):.2f})")

    shared_dihedral_values = {}
    for idx, d in enumerate(DIHEDRALS):
        if "name" in d:
            d_name = d["name"]
            if d_name not in shared_dihedral_values: shared_dihedral_values[d_name] = []
            shared_dihedral_values[d_name].extend(dihedral_values[f"dict_{idx}"])

    derived_priors["dihedrals"] = []
    for idx, d in enumerate(DIHEDRALS):
        vals = shared_dihedral_values[d["name"]] if "name" in d else dihedral_values[f"dict_{idx}"]
        d_type = d.get("type", "cosine")
            
        if len(vals) > 0:
            if d.get("phi0") == "auto":
                sin_vals = np.sin(vals)
                cos_vals = np.cos(vals)
                d["phi0"] = float(np.arctan2(np.mean(sin_vals), np.mean(cos_vals)))
            if d.get("k") == "auto":
                var_phi = np.var(vals)
                d["k"] = float(1.0 / (BETA * var_phi)) if var_phi > 1e-12 else 0.0
        
        derived_priors["dihedrals"].append(d)
        print(f"  -> Dihedral aggiunto: {d['mol_i']}-{d['mol_j']}-{d['mol_k']}-{d['mol_l']} (phi0={d.get('phi0', 0):.4f} rad, k={d.get('k', 0):.2f})")

    # Save priors for simulation
    with open("cg_priors.json", "w") as pf:
        json.dump(derived_priors, pf, indent=4)
    print("[INFO] Salvato file cg_priors.json (da passare poi a run_cg_md.py)")

# =====================================================================
# 3. PASS 2: SOTTRAZIONE PRIOR E SCRITTURA BINARIO
# =====================================================================
print(f"[INFO] Pass 2: Sottrazione forze prior e generazione dataset {args.output}...")

out_dir = os.path.dirname(args.output)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

cached_tables = {}
cached_splines = {}

with open(args.output, "wb") as f:
    num_frames = len(cg_centers_history)
    f.write(struct.pack("i", num_frames))
    
    for ts_idx in range(num_frames):
        if ts_idx % 100 == 0:
            print(f"\r[INFO] Scrittura binario: Frame {ts_idx}/{num_frames}", end="")
            
        box_dim = box_dim_history[ts_idx]
        frame_centers = cg_centers_history[ts_idx]
        frame_forces = cg_forces_history[ts_idx]
        frame_torques = cg_torques_history[ts_idx]
        # Copia superficiale della lista per non alterare lo storico originario
        frame_sites = [list(sites) for sites in sites_data_history[ts_idx]]
        
        # [NEW LOGIC] RECONSTRUCT IDEAL SITES FOR PRIOR SUBTRACTION
        # Applichiamo Kabsch per posizionare i siti ideali con l'orientamento istantaneo
        for mol_idx, r_name in enumerate(mol_resnames):
            if r_name in rigid_bodies_info and "sites" in rigid_bodies_info[r_name]:
                site_names = list(rigid_bodies_info[r_name]["sites"].keys())
                if len(site_names) < 2: continue
                
                ideal_rel = []
                inst_rel = []
                site_indices = []
                
                for s_idx, (st, sp) in enumerate(frame_sites[mol_idx]):
                    if s_idx < len(site_names):
                        s_name = site_names[s_idx]
                        ideal_rel.append(rigid_bodies_info[r_name]["sites"][s_name]["relative_pos_nm"])
                        inst_rel.append(np.array(sp) - frame_centers[mol_idx])
                        site_indices.append(s_idx)
                
                if len(ideal_rel) >= 2:
                    ideal_rel = np.array(ideal_rel)
                    inst_rel = np.array(inst_rel)
                    
                    # R @ ideal_rel \approx inst_rel
                    R = kabsch_align(ideal_rel, inst_rel)
                    ideal_rel_rotated = (R @ ideal_rel.T).T
                    
                    for i_local, s_idx in enumerate(site_indices):
                        new_pos = frame_centers[mol_idx] + ideal_rel_rotated[i_local]
                        frame_sites[mol_idx][s_idx] = (frame_sites[mol_idx][s_idx][0], new_pos)
        
        num_molecules = len(frame_centers)
        num_total_sites = sum(len(s) for s in frame_sites)
        
        f.write(struct.pack("i", num_molecules))
        f.write(struct.pack("i", num_total_sites))
        f.write(struct.pack("3f", *box_dim))
        
        # Copia indipendente per la sottrazione
        res_forces = np.copy(frame_forces)
        res_torques = np.copy(frame_torques)
        
        # 3.1 Sottrazione WCA sui siti virtuali
        has_wca = WCA_SIGMA_VAL > 0 or (isinstance(WCA_OVERRIDES, dict) and len(WCA_OVERRIDES) > 0)
        has_epsilon = WCA_EPSILON > 0 or (isinstance(WCA_OVERRIDES, dict) and any(o.get("epsilon", 0) > 0 for o in WCA_OVERRIDES.values()))
        
        if has_wca and has_epsilon:
            flat_pos = []
            flat_mol = []
            flat_type = []
            for m_idx, sites in enumerate(frame_sites):
                for s_type, s_pos in sites:
                    flat_pos.append(s_pos)
                    flat_mol.append(m_idx)
                    flat_type.append(s_type)
            
            if len(flat_pos) > 0:
                flat_pos = np.array(flat_pos)
                flat_mol = np.array(flat_mol)
                flat_type = np.array(flat_type)
                
                # Retrieve parameters with overrides
                sigmas = np.array([WCA_OVERRIDES.get(str(int(t)), {}).get("sigma", WCA_SIGMA_VAL) for t in flat_type])
                epsilons = np.array([WCA_OVERRIDES.get(str(int(t)), {}).get("epsilon", WCA_EPSILON) for t in flat_type])
                
                diff = flat_pos[:, np.newaxis, :] - flat_pos[np.newaxis, :, :]
                diff -= box_dim * np.round(diff / box_dim)
                dist_sq = np.sum(diff**2, axis=-1)
                
                # Lorentz-Berthelot mixing
                sigma_ij = (sigmas[:, np.newaxis] + sigmas[np.newaxis, :]) / 2.0
                eps_ij = np.sqrt(epsilons[:, np.newaxis] * epsilons[np.newaxis, :])
                r_cut_sq = (sigma_ij * (2.0**(1.0/6.0)))**2
                
                # Only distinct molecules (intra-molecular forces cancel out in rigid bodies)
                idx_i, idx_j = np.where((dist_sq > 1e-6) & (dist_sq < r_cut_sq))
                valid = (idx_i < idx_j) & (flat_mol[idx_i] != flat_mol[idx_j])
                idx_i = idx_i[valid]
                idx_j = idx_j[valid]
                
                for i, j in zip(idx_i, idx_j):
                    mol_i = flat_mol[i]
                    mol_j = flat_mol[j]
                    

                        
                    r_sq = dist_sq[i, j]
                    r = np.sqrt(r_sq)
                    r_hat = diff[i, j] / r
                    s_ij = sigma_ij[i, j]
                    e_ij = eps_ij[i, j]
                    
                    f_scalar = 24.0 * e_ij * (2.0 * (s_ij/r)**12 - (s_ij/r)**6) / r
                    f_vec = f_scalar * r_hat
                    
                    res_forces[mol_i] -= f_vec
                    res_forces[mol_j] += f_vec
                    
                    # Compute torque around the COM of each molecule
                    r_site_i = flat_pos[i] - frame_centers[mol_i]
                    r_site_j = flat_pos[j] - frame_centers[mol_j]
                    
                    res_torques[mol_i] -= np.cross(r_site_i, f_vec)
                    res_torques[mol_j] += np.cross(r_site_j, f_vec)
                
        # 3.2 Sottrazione Legami (con supporto per siti specifici e momento torcente)
        for b in derived_priors["bonds"]:
            i, j = b["mol_i"], b["mol_j"]
            site_i = b.get("site_i", -1)
            site_j = b.get("site_j", -1)

            if i >= num_molecules or j >= num_molecules: continue

            pos_i = frame_centers[i] if site_i == -1 else frame_sites[i][site_i][1]
            pos_j = frame_centers[j] if site_j == -1 else frame_sites[j][site_j][1]

            r_vec = mic_vector(pos_i, pos_j, box_dim)
            r = np.linalg.norm(r_vec)
            if r < 1e-6: continue

            r_hat = r_vec / r
            f_scalar = 0.0
            b_type = b.get("type", "harmonic")

            if b_type == "harmonic":
                k, r0 = b["k"], b["r0"]
                f_scalar = - k * (r - r0)

            elif b_type == "fene":
                k, r0, r_max = b["k"], b["r0"], b["r_max"]
                diff = r - r0
                if abs(diff) >= r_max:
                    f_scalar = 0.0
                else:
                    f_scalar = - k * diff / (1.0 - (diff/r_max)**2)

            elif b_type == "morse":
                D, a, r0 = b["D"], b["a"], b["r0"]
                diff = r - r0
                exp_term = np.exp(-a * diff)
                f_scalar = - 2.0 * a * D * (1.0 - exp_term) * exp_term



            # Safety clamp: prevents a single badly-sampled bin from injecting
            # enormous forces into the residual dataset and crashing training.
            F_MAX_BOND = 10000.0  # kJ/mol/nm
            f_scalar = float(np.clip(f_scalar, -F_MAX_BOND, F_MAX_BOND))

            f_vec = - f_scalar * r_hat

            res_forces[i] -= f_vec
            res_forces[j] += f_vec

            if site_i != -1:
                r_rel_i = mic_vector(frame_centers[i], pos_i, box_dim)
                res_torques[i] -= np.cross(r_rel_i, f_vec)

            if site_j != -1:
                r_rel_j = mic_vector(frame_centers[j], pos_j, box_dim)
                res_torques[j] += np.cross(r_rel_j, f_vec)
                
        # 3.2.1 Sottrazione Angoli
        for a in derived_priors.get("angles", []):
            a_type = a.get("type", "harmonic")
            if a_type in ["ibi", "dbi"]:
                continue
            
            i, j, k_idx = a["mol_i"], a["mol_j"], a["mol_k"]
            if i >= num_molecules or j >= num_molecules or k_idx >= num_molecules: continue
            
            site_i, site_j, site_k = a.get("site_i", -1), a.get("site_j", -1), a.get("site_k", -1)
            pos_i, pos_j, pos_k = frame_centers[i], frame_centers[j], frame_centers[k_idx]
            
            if site_i != -1:
                for st, sp in frame_sites[i]:
                    if st == site_i: pos_i = sp; break
            if site_j != -1:
                for st, sp in frame_sites[j]:
                    if st == site_j: pos_j = sp; break
            if site_k != -1:
                for st, sp in frame_sites[k_idx]:
                    if st == site_k: pos_k = sp; break
            
            r_ji = mic_vector(pos_j, pos_i, box_dim)
            r_jk = mic_vector(pos_j, pos_k, box_dim)
            d_ji = np.linalg.norm(r_ji)
            d_jk = np.linalg.norm(r_jk)
            if d_ji < 1e-6 or d_jk < 1e-6: continue
            
            cos_theta = np.clip(np.dot(r_ji, r_jk) / (d_ji * d_jk), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            sin_theta = np.sqrt(1.0 - cos_theta**2)
            if sin_theta < 1e-6: continue
            
            if a_type == "harmonic":
                dV_dtheta = a["k"] * (theta - a["theta0"])

            else:
                dV_dtheta = 0.0
                
            if abs(dV_dtheta) > 1000.0:
                dV_dtheta = np.sign(dV_dtheta) * 1000.0

            grad_i_cos = r_jk / (d_ji * d_jk) - cos_theta * r_ji / (d_ji**2)
            grad_k_cos = r_ji / (d_ji * d_jk) - cos_theta * r_jk / (d_jk**2)
            
            scalar_force = dV_dtheta / sin_theta
            if abs(scalar_force) > 1000.0:
                scalar_force = np.sign(scalar_force) * 1000.0
                
            f_i = scalar_force * grad_i_cos
            f_k = scalar_force * grad_k_cos
            
            # Explicit vector clip to prevent gradient magnitude explosion
            f_i = np.clip(f_i, -1000.0, 1000.0)
            f_k = np.clip(f_k, -1000.0, 1000.0)
            f_j = -(f_i + f_k)
            
            res_forces[i] -= f_i
            res_forces[j] -= f_j
            res_forces[k_idx] -= f_k
            
            if site_i != -1: res_torques[i] -= np.cross(mic_vector(frame_centers[i], pos_i, box_dim), f_i)
            if site_j != -1: res_torques[j] -= np.cross(mic_vector(frame_centers[j], pos_j, box_dim), f_j)
            if site_k != -1: res_torques[k_idx] -= np.cross(mic_vector(frame_centers[k_idx], pos_k, box_dim), f_k)
            
        # 3.2.2 Sottrazione Diedri
        for d in derived_priors.get("dihedrals", []):
            d_type = d.get("type", "cosine")
            if d_type in ["ibi", "dbi"]:
                continue
                
            i, j, k_idx, l = d["mol_i"], d["mol_j"], d["mol_k"], d["mol_l"]
            if i >= num_molecules or j >= num_molecules or k_idx >= num_molecules or l >= num_molecules: continue
            
            site_i, site_j, site_k, site_l = d.get("site_i", -1), d.get("site_j", -1), d.get("site_k", -1), d.get("site_l", -1)
            pos_i, pos_j, pos_k, pos_l = frame_centers[i], frame_centers[j], frame_centers[k_idx], frame_centers[l]
            
            if site_i != -1:
                for st, sp in frame_sites[i]:
                    if st == site_i: pos_i = sp; break
            if site_j != -1:
                for st, sp in frame_sites[j]:
                    if st == site_j: pos_j = sp; break
            if site_k != -1:
                for st, sp in frame_sites[k_idx]:
                    if st == site_k: pos_k = sp; break
            if site_l != -1:
                for st, sp in frame_sites[l]:
                    if st == site_l: pos_l = sp; break
            b1 = mic_vector(pos_i, pos_j, box_dim)
            b2 = mic_vector(pos_j, pos_k, box_dim)
            b3 = mic_vector(pos_k, pos_l, box_dim)
            
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            n1_norm2 = np.dot(n1, n1)
            n2_norm2 = np.dot(n2, n2)
            if n1_norm2 < 1e-8 or n2_norm2 < 1e-8: continue
            
            b2_norm = np.linalg.norm(b2)
            if b2_norm < 1e-6: continue
            
            m1 = np.cross(n1, b2) / b2_norm
            x = np.dot(n1, n2) / np.sqrt(n1_norm2 * n2_norm2)
            y = np.dot(m1, n2) / np.sqrt(np.dot(m1, m1) * n2_norm2)
            phi = np.arctan2(y, x)
            
            if d_type == "cosine":
                n_mult = d.get("n", 1)
                dV_dphi = d["k"] * n_mult * np.sin(n_mult * phi - d["phi0"])

            else:
                dV_dphi = 0.0
                
            if abs(dV_dphi) > 1000.0:
                dV_dphi = np.sign(dV_dphi) * 1000.0
                
            f_i = (b2_norm / n1_norm2) * dV_dphi * n1
            f_l = -(b2_norm / n2_norm2) * dV_dphi * n2
            
            dot12 = np.dot(b1, b2)
            dot23 = np.dot(b2, b3)
            term1 = (dot12 / (b2_norm**2)) * f_i
            term2 = (dot23 / (b2_norm**2)) * f_l
            
            f_j = -f_i + term1 - term2
            f_k = -f_l - term1 + term2
            
            # Explicit vector clip to prevent any component from exceeding the safety threshold
            f_i = np.clip(f_i, -1000.0, 1000.0)
            f_l = np.clip(f_l, -1000.0, 1000.0)
            f_j = np.clip(f_j, -1000.0, 1000.0)
            f_k = np.clip(f_k, -1000.0, 1000.0)
            
            res_forces[i] -= f_i
            res_forces[j] -= f_j
            res_forces[k_idx] -= f_k
            res_forces[l] -= f_l
            
            if site_i != -1: res_torques[i] -= np.cross(mic_vector(frame_centers[i], pos_i, box_dim), f_i)
            if site_j != -1: res_torques[j] -= np.cross(mic_vector(frame_centers[j], pos_j, box_dim), f_j)
            if site_k != -1: res_torques[k_idx] -= np.cross(mic_vector(frame_centers[k_idx], pos_k, box_dim), f_k)
            if site_l != -1: res_torques[l] -= np.cross(mic_vector(frame_centers[l], pos_l, box_dim), f_l)
            
            
        # 3.2.2 Sottrazione Diedri
        # (Omitted lines in between handled properly)
        
        # 3.3 Clip delle forze residue e scrittura nel file
        # Il clip finale è FONDAMENTALE quando si usa l'IBI, perché la sottrazione dei potenziali IBI (che hanno muri molto ripidi)
        # crea degli artefatti spaventosi (forze > 1000) sui bordi delle distribuzioni.
        if args.clip_forces is not None:
            res_forces = np.clip(res_forces, -args.clip_forces, args.clip_forces)
            res_torques = np.clip(res_torques, -args.clip_forces, args.clip_forces)
        
        for mol_id in range(num_molecules):
            num_sites = len(frame_sites[mol_id])
            f.write(struct.pack("i", mol_id))
            f.write(struct.pack("i", num_sites))
            f.write(struct.pack("3f", *frame_centers[mol_id]))
            f.write(struct.pack("3f", *res_forces[mol_id]))
            f.write(struct.pack("3f", *res_torques[mol_id]))
            for site_type, site_pos in frame_sites[mol_id]:
                f.write(struct.pack("i", site_type))
                f.write(struct.pack("3f", *site_pos))

print("\n[INFO] Conversione completata e forze residue salvate con successo nel dataset!")

with open("rigid_bodies_info.json", "w") as jf:
    json.dump(rigid_bodies_info, jf, indent=4)
print("[INFO] Masse e inerzie salvate in rigid_bodies_info.json!")
