import MDAnalysis as mda
import numpy as np

from geometry_utils import diagonalize_inertia_tensor, minimum_image_distance_matrix
import struct
import json
import argparse
import sys
import os
from scipy.ndimage import gaussian_filter1d



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
RIGID_BODIES_CONFIG = config_data.get("rigid_bodies", {})



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

def resolve_site_position(frame_centers, frame_sites, mol_idx, site_idx):
    """Resolve a prior site reference using the runtime convention.

    site_idx == -1 addresses the rigid-body COM; non-negative values are
    virtual-site indices, not site-type identifiers.
    """
    if site_idx == -1:
        return frame_centers[mol_idx]
    if site_idx < 0 or site_idx >= len(frame_sites[mol_idx]):
        raise IndexError(
            f"Invalid site index {site_idx} for molecule {mol_idx}; "
            f"available sites: {len(frame_sites[mol_idx])}"
        )
    return frame_sites[mol_idx][site_idx][1]

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

def dihedral_energy(pos_i, pos_j, pos_k, pos_l, box_dim, K, n, phi0):
    phi = get_dihedral(pos_i, pos_j, pos_k, pos_l, box_dim)
    return K * (1.0 - np.cos(n * phi - phi0))


def dihedral_forces(pos_i, pos_j, pos_k, pos_l, box_dim, K, n, phi0):
    """Conservative reference forces for the cosine dihedral.

    A central finite difference is deliberately used here.  Preprocessing is
    offline, and this avoids silently subtracting an analytic expression whose
    sign/index convention differs from the ESPResSo dihedral definition.
    """
    positions = np.array([pos_i, pos_j, pos_k, pos_l], dtype=float)
    forces = np.zeros_like(positions)
    eps = 1.0e-6

    def energy(coords):
        return dihedral_energy(*coords, box_dim, K, n, phi0)

    for atom in range(4):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += eps
            minus[atom, axis] -= eps
            forces[atom, axis] = -(energy(plus) - energy(minus)) / (2.0 * eps)

    # Remove round-off drift while preserving internal forces.
    forces -= forces.mean(axis=0, keepdims=True)
    return tuple(forces)


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
principal_axes_lab_by_resname = {}

all_pairwise_distances = {} # Per statistical WCA: (type_i, type_j) -> list of distances

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
        
        if ts_idx == 0 and resname not in rigid_bodies_info:
            total_mass = float(np.sum(masses))
            I_tensor = compute_inertia_tensor(unwrapped_pos, masses, center)
            eigvals, principal_axes = diagonalize_inertia_tensor(I_tensor)
            principal_axes_lab_by_resname[resname] = principal_axes
            rb_config = RIGID_BODIES_CONFIG.get(resname, {})
            rigid_bodies_info[resname] = {
                "schema_version": 2,
                "body_frame": "principal_axes",
                "auto_align_sites": bool(rb_config.get("auto_align_sites", True)),
                "mass_amu": float(total_mass),
                "inertia_amu_nm2": [float(v) for v in eigvals],
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
                    local_by_global = {
                        int(global_index): local_index
                        for local_index, global_index in enumerate(atoms.indices)
                    }
                    mol_site_indices[mol_id][site_name] = [
                        local_by_global[int(global_index)]
                        for global_index in site_atoms.indices
                    ]
        
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
                        "type": int(site_type),
                        "relative_pos_nm": [float(v) for v in relative_pos_nm],
                    }
            sites_for_mol.append((site_type, site_pos))
            
        frame_sites.append(sites_for_mol)
        
    cg_centers_history.append(frame_centers)
    cg_forces_history.append(frame_forces)
    cg_torques_history.append(frame_torques)
    sites_data_history.append(frame_sites)
    
    # --- Estrazione distanze WCA non-bonded ---
    if WCA_SIGMA == "auto":
        flat_pos = []
        flat_types = []
        flat_mols = []
        for m_idx, mol_sites in enumerate(frame_sites):
            for s_type, s_pos in mol_sites:
                flat_pos.append(s_pos)
                flat_types.append(int(s_type))
                flat_mols.append(m_idx)
                
        flat_pos = np.array(flat_pos)
        flat_types = np.array(flat_types)
        flat_mols = np.array(flat_mols)
        
        # Calculate MIC distance matrix
        dist_matrix = minimum_image_distance_matrix(flat_pos, box_dim)
        
        # Mask out intra-molecular distances (same rigid body)
        same_mol_mask = flat_mols[:, None] == flat_mols[None, :]
        dist_matrix[same_mol_mask] = np.inf
        
        # We only want unique pairs (upper triangle)
        i_idx, j_idx = np.triu_indices(len(flat_pos), k=1)
        valid_dist = dist_matrix[i_idx, j_idx]
        
        types_i = flat_types[i_idx]
        types_j = flat_types[j_idx]
        
        t1 = np.minimum(types_i, types_j)
        t2 = np.maximum(types_i, types_j)
        
        # We only care about distances < 1.5 nm for WCA parametrization to save memory
        close_mask = valid_dist < 1.5
        
        if np.any(close_mask):
            valid_dist = valid_dist[close_mask]
            t1 = t1[close_mask]
            t2 = t2[close_mask]
            
            pair_ids = t1 * 10000 + t2
            unique_pairs = np.unique(pair_ids)
            
            for pid in unique_pairs:
                dists = valid_dist[pair_ids == pid]
                pair = (str(int(pid // 10000)), str(int(pid % 10000))) # Must use strings for JSON matching in WCA fit!
                if pair not in all_pairwise_distances:
                    all_pairwise_distances[pair] = []
                # Subsample if too many, to avoid OOM
                if len(dists) > 1000:
                    dists = np.random.choice(dists, 1000, replace=False)
                all_pairwise_distances[pair].extend(dists)

    
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
            pos_i = resolve_site_position(frame_centers, frame_sites, i, site_i)
            pos_j = resolve_site_position(frame_centers, frame_sites, j, site_j)
            
            r_vec = mic_vector(pos_i, pos_j, box_dim)
            r = np.linalg.norm(r_vec)
            bond_distances[b_key].append(r)
            
    for idx, a in enumerate(ANGLES):
        i, j, k = a["mol_i"], a["mol_j"], a["mol_k"]
        if i < len(frame_centers) and j < len(frame_centers) and k < len(frame_centers):
            pos_i = resolve_site_position(frame_centers, frame_sites, i, a.get("site_i", -1))
            pos_j = resolve_site_position(frame_centers, frame_sites, j, a.get("site_j", -1))
            pos_k = resolve_site_position(frame_centers, frame_sites, k, a.get("site_k", -1))
            angle_values[f"dict_{idx}"].append(get_angle(pos_i, pos_j, pos_k, box_dim))
            
    for idx, d in enumerate(DIHEDRALS):
        i, j, k, l = d["mol_i"], d["mol_j"], d["mol_k"], d["mol_l"]
        if i < len(frame_centers) and j < len(frame_centers) and k < len(frame_centers) and l < len(frame_centers):
            pos_i = resolve_site_position(frame_centers, frame_sites, i, d.get("site_i", -1))
            pos_j = resolve_site_position(frame_centers, frame_sites, j, d.get("site_j", -1))
            pos_k = resolve_site_position(frame_centers, frame_sites, k, d.get("site_k", -1))
            pos_l = resolve_site_position(frame_centers, frame_sites, l, d.get("site_l", -1))
            dihedral_values[f"dict_{idx}"].append(get_dihedral(pos_i, pos_j, pos_k, pos_l, box_dim))
            
if WCA_SIGMA == "auto":
    print("\n[INFO] Calcolo parametri WCA geometrici con regolarizzazione gerarchica...")
    import scipy.optimize
    from scipy.ndimage import gaussian_filter1d

    # Trova tutti i tipi unici
    all_types = set()
    for (t1, t2) in all_pairwise_distances.keys():
        all_types.add(t1)
        all_types.add(t2)
    all_types = sorted(list(all_types))
    n_types = len(all_types)
    type_to_idx = {t: i for i, t in enumerate(all_types)}

    empirical_Q1 = {}
    empirical_min = {}
    pair_counts = {}

    for pair, dists in all_pairwise_distances.items():
        if len(dists) > 0:
            empirical_Q1[pair] = np.percentile(dists, 1.0)
            empirical_min[pair] = np.min(dists)
            pair_counts[pair] = len(dists)
        
    # Ottimizzazione globale dei raggi di base R_i
    def cost_func_R(R):
        loss = 0.0
        N0 = 1000.0
        for (t1, t2), q1 in empirical_Q1.items():
            N = pair_counts[(t1, t2)]
            weight = N / (N + N0)
            r_pred = R[type_to_idx[t1]] + R[type_to_idx[t2]]
            loss += weight * (r_pred - q1)**2
        return loss
    
    R_init = np.ones(n_types) * 0.15
    bounds_R = [(0.05, 0.5) for _ in range(n_types)]
    res_R = scipy.optimize.minimize(cost_func_R, R_init, bounds=bounds_R)
    R_opt = res_R.x

    print("  Raggi base R_i ottimizzati:")
    for t, r in zip(all_types, R_opt):
        print(f"    Tipo {t}: {r:.4f} nm")
    
    KB_T = 2.494 # kJ/mol at 300K
    wca_prior_dict = {}

    for (t1, t2), dists in all_pairwise_distances.items():
        r_c = R_opt[type_to_idx[t1]] + R_opt[type_to_idx[t2]]
        sig = r_c / (2.0**(1.0/6.0))
        
        # Bug 8 & 9: Calcolo analitico di epsilon
        # Vogliamo U_WCA(0.9 * r_c) = 10 * k_B * T
        r_guard = 0.9 * r_c
        
        # U(r) = 4 eps [ (sig/r)^12 - (sig/r)^6 ] + eps
        # 10 k_B T = eps * ( 4 * [ (sig/r_guard)^12 - (sig/r_guard)^6 ] + 1 )
        sr = sig / r_guard
        term = 4.0 * (sr**12 - sr**6) + 1.0
        eps = (10.0 * KB_T) / term
        
        # Bug 10: Use gaussian_filter1d for robust extraction
        if len(dists) > 0:
            hist, bin_edges = np.histogram(dists, bins=50)
            smoothed_hist = gaussian_filter1d(hist, sigma=1.0)
            valid_idx = np.where(smoothed_hist > 0)[0]
            if len(valid_idx) > 0:
                r_emp_min = bin_edges[valid_idx[0]]
            else:
                r_emp_min = np.min(dists)
        else:
            r_emp_min = R_opt[type_to_idx[t1]] + R_opt[type_to_idx[t2]]
        
        r_c = R_opt[type_to_idx[t1]] + R_opt[type_to_idx[t2]]
        
        # We compute epsilon such that U_wca(0.9 r_c) = 10 kT
        r_guard = 0.9 * r_c
        sigma = r_c / (2.0**(1.0/6.0))
        # U(r) = 4 * eps * ((sigma/r)^12 - (sigma/r)^6) + eps
        term = (sigma / r_guard)**6
        u_factor = 4.0 * (term**2 - term) + 1.0
        # 10 k_B T in kJ/mol: k_B = 0.00831446 kJ/(mol K)
        kT = 0.00831446 * 300.0
        epsilon = 10.0 * kT / u_factor
        
        q1 = empirical_Q1.get((t1, t2), r_c)
        wca_prior_dict[f"{t1}_{t2}"] = {
            "type_i": t1,
            "type_j": t2,
            "sigma_nm": float(sigma),
            "epsilon_kjmol": float(epsilon),
            "cutoff_nm": float(r_c),
            "empirical_min": float(r_emp_min),
            "q1_nm": float(q1)
        }
        
    print(f"[INFO] Elaborazione parametri WCA completata ({len(wca_prior_dict)} coppie)")
    
    # Print WCA invasion table
    print("\n[INFO] Tabella di Invasione WCA (statistiche sui dati fisici):")
    print(f"{'pair':<8} {'r_c':<8} {'% < r_c':<15} {'% < r_guard':<15}")
    for pair_key, wca_info in wca_prior_dict.items():
        t1 = wca_info["type_i"]
        t2 = wca_info["type_j"]
        r_c = wca_info["cutoff_nm"]
        r_guard = 0.9 * r_c
        
        # Retrieve all distances collected for this pair
        # all_pairwise_distances keys are string tuples: (str(t1), str(t2))
        dists = all_pairwise_distances.get((str(t1), str(t2)), [])
        if not dists:
            # try symmetric key
            dists = all_pairwise_distances.get((str(t2), str(t1)), [])
            
        if len(dists) > 0:
            dists_arr = np.array(dists)
            frac_rc = 100.0 * np.sum(dists_arr < r_c) / len(dists_arr)
            frac_guard = 100.0 * np.sum(dists_arr < r_guard) / len(dists_arr)
            print(f"{t1}-{t2:<6} {r_c:<8.3f} {frac_rc:<15.2f} {frac_guard:<15.2f}")
        else:
            print(f"{t1}-{t2:<6} {r_c:<8.3f} {'N/A':<15} {'N/A':<15}")    
            
    # Bug 11: Inject wca_prior_dict into cg_priors.json and DO NOT write wca_priors.json
    import os
    import json
    if os.path.exists("cg_priors.json"):
        with open("cg_priors.json", "r") as f:
            cg_priors = json.load(f)
    else:
        cg_priors = {}
        
    cg_priors["wca_pairs"] = wca_prior_dict
    
    print(f"[INFO] Elaborazione parametri WCA completata ({len(wca_prior_dict)} coppie)")

print("\n[INFO] Esecuzione allineamento Kabsch per mediare le geometrie dei corpi rigidi...")

for resname, info in rigid_bodies_info.items():
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
    
    if info.get("auto_align_sites", True):
        # Iterative Kabsch average, anchored to the first observed orientation.
        ref = snapshots[0].copy()
        for iteration in range(3):
            aligned_snapshots = []
            for snap in snapshots:
                R = kabsch_align(snap, ref)
                aligned_snapshots.append((R @ snap.T).T)
            ref = np.mean(aligned_snapshots, axis=0)
        source_description = f"mediati {len(snapshots)} snapshot"
    else:
        rb_config = RIGID_BODIES_CONFIG.get(resname, {})
        configured_sites = rb_config.get("sites", {})
        missing = [name for name in site_names if name not in configured_sites]
        if missing:
            raise ValueError(
                f"Rigid body {resname} has auto_align_sites=False but is missing manual sites: {missing}"
            )
        manual = np.asarray(
            [configured_sites[name]["relative_pos_nm"] for name in site_names], dtype=float
        )
        if manual.shape != snapshots[0].shape:
            raise ValueError(f"Invalid manual rigid-body geometry for {resname}: {manual.shape}")
        manual_to_lab = kabsch_align(manual, snapshots[0])
        ref = (manual_to_lab @ manual.T).T
        source_description = "usata geometria manuale"
        
    # Express the ideal geometry in the principal-axis body frame.
    # ESPResSo stores rinertia as three principal moments, so the virtual-site
    # offsets must use the same body-fixed axes.
    principal_axes = principal_axes_lab_by_resname.get(resname)
    if principal_axes is None:
        raise RuntimeError(f"Missing principal axes for rigid body {resname}")
    ref_body = (principal_axes.T @ ref.T).T
    for i, sname in enumerate(site_names):
        info["sites"][sname]["relative_pos_nm"] = [float(v) for v in ref_body[i]]
    print(
        f"  -> {resname}: {source_description}; "
        "siti salvati nel frame degli assi principali."
    )

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

        if WCA_SIGMA == "auto":
            derived_priors["wca_pairs"] = wca_prior_dict
            
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
        # 3.1 Sottrazione WCA usando i parametri di wca_priors.json (36 coppie)
        if WCA_SIGMA == "auto" and 'wca_prior_dict' in locals():
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
                
                diff = flat_pos[:, np.newaxis, :] - flat_pos[np.newaxis, :, :]
                diff -= box_dim * np.round(diff / box_dim)
                dist_sq = np.sum(diff**2, axis=-1)
                
                sigma_ij = np.zeros_like(dist_sq)
                eps_ij = np.zeros_like(dist_sq)
                r_cut_sq = np.zeros_like(dist_sq)
                
                for i in range(len(flat_type)):
                    for j in range(len(flat_type)):
                        t_min = min(int(flat_type[i]), int(flat_type[j]))
                        t_max = max(int(flat_type[i]), int(flat_type[j]))
                        pair_key = f"{t_min}_{t_max}"
                        
                        if pair_key in wca_prior_dict:
                            w = wca_prior_dict[pair_key]
                            sigma_ij[i, j] = w["sigma_nm"]
                            eps_ij[i, j] = w["epsilon_kjmol"]
                            r_cut_sq[i, j] = w["cutoff_nm"]**2
                
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

            pos_i = resolve_site_position(frame_centers, frame_sites, i, site_i)
            pos_j = resolve_site_position(frame_centers, frame_sites, j, site_j)

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
                    raise ValueError(
                        f"FENE bond {i}-{j} is outside its domain: "
                        f"|r-r0|={abs(diff):.6g} >= r_max={r_max:.6g}"
                    )
                f_scalar = - k * diff / (1.0 - (diff/r_max)**2)

            elif b_type == "morse":
                D, a, r0 = b["D"], b["a"], b["r0"]
                diff = r - r0
                exp_term = np.exp(-a * diff)
                f_scalar = - 2.0 * a * D * (1.0 - exp_term) * exp_term


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
            pos_i = resolve_site_position(frame_centers, frame_sites, i, site_i)
            pos_j = resolve_site_position(frame_centers, frame_sites, j, site_j)
            pos_k = resolve_site_position(frame_centers, frame_sites, k_idx, site_k)
            
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
                
            grad_i_cos = r_jk / (d_ji * d_jk) - cos_theta * r_ji / (d_ji**2)
            grad_k_cos = r_ji / (d_ji * d_jk) - cos_theta * r_jk / (d_jk**2)
            
            scalar_force = dV_dtheta / sin_theta
            f_i = scalar_force * grad_i_cos
            f_k = scalar_force * grad_k_cos
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
            pos_i = resolve_site_position(frame_centers, frame_sites, i, site_i)
            pos_j = resolve_site_position(frame_centers, frame_sites, j, site_j)
            pos_k = resolve_site_position(frame_centers, frame_sites, k_idx, site_k)
            pos_l = resolve_site_position(frame_centers, frame_sites, l, site_l)

            if d_type == "cosine":
                f_i, f_j, f_k, f_l = dihedral_forces(
                    pos_i,
                    pos_j,
                    pos_k,
                    pos_l,
                    box_dim,
                    d["k"],
                    d.get("n", 1),
                    d["phi0"],
                )
            else:
                continue

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

        # --- DECOY GENERATION ---
        print(f"\\n[INFO] Generazione Decoy OOD (Deep Core) in corso...")
        decoy_frames = []
        import copy
        import random

        # Number of decoys per pair
        N_DECOYS_PER_PAIR = 256

        # Collect all frames data in memory to easily pick parents for decoys
        # Wait, we already have sites_data_history, forces_history, cg_centers_history, etc.
        # We can just pick random frames from there!

        total_decoys_generated = 0
        for pair_key, wca_info in wca_prior_dict.items():
            t1, t2 = wca_info["type_i"], wca_info["type_j"]
            r_c = wca_info["cutoff_nm"]
            r_emp_min = wca_info["empirical_min"]
        
            # R_OOD is max of 0.75 r_c and something below r_emp_min
            r_ood_max = min(0.95 * r_c, r_emp_min - 0.01)
            r_ood_min = 0.70 * r_c
            if r_ood_max <= r_ood_min:
                # No reliable deep-OOD interval exists for this pair
                print(f"    [Decoy] Skipping pair {t1}-{t2} (rc={r_c:.3f}, r_min={r_emp_min:.3f}) - no valid OOD interval.")
                continue
        
            for _ in range(N_DECOYS_PER_PAIR):
                frame_idx = random.randint(0, len(sites_data_history) - 1)
                
                # Pick two distinct molecules that have t1 and t2
                mol_ids_t1 = []
                mol_ids_t2 = []
                for m_idx, (rname, sites) in enumerate(zip(mol_resnames, sites_data_history[frame_idx])):
                    # Find sites of type t1
                    for (s_type, s_pos) in sites:
                        if s_type == t1: mol_ids_t1.append((m_idx, s_pos))
                        if s_type == t2: mol_ids_t2.append((m_idx, s_pos))
                        
                if not mol_ids_t1 or not mol_ids_t2: continue
                
                # Pick random sites
                m1_idx, pos1 = random.choice(mol_ids_t1)
                m2_idx, pos2 = random.choice(mol_ids_t2)
                
                if m1_idx == m2_idx: continue # must be different molecules
                
                target_r = random.uniform(r_ood_min, r_ood_max)
                
                # Current distance vector from 2 to 1
                vec = pos1 - pos2
                dist = np.linalg.norm(vec)
                if dist < 1e-4: vec = np.array([1.0, 0.0, 0.0])
                else: vec = vec / dist
                
                # We want to translate molecule 2 so that pos2 is at pos1 - target_r * vec
                new_pos2 = pos1 - target_r * vec
                translation = new_pos2 - pos2
                
                # Copy frame data
                decoy_sites = copy.deepcopy(sites_data_history[frame_idx])
                decoy_centers = np.copy(cg_centers_history[frame_idx])
                decoy_forces = np.zeros_like(cg_forces_history[frame_idx]) # SET F_ML = 0 FOR ENTIRE FRAME
                
                # Apply translation to all sites in mol2
                for i in range(len(decoy_sites[m2_idx])):
                    s_type, s_pos = decoy_sites[m2_idx][i]
                    decoy_sites[m2_idx][i] = (s_type, s_pos + translation)
                    
                decoy_centers[m2_idx] += translation
                
                decoy_frames.append((decoy_sites, decoy_centers, decoy_forces, box_dim_history[frame_idx]))
                total_decoys_generated += 1

        print(f"[INFO] Generati {total_decoys_generated} decoy frames.")

        # Write decoys to dataset
        print("[INFO] Scrittura decoy nel binario...")
        with open(args.output, "ab") as f:
            for d_idx, (d_sites, d_centers, d_forces, d_box) in enumerate(decoy_frames):
                num_molecules = len(d_sites)
                num_total_sites = sum(len(sites) for sites in d_sites)
                
                f.write(struct.pack("i", num_molecules))
                f.write(struct.pack("i", num_total_sites))
                f.write(struct.pack("3f", float(d_box[0]), float(d_box[1]), float(d_box[2])))
        
                for mol_id in range(num_molecules):
                    num_sites = len(d_sites[mol_id])
                    f.write(struct.pack("i", mol_id))
                    f.write(struct.pack("i", num_sites))
                    f.write(struct.pack("3f", *d_centers[mol_id]))
                    f.write(struct.pack("3f", *d_forces[mol_id]))
                    f.write(struct.pack("3f", 0.0, 0.0, 0.0)) # torques are zero
                    for site_type, site_pos in d_sites[mol_id]:
                        f.write(struct.pack("i", int(site_type)))
                        f.write(struct.pack("3f", *site_pos))

        print(f"[INFO] Scritti {len(decoy_frames)} decoy nel binario.")
        with open(args.output, "r+b") as f:
            f.seek(0)
            total_frames = len(cg_centers_history) + len(decoy_frames)
            f.write(struct.pack("i", total_frames))
        print("[INFO] Aggiornato il contatore dei frame totali.")


        print("[INFO] Conversione completata e forze residue salvate con successo nel dataset!")


        with open("rigid_bodies_info.json", "w") as jf:
            json.dump(rigid_bodies_info, jf, indent=4)
        print("[INFO] Masse e inerzie salvate in rigid_bodies_info.json!")
