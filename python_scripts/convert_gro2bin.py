import MDAnalysis as mda
import numpy as np
import struct
import json
import argparse

# =====================================================================
# 1. PARSING ARGOMENTI E CARICAMENTO MAPPING
# =====================================================================
parser = argparse.ArgumentParser(description="Converte traiettorie GROMACS in dataset binario per PaiNN-CG")
parser.add_argument("-c", "--topology", type=str, default="topologia.tpr", help="File di topologia (es. .tpr o .gro)")
parser.add_argument("-f", "--trajectory", type=str, default="traiettoria.trr", help="File di traiettoria (es. .trr o .xtc)")
parser.add_argument("-m", "--mapping", type=str, default="cg_mapping.json", help="File JSON di mapping CG")
parser.add_argument("-p", "--priors", type=str, default=None, help="File JSON con definizioni priors (opzionale)")
parser.add_argument("-o", "--output", type=str, default="cg_dataset.bin", help="Nome del file binario di output")
args = parser.parse_args()

try:
    with open(args.mapping, "r") as mf:
        mapping_data = json.load(mf)
except FileNotFoundError:
    print(f"[ERRORE] File di mapping '{args.mapping}' non trovato!")
    exit(1)

priors_data = None
if args.priors:
    try:
        with open(args.priors, "r") as pf:
            priors_data = json.load(pf)
        print(f"[INFO] Trovato file priors: {args.priors}")
    except FileNotFoundError:
        print(f"[ERRORE] File priors '{args.priors}' non trovato!")
        exit(1)

MAPPING_METHOD = mapping_data.get("mapping_method", "COM")
mapping_by_resname = mapping_data.get("residues", {})
site_types = mapping_data.get("site_types", {})

print(f"[INFO] Caricamento MDAnalysis: {args.topology}, {args.trajectory}...")
u = mda.Universe(args.topology, args.trajectory)

ATOMIC_MASSES = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'P': 30.974, 'S': 32.065, 'K': 39.098
}
IONS_MASSES = {
    'NA': 22.990, 'MG': 24.305, 'CL': 35.450, 'FE': 55.845, 
    'ZN': 65.380, 'CU': 63.546, 'BR': 79.904, 'I': 126.904
}

def get_mass(atom_name):
    alpha_chars = ''.join([c for c in atom_name if c.isalpha()]).upper()
    if not alpha_chars: return 12.0
    if alpha_chars in IONS_MASSES: return IONS_MASSES[alpha_chars]
    return ATOMIC_MASSES.get(alpha_chars[0], 12.0)

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
    I = np.zeros((3, 3))
    for pos, mass in zip(positions, masses):
        r = pos - center
        I += mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return I

def mic_vector(pos1, pos2, box_dim):
    """Vector from pos1 to pos2 with Minimum Image Convention"""
    dvec = pos2 - pos1
    return dvec - box_dim * np.round(dvec / box_dim)

# =====================================================================
# 2. ELABORAZIONE E SCRITTURA DEL FILE BINARIO
# =====================================================================
output_bin = args.output
rigid_bodies_info = {}

with open(output_bin, "wb") as f:
    num_frames = len(u.trajectory)
    f.write(struct.pack("i", num_frames))
    print(f"[INFO] Inizio elaborazione di {num_frames} frame con metodo di mapping: {MAPPING_METHOD}...")
    
    for ts in u.trajectory:
        valid_residues = [res for res in u.residues if res.resname in mapping_by_resname]
        num_molecules = len(valid_residues)
        num_total_sites = sum(len(mapping_by_resname[res.resname]) for res in valid_residues)
        box_dim = ts.dimensions[:3] / 10.0 # nm
        
        f.write(struct.pack("i", num_molecules))
        f.write(struct.pack("i", num_total_sites))
        f.write(struct.pack("3f", *box_dim))
        
        # Array temporanei per questo frame (molecole)
        frame_centers = []
        frame_forces = []
        frame_torques = []
        frame_sites_data = [] # list of lists of site info
        
        for mol_id, residue in enumerate(valid_residues):
            resname = residue.resname
            current_mapping = mapping_by_resname[resname]
            num_sites = len(current_mapping)
            
            atoms = residue.atoms
            positions_nm = atoms.positions / 10.0
            forces_nm = atoms.forces * 10.0
            unwrapped_pos = get_unwrapped_positions(positions_nm, box_dim)
            
            try:
                masses = atoms.masses
            except:
                masses = np.array([get_mass(name) for name in atoms.names])
            
            center = compute_com(unwrapped_pos, masses)
            total_force = np.sum(forces_nm, axis=0)
            r_vec = unwrapped_pos - center
            total_torque = np.sum(np.cross(r_vec, forces_nm), axis=0)
            
            frame_centers.append(center)
            frame_forces.append(total_force)
            frame_torques.append(total_torque)
            
            if resname not in rigid_bodies_info:
                total_mass = float(np.sum(masses))
                I_tensor = compute_inertia_tensor(unwrapped_pos, masses, center)
                eigvals = np.linalg.eigvalsh(I_tensor)
                rigid_bodies_info[resname] = {
                    "mass_amu": round(total_mass, 4),
                    "inertia_amu_nm2": [round(v, 4) for v in eigvals],
                    "sites": {}
                }
            
            sites_for_mol = []
            for site_name, atom_names in current_mapping.items():
                if atom_names == ["*"]: site_atoms = atoms
                else:
                    sel = "name " + " ".join(atom_names)
                    site_atoms = atoms.select_atoms(sel)
                
                if len(site_atoms) == 0: continue
                
                if MAPPING_METHOD == "COM":
                    indices = [list(atoms.names).index(n) for n in site_atoms.names]
                    site_pos = compute_com(unwrapped_pos[indices], masses[indices])
                elif MAPPING_METHOD == "COG":
                    indices = [list(atoms.names).index(n) for n in site_atoms.names]
                    site_pos = np.mean(unwrapped_pos[indices], axis=0)
                elif MAPPING_METHOD == "ATOM":
                    ref_name = atom_names[0]
                    if ref_name in list(atoms.names):
                        idx = list(atoms.names).index(ref_name)
                        site_pos = unwrapped_pos[idx]
                    else:
                        site_pos = np.zeros(3)
                
                site_type = site_types[site_name]
                if resname in rigid_bodies_info and site_name not in rigid_bodies_info[resname].get("sites", {}):
                    relative_pos_nm = site_pos - center
                    rigid_bodies_info[resname]["sites"][site_name] = {
                        "type": site_type, "relative_pos_nm": [round(v, 4) for v in relative_pos_nm]
                    }
                sites_for_mol.append((site_type, site_pos))
                
            frame_sites_data.append(sites_for_mol)

        frame_forces = np.array(frame_forces)
        frame_torques = np.array(frame_torques)
        
        # --- CALCOLO DEI PRIORS E SOTTRAZIONE ---
        if priors_data:
            # WCA: applicato tra tutti i centri di massa (COMs) inter-molecolari
            if "wca" in priors_data:
                eps = priors_data["wca"].get("epsilon", 0.0)
                sigma = priors_data["wca"].get("sigma", 0.0)
                if eps > 0 and sigma > 0:
                    r_cut_sq = (sigma * (2.0**(1.0/6.0)))**2
                    for i in range(num_molecules):
                        for j in range(i + 1, num_molecules):
                            r_vec = mic_vector(frame_centers[i], frame_centers[j], box_dim)
                            r_sq = np.sum(r_vec**2)
                            if 1e-6 < r_sq < r_cut_sq:
                                r = np.sqrt(r_sq)
                                r_hat = r_vec / r
                                f_scalar = 24.0 * eps * (2.0 * (sigma/r)**12 - (sigma/r)**6) / r
                                f_vec = - f_scalar * r_hat # F su i da j (repulsiva verso -r_hat)
                                frame_forces[i] -= f_vec
                                frame_forces[j] += f_vec
            
            # BONDS: Armonico e FENE
            if "bonds" in priors_data:
                for b in priors_data["bonds"]:
                    i, j = b["mol_i"], b["mol_j"]
                    if i >= num_molecules or j >= num_molecules: continue
                    
                    r_vec = mic_vector(frame_centers[i], frame_centers[j], box_dim)
                    r = np.linalg.norm(r_vec)
                    if r < 1e-6: continue
                    r_hat = r_vec / r
                    
                    f_scalar = 0.0
                    b_type = b.get("type", "harmonic")
                    
                    if b_type == "harmonic":
                        k = b["k"]
                        r0 = b["r0"]
                        f_scalar = - k * (r - r0)
                        
                    elif b_type == "fene":
                        k = b["k"]
                        r0 = b["r0"]
                        r_max = b["r_max"]
                        diff = r - r0
                        if abs(diff) >= r_max:
                            print(f"[WARNING] FENE limit exceeded frame {ts.frame}, bond {i}-{j}")
                            f_scalar = 0.0
                        else:
                            f_scalar = - k * diff / (1.0 - (diff/r_max)**2)
                    
                    elif b_type == "morse":
                        D = b["D"]
                        a = b["a"]
                        r0 = b["r0"]
                        diff = r - r0
                        # F_r = - 2 * a * D * (1 - e^(-a(r-r0))) * e^(-a(r-r0))
                        exp_term = np.exp(-a * diff)
                        f_scalar = - 2.0 * a * D * (1.0 - exp_term) * exp_term
                    
                    f_vec = - f_scalar * r_hat
                    frame_forces[i] -= f_vec
                    frame_forces[j] += f_vec

        # --- SCRITTURA NEL FILE BINARIO ---
        for mol_id in range(num_molecules):
            num_sites = len(frame_sites_data[mol_id])
            f.write(struct.pack("i", mol_id))
            f.write(struct.pack("i", num_sites))
            f.write(struct.pack("3f", *frame_centers[mol_id]))
            f.write(struct.pack("3f", *frame_forces[mol_id]))
            f.write(struct.pack("3f", *frame_torques[mol_id]))
            for site_type, site_pos in frame_sites_data[mol_id]:
                f.write(struct.pack("i", site_type))
                f.write(struct.pack("3f", *site_pos))

print(f"[INFO] Dataset generato con successo in {output_bin}!")

with open("rigid_bodies_info.json", "w") as jf:
    json.dump(rigid_bodies_info, jf, indent=4)
print("[INFO] Masse e inerzie salvate in rigid_bodies_info.json!")
