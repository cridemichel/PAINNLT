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
parser.add_argument("-o", "--output", type=str, default="cg_dataset.bin", help="Nome del file binario di output")
args = parser.parse_args()

try:
    with open(args.mapping, "r") as mf:
        mapping_data = json.load(mf)
except FileNotFoundError:
    print(f"[ERRORE] File di mapping '{args.mapping}' non trovato!")
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
    # Il Calcio ('CA') e' volutamente omesso per non andare in conflitto con i Carboni Alpha (CA).
    # Per i sistemi con Calcio e' necessario passare il file .tpr a MDAnalysis.
}

def get_mass(atom_name):
    # Fallback element guess da nome atomo
    alpha_chars = ''.join([c for c in atom_name if c.isalpha()]).upper()
    
    if not alpha_chars:
        return 12.0
        
    if alpha_chars in IONS_MASSES:
        return IONS_MASSES[alpha_chars]
        
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
    return np.sum(positions * masses[:, None], axis=0) / np.sum(masses)

def compute_inertia_tensor(positions, masses, com):
    r = positions - com
    Ixx = np.sum(masses * (r[:,1]**2 + r[:,2]**2))
    Iyy = np.sum(masses * (r[:,0]**2 + r[:,2]**2))
    Izz = np.sum(masses * (r[:,0]**2 + r[:,1]**2))
    Ixy = -np.sum(masses * r[:,0] * r[:,1])
    Ixz = -np.sum(masses * r[:,0] * r[:,2])
    Iyz = -np.sum(masses * r[:,1] * r[:,2])
    return np.array([[Ixx, Ixy, Ixz],
                     [Ixy, Iyy, Iyz],
                     [Ixz, Iyz, Izz]])

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
        
        f.write(struct.pack("i", num_molecules))
        f.write(struct.pack("i", num_total_sites))
        
        box_dim = ts.dimensions[:3]
        
        for mol_id, residue in enumerate(valid_residues):
            resname = residue.resname
            current_mapping = mapping_by_resname[resname]
            num_sites = len(current_mapping)
            
            # --- CALCOLO GRANDEZZE FISICHE MOLECOLARI ---
            # MDAnalysis legge nativamente in Angstrom e kJ/(mol*A). Convertiamo in nm e kJ/(mol*nm)
            atoms = residue.atoms
            positions_nm = atoms.positions / 10.0
            forces_nm = atoms.forces * 10.0
            
            unwrapped_pos = get_unwrapped_positions(positions_nm, box_dim / 10.0)
            
            try:
                masses = atoms.masses
            except:
                masses = np.array([get_mass(name) for name in atoms.names])
            
            center = compute_com(unwrapped_pos, masses)
            total_force = np.sum(forces_nm, axis=0)
            
            r_vec = unwrapped_pos - center
            torques_nm = np.cross(r_vec, forces_nm)
            total_torque = np.sum(torques_nm, axis=0)
            
            # Registra inerzia solo al primo incontro della molecola
            if resname not in rigid_bodies_info:
                total_mass = float(np.sum(masses))
                I_tensor = compute_inertia_tensor(unwrapped_pos, masses, center)
                eigvals = np.linalg.eigvalsh(I_tensor)
                # Le coordinate sono gia' in nm, l'inerzia e' nativamente in amu*nm^2
                I_principal_nm2 = eigvals
                rigid_bodies_info[resname] = {
                    "mass_amu": round(total_mass, 4),
                    "inertia_amu_nm2": [round(v, 4) for v in I_principal_nm2],
                    "sites": {}
                }
            
            f.write(struct.pack("i", mol_id))
            f.write(struct.pack("i", num_sites))
            f.write(struct.pack("3f", *center))
            f.write(struct.pack("3f", *total_force))
            f.write(struct.pack("3f", *total_torque))
            
            # --- CALCOLO POSIZIONI DEI SITI CG ---
            for site_name, atom_names in current_mapping.items():
                
                if atom_names == ["*"]:
                    site_atoms = atoms
                else:
                    selection_string = "name " + " ".join(atom_names)
                    site_atoms = atoms.select_atoms(selection_string)
                
                if len(site_atoms) == 0:
                    print(f"[WARNING] Nessun atomo trovato per il sito {site_name} nella molecola {resname} (ID {mol_id})")
                    continue
                
                if MAPPING_METHOD == "COM":
                    # Usare le unwrapped_pos per il site:
                    # Dobbiamo trovare gli indici degli atomi nel sito
                    indices = [list(atoms.names).index(n) for n in site_atoms.names]
                    site_unwrapped_pos = unwrapped_pos[indices]
                    site_masses = masses[indices]
                    site_pos = compute_com(site_unwrapped_pos, site_masses)
                
                elif MAPPING_METHOD == "COG":
                    indices = [list(atoms.names).index(n) for n in site_atoms.names]
                    site_unwrapped_pos = unwrapped_pos[indices]
                    site_pos = np.mean(site_unwrapped_pos, axis=0)
                
                elif MAPPING_METHOD == "ATOM":
                    ref_atom_name = atom_names[0]
                    if ref_atom_name in list(atoms.names):
                        idx = list(atoms.names).index(ref_atom_name)
                        site_pos = unwrapped_pos[idx]
                    else:
                        print(f"[ERRORE] Atomo di riferimento {ref_atom_name} non trovato!")
                        site_pos = np.zeros(3)
                else:
                    raise ValueError(f"Metodo di mapping '{MAPPING_METHOD}' non supportato!")
                
                site_type = site_types[site_name]
                
                if resname in rigid_bodies_info and site_name not in rigid_bodies_info[resname].get("sites", {}):
                    relative_pos_nm = site_pos - center
                    rigid_bodies_info[resname]["sites"][site_name] = {
                        "type": site_type,
                        "relative_pos_nm": [round(v, 4) for v in relative_pos_nm]
                    }
                
                f.write(struct.pack("i", site_type))
                f.write(struct.pack("3f", *site_pos))

print(f"[INFO] Dataset generato con successo in {output_bin}!")

# Salva il file info dei corpi rigidi
with open("rigid_bodies_info.json", "w") as jf:
    json.dump(rigid_bodies_info, jf, indent=4)
print("[INFO] Masse e inerzie salvate in rigid_bodies_info.json!")

