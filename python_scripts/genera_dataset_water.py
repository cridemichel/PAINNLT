import MDAnalysis as mda
import numpy as np
import struct

# =====================================================================
# 1. IMPOSTAZIONI E MAPPING PER TEST ACQUA
# =====================================================================
topology_file = "conf.gro"
trajectory_file = "traiettoria.trr" 

print("[INFO] Caricamento universo MDAnalysis...")
u = mda.Universe(topology_file, trajectory_file)

MAPPING_METHOD = "COM" 

mapping_by_resname = {
    "SOL": {  
        "CG_WATER": ["OW", "HW1", "HW2"] 
    }
}

site_types = {
    "CG_WATER": 0
}

def get_unwrapped_positions(atoms, box_dim):
    # Clona le posizioni per non modificare la traiettoria originale
    pos = atoms.positions.copy()
    ref = pos[0]
    for i in range(1, len(pos)):
        dvec = pos[i] - ref
        dvec -= box_dim * np.round(dvec / box_dim)
        pos[i] = ref + dvec
    return pos

def compute_com(pos, masses):
    return np.sum(pos * masses[:, None], axis=0) / np.sum(masses)


# =====================================================================
# 2. ELABORAZIONE E SCRITTURA DEL FILE BINARIO (OTTIMIZZATO)
# =====================================================================
output_bin = "cg_dataset.bin"

# 🌟 VALIDAZIONE ESEGUITA UNA SOLA VOLTA (Aumento drastico delle prestazioni)
valid_residues = [res for res in u.residues if res.resname in mapping_by_resname]
num_molecules = len(valid_residues)
num_total_sites = sum(len(mapping_by_resname[res.resname]) for res in valid_residues)

if num_molecules == 0:
    print("[ERROR] Nessun residuo valido trovato con i resname forniti. Controlla il file di topologia.")
    exit(1)

with open(output_bin, "wb") as f:
    
    num_frames = len(u.trajectory)
    f.write(struct.pack("i", num_frames))
    
    print(f"[INFO] Inizio elaborazione di {num_frames} frame...")
    print(f"[INFO] Molecole valide per frame: {num_molecules} | Siti CG totali per frame: {num_total_sites}")
    print(f"[INFO] Metodo di mapping scelto: {MAPPING_METHOD}")
    
    for frame_idx, ts in enumerate(u.trajectory):
        # Scrittura header del singolo frame
        f.write(struct.pack("i", num_molecules))
        f.write(struct.pack("i", num_total_sites))
        
        for mol_id, residue in enumerate(valid_residues):
            resname = residue.resname
            current_mapping = mapping_by_resname[resname]
            num_sites = len(current_mapping)

            # --- CALCOLO GRANDEZZE FISICHE MOLECOLARI (ALL-ATOM) ---
            atoms = residue.atoms
            box_dim = ts.dimensions[:3]
            unwrapped_pos = get_unwrapped_positions(atoms, box_dim)
            positions_nm = unwrapped_pos / 10.0
            forces_nm = atoms.forces * 10.0
            
            # Centro di massa molecolare all-atom con PBC corrette
            center = compute_com(positions_nm, atoms.masses)
            
            # Forza risultante sulla molecola
            total_force = np.sum(forces_nm, axis=0)
            
            # Calcolo del momento torcente rispetto al COM corretto
            r_vec = positions_nm - center
            torques_nm = np.cross(r_vec, forces_nm)
            total_torque = np.sum(torques_nm, axis=0)
            
            # Scrittura dati molecolari
            f.write(struct.pack("i", mol_id))
            f.write(struct.pack("i", num_sites))
            f.write(struct.pack("3f", *center))
            f.write(struct.pack("3f", *total_force))
            f.write(struct.pack("3f", *total_torque)) 
            
            # --- CALCOLO POSIZIONI DEI SITI CG ---
            for site_name, atom_names in current_mapping.items():
                
                selection_string = "name " + " ".join(atom_names)
                site_atoms = atoms.select_atoms(selection_string)
                
                if len(site_atoms) == 0:
                    continue
                # Calcolo posizione del sito CG usando posizioni unwrapped
                indices = [list(atoms.names).index(name) for name in atom_names]
                site_pos_unwrapped = positions_nm[indices]
                site_masses = atoms.masses[indices]
                
                if MAPPING_METHOD == "COM":
                    site_pos = compute_com(site_pos_unwrapped, site_masses)
                elif MAPPING_METHOD == "COG":
                    site_pos = np.mean(site_pos_unwrapped, axis=0)
                elif MAPPING_METHOD == "ATOM":
                    site_pos = site_pos_unwrapped[0]
                
                site_type = site_types[site_name]
                
                # Scrittura dati del sito
                f.write(struct.pack("i", site_type))
                f.write(struct.pack("3f", *site_pos))

        if (frame_idx + 1) % 50 == 0 or (frame_idx + 1) == num_frames:
            print(f" -> Processati {frame_idx + 1}/{num_frames} frame...")

print(f"\n[INFO] SUCCESSO! Dataset binario aggiornato generato in: {output_bin}")
