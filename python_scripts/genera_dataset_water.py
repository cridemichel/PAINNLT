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
            positions_aa = atoms.positions
            forces_aa = atoms.forces
            
            # Centro di massa molecolare all-atom
            center = atoms.center_of_mass()
            # Forza risultante sulla molecola (somma vettoriale delle forze atomiche)
            total_force = np.sum(forces_aa, axis=0)
            
            # Calcolo del momento torcente reale all-atom rispetto al centro di massa
            r_vec = positions_aa - center
            torques_aa = np.cross(r_vec, forces_aa)
            total_torque = np.sum(torques_aa, axis=0)
            
            # Scrittura dati molecolari (Formato: int, int, 3f, 3f, 3f)
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
                
                if MAPPING_METHOD == "COM":
                    site_pos = site_atoms.center_of_mass()
                elif MAPPING_METHOD == "COG":
                    site_pos = site_atoms.center_of_geometry()
                elif MAPPING_METHOD == "ATOM":
                    ref_atom_name = atom_names[0]
                    ref_atom = atoms.select_atoms(f"name {ref_atom_name}")
                    site_pos = ref_atom.positions[0] if len(ref_atom) > 0 else np.zeros(3)
                
                site_type = site_types[site_name]
                
                # Scrittura dati del sito (Formato: int, 3f)
                f.write(struct.pack("i", site_type))
                f.write(struct.pack("3f", *site_pos))

        if (frame_idx + 1) % 50 == 0 or (frame_idx + 1) == num_frames:
            print(f" -> Processati {frame_idx + 1}/{num_frames} frame...")

print(f"\n[INFO] SUCCESSO! Dataset binario aggiornato generato in: {output_bin}")
