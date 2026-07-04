import MDAnalysis as mda
import numpy as np
import struct

# =====================================================================
# 1. IMPOSTAZIONI E MAPPING
# =====================================================================
topology_file = "topologia.tpr"
trajectory_file = "traiettoria.trr" 
u = mda.Universe(topology_file, trajectory_file)

# --- SCELTA DELLA STRATEGIA DI MAPPING ---
# Scegli tra:
# "COM"  -> Centro di Massa (consigliato per force-matching)
# "COG"  -> Centro di Geometria (media aritmetica delle posizioni)
# "ATOM" -> Atomo di riferimento (usa l'esatta posizione del PRIMO atomo elencato nella lista)
MAPPING_METHOD = "COM" 

mapping_by_resname = {
    "GUA": {  
        # Se usi il metodo "ATOM", l'atomo di riferimento sarà il primo della lista (es. "N9" per CG_G1)
        "CG_G1": ["N9", "C8"], # questo vuol dire che gli atomi N9 e C8 vengono "raggruppati" in CG_G1
        "CG_G2": ["N7", "C5"],
        "CG_G3": ["C4", "N3"],
        "CG_G4": ["C2", "N1"],
        "CG_G5": ["C6", "O6"],
        "CG_G6": ["C1*", "C2*", "C3*"] 
    },
    "ETH": {
        "CG_CH3": ["C1", "H1", "H2", "H3"],
        "CG_CH2": ["C2", "H4", "H5"],
        "CG_OH":  ["O1", "H6"]
    }
}

site_types = {
    "CG_G1": 0, "CG_G2": 1, "CG_G3": 2, "CG_G4": 3, "CG_G5": 4, "CG_G6": 5,
    "CG_CH3": 6, "CG_CH2": 7, "CG_OH": 8
}

# =====================================================================
# 2. ELABORAZIONE E SCRITTURA DEL FILE BINARIO
# =====================================================================
output_bin = "cg_dataset.bin"

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
        
        for mol_id, residue in enumerate(valid_residues):
            resname = residue.resname
            current_mapping = mapping_by_resname[resname]
            num_sites = len(current_mapping)
            
            # --- CALCOLO GRANDEZZE FISICHE MOLECOLARI ---
            atoms = residue.atoms
            positions_aa = atoms.positions
            forces_aa = atoms.forces
            
            center = atoms.center_of_mass()
            total_force = np.sum(forces_aa, axis=0)
            
            r_vec = positions_aa - center
            torques_aa = np.cross(r_vec, forces_aa)
            total_torque = np.sum(torques_aa, axis=0)
            
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
                    print(f"[WARNING] Nessun atomo trovato per il sito {site_name} nella molecola {resname} (ID {mol_id})")
                    continue
                
                # --- LOGICA DI MAPPING CONFIGURABILE ---
                if MAPPING_METHOD == "COM":
                    site_pos = site_atoms.center_of_mass()
                
                elif MAPPING_METHOD == "COG":
                    site_pos = site_atoms.center_of_geometry()
                
                elif MAPPING_METHOD == "ATOM":
                    # MDAnalysis potrebbe non mantenere l'ordine della stringa di selezione.
                    # Per essere sicuri, selezioniamo esplicitamente il PRIMO atomo della lista
                    ref_atom_name = atom_names[0]
                    ref_atom = atoms.select_atoms(f"name {ref_atom_name}")
                    
                    if len(ref_atom) == 0:
                        print(f"[ERRORE] Atomo di riferimento {ref_atom_name} non trovato!")
                        site_pos = np.zeros(3) # Fallback di sicurezza
                    else:
                        site_pos = ref_atom.positions[0]
                
                else:
                    raise ValueError(f"Metodo di mapping '{MAPPING_METHOD}' non supportato!")
                
                site_type = site_types[site_name]
                
                f.write(struct.pack("i", site_type))
                f.write(struct.pack("3f", *site_pos))

print(f"[INFO] Dataset generato con successo in {output_bin}!")
