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

# Metodo di mapping: per l'acqua mappata su un singolo sito, 
# il Centro di Massa (COM) è la scelta ideale.
MAPPING_METHOD = "COM" 

# Mappiamo l'intera molecola d'acqua su un singolo sito CG
mapping_by_resname = {
    "SOL": {  
        "CG_WATER": ["OW", "HW1", "HW2"] 
    }
}

site_types = {
    "CG_WATER": 0
}

# =====================================================================
# 2. ELABORAZIONE E SCRITTURA DEL FILE BINARIO
# =====================================================================
output_bin = "cg_dataset.bin"

with open(output_bin, "wb") as f:
    
    num_frames = len(u.trajectory)
    f.write(struct.pack("i", num_frames))
    
    print(f"[INFO] Inizio elaborazione di {num_frames} frame...")
    print(f"[INFO] Metodo di mapping scelto: {MAPPING_METHOD}")
    
    for ts in u.trajectory:
        # Troviamo tutte le molecole d'acqua
        valid_residues = [res for res in u.residues if res.resname in mapping_by_resname]
        num_molecules = len(valid_residues)
        num_total_sites = sum(len(mapping_by_resname[res.resname]) for res in valid_residues)
        
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
            
            center = atoms.center_of_mass()
            total_force = np.sum(forces_aa, axis=0)
            
            # Momento torcente
            r_vec = positions_aa - center
            torques_aa = np.cross(r_vec, forces_aa)
            total_torque = np.sum(torques_aa, axis=0)
            
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
                    print(f"[WARNING] Nessun atomo trovato per {site_name} (ID {mol_id})")
                    continue
                
                # --- LOGICA DI MAPPING ---
                if MAPPING_METHOD == "COM":
                    site_pos = site_atoms.center_of_mass()
                elif MAPPING_METHOD == "COG":
                    site_pos = site_atoms.center_of_geometry()
                elif MAPPING_METHOD == "ATOM":
                    ref_atom_name = atom_names[0]
                    ref_atom = atoms.select_atoms(f"name {ref_atom_name}")
                    site_pos = ref_atom.positions[0] if len(ref_atom) > 0 else np.zeros(3)
                
                site_type = site_types[site_name]
                
                # Scrittura dati del sito
                f.write(struct.pack("i", site_type))
                f.write(struct.pack("3f", *site_pos))

print(f"[INFO] SUCCESSO! Dataset binario generato correttamente in: {output_bin}")
