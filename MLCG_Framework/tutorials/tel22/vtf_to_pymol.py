#!/usr/bin/env python3
"""
Convertitore VTF -> PDB per PyMOL
Questo script analizza la traiettoria Coarse-Grained (.vtf) di ESPResSo e il dataset 
binario di inizializzazione per estrarre esclusivamente i veri Centri di Massa (COM) 
dei nucleotidi, ripulendo la traiettoria da tutti i Virtual Sites fittizi.

Genera:
1. cg_trajectory_clean.pdb: Una traiettoria PDB multi-modello pronta per essere animata.
2. load_tel22_pymol.pml: Uno script di visualizzazione per PyMOL che applica
   automaticamente i colori alle catene e formatta i filamenti come tubi continui (cartoon).
"""

import sys
import os
import struct
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convertitore VTF -> PDB per PyMOL")
    parser.add_argument("--style", choices=["tube", "spheres"], default="tube", 
                        help="Stile di visualizzazione in PyMOL (default: tube)")
    args = parser.parse_args()
    
    dataset_file = "tel22_dataset.bin"
    vtf_file = "cg_trajectory.vtf"
    
    if not os.path.exists(dataset_file) or not os.path.exists(vtf_file):
        print(f"Errore: File {dataset_file} o {vtf_file} non trovati.")
        print("Assicurati di aver eseguito prima la simulazione ESPResSo!")
        sys.exit(1)

    print("Analisi del layout delle molecole...")
    mol_com_parts = {}
    with open(dataset_file, "rb") as f:
        num_frames, num_molecules, num_total_sites = struct.unpack("i"*3, f.read(12))
        f.read(12)
        idx = 0
        for mol_idx in range(num_molecules):
            mol_id, num_sites = struct.unpack("ii", f.read(8))
            f.read(36)
            for s in range(num_sites): f.read(16)
            # ESPResSo assegna gli ID sequenzialmente. Il COM viene aggiunto per primo.
            mol_com_parts[mol_idx] = idx
            idx += 1 + num_sites
            
    print("Estrazione dei frame dalla traiettoria VTF...")
    frames = []
    current_frame = []
    with open(vtf_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("timestep"):
                if current_frame:
                    frames.append(np.array(current_frame))
                    current_frame = []
            elif line[0].isdigit() or line[0] == "-":
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        # parts[0] è l'ID atomo, parts[1..3] sono x, y, z
                        coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                        current_frame.append(coords)
                    except ValueError:
                        pass

    if current_frame:
        frames.append(np.array(current_frame))
        
    print(f"Letti {len(frames)} frame. Scrittura del file PDB...")
    
    pdb_filename = "cg_trajectory_clean.pdb"
    with open(pdb_filename, "w") as out:
        for f_idx, frame in enumerate(frames):
            out.write(f"MODEL {f_idx+1}\n")
            atom_count = 1
            # TEL22 ha 10 filamenti, ciascuno composto da 22 nucleotidi.
            # Questo assicura che PyMOL unisca correttamente la spina dorsale di ogni catena.
            for strand in range(10):
                start_mol_idx = strand * 22
                end_mol_idx = start_mol_idx + 21
                com_indices = [mol_com_parts[m] for m in range(start_mol_idx, end_mol_idx + 1)]
                
                chain_id = chr(ord('A') + strand)
                for i, idx in enumerate(com_indices):
                    res_num = i + 1
                    coords = frame[idx] * 10.0 # Converti nm in Angstroms per PyMOL
                    out.write(f"ATOM  {atom_count:5d}  CA  CG  {chain_id}{res_num:4d}    {coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00  0.00           C\n")
                    atom_count += 1
            out.write("ENDMDL\n")
            
    print(f"File '{pdb_filename}' generato con successo.")
    
    # Genera i comandi di base per lo stile
    style_cmds = """\
# Mostra il DNA come tubi continui (backbone)
show cartoon, tel22_cg
cartoon tube, tel22_cg
set cartoon_tube_radius, 0.4
color grey, tel22_cg

# Mostra le sferette (Coarse-Grained beads) per tutto
show spheres, tel22_cg
set sphere_scale, 0.8, tel22_cg

# Nascondi le sferette per i linker AAT e le adenine terminali (residui non-guanine)
# Le guanine sono 2,3,4, 8,9,10, 14,15,16, 20,21,22. Tutto il resto è linker.
hide spheres, resi 1+5-7+11-13+17-19"""
    # Genera le selezioni e i legami per tutti i 10 filamenti
    t1_ids, t2_ids, t3_ids = [], [], []
    bond_cmds = []
    
    first_frame = frames[0]
    
    def get_pca_perimeter(mol_indices):
        # 1. Estrai le coordinate 3D vere
        pts = np.array([first_frame[mol_com_parts[m]] for m in mol_indices])
        # 2. Centra i punti
        center = np.mean(pts, axis=0)
        pts_centered = pts - center
        # 3. Calcola il piano migliore (PCA)
        cov = np.cov(pts_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        v1 = eigenvectors[:, -1]
        v2 = eigenvectors[:, -2]
        # 4. Proietta in 2D
        proj_2d = np.array([[np.dot(p, v1), np.dot(p, v2)] for p in pts_centered])
        # 5. Calcola gli angoli nel piano per trovare il perimetro convesso (senza incroci!)
        angles = np.arctan2(proj_2d[:, 1], proj_2d[:, 0])
        order = np.argsort(angles)
        
        # Restituisci gli ID PyMOL (1-based) nell'ordine corretto
        return [mol_indices[i] + 1 for i in order]

    for strand in range(10):
        offset = strand * 22
        
        # Le 3 tetradi biologiche (vere) dedotte dalla struttura atomistica 143D 
        # (topologia basket anti-parallela). I filamenti "up" e "down" causano 
        # uno scivolamento delle guanine tra la tetrade superiore e inferiore.
        # Gli indici qui sotto sono "mol_idx" (0-based, quindi G2 = 1, G10 = 9).
        
        # Top Tetrad: G2, G10, G14, G22
        m_t1 = [offset + 1, offset + 9, offset + 13, offset + 21]
        
        # Middle Tetrad: G3, G9, G15, G21
        m_t2 = [offset + 2, offset + 8, offset + 14, offset + 20]
        
        # Bottom Tetrad: G4, G8, G16, G20
        m_t3 = [offset + 3, offset + 7, offset + 15, offset + 19]
        
        # Aggiungiamo agli ID globali (1-based) per colorare
        t1_ids.extend([m+1 for m in m_t1])
        t2_ids.extend([m+1 for m in m_t2])
        t3_ids.extend([m+1 for m in m_t3])
        
        # Troviamo il perimetro ottimo proiettando sul piano complanare (PCA)
        for m_tetrad in [m_t1, m_t2, m_t3]:
            opt_ids = get_pca_perimeter(m_tetrad)
            for i in range(4):
                bond_cmds.append(f"bond id {opt_ids[i]}, id {opt_ids[(i+1)%4]}")

    t1_sel = "+".join(map(str, t1_ids))
    t2_sel = "+".join(map(str, t2_ids))
    t3_sel = "+".join(map(str, t3_ids))
    bond_cmds_str = "\n".join(bond_cmds)

    # Genera lo script di visualizzazione per PyMOL
    pml_filename = "load_tel22_pymol.pml"
    pml_content = f"""\
# Script generato automaticamente per visualizzare il TEL22 in PyMOL
load {pdb_filename}, tel22_cg

# Nasconde tutte le rappresentazioni di default
hide all

{style_cmds}

# Colora il backbone
color grey50, tel22_cg

# Seleziona e colora le 3 Tetradi per tutti i 10 filamenti
select tetrad_1, id {t1_sel}
color red, tetrad_1

select tetrad_2, id {t2_sel}
color green, tetrad_2

select tetrad_3, id {t3_sel}
color blue, tetrad_3

# Unisci le guanine della stessa tetrade perimetralmente per evitare incroci diagonali
{bond_cmds_str}

# Mostra i legami del quadrato come bastoncini sottili
show sticks, tetrad_1 or tetrad_2 or tetrad_3
set stick_radius, 0.2

# Migliora la resa visiva
orient tel22_cg
bg_color white
set ray_opaque_background, on
set depth_cue, on
set spec_reflect, 0.5

# Imposta il player di animazione
mset 1 -x
mplay

print("==========================================================")
print("Traiettoria Coarse-Grained caricata con visualizzazione a sfere!")
print("Le 3 tetradi sono colorate in Rosso, Verde e Blu per TUTTI i filamenti.")
print("I legami evidenziano la corretta topologia planare quadrata.")
print("Premi Play in basso a destra per animare il movimento.")
print("==========================================================")
"""
    with open(pml_filename, "w") as out:
        out.write(pml_content)
        
    print(f"File '{pml_filename}' generato con successo.")
    print(f"\nFatto! Ora puoi semplicemente aprire {pml_filename} in PyMOL!")

if __name__ == "__main__":
    main()
