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
    
    # Genera i comandi di stile in base alla scelta dell'utente
    if args.style == "tube":
        style_cmds = """\
# Mostra il DNA come tubi continui
show cartoon, tel22_cg
cartoon tube, tel22_cg
set cartoon_tube_radius, 0.5"""
    else:
        style_cmds = """\
# Mostra il DNA come sferette (Coarse-Grained beads)
show spheres, tel22_cg
set sphere_scale, 0.8, tel22_cg"""

    # Genera lo script di visualizzazione per PyMOL
    pml_filename = "load_tel22_pymol.pml"
    pml_content = f"""\
# Script generato automaticamente per visualizzare il TEL22 in PyMOL
load {pdb_filename}, tel22_cg

# Nasconde le rappresentazioni a linee o sfere disconnesse
hide all

{style_cmds}

# Colora ogni filamento con un colore distinto per seguire l'unfolding
color red, chain A
color green, chain B
color blue, chain C
color yellow, chain D
color cyan, chain E
color magenta, chain F
color orange, chain G
color purple, chain H
color pink, chain I
color teal, chain J

# Centra la visuale
orient tel22_cg
bg_color white
set ray_opaque_background, on

# Imposta il player di animazione (se ci sono piu frame)
mset 1 -x
mplay

echo "=========================================================="
echo "Traiettoria Coarse-Grained caricata!"
echo "Premi Play in basso a destra per animare l'unfolding."
echo "=========================================================="
"""
    with open(pml_filename, "w") as out:
        out.write(pml_content)
        
    print(f"File '{pml_filename}' generato con successo.")
    print(f"\nFatto! Ora puoi semplicemente aprire {pml_filename} in PyMOL!")

if __name__ == "__main__":
    main()
