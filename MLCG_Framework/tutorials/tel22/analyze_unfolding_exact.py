import numpy as np
import struct

# First, read the exact particle indices from the binary dataset just like run_cg_md.py did
mol_com_parts = {}
mol_vs_parts = {}

dataset_file = "tel22_dataset.bin"

with open(dataset_file, "rb") as f:
    num_frames = struct.unpack("i", f.read(4))[0]
    num_molecules = struct.unpack("i", f.read(4))[0] # Should be 220
    num_total_sites = struct.unpack("i", f.read(4))[0]
    box_dim = struct.unpack("3f", f.read(12))
    
    # We just want to know how many sites each molecule has to calculate indices
    # Because run_cg_md.py adds: p_com, then p_vs1, p_vs2...
    
    current_atom_idx = 0
    
    for mol_idx in range(num_molecules):
        mol_id = struct.unpack("i", f.read(4))[0]
        num_sites = struct.unpack("i", f.read(4))[0]
        
        # skip center, force, torque
        f.read(12 + 12 + 12)
        
        # skip sites
        for s in range(num_sites):
            f.read(4 + 12)
            
        # Particle added: COM
        mol_com_parts[mol_idx] = current_atom_idx
        current_atom_idx += 1
        
        # Particles added: VS
        for s in range(num_sites):
            mol_vs_parts[(mol_idx, s)] = current_atom_idx
            current_atom_idx += 1

print(f"Total atoms expected based on initialization logic: {current_atom_idx}")

# Read the VTF frames
vtf_file = "cg_trajectory.vtf"
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
                    coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                    current_frame.append(coords)
                except ValueError:
                    pass

if current_frame:
    frames.append(np.array(current_frame))

# Unfolding analysis
first_frame = frames[0]
last_frame = frames[-1]

print("\n--- Analisi End-to-End Distance e Raggio di Girazione (Rg) CORETTA ---")
# There are 10 DNA strands, each consists of 22 consecutive "molecules" (nucleotides) in the dataset
unfolded_count = 0
for strand in range(10):
    start_mol_idx = strand * 22
    end_mol_idx = start_mol_idx + 21
    
    # Get COM indices for this strand
    com_indices = [mol_com_parts[m] for m in range(start_mol_idx, end_mol_idx + 1)]
    
    # End-to-end distance (COM of first nucleotide to COM of last nucleotide)
    e2e_first = np.linalg.norm(first_frame[com_indices[0]] - first_frame[com_indices[-1]])
    e2e_last = np.linalg.norm(last_frame[com_indices[0]] - last_frame[com_indices[-1]])
    
    # Radius of Gyration of the COMs
    com_coords_first = first_frame[com_indices]
    center_first = np.mean(com_coords_first, axis=0)
    rg_first = np.sqrt(np.mean(np.sum((com_coords_first - center_first)**2, axis=1)))
    
    com_coords_last = last_frame[com_indices]
    center_last = np.mean(com_coords_last, axis=0)
    rg_last = np.sqrt(np.mean(np.sum((com_coords_last - center_last)**2, axis=1)))
    
    # Heuristic
    is_unfolded = rg_last > rg_first * 1.5 or e2e_last > 4.0
    status = "UNFOLDED ⚠️" if is_unfolded else "Folded ✅"
    if is_unfolded:
        unfolded_count += 1
        
    print(f"Strand {strand+1:2d}: Rg ({rg_first:.2f} -> {rg_last:.2f} nm) | E2E ({e2e_first:.2f} -> {e2e_last:.2f} nm) => {status}")

print(f"\nTotale molecole unfoldate: {unfolded_count} su 10")

# Write a clean PDB of the COM trajectory to view in PyMOL
pdb_filename = "cg_com_trajectory.pdb"
with open(pdb_filename, "w") as out:
    for f_idx, frame in enumerate(frames):
        out.write(f"MODEL {f_idx+1}\n")
        atom_count = 1
        for strand in range(10):
            start_mol_idx = strand * 22
            end_mol_idx = start_mol_idx + 21
            com_indices = [mol_com_parts[m] for m in range(start_mol_idx, end_mol_idx + 1)]
            for i, idx in enumerate(com_indices):
                res_num = i + 1
                chain_id = chr(ord('A') + strand)
                coords = frame[idx] * 10.0 # Convert nm to Angstroms for PDB
                # Write a standard ATOM record (fake C-alpha for rendering)
                out.write(f"ATOM  {atom_count:5d}  CA  CG  {chain_id}{res_num:4d}    {coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00  0.00           C\n")
                atom_count += 1
        out.write("ENDMDL\n")
print(f"Wrote {pdb_filename} for PyMOL visualization!")
