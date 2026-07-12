import sys

def convert_vtf_to_pdb(vtf_file, pdb_file):
    with open(vtf_file, 'r') as f:
        lines = f.readlines()
        
    out = open(pdb_file, 'w')
    
    atoms = []
    bonds = []
    
    frame = 0
    reading_coords = False
    atom_idx = 0
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        if parts[0] == "atom":
            # atom 0 radius 1.0 name C type 0
            # or atom 0:10 radius 1.0
            idx_str = parts[1]
            if ":" in idx_str:
                start, end = map(int, idx_str.split(":"))
                for i in range(start, end+1):
                    atoms.append({"id": i, "type": "C"}) # placeholder
            else:
                atoms.append({"id": int(idx_str), "type": "C"})
        elif parts[0] == "bond":
            # bond 0:1
            if ":" in parts[1]:
                a, b = parts[1].split(":")
                bonds.append((int(a), int(b)))
        elif parts[0] == "timestep":
            if reading_coords:
                out.write("ENDMDL\n")
            frame += 1
            out.write(f"MODEL {frame}\n")
            reading_coords = True
            atom_idx = 0
        elif reading_coords:
            # coordinates
            if len(parts) == 3:
                x, y, z = map(float, parts)
                # PDB ATOM record
                # ATOM  %5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s
                idx = atom_idx + 1
                aname = "CA"
                resname = "UNK"
                resid = idx
                out.write(f"ATOM  {idx:5d}  {aname:<3s} {resname:3s} A{resid:4d}    {x*10:8.3f}{y*10:8.3f}{z*10:8.3f}  1.00  0.00           C\n")
                atom_idx += 1
                
    if reading_coords:
        out.write("ENDMDL\n")
        
    for b in bonds:
        # PDB CONECT uses 1-based indexing
        out.write(f"CONECT{b[0]+1:5d}{b[1]+1:5d}\n")
        
    out.close()
    print(f"Converted {frame} frames to {pdb_file}")

if __name__ == "__main__":
    convert_vtf_to_pdb("cg_trajectory.vtf", "cg_trajectory.pdb")
