import numpy as np

n1_coords = {}
o6_coords = {}

with open("../../tutorials/tel22/143D.pdb", 'r') as f:
    for line in f:
        if line.startswith("ATOM") and line[17:20] == " DG":
            atom_name = line[12:16].strip()
            res_id = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if atom_name == "N1":
                n1_coords[res_id] = np.array([x, y, z])
            elif atom_name == "O6":
                o6_coords[res_id] = np.array([x, y, z])

print("H-bonds (N1 to O6 < 3.5 A):")
for g1 in n1_coords:
    for g2 in o6_coords:
        if g1 != g2:
            dist = np.linalg.norm(n1_coords[g1] - o6_coords[g2])
            if dist < 3.5:
                print(f"G{g1} (N1) -> G{g2} (O6): {dist:.2f} A")

