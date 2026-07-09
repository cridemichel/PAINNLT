import numpy as np

coords = {}
with open("../../tutorials/tel22/143D.pdb", 'r') as f:
    for line in f:
        if line.startswith("ATOM") and line[17:20] == " DG":
            atom_name = line[12:16].strip()
            res_id = int(line[22:26])
            if atom_name == "N1":
                coords[res_id] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

for g1 in coords:
    distances = []
    for g2 in coords:
        if g1 != g2:
            distances.append((g2, np.linalg.norm(coords[g1] - coords[g2])))
    distances.sort(key=lambda x: x[1])
    # The two closest should be the H-bond partners in the tetrad
    # The 3rd closest is usually the diagonally opposite one in the tetrad
    print(f"G{g1} is closest to: {distances[:3]}")

