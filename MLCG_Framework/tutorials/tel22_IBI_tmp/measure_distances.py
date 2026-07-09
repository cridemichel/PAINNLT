import numpy as np

def read_first_model(pdb_file):
    coords = {}
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if line.startswith("ATOM"):
                atom_id = int(line[6:11])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords[atom_id] = np.array([x, y, z])
    return coords

coords = read_first_model("cg_trajectory_clean.pdb")
g_ids = [2, 8, 14, 20]
print("Distances for Tetrad 1:")
for i in range(4):
    for j in range(i+1, 4):
        id1 = g_ids[i]
        id2 = g_ids[j]
        dist = np.linalg.norm(coords[id1] - coords[id2])
        print(f"{id1} - {id2}: {dist:.2f} Angstroms")
