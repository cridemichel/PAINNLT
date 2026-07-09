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
pts = np.array([coords[i] for i in g_ids])

# Center
center = np.mean(pts, axis=0)
pts_centered = pts - center

# PCA
cov = np.cov(pts_centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Top 2 eigenvectors
v1 = eigenvectors[:, -1]
v2 = eigenvectors[:, -2]

# Project
proj_2d = []
for p in pts_centered:
    proj_2d.append([np.dot(p, v1), np.dot(p, v2)])
proj_2d = np.array(proj_2d)

# Angles
angles = np.arctan2(proj_2d[:, 1], proj_2d[:, 0])

# Sort
order = np.argsort(angles)
ordered_ids = [g_ids[i] for i in order]

print(f"Optimal 2D projected order: {ordered_ids}")
