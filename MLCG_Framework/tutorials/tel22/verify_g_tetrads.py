import numpy as np
import os
import sys

# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "numpy",
# ]
# ///

def best_fit_plane(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    U, S, Vt = np.linalg.svd(centered)
    normal = Vt[-1]
    return centroid, normal

def analyze_tetrad(points):
    """
    points: 4x3 array of coordinates (G_a, G_b, G_c, G_d)
    Returns:
    - planarity_rmsd: RMS distance of points to the best fit plane
    - avg_edge: average distance between adjacent guanines
    - avg_diag: average distance of the diagonals
    """
    centroid, normal = best_fit_plane(points)
    
    # Distance to plane
    dists_to_plane = np.dot(points - centroid, normal)
    planarity_rmsd = np.sqrt(np.mean(dists_to_plane**2))
    
    # Edges: a-b, b-c, c-d, d-a
    edges = [
        np.linalg.norm(points[0]-points[1]),
        np.linalg.norm(points[1]-points[2]),
        np.linalg.norm(points[2]-points[3]),
        np.linalg.norm(points[3]-points[0])
    ]
    avg_edge = np.mean(edges)
    std_edge = np.std(edges)
    
    # Diagonals: a-c, b-d
    diagonals = [
        np.linalg.norm(points[0]-points[2]),
        np.linalg.norm(points[1]-points[3])
    ]
    avg_diag = np.mean(diagonals)
    
    return planarity_rmsd, avg_edge, std_edge, avg_diag, centroid

def main():
    # Read the PDB containing only COM particles
    pdb_file = "cg_com_trajectory.pdb"
    if not os.path.exists(pdb_file):
        print(f"File {pdb_file} not found.")
        sys.exit(1)
        
    # We only care about the LAST frame
    # A PDB from ESPResSo/MDAnalysis might have multiple frames separated by MODEL/ENDMDL
    # We parse the file and keep overwriting `coords` until the end to get the last frame.
    
    current_frame_coords = []
    last_frame_coords = []
    
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                current_frame_coords = []
            elif line.startswith("ATOM"):
                parts = line.split()
                # PDB ATOM line format usually has X Y Z at specific columns, 
                # but split() works if we just want the float values.
                # Standard PDB: X is 30-38, Y is 38-46, Z is 46-54
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                current_frame_coords.append([x, y, z])
            elif line.startswith("ENDMDL"):
                last_frame_coords = current_frame_coords
    
    if not last_frame_coords:
        last_frame_coords = current_frame_coords
        
    coords = np.array(last_frame_coords) / 10.0 # Convert from Angstrom to nm
    
    n_strands = 10
    n_residues = 22
    
    if len(coords) != n_strands * n_residues:
        print(f"Error: Expected {n_strands*n_residues} atoms, found {len(coords)}")
        sys.exit(1)
        
    # The G-tetrads are formed by residues (1-indexed):
    # T1: 2, 8, 14, 20
    # T2: 3, 9, 15, 21
    # T3: 4, 10, 16, 22
    # In 0-indexed arrays:
    t1_idx = [1, 7, 13, 19]
    t2_idx = [2, 8, 14, 20]
    t3_idx = [3, 9, 15, 21]
    
    print(f"{'Strand':<8} | {'Tetrad':<8} | {'Planarity RMSD (nm)':<20} | {'Avg Edge (nm)':<15} | {'Avg Diag (nm)':<15} | {'Edge/Diag Ratio':<15}")
    print("-" * 95)
    
    tetrad_centroids = []
    
    for strand in range(n_strands):
        offset = strand * n_residues
        
        for t_name, t_idx in [("T1", t1_idx), ("T2", t2_idx), ("T3", t3_idx)]:
            global_idx = [offset + i for i in t_idx]
            tetrad_coords = coords[global_idx]
            
            p_rmsd, a_edge, s_edge, a_diag, centroid = analyze_tetrad(tetrad_coords)
            tetrad_centroids.append(centroid)
            
            ratio = a_edge / a_diag
            # For a perfect square, side/diag = 1 / sqrt(2) = 0.707
            
            print(f"Strand {strand:<1} | {t_name:<8} | {p_rmsd:<20.4f} | {a_edge:<15.4f} | {a_diag:<15.4f} | {ratio:<15.4f}")
            
    # Calculate stacking distances between T1-T2 and T2-T3 for each strand
    print("\n--- Stacking Distances (nm) ---")
    for strand in range(n_strands):
        c1 = tetrad_centroids[strand*3 + 0]
        c2 = tetrad_centroids[strand*3 + 1]
        c3 = tetrad_centroids[strand*3 + 2]
        
        dist_12 = np.linalg.norm(c1 - c2)
        dist_23 = np.linalg.norm(c2 - c3)
        
        print(f"Strand {strand}: T1-T2 = {dist_12:.3f} nm, T2-T3 = {dist_23:.3f} nm")

if __name__ == "__main__":
    main()
