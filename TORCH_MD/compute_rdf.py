import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

def compute_rdf(trajectory, box_size, r_max, bins):
    """
    Calcola la funzione di distribuzione radiale g(r).
    trajectory: numpy array (N_frames, N_atoms, 3)
    """
    n_frames, n_atoms, _ = trajectory.shape
    density = n_atoms / (box_size**3)
    dr = r_max / bins
    radii = np.linspace(0, r_max, bins + 1)
    centers = (radii[:-1] + radii[1:]) / 2
    
    g_r = np.zeros(bins)
    
    for frame in range(n_frames):
        pos = trajectory[frame]
        # Minimum image convention per distanze a coppie
        # pdist calcola tutte le distanze i < j (N(N-1)/2 coppie)
        # Lo implementiamo calcolando la differenza usando l'array intero:
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # Applica minimum image convention se box_size non è None
        if box_size is not None:
            diff = diff - box_size * np.round(diff / box_size)
            
        distances = np.linalg.norm(diff, axis=-1)
        # Prendi solo la parte triangolare superiore per evitare auto-distanze
        i, j = np.triu_indices(n_atoms, k=1)
        dist_values = distances[i, j]
        
        # Istogramma delle distanze
        hist, _ = np.histogram(dist_values, bins=radii)
        g_r += hist

    # Normalizzazione
    # Per ogni bin, il volume del guscio sferico è 4/3 pi ((r+dr)^3 - r^3)
    vol_shell = (4/3) * np.pi * (radii[1:]**3 - radii[:-1]**3)
    # Numero atteso di particelle nel guscio per un gas ideale
    # = densità * volume_shell
    # Visto che calcoliamo N(N-1)/2 coppie, per ottenere il numero medio di vicini 
    # per particella dovremmo moltiplicare hist per 2 e dividere per n_atoms.
    # Quindi g(r) = (hist * 2 / n_atoms) / (numero atteso in un gas ideale)
    expected = density * vol_shell
    g_r = (g_r * 2.0 / (n_atoms * n_frames)) / expected
    
    return centers, g_r

def main():
    box_size = 2.0 # nm
    r_max = box_size / 2.0
    bins = 100
    
    print("[INFO] Calcolo RDF del dataset originale (GROMACS)...")
    data_gmx = np.load('dataset.npz')
    traj_gmx = data_gmx['pos']
    r_gmx, g_gmx = compute_rdf(traj_gmx, box_size, r_max, bins)
    
    print("[INFO] Calcolo RDF della simulazione neurale (TorchMD-Net)...")
    traj_nn = np.load('trajectory.npy')
    # Prendi solo la seconda metà della traiettoria per l'equilibrio
    equilibration = len(traj_nn) // 2
    traj_nn = traj_nn[equilibration:]
    r_nn, g_nn = compute_rdf(traj_nn, box_size, r_max, bins)
    
    print("[INFO] Generazione del grafico plot_rdf.png...")
    plt.figure(figsize=(8, 6))
    plt.plot(r_gmx, g_gmx, label="GROMACS (Reference)", color='black', linewidth=2)
    plt.plot(r_nn, g_nn, label="TorchMD-Net (SchNet)", color='red', linestyle='--', linewidth=2)
    
    plt.xlabel('Distanza r (nm)')
    plt.ylabel('g(r)')
    plt.title('Funzione di Distribuzione Radiale')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_rdf.png', dpi=300)

if __name__ == "__main__":
    main()
