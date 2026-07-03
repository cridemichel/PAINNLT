import numpy as np
import struct
import os

def npz_to_bin_split(npz_path, out_prefix, train_ratio=0.8, max_frames=None):
    print(f"Caricamento di {npz_path}...")
    data = np.load(npz_path)

    # I numeri atomici di solito sono uguali per tutti i frame in MD17
    numbers = data['nuclear_charges'].astype(np.int32) 
    coords = data['coords'].astype(np.float64)      # (Frames, Atomi, 3)
    energies = data['energies'].astype(np.float64)  # (Frames,)
    forces = data['forces'].astype(np.float64)      # (Frames, Atomi, 3)

    num_frames_tot = coords.shape[0]
    num_atoms = coords.shape[1]
    
    if max_frames is not None:
        num_frames_tot = min(num_frames_tot, max_frames)
        
    # Calcolo split
    num_train = int(num_frames_tot * train_ratio)
    num_val = num_frames_tot - num_train

    # Calcolo AtomRef tramite Least Squares sul Training Set
    unique_z = np.unique(numbers)
    counts = {z: np.sum(numbers == z) for z in unique_z}
    
    X_train = np.zeros((num_train, len(unique_z)), dtype=np.float64)
    for i, z in enumerate(unique_z):
        X_train[:, i] = counts[z]
    
    y_train = energies[:num_train]
    atomref_coeffs, residuals, rank, s = np.linalg.lstsq(X_train, y_train, rcond=None)
    atomref_dict = {z: coeff for z, coeff in zip(unique_z, atomref_coeffs)}
    print(f"AtomRef calcolati sul training set: {atomref_dict}")
    
    # Applichiamo l'offset su TUTTO il dataset (sia train che val)
    baseline_energy = sum(counts[z] * atomref_dict[z] for z in unique_z)
    energies_centered = energies - baseline_energy
    
    print(f"Energia media prima del centering: {np.mean(energies):.6f}")
    print(f"Energia media dopo il centering: {np.mean(energies_centered):.6f}")
    
    print(f"Dataset totale: {num_frames_tot} frames ({num_atoms} atomi).")
    print(f" -> Training: {num_train} frames")
    print(f" -> Validation: {num_val} frames")

    # Funzione helper per scrivere un blocco di frame in binario
    def write_binary(filename, start_idx, end_idx):
        num_frames = end_idx - start_idx
        with open(filename, "wb") as f:
            # 1. HEADER: Scriviamo il numero di frame e di atomi come interi a 32 bit
            f.write(struct.pack('ii', num_frames, num_atoms))
            
            # 2. Per ogni frame, scriviamo i dati in formato binario continuo
            for i in range(start_idx, end_idx):
                # Energia (1 double)
                f.write(struct.pack('d', energies_centered[i]))
                # Numeri atomici (N int)
                f.write(numbers.tobytes())
                # Coordinate (N*3 double)
                f.write(coords[i].tobytes())
                # Forze (N*3 double)
                f.write(forces[i].tobytes())
                
        print(f"Salvato {filename} ({num_frames} frames)")

    # Creazione dei due file binari
    train_file = f"{out_prefix}_train.bin"
    val_file = f"{out_prefix}_val.bin"
    
    write_binary(train_file, 0, num_train)
    write_binary(val_file, num_train, num_frames_tot)

# ESECUZIONE
npz_to_bin_split("rmd17/npz_data/rmd17_ethanol.npz", "ethanol", train_ratio=0.8, max_frames=10000)
