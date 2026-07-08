import numpy as np
import struct
import matplotlib.pyplot as plt

def main():
    print("[INFO] Parsing dataset to get COM indices...")
    dataset_file = "tel22_dataset.bin"
    mol_com_parts = {}
    
    with open(dataset_file, "rb") as f:
        num_frames = struct.unpack("i", f.read(4))[0]
        num_molecules = struct.unpack("i", f.read(4))[0]
        num_total_sites = struct.unpack("i", f.read(4))[0]
        box_dim = struct.unpack("3f", f.read(12))
        
        current_atom_idx = 0
        for mol_idx in range(num_molecules):
            mol_id = struct.unpack("i", f.read(4))[0]
            num_sites = struct.unpack("i", f.read(4))[0]
            f.read(12 + 12 + 12) # skip center, force, torque
            for s in range(num_sites):
                f.read(4 + 12)
                
            # Particle added: COM
            mol_com_parts[mol_idx] = current_atom_idx
            current_atom_idx += 1
            
            # Particles added: VS
            current_atom_idx += num_sites

    print("[INFO] Parsing VTF trajectory...")
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
        
    num_frames = len(frames)
    print(f"[INFO] Trovati {num_frames} frames nella traiettoria.")

    # Inizializziamo l'array dei Raggi di Girazione: shape (10 strands, num_frames)
    rg_timeseries = np.zeros((10, num_frames))
    
    for f_idx, frame in enumerate(frames):
        for strand in range(10):
            start_mol_idx = strand * 22
            end_mol_idx = start_mol_idx + 21
            com_indices = [mol_com_parts[m] for m in range(start_mol_idx, end_mol_idx + 1)]
            
            com_coords = frame[com_indices]
            center = np.mean(com_coords, axis=0)
            rg = np.sqrt(np.mean(np.sum((com_coords - center)**2, axis=1)))
            rg_timeseries[strand, f_idx] = rg

    print("[INFO] Salvataggio plot...")
    plt.figure(figsize=(10, 6))
    
    # Plot delle 10 time series
    for strand in range(10):
        plt.plot(rg_timeseries[strand], lw=2, alpha=0.8, label=f"Strand {strand+1}")
        
    # Calcolo la media su tutti i 10 strand per ogni frame
    mean_rg = np.mean(rg_timeseries, axis=0)
    plt.plot(mean_rg, color='black', lw=3, linestyle='--', label='Media (Mean Rg)')

    plt.title("Radius of Gyration (Rg) del TEL22 durante la MD MLCG", fontsize=14)
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Radius of Gyration (nm)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_img = "output/tel22_rg_timeseries.png"
    plt.savefig(out_img, dpi=300)
    print(f"[INFO] Plot salvato in: {out_img}")

if __name__ == "__main__":
    main()
