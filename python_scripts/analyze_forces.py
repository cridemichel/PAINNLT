import struct
import numpy as np

def analyze_dataset_variance(bin_filepath):
    print(f"[INFO] Lettura del file {bin_filepath} in corso...")
    
    forces = []
    
    try:
        with open(bin_filepath, "rb") as f:
            # Leggi numero di frame
            num_frames_data = f.read(4)
            if not num_frames_data:
                print("File vuoto o non trovato.")
                return
            num_frames = struct.unpack("i", num_frames_data)[0]
            print(f"[INFO] Trovati {num_frames} frame. Estrazione forze...")
            
            for _ in range(num_frames):
                num_molecules = struct.unpack("i", f.read(4))[0]
                num_total_sites = struct.unpack("i", f.read(4))[0]
                
                for _ in range(num_molecules):
                    mol_id = struct.unpack("i", f.read(4))[0]
                    num_sites = struct.unpack("i", f.read(4))[0]
                    
                    center = struct.unpack("3f", f.read(12))
                    total_force = struct.unpack("3f", f.read(12))  # XYZ della forza
                    total_torque = struct.unpack("3f", f.read(12))
                    
                    forces.append(total_force)
                    
                    # Salta la lettura dei siti (non ci serve per l'analisi delle forze target)
                    for _ in range(num_sites):
                        f.read(4 + 12) # int (type) + 3 floats (xyz)
                        
    except FileNotFoundError:
        print(f"[ERRORE] File {bin_filepath} non trovato.")
        return

    forces = np.array(forces)
    
    # --- CALCOLI STATISTICI ---
    # 1. Forza media vettoriale (in un liquido all'equilibrio deve essere vicina a zero)
    mean_force = np.mean(forces, axis=0)
    
    # 2. MAE predicendo sempre 0 (quello che la rete fa all'Epoca 1)
    mae_zero = np.mean(np.abs(forces))
    
    # 3. MAE predicendo il vettore medio (baseline statistica)
    mae_mean = np.mean(np.abs(forces - mean_force))
    
    # 4. Deviazione Standard (quanto le molecole "urtano" violentemente)
    std_dev = np.std(forces)
    
    # 5. Calcolo dei percentili per trovare gli "outlier" estremi
    magnitudes = np.linalg.norm(forces, axis=1)
    p95 = np.percentile(magnitudes, 95)
    p99 = np.percentile(magnitudes, 99)

    print("\n" + "="*50)
    print(" 📊 ANALISI STATISTICA DELLE FORZE (kJ/mol*nm)")
    print("="*50)
    print(f"Molecole totali analizzate : {len(forces)}")
    print(f"Forza media (X, Y, Z)      : [{mean_force[0]:.2f}, {mean_force[1]:.2f}, {mean_force[2]:.2f}]")
    print("-" * 50)
    print(f"🎯 MAE Iniziale (Epoca 1)   : {mae_zero:.2f}  <-- Il tuo ~246!")
    print(f"📉 MAE Baseline (Media)     : {mae_mean:.2f}")
    print(f"🌊 Deviazione Standard (σ)  : {std_dev:.2f}")
    print("-" * 50)
    print(f"Forza massima registrata   : {np.max(magnitudes):.2f}")
    print(f"Il 95% delle forze è sotto : {p95:.2f}")
    print(f"Il 99% delle forze è sotto : {p99:.2f}")
    print("="*50)

if __name__ == "__main__":
    analyze_dataset_variance("cg_dataset.bin")
