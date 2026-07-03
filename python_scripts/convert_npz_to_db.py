import argparse
import os
import numpy as np
from ase import Atoms
from ase.db import connect

def main():
    parser = argparse.ArgumentParser(
        description="Converte un dataset rMD17 .npz in un database ASE .db per SchNetPack, includendo le unità di misura corrette."
    )
    parser.add_argument("input_file", type=str, help="Il percorso del file .npz di rMD17")
    parser.add_argument("--distance_unit", type=str, default="Angstrom")
    parser.add_argument("--energy_unit", type=str, default="kcal/mol")
    
    args = parser.parse_args()
    npz_file_path = args.input_file

    if not os.path.exists(npz_file_path):
        print(f"Errore: Il file '{npz_file_path}' non esiste.")
        return

    nome_base, _ = os.path.splitext(npz_file_path)
    db_file_path = nome_base + ".db"

    print(f"Lettura del file {npz_file_path}...")
    dataset = np.load(npz_file_path)

    positions = dataset['coords']            
    forces = dataset['forces']                
    atomic_numbers = dataset['nuclear_charges'] 
    energies = dataset['energies']            

    num_frames = positions.shape[0]
    print(f"Trovati {num_frames} frame. Creazione del file {db_file_path}...")

    # Rimuoviamo il vecchio db se esiste per evitare conflitti di tabelle
    if os.path.exists(db_file_path):
        os.remove(db_file_path)

    with connect(db_file_path) as db:
        
        # NOTA CRUCIALE: Usiamo esattamente i nomi atomici di metadati privati 
        # che SchNetPack cerca internamente per mappare le proprietà.
        db.metadata = {
            "_distance_unit": args.distance_unit,
            "_property_unit_dict": {
                "energy": args.energy_unit,
                "forces": f"{args.energy_unit}/{args.distance_unit}"
            }
        }
        
        for i in range(num_frames):
            pos_frame = positions[i]
            forces_frame = forces[i]
            energy_frame = energies[i]
            
            atoms = Atoms(
                numbers=atomic_numbers, 
                positions=pos_frame, 
                pbc=False
            )
            
            # Salviamo l'energia impacchettata come array NumPy di dimensione (1,)
            db.write(atoms, data={
                'energy': np.array([energy_frame], dtype=np.float64), 
                'forces': forces_frame
            })
            
            if (i + 1) % 10000 == 0:
                print(f"Frame {i + 1}/{num_frames} elaborato...")

    print(f"\nConversione completata! File pronto: {db_file_path}")

if __name__ == "__main__":
    main()
