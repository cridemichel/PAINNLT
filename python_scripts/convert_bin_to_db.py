import numpy as np
from ase import Atoms
from ase.db import connect

# 1. PARAMETRI DEL TUO SISTEMA
bin_file_path = "traiettoria.bin"
db_file_path = "dataset_coarse_grained.db"
num_atomi = 150       # Sostituisci con il numero di siti/atomi del tuo sistema
box_size = 10.0       # Sostituisci con la dimensione del tuo box (in Angstrom)

# Ipotizziamo che tu sappia già i tipi atomici (Z) per i tuoi siti
# Es. 1 per il sito reale, 2 per i siti virtuali, ecc.
tipi_atomici = np.ones(num_atomi, dtype=int) 

print(f"Inizio conversione da {bin_file_path} a {db_file_path}...")

# 2. LETTURA DEL FILE .BIN
# Assumiamo che il file .bin contenga array in float32. 
# Modifica 'dtype' se i tuoi dati sono in double (float64)
# Questa logica dipende ESATTAMENTE da come hai scritto il file .bin!
dati_grezzi = np.fromfile(bin_file_path, dtype=np.float32)

# Ipotizziamo che ogni frame contenga: posizioni (N*3) e forze (N*3).
# Quindi ogni frame è lungo num_atomi * 6.
elementi_per_frame = num_atomi * 6 
num_frames = len(dati_grezzi) // elementi_per_frame

# Rimodelliamo l'array grezzo in (num_frames, num_atomi, 6)
dati_strutturati = dati_grezzi.reshape((num_frames, num_atomi, 6))

# 3. CREAZIONE DEL DATABASE ASE
with connect(db_file_path) as db:
    for i in range(num_frames):
        # Estraiamo posizioni (prime 3 colonne) e forze (ultime 3 colonne)
        posizioni = dati_strutturati[i, :, 0:3]
        forze = dati_strutturati[i, :, 3:6]
        
        # Creiamo l'oggetto Atoms di ASE
        atoms = Atoms(
            numbers=tipi_atomici, 
            positions=posizioni, 
            cell=[box_size, box_size, box_size], 
            pbc=True
        )
        
        # Scriviamo nel database. 
        # I target personalizzati (forze, mol_id, momenti torcenti) vanno nel dict 'data'
        db.write(atoms, data={'forces': forze})
        
        if i % 100 == 0:
            print(f"Frame {i}/{num_frames} elaborato...")

print("Conversione completata con successo! Il file .db è pronto per SchNetPack.")
