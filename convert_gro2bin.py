import numpy as np
import MDAnalysis as mda
import struct
import os

def calcola_proprieta_gruppo(positions, forces, masses):
    """
    Calcola il Centro di Massa, gli Assi Principali e proietta le forze
    per un singolo gruppo di atomi preservando Forze Nette e Momenti Torcenti (Torque).
    """
    total_mass = np.sum(masses)
    # 1. Centro di Massa (Sito 1)
    com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
    
    # Forza netta sul Centro di Massa
    net_force = np.sum(forces, axis=0)
    
    # 2. Calcolo degli Assi Principali tramite Tensore d'Inerzia
    pos_rel = positions - com
    inertia_tensor = np.zeros((3, 3))
    for m, r in zip(masses, pos_rel):
        inertia_tensor += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    
    # Autovettori (gli assi principali sono le colonne di v)
    _, v = np.linalg.eigh(inertia_tensor)
    asse_x = v[:, 0] # Primo asse principale
    asse_y = v[:, 1] # Secondo asse principale
    
    # 3. Posizionamento dei Siti Virtuali (Distanza fissa di 1.0 Angstrom dal COM)
    d = 1.0 
    sito2_pos = com + d * asse_x
    sito3_pos = com + d * asse_y
    
    # 4. Proiezione del Torque sulle forze dei siti virtuali
    # Calcoliamo il Torque totale generato dalle forze atomiche rispetto al COM
    torque_totale = np.sum(np.cross(pos_rel, forces), axis=0)
    
    # Ripartiamo il torque applicando forze opposte sui siti virtuali (coppia di forze)
    # Per semplicità geometrica e stabilità del Force Matching:
    f_sito2 = np.cross(torque_totale, asse_x) / (2.0 * d)
    f_sito3 = np.cross(torque_totale, asse_y) / (2.0 * d)
    
    # La forza sul sito 1 (COM) assorbe il resto per garantire la risultante traslazionale
    f_sito1 = net_force - (f_sito2 + f_sito3)
    
    coords = np.vstack([com, sito2_pos, sito3_pos])
    sito_forces = np.vstack([f_sito1, f_sito2, f_sito3])
    
    return coords, sito_forces

def converti_gromacs_a_cg_bin(tpr_path, trr_path, out_bin_path, index_gruppi):
    """
    Legge una traiettoria GROMACS e genera il file binario per PaiNN C++.
    index_gruppi: lista di liste contenente gli indici (0-indexed) degli atomi 
                  che compongono ciascun macro-gruppo Coarse-Grained.
    """
    print(f"Caricamento topologia {tpr_path} e traiettoria {trr_path}...")
    u = mda.Universe(tpr_path, trr_path)
    
    num_frames = len(u.trajectory)
    num_groups = len(index_gruppi)
    num_sites_tot = num_groups * 3 # Ogni gruppo produce 3 siti nel file binario
    
    # Definiamo gli ID dei tipi per PaiNN (es. Tipo 1 per il COM, Tipo 2 e 3 per i siti virtuali dell'asse X e Y)
    # Questo array si ripete identico per ogni frame
    atomic_numbers = []
    for g_idx in range(num_groups):
        atomic_numbers.extend([1, 2, 3]) # ID arbitrari per la GNN
    atomic_numbers = np.array(atomic_numbers, dtype=np.int32)
    
    print(f"Traiettoria rilevata: {num_frames} frame.")
    print(f"Struttura CG: {num_groups} gruppi -> {num_sites_tot} siti totali nel grafo PaiNN.")
    
    # Apertura del file binario in scrittura
    with open(out_bin_path, "wb") as f:
        # 1. HEADER: Numero di frame e Numero totale di SITI (nodi del grafo)
        f.write(struct.pack('ii', num_frames, num_sites_tot))
        
        # 2. LOOP SUI FRAME
        for frame_idx, ts in enumerate(u.trajectory):
            # Nota: GROMACS salva le forze nel file .trr. MDAnalysis le espone in u.atoms.forces
            if not hasattr(u.atoms, 'forces'):
                raise RuntimeError("Il file di traiettoria non contiene le forze! Assicurati di usare un file .trr compilato con nstfout > 0.")
            
            frame_coords = []
            frame_forces = []
            
            # Per questa simulazione CG l'energia potenziale totale del frame all-atom (se disponibile) 
            # può essere estratta, altrimenti usiamo 0.0 (ci concentreremo sul Force Matching tramite i gradienti)
            energia_potenziale = 0.0 
            
            for gruppo in index_gruppi:
                # Estraiamo posizioni, forze e masse degli atomi appartenenti a questo blocco CG
                ag = u.atoms[gruppo]
                pos = ag.positions      # In Angstrom
                forces = ag.forces      # In kJ/(mol * A) - Unità standard di MDAnalysis per GROMACS
                masses = ag.masses      # In unità di massa atomica
                
                # Elaborazione geometrica e Force Matching
                coords_cg, forze_cg = calcola_proprieta_gruppo(pos, forces, masses)
                
                frame_coords.append(coords_cg)
                frame_forces.append(forze_cg)
            
            # Compattiamo i dati del frame corrente
            frame_coords = np.vstack(frame_coords).astype(np.float64) # (NumSiti, 3)
            frame_forces = np.vstack(frame_forces).astype(np.float64) # (NumSiti, 3)
            
            # Scrittura binaria continua (coerente con il lettore C++)
            # Energia (1 double)
            f.write(struct.pack('d', energia_potenziale))
            # Numeri atomici (N int)
            f.write(atomic_numbers.tobytes())
            # Coordinate (N*3 double)
            f.write(frame_coords.tobytes())
            # Forze (N*3 double)
            f.write(frame_forces.tobytes())
            
            if (frame_idx + 1) % 100 == 0 or (frame_idx + 1) == num_frames:
                print(f" -> Elaborati {frame_idx + 1}/{num_frames} frame...")

    print(f"Conversione completata! File salvato in: {out_bin_path}")

# =====================================================================
# ESEMPIO DI UTILIZZO
# =====================================================================
if __name__ == "__main__":
    # Supponiamo di avere una molecola mappata in 2 macro-gruppi.
    # Gruppo 0 contiene gli atomi con indice 0, 1, 2, 3, 4
    # Gruppo 1 contiene gli atomi con indice 5, 6, 7, 8
    mappatura_beads = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8]
    ]
    
    # Sostituisci con i percorsi reali dei tuoi file GROMACS (.tpr e .trr con FORZE)
    converti_gromacs_a_cg_bin(
        tpr_path="simulazione_aa.tpr",
        trr_path="simulazione_aa.trr", 
        out_bin_path="coarse_grained_train.bin",
        index_gruppi=mappatura_beads
    )
