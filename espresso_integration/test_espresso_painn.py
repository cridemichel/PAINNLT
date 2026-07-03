import espressomd
import espressomd.painn
import numpy as np
import os

# 1. Inizializzazione del sistema ESPResSo
system = espressomd.System(box_l=[12.0, 12.0, 12.0])
system.time_step = 0.01
system.cell_system.skin = 0.4

# 2. Creazione delle particelle (es. un molecola fittizia di etanolo con 9 atomi)
# Per semplicità, posizioniamo 9 particelle a caso all'interno della scatola
np.random.seed(42)
positions = np.random.rand(9, 3) * 5.0
atomic_numbers = [6, 6, 8, 1, 1, 1, 1, 1, 1] # C2H5OH

for i in range(9):
    # type è usato per l'atomic number nel nostro potenziale PaiNN
    system.part.add(pos=positions[i], type=atomic_numbers[i])

# 3. Attivazione delle Verlet Lists (ESPResSo richiede almeno un'interazione per costruire la lista dei vicini)
for i in range(10):
    for j in range(i, 10):
        system.non_bonded_inter[i, j].lennard_jones.set_params(epsilon=0.0, sigma=1.0, cutoff=5.0, shift=0.0)

# 4. Attivazione del Potenziale PaiNN
model_path = "best_painn_etanolo.pt"

if not os.path.exists(model_path):
    print(f"ATTENZIONE: {model_path} non trovato. Questo test verificherà solo il caricamento se il file esiste.")
else:
    # Parametri: path, num_atoms (embedding), hidden_channels, n_layers, num_rbf, cutoff, device
    espressomd.painn.activate_painn_potential(
    model_path="best_painn_etanolo.pt", 
    num_atoms=100, 
    hidden_channels=128, 
    n_layers=3, 
    num_rbf=20, 
    cutoff=5.0, 
    device="cpu"
)

    # 4. Calcolo delle forze
    # Eseguiamo 0 step di integrazione solo per forzare l'aggiornamento e il calcolo delle forze
    system.integrator.run(0)

    # 5. Stampa dei risultati
    print("\nForze calcolate dal modello PaiNN:")
    for p in system.part:
        print(f"Atomo {p.id} (Z={p.type}): {p.f}")
