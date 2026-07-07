import espressomd
import espressomd.painn
import numpy as np
import struct
import math
import json

def read_first_frame_md17_bin(filepath):
    """Legge il primo frame dal dataset MD17 binario."""
    with open(filepath, "rb") as f:
        num_frames = struct.unpack("i", f.read(4))[0]
        num_atoms = struct.unpack("i", f.read(4))[0]
        
        energy = struct.unpack("d", f.read(8))[0]
        atomic_numbers = np.frombuffer(f.read(num_atoms * 4), dtype=np.int32)
        coords = np.frombuffer(f.read(num_atoms * 3 * 8), dtype=np.float64).reshape((num_atoms, 3))
        forces = np.frombuffer(f.read(num_atoms * 3 * 8), dtype=np.float64).reshape((num_atoms, 3))
        
        return atomic_numbers, coords, energy, forces

print("=== TEST 1: SINGLE-POINT FORCES VALIDATION ===")

# 1. Carica il frame di ground-truth
try:
    atomic_numbers, coords, true_energy, true_forces = read_first_frame_md17_bin("ethanol_val.bin")
    num_atoms = len(atomic_numbers)
    print(f"Caricato dataset con {num_atoms} atomi.")
except FileNotFoundError:
    print("Errore: ethanol_val.bin non trovato! Assicurati di eseguire dalla root.")
    exit(1)

# 2. Setup Sistema ESPResSo
# Creiamo un box grande a sufficienza per contenere la molecola isolata
box_size = 20.0 
system = espressomd.System(box_l=[box_size, box_size, box_size])
system.time_step = 0.001
system.cell_system.skin = 0.4

# Spostiamo il centro di massa della molecola al centro del box
center_of_mass = np.mean(coords, axis=0)
shift = (box_size / 2.0) - center_of_mass
coords_shifted = coords + shift

# Aggiunta particelle
for i in range(num_atoms):
    system.part.add(pos=coords_shifted[i], type=int(atomic_numbers[i]))


# read json file with model parameters
config_path = "best_painn_etanolo_config.json"

try:
    with open(config_path, "r") as f:
        nn_config = json.load(f)
    
    # Stampa verbosa per il controllo scientifico dei parametri
    print("\n" + "="*60)
    print("   PARAMETRI ARCHITETTURA PAINN CARICATI DAL JSON")
    print("="*60)
    print(f" * Number of species (num_species):       {nn_config['num_species']}")
    print(f" * Hidden Channels (dim):          {nn_config['hidden_channels']}")
    print(f" * Number of Layers (n_layers):        {nn_config['n_layers']}")
    print(f" * Gaussian, bases (num_rbf):       {nn_config['num_rbf']}")
    print(f" * Cutoff Radfius (cutoff):      {nn_config['cutoff']} Å")
    print("="*60 + "\n")

except FileNotFoundError:
    print(f"\nCRITICAL ERROR: Configuration file '{config_path}' does not exist.")
    print("Make sure you have run the C++ training at least once to generate it.")
    exit(1)


# 3. Verlet Lists Hack (Interazioni Dummy)
# Obbligatorio in ESPResSo per fargli creare le Verlet lists 
# per particelle che altrimenti non interagirebbero
for i in range(10):
    for j in range(i, 10):
        system.non_bonded_inter[i, j].lennard_jones.set_params(epsilon=0.0, sigma=1.0, cutoff=float(nn_config['cutoff']), shift="auto")

# 4. Attivazione Modello PaiNN
espressomd.painn.activate_painn_potential(
    model_path="best_painn_etanolo.pt", 
    num_species=int(nn_config['num_species']), 
    hidden_channels=int(nn_config['hidden_channels']), 
    n_layers=int(nn_config['n_layers']), 
    num_rbf=int(nn_config['num_rbf']), 
    cutoff=float(nn_config['cutoff']), 
    device="cpu"  # Forza CPU per compatibilità MacOS
)

# 5. Esegui il calcolo delle forze
system.integrator.run(0)

# 6. Estrazione Forze ed Errori
esp_forces = np.zeros((num_atoms, 3))
for p in system.part:
    esp_forces[p.id] = p.f

mae = np.mean(np.abs(esp_forces - true_forces))
mse = np.mean(np.square(esp_forces - true_forces))

print("\n--- RISULTATI ---")
for i in range(num_atoms):
    print(f"Atomo {i:2d} (Z={atomic_numbers[i]:2d}):")
    print(f"  PaiNN+ESPResSo: {esp_forces[i][0]:9.4f}  {esp_forces[i][1]:9.4f}  {esp_forces[i][2]:9.4f}")
    print(f"  Ground Truth:   {true_forces[i][0]:9.4f}  {true_forces[i][1]:9.4f}  {true_forces[i][2]:9.4f}")
    print(f"  Diff Assoluta:  {abs(esp_forces[i][0]-true_forces[i][0]):9.4f}  {abs(esp_forces[i][1]-true_forces[i][1]):9.4f}  {abs(esp_forces[i][2]-true_forces[i][2]):9.4f}")

print("\n--- METRICHE GLOBALI ---")
print(f"Mean Absolute Error (MAE) Forze: {mae:.6f} (dovrebbe rispecchiare le performance in training/validation)")
print(f"Mean Squared Error  (MSE) Forze: {mse:.6f}")
