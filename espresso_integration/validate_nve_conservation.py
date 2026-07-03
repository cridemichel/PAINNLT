import espressomd
import espressomd.painn
import numpy as np
import struct
import time
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

print("=== TEST 2: NVE ENERGY CONSERVATION ===")

# 1. Carica il frame
try:
    atomic_numbers, coords, _, _ = read_first_frame_md17_bin("ethanol_val.bin")
    num_atoms = len(atomic_numbers)
except FileNotFoundError:
    print("Errore: ethanol_val.bin non trovato! Assicurati di eseguire dalla root.")
    exit(1)

# 2. Setup Sistema ESPResSo
box_size = 20.0 
system = espressomd.System(box_l=[box_size, box_size, box_size])
system.time_step = 0.001  # Time step piccolissimo per NVE con masse unitarie
system.cell_system.skin = 0.4

# Spostiamo il centro di massa della molecola al centro del box
center_of_mass = np.mean(coords, axis=0)
shift = (box_size / 2.0) - center_of_mass
coords_shifted = coords + shift

# Aggiunta particelle con velocità iniziali casuali
np.random.seed(42)
velocities = np.random.randn(num_atoms, 3) * 0.01  # Velocità iniziali piccole

for i in range(num_atoms):
    system.part.add(pos=coords_shifted[i], type=int(atomic_numbers[i]), v=velocities[i])


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


# Estraiamo il cutoff convertendolo in float per sicurezza
model_cutoff = float(nn_config["cutoff"])
# 3. Verlet Lists Hack
for i in range(10):
    for j in range(i, 10):
        system.non_bonded_inter[i, j].lennard_jones.set_params(epsilon=0.0, sigma=1.0, cutoff=model_cutoff, shift=0.0)

# 4. Attivazione Modello PaiNN
espressomd.painn.activate_painn_potential(
    model_path="best_painn_etanolo.pt", 
    num_species=int(nn_config["num_species"]), 
    hidden_channels=int(nn_config["hidden_channels"]), 
    n_layers=int(nn_config["n_layers"]), 
    num_rbf=int(nn_config["num_rbf"]), 
    cutoff=model_cutoff, 
    device="cpu"
)

# 5. Simulazione NVE (Nessun Termostato)
print("Inizio integrazione NVE per 1000 step...")
print(f"{'Step':>8} | {'E_kin':>12} | {'E_pot (PaiNN)':>15} | {'E_tot':>15}")
print("-" * 59)

# Assicuriamoci che le forze siano calcolate al t=0
system.integrator.run(0)

steps = 1000
log_interval = 100

for step in range(0, steps + 1, log_interval):
    if step > 0:
        system.integrator.run(log_interval)
    
    # E_kin è gestito da ESPResSo
    e_kin = system.analysis.energy()["kinetic"]
    
    # E_pot è letto direttamente dalla libreria custom C++ PaiNN
    e_pot = espressomd.painn.get_painn_energy()
    
    e_tot = e_kin + e_pot
    
    print(f"{step:8d} | {e_kin:12.6f} | {e_pot:15.6f} | {e_tot:15.6f}")

print("\nSimulazione completata!")
print("Se E_tot rimane costante, il potenziale è perfettamente conservativo!")
