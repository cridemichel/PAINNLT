import espressomd
import espressomd.painn
import numpy as np
import json
import time

print("=== TEST NVE ENERGY CONSERVATION (COARSE-GRAINED) ===")

# 1. Generiamo un reticolo cubico per testare l'integratore, poiché il dataset originale
# contiene particelle sovrapposte o molto ravvicinate (artefatti del calcolo del COM con PBC)
box_size = 2.0  # Box size in nm per l'acqua
coords = []
L = int(np.floor(box_size / 0.4))
for x in range(L):
    for y in range(L):
        for z in range(L):
            coords.append([x*0.4+0.1, y*0.4+0.1, z*0.4+0.1])
coords = np.array(coords)
num_atoms = len(coords)
atomic_numbers = np.zeros(num_atoms, dtype=np.int32)

print(f"Sistema Inizializzato con {num_atoms} siti virtuali/particelle CG su un reticolo {L}x{L}x{L}.")

# 2. Setup Sistema ESPResSo
system = espressomd.System(box_l=[box_size, box_size, box_size])
system.time_step = 0.001  # Time step piccolissimo per conservazione dell'energia
system.cell_system.skin = 0.4

# Aggiunta particelle con velocità iniziali casuali
np.random.seed(42)
velocities = np.random.randn(num_atoms, 3) * 0.01

for i in range(num_atoms):
    system.part.add(pos=coords[i], type=int(atomic_numbers[i]), v=velocities[i])

# 3. Lettura JSON con parametri del modello CG
config_path = "../best_cg_model_config.json"

try:
    with open(config_path, "r") as f:
        nn_config = json.load(f)
    
    print("\n" + "="*60)
    print("   PARAMETRI ARCHITETTURA PAINN CG CARICATI DAL JSON")
    print("="*60)
    print(f" * Number of species (num_species): {nn_config['num_species']}")
    print(f" * Cutoff Radius (cutoff):          {nn_config['cutoff']} nm")
    print("="*60 + "\n")

except FileNotFoundError:
    print(f"\nCRITICAL ERROR: Configuration file '{config_path}' does not exist.")
    exit(1)

model_cutoff = float(nn_config["cutoff"])

# 4. Verlet Lists Hack (ESPResSo richiede un'interazione fittizia per creare la lista dei vicini)
for i in range(10):
    for j in range(i, 10):
        system.non_bonded_inter[i, j].lennard_jones.set_params(
            epsilon=0.0, sigma=1.0, cutoff=model_cutoff, shift="auto"
        )

# 5. Attivazione Modello PaiNN CG
model_path = "../best_cg_model.pt"
try:
    espressomd.painn.activate_painn_potential(
        model_path=model_path, 
        num_species=int(nn_config["num_species"]), 
        hidden_channels=int(nn_config["hidden_channels"]), 
        n_layers=int(nn_config["n_layers"]), 
        num_rbf=int(nn_config["num_rbf"]), 
        cutoff=model_cutoff, 
        device="cpu"
    )
except Exception as e:
    print(f"Errore caricamento modello: {e}")
    exit(1)


# 6. Simulazione NVE (Nessun Termostato)
print("Inizio integrazione NVE per 1000 step...")
print(f"{'Step':>8} | {'E_kin':>12} | {'E_pot (PaiNN)':>15} | {'E_tot':>15}")
print("-" * 59)

# Assicuriamoci che le forze siano calcolate al t=0
system.integrator.run(0)

steps = 1000
log_interval = 100
energies = []

for step in range(0, steps + 1, log_interval):
    if step > 0:
        system.integrator.run(log_interval)
    
    # E_kin è gestito da ESPResSo
    e_kin = system.analysis.energy()["kinetic"]
    
    # E_pot è letto direttamente dalla libreria custom C++ PaiNN
    e_pot = espressomd.painn.get_painn_energy()
    
    e_tot = e_kin + e_pot
    energies.append(e_tot)
    
    print(f"{step:8d} | {e_kin:12.6f} | {e_pot:15.6f} | {e_tot:15.6f}")

energies = np.array(energies)
drift = np.max(energies) - np.min(energies)
print("\nSimulazione completata!")
print(f"Drift massimo dell'energia totale: {drift:.6f}")
print("Se E_tot rimane costante, il potenziale CG è perfettamente conservativo!")
