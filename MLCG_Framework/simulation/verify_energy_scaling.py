import espressomd
import espressomd.painn
import espressomd.interactions
import numpy as np
import json
import os

# ==============================================================================
# SCRIPT DI TEST: Conservazione dell'Energia e Scaling Quadratico nel tempo
# ==============================================================================
# Questo script calcola la fluttuazione dell'energia totale (E_kin + E_pot + E_prior)
# al variare del time step (dt). Se le interazioni C++ (PyTorch e Priors) sono 
# implementate correttamente, il Velocity Verlet garantirà che l'errore dell'energia
# scali in modo quadratico con il time step O(dt^2). 
# Dimezzando il dt, la deviazione standard dell'energia totale deve ridursi di 1/4 (~0.25).

print("=" * 80)
print(" VERIFICA SCALING QUADRATICO ENERGIA - NVE ENSEMBLE (TEL22 + WATER)")
print("=" * 80)

# 1. Configurazione del Sistema (Placeholder per TEL22 + Acqua)
# Quando la configurazione GROMACS sarà pronta, caricare le posizioni qui.
# Per ora generiamo un reticolo fittizio per il test.
box_size = 11.0  # nm
coords = []
L = 3
for x in range(L):
    for y in range(L):
        for z in range(L):
            coords.append([x*0.5, y*0.5, z*0.5])
coords = np.array(coords)[:9] # prendiamo esattamente 9 atomi come l'etanolo
num_atoms = len(coords)
atomic_numbers = np.zeros(num_atoms, dtype=np.int32)

np.random.seed(42)
initial_velocities = np.random.randn(num_atoms, 3) * 0.01

dt_values = [0.004, 0.002, 0.001, 0.0005, 0.00025, 0.000125]
physical_time = 0.02 # Tempo totale simulato (ps) costante per ogni run

print(f"\n{ 'dt (fs)':>10} | {'Steps':>6} | {'Std(E_tot)':>12} | {'Max(E_tot)-Min(E_tot)':>22} | {'Ratio (Std / prev Std)':>22}")
print("-" * 80)

prev_std = None

system = espressomd.System(box_l=[box_size, box_size, box_size])
system.cell_system.skin = 0.4

# 2. Caricamento Parametri Rete PaiNN
config_path = "../training/cg_model_config.json"
try:
    with open(config_path, "r") as f:
        nn_config = json.load(f)
except FileNotFoundError:
    print(f"\n[ERRORE] File '{config_path}' non trovato.")
    exit(1)

for i in range(10):
    for j in range(i, 10):
        system.non_bonded_inter[i, j].lennard_jones.set_params(
            epsilon=0.0, sigma=1.0, cutoff=float(nn_config.get('cutoff', 5.0)), shift="auto")

# 3. Attivazione PyTorch ML Potential
model_path = "../training/best_cg_model.pt"
if os.path.exists(model_path):
    espressomd.painn.activate_painn_potential(
        model_path=model_path,
        num_species=int(nn_config.get('num_species', 10)), 
        hidden_channels=int(nn_config.get('hidden_channels', 128)), 
        n_layers=int(nn_config.get('n_layers', 3)), 
        num_rbf=int(nn_config.get('num_rbf', 20)), 
        cutoff=float(nn_config.get('cutoff', 5.0)), 
        apply_envelope=nn_config.get("apply_envelope", False),
        use_bias=nn_config.get("use_bias", False),
        toxvaerd_alpha=nn_config.get("toxvaerd_alpha", 0.0),
        device="cpu"
    )
else:
    print(f"[WARNING] Modello {model_path} non trovato. L'energia ML sarà ignorata (test solo Priors).")

# (Opzionale) Qui potresti caricare anche i Prior topologici per TEL22 
# da "../preprocessing/cg_priors.json" e aggiungerli al sistema.

# 4. Ciclo di Simulazione per Scaling NVE
for dt in dt_values:
    system.time_step = dt
    
    # Reset sistema
    system.part.clear()
    for i in range(num_atoms):
        system.part.add(pos=coords[i], type=int(atomic_numbers[i]), v=initial_velocities[i])
    
    steps = int(physical_time / dt)
    energies = []
    
    # Step iniziale
    system.integrator.run(0)
    e_kin = system.analysis.energy()["kinetic"]
    e_bonded = system.analysis.energy()["bonded"] 
    e_nonbonded = system.analysis.energy()["non_bonded"]
    e_pot_ml = espressomd.painn.get_painn_energy() if os.path.exists(model_path) else 0.0
    
    energies.append(e_kin + e_bonded + e_nonbonded + e_pot_ml)
    
    for _ in range(steps):
        system.integrator.run(1)
        e_kin = system.analysis.energy()["kinetic"]
        e_bonded = system.analysis.energy()["bonded"]
        e_nonbonded = system.analysis.energy()["non_bonded"]
        e_pot_ml = espressomd.painn.get_painn_energy() if os.path.exists(model_path) else 0.0
        
        energies.append(e_kin + e_bonded + e_nonbonded + e_pot_ml)
        
    energies = np.array(energies)
    std_e = np.std(energies)
    delta_e = np.max(energies) - np.min(energies)
    
    if dt == dt_values[0]:
        print(f"DEBUG E_tot trace (dt={dt}): {energies[:5]}")
    
    ratio = ""
    if prev_std is not None and prev_std > 0:
        val = std_e / prev_std
        ratio = f"{val:.4f} (atteso ~0.25)"
        
    print(f"{dt*1000:9.4f} | {steps:6d} | {std_e:12.6f} | {delta_e:22.6f} | {ratio:>22}")
    prev_std = std_e

print("=" * 80)
