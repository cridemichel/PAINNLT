import espressomd
import espressomd.painn
import numpy as np
import struct
import json
import matplotlib.pyplot as plt

# Generiamo un reticolo cubico per testare l'integratore, poiché il dataset originale
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
coords_shifted = coords

# Impostiamo velocità iniziali identiche per tutti i run
np.random.seed(42)
initial_velocities = np.random.randn(num_atoms, 3) * 0.01

dt_values = [0.004, 0.002, 0.001, 0.0005, 0.00025, 0.000125]
physical_time = 0.02 # Ridotto per esecuzione rapida su 125 molecole

print(f"{'dt':>10} | {'Steps':>6} | {'Std(E_tot)':>12} | {'Max(E_tot)-Min(E_tot)':>22} | {'Ratio (Std / prev Std)':>22}", flush=True)
print("-" * 80, flush=True)

prev_std = None

system = espressomd.System(box_l=[box_size, box_size, box_size])
system.cell_system.skin = 0.4

# read json file with model parameters
config_path = "best_cg_model_config.json"

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


for i in range(10):
    for j in range(i, 10):
        system.non_bonded_inter[i, j].lennard_jones.set_params(epsilon=0.0, sigma=1.0, 
                                                               cutoff=float(nn_config['cutoff']), shift="auto")
        
espressomd.painn.activate_painn_potential(
    model_path="best_cg_model.pt", 
    num_species=int(nn_config['num_species']), hidden_channels=int(nn_config['hidden_channels']), 
    n_layers=int(nn_config['n_layers']), num_rbf=int(nn_config['num_rbf']), cutoff=float(nn_config['cutoff']), 
    device="cpu"
)

plot_dts = []
plot_stds = []

for dt in dt_values:
    system.time_step = dt
    
    system.part.clear()
    for i in range(num_atoms):
        system.part.add(pos=coords_shifted[i], type=int(atomic_numbers[i]), v=initial_velocities[i])
    
    steps = int(physical_time / dt)
    energies = []
    
    system.integrator.run(0)
    e_kin = system.analysis.energy()["kinetic"]
    e_pot = espressomd.painn.get_painn_energy()
    energies.append(e_kin + e_pot)
    
    for _ in range(steps):
        system.integrator.run(1)
        e_kin = system.analysis.energy()["kinetic"]
        e_pot = espressomd.painn.get_painn_energy()
        energies.append(e_kin + e_pot)
        
    energies = np.array(energies)
    std_e = np.std(energies)
    delta_e = np.max(energies) - np.min(energies)
    
    ratio = ""
    if prev_std is not None:
        # Nel velocity verlet, errore scala con dt^2. 
        # Se dimezziamo dt, la fluttuazione dovrebbe ridursi di un fattore 4 (circa 0.25).
        # Calcoliamo il ratio rispetto al precedente (che aveva dt doppio)
        val = std_e / prev_std
        ratio = f"{val:.4f} (atteso ~0.25)"
        
    print(f"{dt:10.6f} | {steps:6d} | {std_e:12.6f} | {delta_e:22.6f} | {ratio:>22}")
    prev_std = std_e
    plot_dts.append(dt * 1000) # fs
    plot_stds.append(std_e)

print("=" * 80)

# Generazione Plot log-log
plt.figure(figsize=(8, 6))
plt.loglog(plot_dts, plot_stds, 'o-', markersize=8, label="Misurazioni")

# Fit lineare per calcolare pendenza
coeffs = np.polyfit(np.log(plot_dts), np.log(plot_stds), 1)
slope = coeffs[0]

# Retta O(dt^2) passante per il primo punto
ref_stds = plot_stds[-1] * (np.array(plot_dts) / plot_dts[-1])**2
plt.loglog(plot_dts, ref_stds, '--', color='red', label="Scaling Teorico (dt$^2$)")

plt.xlabel("Timestep dt (fs)")
plt.ylabel(r"Varianza E Totale, $\sigma(E)$ (kJ/mol)")
plt.title(f"Scaling dell'errore (Pendenza log-log: {slope:.2f})")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig("scaling_plot.png", dpi=300, bbox_inches='tight')
print("Plot salvato in scaling_plot.png")
