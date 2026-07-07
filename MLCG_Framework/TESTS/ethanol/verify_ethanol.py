import espressomd
import espressomd.painn
import numpy as np
import struct
import json

def read_first_frame_md17_bin(filepath):
    with open(filepath, "rb") as f:
        num_frames = struct.unpack("i", f.read(4))[0]
        num_atoms = struct.unpack("i", f.read(4))[0]
        
        energy = struct.unpack("d", f.read(8))[0]
        atomic_numbers = np.frombuffer(f.read(num_atoms * 4), dtype=np.int32)
        coords = np.frombuffer(f.read(num_atoms * 3 * 8), dtype=np.float64).reshape((num_atoms, 3))
        
        return atomic_numbers, coords

atomic_numbers, coords = read_first_frame_md17_bin("ethanol_val.bin")
num_atoms = len(atomic_numbers)
box_size = 20.0 
center_of_mass = np.mean(coords, axis=0)
shift = (box_size / 2.0) - center_of_mass
coords_shifted = coords + shift

# Impostiamo velocità iniziali identiche per tutti i run
np.random.seed(42)
initial_velocities = np.random.randn(num_atoms, 3) * 0.01

dt_values = [0.004, 0.002, 0.001, 0.0005, 0.00025, 0.000125]
physical_time = 0.2

print(f"{'dt':>10} | {'Steps':>6} | {'Std(E_tot)':>12} | {'Max(E_tot)-Min(E_tot)':>22} | {'Ratio (Std / prev Std)':>22}")
print("-" * 80)

prev_std = None

system = espressomd.System(box_l=[box_size, box_size, box_size])
system.cell_system.skin = 0.4

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


for i in range(10):
    for j in range(i, 10):
        system.non_bonded_inter[i, j].lennard_jones.set_params(epsilon=0.0, sigma=1.0, 
                                                               cutoff=float(nn_config['cutoff']), shift="auto")
        
espressomd.painn.activate_painn_potential(
    model_path="best_painn_etanolo.pt", 
    num_species=int(nn_config['num_species']), hidden_channels=int(nn_config['hidden_channels']), 
    n_layers=int(nn_config['n_layers']), num_rbf=int(nn_config['num_rbf']), cutoff=float(nn_config['cutoff']), 
    device="cpu"
)

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
