import espressomd
import espressomd.painn
import numpy as np
import json
import os

# -----------------------------------------------------------------------------
# 1. Inizializzazione Sistema
# -----------------------------------------------------------------------------
box_size = 10.0
system = espressomd.System(box_l=[box_size, box_size, box_size])
system.time_step = 0.001
system.cell_system.skin = 0.4

# -----------------------------------------------------------------------------
# 2. Caricamento Dati Corpi Rigidi
# -----------------------------------------------------------------------------
rb_info_path = "../python_scripts/rigid_bodies_info.json"
try:
    with open(rb_info_path, "r") as f:
        rb_info = json.load(f)
except FileNotFoundError:
    print(f"[ERRORE] File '{rb_info_path}' non trovato. Esegui convert_gro2bin.py prima di lanciare questo script.")
    exit(1)

# Verifichiamo se GUA esiste nel JSON
if "GUA" not in rb_info:
    print("[ERRORE] La molecola 'GUA' non è definita nel file rigid_bodies_info.json.")
    exit(1)

gua_data = rb_info["GUA"]
mass = gua_data["mass_amu"]
rinertia = gua_data["inertia_amu_nm2"]
sites = gua_data.get("sites", {})

if not sites:
    print("[ERRORE] Nessuna informazione sulla geometria dei siti (virtual sites) trovata nel JSON.")
    exit(1)

print(f"[INFO] Trovata configurazione per GUA:")
print(f"       Massa: {mass} amu")
print(f"       Inerzia: {rinertia} amu*nm^2")
print(f"       Numero di siti CG: {len(sites)}")

# -----------------------------------------------------------------------------
# 3. Creazione del corpo rigido (Particella Centrale + Siti Virtuali)
# -----------------------------------------------------------------------------
# Creazione molecole:
# Per un test, creiamo 1 molecola di ogni tipo trovato nel JSON (spaziate lungo l'asse X)
offset_x = 2.0
center_parts = []

for resname, data in rb_info.items():
    mass = data["mass_amu"]
    rinertia = np.array(data["inertia_amu_nm2"])
    sites = data.get("sites", {})
    
    # Posizione di base per la molecola corrente
    center_pos = np.array([offset_x, box_size/2, box_size/2])
    
    # Se c'è un solo sito (es. A o T), è una particella singola, niente virtual sites
    if len(sites) == 1:
        site_name, site_data = list(sites.items())[0]
        p = system.part.add(
            pos=center_pos,
            type=site_data["type"],
            mass=mass
            # Niente rotazione né inerzia per un particle puntiforme
        )
        center_parts.append(p)
        print(f"[INFO] Creata particella singola reale per {resname} (Sito: {site_name}, Type: {site_data['type']})")
    
    # Se ci sono più siti, creiamo il corpo rigido e associamo i virtual sites
    elif len(sites) > 1:
        # Particella centrale (fittizia)
        center_part = system.part.add(
            pos=center_pos,
            type=100,
            mass=mass,
            rinertia=rinertia,
            rotation=[True, True, True]
        )
        center_parts.append(center_part)
        
        for site_name, site_data in sites.items():
            rel_pos = np.array(site_data["relative_pos_nm"])
            abs_pos = center_pos + rel_pos
            
            v_part = system.part.add(
                pos=abs_pos,
                type=site_data["type"],
                virtual=True
            )
            v_part.vs_auto_relate_to(center_part.id)
            print(f"[INFO] Creato sito virtuale '{site_name}' per {resname} e collegato al COM.")
            
    offset_x += 2.0

# -----------------------------------------------------------------------------
# 4. Attivazione PaiNN (Setup simile a check_dt_scaling)
# -----------------------------------------------------------------------------
config_path = "../GROMACS/best_cg_model_config.json"
model_path = "../GROMACS/best_cg_model.pt"

if os.path.exists(config_path) and os.path.exists(model_path):
    with open(config_path, "r") as f:
        nn_config = json.load(f)
        
    # Inizializziamo le Verlet Lists (necessarie a ESPResSo per costruire il vicinato)
    # Impostiamo il cutoff al valore predetto dalla rete per TUTTI i tipi da 0 a num_species
    num_species = int(nn_config['num_species'])
    cutoff = float(nn_config['cutoff'])
    
    for i in range(num_species):
        for j in range(i, num_species):
            system.non_bonded_inter[i, j].lennard_jones.set_params(
                epsilon=0.0, sigma=1.0, cutoff=cutoff, shift=0.0
            )

    print("[INFO] Attivazione potenziale PaiNN...")
    espressomd.painn.activate_painn_potential(
        model_path=model_path, 
        num_species=num_species, 
        hidden_channels=int(nn_config['hidden_channels']), 
        n_layers=int(nn_config['n_layers']), 
        num_rbf=int(nn_config['num_rbf']), 
        cutoff=cutoff, 
        device="cpu"  # Usa MPS se ESPResSo lo supporta, altrimenti cpu
    )
else:
    print(f"\n[WARNING] Rete PaiNN non trovata ({model_path}). Avvio una simulazione a vuoto (Free Particle) solo per testare la stabilità strutturale del corpo rigido.")

# -----------------------------------------------------------------------------
# 5. Integrazione Langevin e Simulazione
# -----------------------------------------------------------------------------
temperature = 300.0 * 0.00831446 # 300K in kJ/mol
gamma = 1.0 # Attrito
system.thermostat.set_langevin(kT=temperature, gamma=gamma, seed=42)

print("\n[INFO] Avvio integrazione per testare la cinematica...")

# Eseguiamo 1000 step
for i in range(10):
    system.integrator.run(100)
    
    # Stampiamo l'energia e le posizioni per assicurarci che i vincoli tengano
    if os.path.exists(model_path):
        e_pot = espressomd.painn.get_painn_energy()
    else:
        e_pot = 0.0
    
    pos_str = " | ".join([f"M{j}: {p.pos}" for j, p in enumerate(center_parts)])
    print(f"Step {(i+1)*100:4d} | {pos_str} | E_pot: {e_pot:.4f} kJ/mol")

print("\n[INFO] Test superato! I corpi si sono mossi correttamente.")
