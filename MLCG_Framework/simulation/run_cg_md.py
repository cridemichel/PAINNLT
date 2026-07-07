import espressomd
import espressomd.painn
import espressomd.interactions
import json
import argparse

parser = argparse.ArgumentParser(description="Run MLCG Framework MD with PyTorch and ESPResSo")
parser.add_argument("-m", "--model", type=str, default="../training/best_cg_model.pt", help="Path al modello PyTorch C++ (.pt)")
parser.add_argument("-p", "--priors", type=str, default="../preprocessing/cg_priors.json", help="Path al file generato dalla Boltzmann Inversion")
args = parser.parse_args()

system = espressomd.System(box_l=[10.0, 10.0, 10.0])
system.time_step = 0.002 # 2 fs
system.cell_system.skin = 0.4

# ... qui andrà la logica per creare le particelle della topologia CG ...
# system.part.add(pos=[...], type=0)

# Caricamento Priors
with open(args.priors, "r") as f:
    priors_data = json.load(f)

# 1. Caricamento WCA
if "wca" in priors_data:
    sigma = priors_data["wca"]["sigma"]
    eps = priors_data["wca"]["epsilon"]
    if sigma > 0 and eps > 0:
        # Applica WCA tra tutti i tipi definiti nel sistema CG
        # (Da configurare dinamicamente in base ai tipi)
        pass

# 2. Caricamento Harmonic Priors
if "bonds" in priors_data:
    for b in priors_data["bonds"]:
        mol_i = b["mol_i"]
        mol_j = b["mol_j"]
        k = b["k"]
        r0 = b["r0"]
        
        # Creazione del legame in ESPResSo
        hb = espressomd.interactions.HarmonicBond(k=k, r_0=r0)
        system.bonded_inter.add(hb)
        
        # Applicazione topologica: lega specificamente mol_i con mol_j
        # system.part.by_id(mol_i).add_bond((hb, mol_j))
        print(f"[INFO] Aggiunto Harmonic Prior tra {mol_i} e {mol_j} (k={k:.2f}, r0={r0:.4f})")

# 3. Attivazione Modello PyTorch C++
print(f"[INFO] Attivazione PaiNN ML Potential: {args.model}")
espressomd.painn.activate_painn_potential(
    system=system,
    model_path=args.model,
    cutoff=0.6,
    device="cpu" # cpu o mps o cuda
)

print("[INFO] Simulazione pronta (da completare con la topologia effettiva).")
