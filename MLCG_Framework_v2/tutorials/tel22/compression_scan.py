import sys
# Set fallback so we can run PaiNN without full GPU requirements
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import espressomd
import espressomd.painn
import numpy as np
import json

def run_scan():
    with open("cg_priors.json") as f:
        cg_priors = json.load(f)
        wca_priors = cg_priors.get("wca_pairs", {})
    with open("tel22_training_config.json") as f:
        nn_config = json.load(f)
        
    system = espressomd.System(box_l=[10.0, 10.0, 10.0])
    system.time_step = 0.001
    system.cell_system.skin = 0.4
    
    ml_cutoff = nn_config["cutoff"]
    for i in range(nn_config["num_species"]):
        for j in range(i, nn_config["num_species"]):
            system.non_bonded_inter[i, j].soft_sphere.set_params(a=0.0, n=1, cutoff=ml_cutoff, offset=0.0)

    system.part.add(id=0, pos=[5.0, 5.0, 5.0], type=0)
    system.part.add(id=1, pos=[5.0, 5.0, 5.5], type=0)

    espressomd.painn.activate_painn_potential(
        model_path="tel22_model.pt",
        num_species=nn_config["num_species"],
        hidden_channels=nn_config["hidden_channels"],
        n_layers=nn_config["n_layers"],
        num_rbf=nn_config["num_rbf"],
        cutoff=ml_cutoff,
        max_num_neighbors=100
    )

    print("r/rc   | r (nm) | F_ML     | F_WCA    | F_tot    | rho")
    print("-" * 65)

    for pair_key, wca_info in wca_priors.items():
        t1 = wca_info["type_i"]
        t2 = wca_info["type_j"]
        sig = wca_info["sigma_nm"]
        eps = wca_info["epsilon_kjmol"]
        rc = wca_info["cutoff_nm"]
        
        system.part.by_id(0).type = t1
        system.part.by_id(1).type = t2
        
        print(f"\\n--- Pair {t1}-{t2} (rc={rc:.3f} nm, eps={eps:.1f}) ---")
        
        for r_ratio in [1.2, 1.1, 1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
            r = rc * r_ratio
            system.part.by_id(1).pos = [5.0, 5.0, 5.0 + r]
            system.integrator.run(0) # Evaluate forces
            
            # WCA analytical
            if r < rc:
                sr6 = (sig / r)**6
                F_wca_mag = 24 * eps * (2*sr6**2 - sr6) / r
            else:
                F_wca_mag = 0.0
                
            # ML force
            # vec is from 0 to 1 -> +z
            # F_rad = force on 0 in -z direction
            f_ml = system.part.by_id(0).f[2]
            F_rad_ml = -f_ml
            
            F_tot = F_wca_mag + F_rad_ml
            rho = abs(F_rad_ml) / abs(F_wca_mag) if F_wca_mag > 1e-6 else 999.999
            
            print(f"{r_ratio:.2f}   | {r:.3f}  | {F_rad_ml:8.1f} | {F_wca_mag:8.1f} | {F_tot:8.1f} | {rho:.3f}")

if __name__ == "__main__":
    run_scan()
