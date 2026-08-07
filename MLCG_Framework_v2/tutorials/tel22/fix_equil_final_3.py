import re

filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/simulation/equilibrate.py"
with open(filepath, "r") as f:
    content = f.read()

old_wca = """print("[INFO] Adding priors...")
# WCA
import math
wca = priors.get("wca", {})
has_wca = wca.get("sigma", 0.0) > 0 or len(wca.get("overrides", {})) > 0
if wca.get("epsilon", 0.0) > 0 and has_wca:
    for i in range(nn_config["num_species"]):
        sigma_i = wca.get("overrides", {}).get(str(i), {}).get("sigma", wca["sigma"])
        eps_i = wca.get("overrides", {}).get(str(i), {}).get("epsilon", wca["epsilon"])
        for j in range(i, nn_config["num_species"]):
            sigma_j = wca.get("overrides", {}).get(str(j), {}).get("sigma", wca["sigma"])
            eps_j = wca.get("overrides", {}).get(str(j), {}).get("epsilon", wca["epsilon"])
            
            sigma_mix = (sigma_i + sigma_j) / 2.0
            eps_mix = math.sqrt(eps_i * eps_j)
            
            system.non_bonded_inter[i, j].lennard_jones.set_params(
                epsilon=eps_mix, sigma=sigma_mix,
                cutoff=sigma_mix * (2.0**(1/6)), shift="auto"
            )"""

new_wca = """print("[INFO] Adding priors...")
# WCA from wca_priors.json
import json
import os

wca_priors_path = "wca_priors.json"
if os.path.exists(wca_priors_path):
    print(f"[INFO] Loading statistical WCA priors from {wca_priors_path}")
    with open(wca_priors_path, "r") as f:
        wca_dict = json.load(f)
        
    for pair_key, wca_info in wca_dict.items():
        type_i = wca_info["type_i"]
        type_j = wca_info["type_j"]
        sig = wca_info["sigma_nm"]
        eps = wca_info["epsilon_kjmol"]
        cut = wca_info["cutoff_nm"]
        
        system.non_bonded_inter[type_i, type_j].lennard_jones.set_params(
            epsilon=eps, sigma=sig,
            cutoff=cut, shift="auto"
        )
else:
    print(f"[WARNING] {wca_priors_path} not found! No WCA will be applied.")"""

if old_wca in content:
    content = content.replace(old_wca, new_wca)
    with open(filepath, "w") as f:
        f.write(content)
    print("Fixed equilibrate.py successfully!")
else:
    print("Could not find block in equilibrate.py")
