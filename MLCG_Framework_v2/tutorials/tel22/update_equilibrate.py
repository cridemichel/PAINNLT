import re

filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/simulation/equilibrate.py"
with open(filepath, "r") as f:
    content = f.read()

# Trova tutti i blocchi relativi al WCA pre-esistente
start_idx = content.find('print("[INFO] Adding priors...")')
end_idx = content.find('# Bonds (Harmonic, FENE, Morse)')

if start_idx != -1 and end_idx != -1:
    old_block = content[start_idx:end_idx]
    new_block = """print("[INFO] Adding priors...")
# WCA from cg_priors.json (Unified truth)
import json
import os

cg_priors_path = "cg_priors.json"
if os.path.exists(cg_priors_path):
    print(f"[INFO] Loading unified WCA priors from {cg_priors_path}")
    with open(cg_priors_path, "r") as f:
        cg_priors = json.load(f)
        
    wca_dict = cg_priors.get("wca_pairs", {})
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
    print(f"[WARNING] {cg_priors_path} not found! No WCA will be applied.")

"""
    content = content.replace(old_block, new_block)
    with open(filepath, "w") as f:
        f.write(content)
    print("Patched equilibrate.py logic!")
else:
    print("Block not found in equilibrate.py")
