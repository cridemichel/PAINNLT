import re

filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/simulation/run_cg_md.py"
with open(filepath, "r") as f:
    content = f.read()

# Replace the specific wca loading block in run_cg_md
old_wca = """print("[INFO] Adding non-bonded priors (WCA)...")
# TODO: Inserire qui l'implementazione WCA corretta una volta stabilizzata.
import math
wca = priors.get("wca", {})
has_wca = wca.get("sigma", 0.0) > 0 or len(wca.get("overrides", {})) > 0

import json
if has_wca:
    with open("wca_priors.json") as f:
        wca_dict = json.load(f)
        
    for pair_key, wca_info in wca_dict.items():
        t_i, t_j = wca_info["type_i"], wca_info["type_j"]
        eps = wca_info["epsilon_kjmol"]
        sig = wca_info["sigma_nm"]
        rcut = wca_info["cutoff_nm"]
        system.non_bonded_inter[t_i, t_j].lennard_jones.set_params(
            epsilon=eps, sigma=sig, cutoff=rcut, shift="auto"
        )"""

new_wca = """print("[INFO] Adding non-bonded priors (WCA)...")
import json
with open("wca_priors.json") as f:
    wca_dict = json.load(f)

for pair_key, wca_info in wca_dict.items():
    t_i, t_j = wca_info["type_i"], wca_info["type_j"]
    eps = wca_info["epsilon_kjmol"]
    sig = wca_info["sigma_nm"]
    rcut = wca_info["cutoff_nm"]
    system.non_bonded_inter[t_i, t_j].lennard_jones.set_params(
        epsilon=eps, sigma=sig, cutoff=rcut, shift="auto"
    )"""

if old_wca in content:
    content = content.replace(old_wca, new_wca)
    with open(filepath, "w") as f:
        f.write(content)
    print("Fixed run_cg_md.py successfully!")
else:
    print("Could not find block in run_cg_md.py")
