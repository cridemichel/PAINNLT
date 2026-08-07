import json

with open("wca_priors.json") as f:
    wca_dict = json.load(f)
    
new_wca_block = """print("[INFO] Adding priors...")
# WCA
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
    )

# Bonds (Harmonic, FENE, Morse)"""

filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/simulation/equilibrate.py"
with open(filepath, "r") as f:
    content = f.read()

import re
content = re.sub(r'print\("\[INFO\] Adding priors\.\.\."\).*?# Bonds \(Harmonic, FENE, Morse\)', new_wca_block, content, flags=re.DOTALL)

with open(filepath, "w") as f:
    f.write(content)
