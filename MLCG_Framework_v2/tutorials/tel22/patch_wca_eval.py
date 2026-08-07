import json

filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/preprocessing/build_cg_dataset.py"
with open(filepath, "r") as f:
    content = f.read()

# Trova il blocco di codice incriminato nel pass 3
old_wca_subtraction = """        has_wca = WCA_SIGMA_VAL > 0 or (isinstance(WCA_OVERRIDES, dict) and len(WCA_OVERRIDES) > 0)
        has_epsilon = WCA_EPSILON > 0 or (isinstance(WCA_OVERRIDES, dict) and any(o.get("epsilon", 0) > 0 for o in WCA_OVERRIDES.values()))
        
        if has_wca and has_epsilon:
            flat_pos = []
            flat_mol = []
            flat_type = []
            for m_idx, sites in enumerate(frame_sites):
                for s_type, s_pos in sites:
                    flat_pos.append(s_pos)
                    flat_mol.append(m_idx)
                    flat_type.append(s_type)
            
            if len(flat_pos) > 0:
                flat_pos = np.array(flat_pos)
                flat_mol = np.array(flat_mol)
                flat_type = np.array(flat_type)
                
                # Retrieve parameters with overrides
                sigmas = np.array([WCA_OVERRIDES.get(str(int(t)), {}).get("sigma", WCA_SIGMA_VAL) for t in flat_type])
                epsilons = np.array([WCA_OVERRIDES.get(str(int(t)), {}).get("epsilon", WCA_EPSILON) for t in flat_type])
                
                diff = flat_pos[:, np.newaxis, :] - flat_pos[np.newaxis, :, :]
                diff -= box_dim * np.round(diff / box_dim)
                dist_sq = np.sum(diff**2, axis=-1)
                
                # Lorentz-Berthelot mixing
                sigma_ij = (sigmas[:, np.newaxis] + sigmas[np.newaxis, :]) / 2.0
                eps_ij = np.sqrt(epsilons[:, np.newaxis] * epsilons[np.newaxis, :])
                r_cut_sq = (sigma_ij * (2.0**(1.0/6.0)))**2"""

new_wca_subtraction = """        # 3.1 Sottrazione WCA usando i parametri di wca_priors.json (36 coppie)
        if WCA_SIGMA == "auto" and 'wca_prior_dict' in locals():
            flat_pos = []
            flat_mol = []
            flat_type = []
            for m_idx, sites in enumerate(frame_sites):
                for s_type, s_pos in sites:
                    flat_pos.append(s_pos)
                    flat_mol.append(m_idx)
                    flat_type.append(s_type)
            
            if len(flat_pos) > 0:
                flat_pos = np.array(flat_pos)
                flat_mol = np.array(flat_mol)
                flat_type = np.array(flat_type)
                
                diff = flat_pos[:, np.newaxis, :] - flat_pos[np.newaxis, :, :]
                diff -= box_dim * np.round(diff / box_dim)
                dist_sq = np.sum(diff**2, axis=-1)
                
                sigma_ij = np.zeros_like(dist_sq)
                eps_ij = np.zeros_like(dist_sq)
                r_cut_sq = np.zeros_like(dist_sq)
                
                for i in range(len(flat_type)):
                    for j in range(len(flat_type)):
                        t_min = min(int(flat_type[i]), int(flat_type[j]))
                        t_max = max(int(flat_type[i]), int(flat_type[j]))
                        pair_key = f"{t_min}_{t_max}"
                        
                        if pair_key in wca_prior_dict:
                            w = wca_prior_dict[pair_key]
                            sigma_ij[i, j] = w["sigma_nm"]
                            eps_ij[i, j] = w["epsilon_kjmol"]
                            r_cut_sq[i, j] = w["cutoff_nm"]**2"""

import re
if old_wca_subtraction in content:
    content = content.replace(old_wca_subtraction, new_wca_subtraction)
    with open(filepath, "w") as f:
        f.write(content)
    print("Patched WCA force subtraction!")
else:
    print("Could not find exact block to replace, check script...")
