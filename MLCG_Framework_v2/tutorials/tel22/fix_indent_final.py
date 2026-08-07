filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/preprocessing/build_cg_dataset.py"
with open(filepath, "r") as f:
    content = f.read()

start_marker = '    if WCA_SIGMA == "auto":'
end_marker = '    print("\\n[INFO] Esecuzione allineamento Kabsch per mediare le geometrie dei corpi rigidi...")'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    old_block = content[start_idx:end_idx]
    
    new_block = """    if WCA_SIGMA == "auto":
        print("\\n[INFO] Calcolo parametri WCA geometrici con regolarizzazione gerarchica...")
        import scipy.optimize
        from scipy.ndimage import gaussian_filter1d
        
        # Trova tutti i tipi unici
        all_types = set()
        for (t1, t2) in all_pairwise_distances.keys():
            all_types.add(t1)
            all_types.add(t2)
        all_types = sorted(list(all_types))
        n_types = len(all_types)
        type_to_idx = {t: i for i, t in enumerate(all_types)}
        
        empirical_Q1 = {}
        empirical_min = {}
        pair_counts = {}
        
        for pair, dists in all_pairwise_distances.items():
            if len(dists) > 0:
                empirical_Q1[pair] = np.percentile(dists, 1.0)
                empirical_min[pair] = np.min(dists)
                pair_counts[pair] = len(dists)
                
        # Ottimizzazione globale dei raggi di base R_i
        def cost_func_R(R):
            loss = 0.0
            for (t1, t2), q1 in empirical_Q1.items():
                N = pair_counts[(t1, t2)]
                weight = 50.0 / (N + 50.0) # Bug 7: weight = N0 / (N_ij + N0)
                r_pred = R[type_to_idx[t1]] + R[type_to_idx[t2]]
                loss += weight * (r_pred - q1)**2
            return loss
            
        R_init = np.ones(n_types) * 0.15
        bounds_R = [(0.05, 0.5) for _ in range(n_types)]
        res_R = scipy.optimize.minimize(cost_func_R, R_init, bounds=bounds_R)
        R_opt = res_R.x
        
        print("  Raggi base R_i ottimizzati:")
        for t, r in zip(all_types, R_opt):
            print(f"    Tipo {t}: {r:.4f} nm")
            
        KB_T = 2.494 # kJ/mol at 300K
        wca_prior_dict = {}
        
        for (t1, t2), dists in all_pairwise_distances.items():
            r_c = R_opt[type_to_idx[t1]] + R_opt[type_to_idx[t2]]
            sig = r_c / (2.0**(1.0/6.0))
            
            # Bug 8 & 9: Calcolo analitico di epsilon
            # Vogliamo U_WCA(0.9 * r_c) = 10 * k_B * T
            r_guard = 0.9 * r_c
            
            # U(r) = 4 eps [ (sig/r)^12 - (sig/r)^6 ] + eps
            # 10 k_B T = eps * ( 4 * [ (sig/r_guard)^12 - (sig/r_guard)^6 ] + 1 )
            sr = sig / r_guard
            term = 4.0 * (sr**12 - sr**6) + 1.0
            eps = (10.0 * KB_T) / term
            
            # Bug 10: Use gaussian_filter1d for robust extraction
            if len(dists) > 0:
                hist, bin_edges = np.histogram(dists, bins=50)
                smoothed_hist = gaussian_filter1d(hist, sigma=1.0)
                valid_idx = np.where(smoothed_hist > 0)[0]
                if len(valid_idx) > 0:
                    robust_min = bin_edges[valid_idx[0]]
                else:
                    robust_min = r_c
            else:
                robust_min = r_c
                
            wca_prior_dict[f"{t1}_{t2}"] = {
                "type_i": int(t1), "type_j": int(t2),
                "sigma_nm": float(sig), "epsilon_kjmol": float(eps),
                "cutoff_nm": float(r_c), "r_guard_nm": float(r_guard),
                "n_samples": len(dists), "empirical_min": float(robust_min)
            }
            
        # Bug 11: Inject wca_prior_dict into cg_priors.json and DO NOT write wca_priors.json
        import os
        import json
        if os.path.exists("cg_priors.json"):
            with open("cg_priors.json", "r") as f:
                cg_priors = json.load(f)
        else:
            cg_priors = {}
            
        cg_priors["wca_pairs"] = wca_prior_dict
        
        with open("cg_priors.json", "w") as f:
            json.dump(cg_priors, f, indent=4)
        print(f"[INFO] Salvati parametri WCA in cg_priors.json sotto 'wca_pairs' ({len(wca_prior_dict)} coppie)")
"""
    content = content.replace(old_block, new_block + "\n")
    with open(filepath, "w") as f:
        f.write(content)
    print("Fixed indentation!")
else:
    print("Could not find the block to fix")
