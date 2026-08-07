filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/preprocessing/build_cg_dataset.py"
with open(filepath, "r") as f:
    content = f.read()

# We want to inject it after `sites_data_history.append(frame_sites)`
injection_point = "    sites_data_history.append(frame_sites)"

if injection_point not in content:
    print("Injection point not found!")
    exit(1)

new_logic = """    sites_data_history.append(frame_sites)
    
    # --- Estrazione distanze WCA non-bonded ---
    if WCA_SIGMA == "auto":
        flat_pos = []
        flat_types = []
        flat_mols = []
        for m_idx, mol_sites in enumerate(frame_sites):
            for s_type, s_pos in mol_sites:
                flat_pos.append(s_pos)
                flat_types.append(int(s_type))
                flat_mols.append(m_idx)
                
        flat_pos = np.array(flat_pos)
        flat_types = np.array(flat_types)
        flat_mols = np.array(flat_mols)
        
        # Calculate MIC distance matrix
        dist_matrix = minimum_image_distance_matrix(flat_pos, box_dim)
        
        # Mask out intra-molecular distances (same rigid body)
        same_mol_mask = flat_mols[:, None] == flat_mols[None, :]
        dist_matrix[same_mol_mask] = np.inf
        
        # We only want unique pairs (upper triangle)
        i_idx, j_idx = np.triu_indices(len(flat_pos), k=1)
        valid_dist = dist_matrix[i_idx, j_idx]
        
        types_i = flat_types[i_idx]
        types_j = flat_types[j_idx]
        
        t1 = np.minimum(types_i, types_j)
        t2 = np.maximum(types_i, types_j)
        
        # We only care about distances < 1.5 nm for WCA parametrization to save memory
        close_mask = valid_dist < 1.5
        
        if np.any(close_mask):
            valid_dist = valid_dist[close_mask]
            t1 = t1[close_mask]
            t2 = t2[close_mask]
            
            pair_ids = t1 * 10000 + t2
            unique_pairs = np.unique(pair_ids)
            
            for pid in unique_pairs:
                dists = valid_dist[pair_ids == pid]
                pair = (str(int(pid // 10000)), str(int(pid % 10000))) # Must use strings for JSON matching in WCA fit!
                if pair not in all_pairwise_distances:
                    all_pairwise_distances[pair] = []
                # Subsample if too many, to avoid OOM
                if len(dists) > 1000:
                    dists = np.random.choice(dists, 1000, replace=False)
                all_pairwise_distances[pair].extend(dists)
"""
content = content.replace(injection_point, new_logic)

with open(filepath, "w") as f:
    f.write(content)
print("Injected non-bonded pairwise distance calculation.")
