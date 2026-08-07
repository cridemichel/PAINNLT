import re

with open("../../simulation/run_cg_md.py", "r") as f:
    code = f.read()

# Add scipy import
code = code.replace("import numpy as np", "import numpy as np\nfrom scipy.spatial.distance import pdist, squareform")

diagnostic_func = """
def log_diagnostics(step):
    pos = []
    types = []
    mol_ids = []
    forces = []
    pids = []
    for p in system.part:
        if p.is_virtual:
            pos.append(p.pos)
            types.append(p.type)
            mol_ids.append(p.mol_id)
            forces.append(p.f)
            pids.append(p.id)
            
    pos = np.array(pos)
    types = np.array(types)
    mol_ids = np.array(mol_ids)
    forces = np.array(forces)
    pids = np.array(pids)
    
    dist_matrix = squareform(pdist(pos))
    mask = mol_ids[:, None] == mol_ids[None, :]
    dist_matrix[mask] = np.inf
    np.fill_diagonal(dist_matrix, np.inf)
    
    num_species = nn_config["num_species"]
    min_dists = {}
    
    global_min_dist = np.inf
    global_min_pair = None
    global_min_pids = None
    
    for i in range(num_species):
        for j in range(i, num_species):
            mask_types = (types[:, None] == i) & (types[None, :] == j)
            # Make symmetric
            mask_types = mask_types | ((types[:, None] == j) & (types[None, :] == i))
            
            valid_dists = dist_matrix[mask_types]
            if len(valid_dists) > 0:
                m_d = np.min(valid_dists)
                min_dists[(i, j)] = m_d
                if m_d < global_min_dist:
                    global_min_dist = m_d
                    global_min_pair = (i, j)
                    # Find the exact particles
                    # Get indices where mask_types and dist_matrix == m_d
                    idx1, idx2 = np.where((dist_matrix == m_d) & mask_types)
                    if len(idx1) > 0:
                        global_min_pids = (pids[idx1[0]], pids[idx2[0]])
            else:
                min_dists[(i, j)] = np.inf
                
    f_max = np.max(np.linalg.norm(forces, axis=1))
    
    return global_min_dist, global_min_pair, global_min_pids, f_max

"""

# Insert diagnostic_func before measure_energies
code = code.replace("def measure_energies():", diagnostic_func + "\ndef measure_energies():")

# Modify logging to include diagnostic
replace_logging = """        if step % 10 == 0:
            e_tot, e_kin, e_kin_trans, e_kin_rot, e_class, e_ml = measure_energies()
            g_dist, g_pair, g_pids, f_max = log_diagnostics(step)
            energy_file.write(f"{step},{e_tot},{e_kin},{e_kin_trans},{e_kin_rot},{e_class},{e_ml},{g_dist},{g_pair},{g_pids},{f_max}\\n")
            energy_file.flush()
            print(f"\\r[INFO] MD Progress: {step}/{args.steps} steps. E_tot={e_tot:.3f}, E_kin={e_kin:.3f}, min_dist={g_dist:.3f} nm (types {g_pair}), F_max={f_max:.1f}", end="", flush=True)"""

code = code.replace("""        if step % 10 == 0:
            e_tot, e_kin, e_kin_trans, e_kin_rot, e_class, e_ml = measure_energies()
            energy_file.write(f"{step},{e_tot},{e_kin},{e_kin_trans},{e_kin_rot},{e_class},{e_ml}\\n")
            energy_file.flush()
            print(f"\\r[INFO] MD Progress: {step}/{args.steps} steps. E_tot={e_tot:.3f}", end="", flush=True)""", replace_logging)

code = code.replace("""        energy_file.write("Step,E_tot,E_kin,E_kin_trans,E_kin_rot,E_class,E_ml\\n")""", """        energy_file.write("Step,E_tot,E_kin,E_kin_trans,E_kin_rot,E_class,E_ml,min_dist,min_pair,min_pids,f_max\\n")""")

with open("../../simulation/run_cg_md.py", "w") as f:
    f.write(code)

