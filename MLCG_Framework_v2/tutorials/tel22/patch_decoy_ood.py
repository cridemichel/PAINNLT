import json

filepath = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/preprocessing/build_cg_dataset.py"
with open(filepath, "r") as f:
    content = f.read()

old_decoy_logic = """    r_ood_max = min(0.95 * r_c, r_emp_min - 0.01)
    r_ood_min = 0.75 * r_c
    if r_ood_max <= r_ood_min:
        r_ood_max = r_ood_min + 0.02"""

new_decoy_logic = """    r_ood_max = min(0.95 * r_c, r_emp_min - 0.01)
    r_ood_min = 0.70 * r_c
    if r_ood_max <= r_ood_min:
        # No reliable deep-OOD interval exists for this pair
        print(f"    [Decoy] Skipping pair {t1}-{t2} (rc={r_c:.3f}, r_min={r_emp_min:.3f}) - no valid OOD interval.")
        continue"""

import re
if old_decoy_logic in content:
    content = content.replace(old_decoy_logic, new_decoy_logic)
    with open(filepath, "w") as f:
        f.write(content)
    print("Patched Decoy OOD logic!")
else:
    print("Could not find decoy logic to replace, check script...")
