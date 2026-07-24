import subprocess
import numpy as np
import re
import os

dts = [0.012, 0.008, 0.004, 0.002, 0.001]
# We want constant physical time. 1 ps is enough.
# steps = 1 / dt
steps = [83, 125, 250, 500, 1000]

print(f"{'dt':>8} | {'Steps':>6} | {'Std(E_tot)':>12} | {'Ratio':>8}")
print("-" * 45)

prev_std = None

# Change dir to the tel22 tutorial so we can read the json files
os.chdir("/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/tutorials/tel22")

for dt, step in zip(dts, steps):
    pypresso = "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/espresso/build/pypresso"
    cmd = [
        pypresso, "/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework_v2/simulation/run_cg_md.py",
        "--config", "tel22_training_config.json",
        "--model", "tel22_model.pt",
        "--priors", "cg_priors.json",
        "--rb_info", "rigid_bodies_info.json",
        "--dataset", "tel22_dataset.bin",
        "--checkpoint", "equilibrated.npz",
        "--dt", str(dt),
        "--steps", str(step),
        "--nve",
        "--device", "mps",
        "--no_log"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    e_tot_vals = []
    for line in result.stdout.split('\n'):
        if "E_tot:" in line:
            # Extract E_tot value
            match = re.search(r'E_tot:\s*([-\d\.]+)', line)
            if match:
                e_tot_vals.append(float(match.group(1)))
                
    if len(e_tot_vals) < 2:
        print(f"{dt:>8} | {step:>6} | {'FAILED':>12} | {'-':>8}")
        # Print stderr to understand why it failed
        print(result.stderr)
        continue
        
    # Ignore the first 10% of steps to allow initial slight adjustments
    skip = max(1, len(e_tot_vals) // 10)
    e_tot_vals = np.array(e_tot_vals[skip:])
    std = np.std(e_tot_vals)
    
    ratio = std / prev_std if prev_std is not None else 0.0
    ratio_str = f"{ratio:.3f}" if prev_std is not None else "-"
    
    print(f"{dt:>8} | {step:>6} | {std:>12.6f} | {ratio_str:>8}")
    prev_std = std
