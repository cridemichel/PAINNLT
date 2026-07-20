import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

dts = [0.002, 0.004, 0.006, 0.008, 0.01]
stds = []

for dt in dts:
    steps = int(0.5 / dt) # 0.5 ps
    print(f"Running NVE with dt={dt} for {steps} steps...")
    
    cmd = [
        "../../../espresso/build/pypresso", "../../simulation/run_cg_md.py",
        "--model", "tel22_model_fixed.pt",
        "--config", "tel22_training_config.json",
        "--priors", "cg_priors.json",
        "--rb_info", "rigid_bodies_info.json",
        "--dataset", "tel22_dataset.bin",
        "--checkpoint", "equilibrated.npz",
        "--dt", str(dt),
        "--steps", str(steps),
        "--device", "cpu",
        "--nve"
    ]
    
    # Run the simulation
    subprocess.run(cmd, check=True)
    
    # Analyze energy
    df = pd.read_csv("energy.csv", header=0, names=["Step", "E_tot", "E_kin", "E_kin_trans", "E_kin_rot"])
    std = df["E_tot"].std()
    stds.append(std)
    
    # Rename to keep track
    os.rename("energy.csv", f"energy_{dt}.csv")
    print(f"dt={dt} -> std={std}")

dts = np.array(dts)
stds = np.array(stds)

plt.figure(figsize=(8,6))
plt.loglog(dts, stds, marker='o', label='ML NVE')

ref_y = stds[0] * (dts / dts[0])**2
plt.loglog(dts, ref_y, linestyle='--', color='gray', label='O(dt^2)')

plt.xlabel('Timestep dt (ps)')
plt.ylabel('Std of Total Energy (kJ/mol)')
plt.title('Energy Conservation Scaling (0.5 ps) (tel22 + PaiNN)')
plt.legend()
plt.grid(True, which="both", ls="-")

plot_path = '/Users/demichel/.gemini/antigravity/brain/d54d813a-ae54-4ca6-9922-6179a054f737/energy_conservation_tel22_nve_ml_0.5ps_high_dt.png'
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
