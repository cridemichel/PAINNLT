import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The integration timesteps to test (in ps)
dt_values = [0.0005, 0.001, 0.002, 0.005]

# Compensated kT: target is 300K (2.49 kJ/mol). 
# Due to noise heating, we compensate by requesting ~168K (1.39 kJ/mol)
kT_comp = 1.39 
total_time_ps = 5.0 # Let's run for 5 ps to see the stability

print(f"Starting NVT Temperature Compensation Test (kT_input={kT_comp} kJ/mol, T_tot={total_time_ps} ps)...")

plt.figure(figsize=(10, 6))

for dt in dt_values:
    steps = int(total_time_ps / dt)
    print(f"Running NVT simulation with dt = {dt} ps (steps = {steps})...")
    
    cmd = [
        "../../../espresso/build/pypresso",
        "../../simulation/run_cg_md.py",
        "--model", "../../models/painn_cg_model.pt",
        "--config", "tel22_training_config.json",
        "--priors", "cg_priors.json",
        "--rb_info", "rigid_bodies_info.json",
        "--dataset", "tel22_dataset.bin",
        "--checkpoint", "equilibrated.npz",
        "--dt", str(dt),
        "--steps", str(steps),
        "--kT", str(kT_comp) # Thermostat is ON (no --nve flag)
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    df = pd.read_csv("energy.csv", header=0, names=["Step", "E_tot", "E_kin", "E_kin_trans", "E_kin_rot"])
    
    # Time array in ps
    time_ps = df["Step"] * dt
    
    mean_ekin = df['E_kin'].iloc[10:].mean()
    print(f"  -> Mean E_kin: {mean_ekin:.1f} kJ/mol (Target ~ 1200 kJ/mol for 300K)")
    
    plt.plot(time_ps, df["E_kin"], label=f'dt={dt} ps ($\mu$={mean_ekin:.0f})')

plt.axhline(1200, color='r', linestyle='--', label='Target 300K (1200 kJ/mol)')
plt.xlabel('Time (ps)', fontsize=12)
plt.ylabel('Kinetic Energy (kJ/mol)', fontsize=12)
plt.title(f'NVT Temperature Compensation Test (Input kT = {kT_comp})', fontsize=14)
plt.grid(True, alpha=0.5)
plt.legend(fontsize=11)

plot_filename = "temperature_compensation_nvt.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_filename}")
