import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The integration timesteps to test (in ps)
dt_values = [0.0005, 0.001, 0.002, 0.005]
std_devs = []

total_time_ps = 1.0

print(f"Starting NVE Energy Conservation Verification on tel22 (Total Time: {total_time_ps} ps)...")

for dt in dt_values:
    steps = int(total_time_ps / dt)
    print(f"Running simulation with dt = {dt} ps (steps = {steps})...")
    
    # Run the MD script without --model (Classical mode) and with --nve
    cmd = [
        "../../../espresso/build/pypresso",
        "../../simulation/run_cg_md.py",
        "--config", "tel22_training_config.json",
        "--priors", "cg_priors.json",
        "--rb_info", "rigid_bodies_info.json",
        "--dataset", "tel22_dataset.bin",
        "--checkpoint", "equilibrated.npz",
        "--dt", str(dt),
        "--steps", str(steps),
        "--nve"
    ]
    
    # We don't want the output to flood the console
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Read the output energies ignoring the broken header and specifying names
    df = pd.read_csv("energy.csv", header=0, names=["Step", "E_tot", "E_kin", "E_kin_trans", "E_kin_rot"])
    
    # Calculate the standard deviation of the total energy
    # We skip the very first few steps just in case of initial equilibration artifacts
    std_e = df['E_tot'].iloc[5:].std()
    
    print(f"  -> std(E_tot) = {std_e:.5f} kJ/mol")
    std_devs.append(std_e)

print("Simulations complete. Generating plot...")

dt_values = np.array(dt_values)
std_devs = np.array(std_devs)

plt.figure(figsize=(8, 6))

# Plot the measured standard deviations
plt.plot(dt_values, std_devs, 'o-', label='Measured $\sigma(E_{tot})$', markersize=8)

# Fit a line in log-log space to find the scaling exponent (O(dt^x))
log_dt = np.log(dt_values)
log_std = np.log(std_devs)
slope, intercept = np.polyfit(log_dt, log_std, 1)

print(f"Fitted slope in log-log plot: {slope:.3f} (Expected: ~2.0)")

# Plot the theoretical O(dt^2) reference line starting from the first point
ref_std = std_devs[0] * (dt_values / dt_values[0])**2
plt.plot(dt_values, ref_std, 'r--', label='Theoretical $\mathcal{O}(dt^2)$ scaling')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time step $dt$ (ps)', fontsize=12)
plt.ylabel('Standard Deviation of Total Energy (kJ/mol)', fontsize=12)
plt.title(f'NVE Energy Conservation Verification (tel22)\nFitted scaling exponent: {slope:.2f}', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(fontsize=12)

# Save the plot
plot_filename = "energy_conservation_tel22_nve.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_filename}")

# Create markdown for the user
with open("nve_results.md", "w") as f:
    f.write(f"# NVE Energy Conservation Verification (tel22)\n\n")
    f.write(f"The energy conservation test yielded a scaling exponent of **{slope:.2f}** (expected ~2.0).\n\n")
    f.write(f"![NVE Scaling Plot]({os.path.abspath(plot_filename)})\n")
