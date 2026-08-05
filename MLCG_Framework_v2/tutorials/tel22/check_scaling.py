import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os

dts = [0.002, 0.004, 0.006, 0.008, 0.010, 0.012]
total_time = 2.0 # 2.0 ps

std_devs = []

for dt in dts:
    steps = int(total_time / dt)
    print(f"\n--- Running NVE for dt = {dt} ps, steps = {steps} ---")
    
    # Run the MD simulation
    cmd = [
        "../../espresso/build/pypresso", "../../simulation/run_cg_md.py",
        "--model", "tel22_model.pt",
        "--config", "tel22_training_config.json",
        "--priors", "cg_priors.json",
        "--rb_info", "rigid_bodies_info.json",
        "--dataset", "tel22_dataset.bin",
        "--checkpoint", "equilibrated.npz",
        "--nve",
        "--allow_nonconservative_tables",
        "--dt", str(dt),
        "--steps", str(steps),
        "--device", "cpu"
    ]
    
    # Capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output to extract E_tot
    energies = []
    for line in result.stdout.split('\n'):
        if "[INFO] Step " in line and "E_tot:" in line:
            parts = line.split("|")
            for p in parts:
                if "E_tot:" in p:
                    val = float(p.split("E_tot:")[1].strip())
                    energies.append(val)
                    
    if len(energies) > 0:
        std_e = np.std(energies)
        std_devs.append(std_e)
        print(f"-> Std(E) = {std_e:.5e} kJ/mol")
    else:
        print("-> Error: could not parse energies!")
        print("STDERR:", result.stderr)
        std_devs.append(np.nan)

# Plotting the scaling
plt.figure(figsize=(8, 6))

log_dts = np.log10(dts)
log_stds = np.log10(std_devs)

plt.scatter(log_dts, log_stds, color='blue', s=100, zorder=5, label='Simulation (Analytical Priors)')

# Fit a line to get the slope
valid = ~np.isnan(log_stds)
if np.sum(valid) > 1:
    slope, intercept = np.polyfit(log_dts[valid], log_stds[valid], 1)
    
    x_fit = np.linspace(min(log_dts[valid])-0.1, max(log_dts[valid])+0.1, 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f'Fit: slope = {slope:.2f} (Expected: ~2.0)')

plt.xlabel('log10(dt) [ps]')
plt.ylabel('log10(Std(E_tot)) [kJ/mol]')
plt.title('Energy Conservation Scaling (Verlet Integrator + Toxvaerd Cutoff)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('energy_conservation_scaling_2ps.png', dpi=300)
print("\n[INFO] Plot saved to energy_conservation_scaling_2ps.png")
