import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os

dts = [0.004, 0.006, 0.008, 0.01]
total_time = 0.5 # 0.5 ps

std_devs_toxvaerd = []

for dt in dts:
    steps = int(total_time / dt)
    print(f"\n--- Running NVE for dt = {dt} ps, steps = {steps} ---")
    
    # Run the MD simulation with the current settings (use_bias=false, toxvaerd_alpha=0.1)
    cmd = [
        "../../../espresso/build/pypresso", "../../simulation/run_cg_md.py",
        "--model", "tel22_model.pt",
        "--config", "tel22_training_config.json",
        "--priors", "cg_priors.json",
        "--rb_info", "rigid_bodies_info.json",
        "--dataset", "tel22_dataset.bin",
        "--checkpoint", "equilibrated.npz",
        "--nve",
        "--dt", str(dt),
        "--steps", str(steps),
        "--device", "cpu"
    ]
    
    # Capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output to extract E_tot from energy.csv
    energies = []
    if os.path.exists("energy.csv"):
        with open("energy.csv", "r") as f:
            lines = f.readlines()[1:] # skip header
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    energies.append(float(parts[1]))
    
    if len(energies) > 0:
        std_e = np.std(energies)
        std_devs_toxvaerd.append(std_e)
        print(f"-> Std(E) = {std_e:.5e} kJ/mol")
    else:
        print("-> Error: could not parse energies!")
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)
        std_devs_toxvaerd.append(np.nan)

# The values we obtained yesterday for bias=0 (Option 1) as Reference
std_devs_bias_0 = [6.0967e-03, 1.3857e-02, 2.4842e-02, 3.9022e-02]

plt.figure(figsize=(9, 7))

log_dts = np.log10(dts)
log_stds_tox = np.log10(std_devs_toxvaerd)
log_stds_bias0 = np.log10(std_devs_bias_0)

plt.scatter(log_dts, log_stds_tox, color='blue', s=100, zorder=5, label='Current (Toxvaerd C4, bias=0)')
plt.scatter(log_dts, log_stds_bias0, color='green', marker='^', s=100, zorder=5, label='Reference (Cosine C1, bias=0)')

# Theoretical slope 2 line (passes through the first point)
x_line = np.linspace(min(log_dts)-0.1, max(log_dts)+0.1, 100)
y_line_ideal = 2.0 * (x_line - log_dts[0]) + log_stds_bias0[0]
plt.plot(x_line, y_line_ideal, color='red', linestyle='--', linewidth=2, label=r'Ideal Verlet Scaling ($\mathcal{O}(dt^2)$)')

# Fit a line for Toxvaerd
valid_tox = ~np.isnan(log_stds_tox)
if np.sum(valid_tox) > 1:
    slope_tox, intercept_tox = np.polyfit(log_dts[valid_tox], log_stds_tox[valid_tox], 1)
    y_fit_tox = slope_tox * x_line + intercept_tox
    plt.plot(x_line, y_fit_tox, color='blue', linestyle=':', alpha=0.7, label=f'Fit Toxvaerd C4 (slope = {slope_tox:.2f})')

# Fit a line for Bias=0 Reference
slope_b0, intercept_b0 = np.polyfit(log_dts, log_stds_bias0, 1)
y_fit_b0 = slope_b0 * x_line + intercept_b0
plt.plot(x_line, y_fit_b0, color='green', linestyle=':', alpha=0.7, label=f'Fit Cosine C1 (slope = {slope_b0:.2f})')

plt.xlabel('log10(dt) [ps]')
plt.ylabel('log10(Std(E_tot)) [kJ/mol]')
plt.title('Energy Conservation Scaling (Verlet Integrator)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('energy_conservation_toxvaerd_scaling.png', dpi=300)
print("\n[INFO] Plot saved to energy_conservation_toxvaerd_scaling.png")
