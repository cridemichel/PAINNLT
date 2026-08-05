import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os

dt = 0.004
total_time = 10.0 # 10 ps
steps = int(total_time / dt)

print(f"--- Running NVE for drift analysis: dt = {dt} ps, time = {total_time} ps, steps = {steps} ---")

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

result = subprocess.run(cmd, capture_output=True, text=True)

times = []
e_tots = []
e_kins = []
e_pots = []

for line in result.stdout.split('\n'):
    if "[INFO] Step " in line and "E_tot:" in line:
        try:
            step_str = line.split("|")[0].split("Step")[1].strip()
            step_idx = int(step_str.split("/")[0])
            t = step_idx * dt
            
            parts = line.split("|")
            e_kin = float([p for p in parts if "E_kin" in p][0].split(":")[1].strip())
            e_pot = float([p for p in parts if "E_pot" in p][0].split(":")[1].strip())
            e_tot = float([p for p in parts if "E_tot" in p][0].split(":")[1].strip())
            
            times.append(t)
            e_kins.append(e_kin)
            e_pots.append(e_pot)
            e_tots.append(e_tot)
        except Exception as e:
            pass

if not times:
    print("Error: could not parse energies!")
    print(result.stderr)
    exit(1)

times = np.array(times)
e_tots = np.array(e_tots)
e_kins = np.array(e_kins)
e_pots = np.array(e_pots)

# Calculate drift using linear regression
slope, intercept = np.polyfit(times, e_tots, 1)
drift_rate = slope # kJ/(mol*ps)

print(f"Mean E_tot: {np.mean(e_tots):.2f} kJ/mol")
print(f"Energy Drift Rate: {drift_rate:.5e} kJ/(mol*ps)")
print(f"Total Drift over {total_time} ps: {drift_rate * total_time:.5e} kJ/mol")

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(times, e_tots - e_tots[0], label='E_tot (centered)', color='black', linewidth=1.5)
plt.plot(times, slope * times + intercept - e_tots[0], color='red', linestyle='--', label=f'Drift Fit: {drift_rate:.2e} kJ/(mol*ps)')
plt.ylabel('ΔE [kJ/mol]')
plt.title(f'NVE Energy Drift (dt = {dt} ps)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(times, e_kins, label='E_kin', color='orange', alpha=0.7)
plt.plot(times, e_pots, label='E_pot', color='blue', alpha=0.7)
plt.xlabel('Time [ps]')
plt.ylabel('Energy [kJ/mol]')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

plt.tight_layout()
out_png = '/Users/demichel/.gemini/antigravity/brain/88f2c4a4-4efd-4e31-b158-879a8540a940/energy_drift_v2.png'
plt.savefig(out_png, dpi=300)
print(f"Plot saved to {out_png}")
