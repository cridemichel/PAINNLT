import numpy as np
import matplotlib.pyplot as plt

dts = [0.004, 0.006, 0.008, 0.01]
log_dts = np.log10(dts)

# "Toxvaerd C4, bias=0" (user requested data)
std_devs_tox = [4.33150e-03, 9.77123e-03, 1.73719e-02, 2.69123e-02]
log_stds_tox = np.log10(std_devs_tox)

# "Cosine C1, bias=0"
std_devs_cos = [6.0967e-03, 1.3857e-02, 2.4842e-02, 3.9022e-02]
log_stds_cos = np.log10(std_devs_cos)

# "Toxvaerd Envelope, bias=1" (best data)
std_devs_env = [1.89995e-03, 3.98508e-03, 6.84969e-03, 1.08299e-02]
log_stds_env = np.log10(std_devs_env)

plt.figure(figsize=(10, 8))

# Plot Toxvaerd Envelope
plt.scatter(log_dts, log_stds_env, color='purple', marker='s', s=100, zorder=5, label='Toxvaerd Cutoff + Envelope (bias=1)')
# Plot Toxvaerd C4, bias=0
plt.scatter(log_dts, log_stds_tox, color='blue', s=100, zorder=5, label='Toxvaerd Cutoff (bias=0)')
# Plot Cosine C1, bias=0
plt.scatter(log_dts, log_stds_cos, color='green', marker='^', s=100, zorder=5, label='Reference (Cosine Cutoff, bias=0)')

# Ideal Verlet Scaling
x_ideal = np.array([-2.5, -1.9])
y_ideal = 2.0 * x_ideal + 2.08
plt.plot(x_ideal, y_ideal, 'r--', linewidth=2, label='Ideal Verlet Scaling ($\mathcal{O}(dt^2)$)')

# Fits
slope_env, intercept_env = np.polyfit(log_dts, log_stds_env, 1)
plt.plot(log_dts, slope_env * log_dts + intercept_env, 'purple', linestyle=':', alpha=0.7, label=f'Fit Tox. Env (slope = {slope_env:.2f})')

slope_tox, intercept_tox = np.polyfit(log_dts, log_stds_tox, 1)
plt.plot(log_dts, slope_tox * log_dts + intercept_tox, 'b:', alpha=0.7, label=f'Fit Toxvaerd C4 (slope = {slope_tox:.2f})')

slope_cos, intercept_cos = np.polyfit(log_dts, log_stds_cos, 1)
plt.plot(log_dts, slope_cos * log_dts + intercept_cos, 'g:', alpha=0.7, label=f'Fit Cosine C1 (slope = {slope_cos:.2f})')

plt.xlabel('log10(dt) [ps]')
plt.ylabel('log10(Std(E_tot)) [kJ/mol]')
plt.title('Energy Conservation Scaling (Verlet Integrator)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('/Users/demichel/.gemini/antigravity/brain/d54d813a-ae54-4ca6-9922-6179a054f737/energy_conservation_toxvaerd_scaling.png', dpi=150, bbox_inches='tight')
