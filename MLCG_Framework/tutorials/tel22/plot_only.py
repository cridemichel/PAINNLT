import numpy as np
import matplotlib.pyplot as plt

dts = [0.004, 0.006, 0.008, 0.01]
# Data obtained from the run with Envelope and Bias:
std_devs_toxvaerd = [1.89995e-03, 3.98508e-03, 6.84969e-03, 1.08299e-02]
# Data from Cosine C1 (no bias):
std_devs_bias_0 = [6.0967e-03, 1.3857e-02, 2.4842e-02, 3.9022e-02]
# Data from Toxvaerd C4 (no bias):
std_devs_toxvaerd_bias0 = [3.82771e-02, 8.60608e-02, 1.52302e-01, 2.35870e-01]

plt.figure(figsize=(10, 8))

log_dts = np.log10(dts)
log_stds_tox = np.log10(std_devs_toxvaerd)
log_stds_bias0 = np.log10(std_devs_bias_0)
log_stds_tox_bias0 = np.log10(std_devs_toxvaerd_bias0)

plt.scatter(log_dts, log_stds_tox, color='blue', s=100, zorder=5, label='ML + Toxvaerd Env (bias=1, trained with env)')
plt.scatter(log_dts, log_stds_bias0, color='green', marker='^', s=100, zorder=5, label='ML + Cosine C1 (bias=0, no env training)')
plt.scatter(log_dts, log_stds_tox_bias0, color='orange', marker='s', s=100, zorder=5, label='ML + Toxvaerd Env (bias=0, no env training)')

# Theoretical slope 2 line (passes through the first point)
x_line = np.linspace(min(log_dts)-0.1, max(log_dts)+0.1, 100)
y_line_ideal = 2.0 * (x_line - log_dts[0]) + log_stds_tox[0]
plt.plot(x_line, y_line_ideal, color='red', linestyle='--', linewidth=2, label=r'Ideal Verlet Scaling ($\mathcal{O}(dt^2)$)')

# Fit a line for Toxvaerd (bias=1, env=1)
slope_tox, intercept_tox = np.polyfit(log_dts, log_stds_tox, 1)
y_fit_tox = slope_tox * x_line + intercept_tox
plt.plot(x_line, y_fit_tox, color='blue', linestyle=':', alpha=0.7, label=f'Fit Toxvaerd (bias=1) (slope = {slope_tox:.2f})')

# Fit a line for Bias=0 Reference
slope_b0, intercept_b0 = np.polyfit(log_dts, log_stds_bias0, 1)
y_fit_b0 = slope_b0 * x_line + intercept_b0
plt.plot(x_line, y_fit_b0, color='green', linestyle=':', alpha=0.7, label=f'Fit Cosine C1 (bias=0) (slope = {slope_b0:.2f})')

# Fit a line for Toxvaerd (bias=0)
slope_t0, intercept_t0 = np.polyfit(log_dts, log_stds_tox_bias0, 1)
y_fit_t0 = slope_t0 * x_line + intercept_t0
plt.plot(x_line, y_fit_t0, color='orange', linestyle=':', alpha=0.7, label=f'Fit Toxvaerd (bias=0) (slope = {slope_t0:.2f})')

plt.xlabel('log10(dt) [ps]')
plt.ylabel('log10(Std(E_tot)) [kJ/mol]')
plt.title('Energy Conservation Scaling (Verlet Integrator)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('energy_conservation_toxvaerd_scaling_full.png', dpi=300)
print("\n[INFO] Plot saved to energy_conservation_toxvaerd_scaling_full.png")
