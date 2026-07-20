import numpy as np
import matplotlib.pyplot as plt
import os

dts = [0.004, 0.006, 0.008, 0.01]
std_devs_envelope = [5.99947e-03, 9.92789e-03, 1.81105e-02, 2.74955e-02]

# The values we obtained yesterday for bias=0 (Option 1)
std_devs_bias_0 = [6.0967e-03, 1.3857e-02, 2.4842e-02, 3.9022e-02]

plt.figure(figsize=(9, 7))

log_dts = np.log10(dts)
log_stds_env = np.log10(std_devs_envelope)
log_stds_bias0 = np.log10(std_devs_bias_0)

plt.scatter(log_dts, log_stds_env, color='blue', s=100, zorder=5, label='Simulation (Envelope Active)')
plt.scatter(log_dts, log_stds_bias0, color='green', marker='^', s=100, zorder=5, label='Simulation (Bias=0, No Envelope)')

# Theoretical slope 2 line (passes through the first envelope point)
x_line = np.linspace(min(log_dts)-0.1, max(log_dts)+0.1, 100)
# y - y1 = m*(x - x1) -> y = 2*(x - x1) + y1
y_line_ideal = 2.0 * (x_line - log_dts[0]) + log_stds_env[0]
plt.plot(x_line, y_line_ideal, color='red', linestyle='--', linewidth=2, label='Ideal Verlet Scaling ($\mathcal{O}(dt^2)$)')

# Fit a line for Envelope
slope_env, intercept_env = np.polyfit(log_dts, log_stds_env, 1)
y_fit_env = slope_env * x_line + intercept_env
plt.plot(x_line, y_fit_env, color='blue', linestyle=':', alpha=0.7, label=f'Fit Envelope (slope = {slope_env:.2f})')

# Fit a line for Bias=0
slope_b0, intercept_b0 = np.polyfit(log_dts, log_stds_bias0, 1)
y_fit_b0 = slope_b0 * x_line + intercept_b0
plt.plot(x_line, y_fit_b0, color='green', linestyle=':', alpha=0.7, label=f'Fit Bias=0 (slope = {slope_b0:.2f})')

plt.xlabel('log10(dt) [ps]')
plt.ylabel('log10(Std(E_tot)) [kJ/mol]')
plt.title('Energy Conservation Scaling (Verlet Integrator)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('energy_conservation_envelope_scaling_v2.png', dpi=300)
print("\n[INFO] Plot saved to energy_conservation_envelope_scaling_v2.png")
