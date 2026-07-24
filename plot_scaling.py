import matplotlib.pyplot as plt
import numpy as np
import os

dt = np.array([0.012, 0.008, 0.004, 0.002, 0.001])
std_E = np.array([0.023041, 0.010327, 0.002572, 0.000654, 0.000170])

# Plot on log-log scale
plt.figure(figsize=(8, 6))

# Theoretical O(dt^2) line
# We align it with the first point for visual comparison
theoretical_std = std_E[0] * (dt / dt[0])**2

plt.plot(dt, theoretical_std, 'k--', label='Theoretical $\\mathcal{O}(dt^2)$')
plt.plot(dt, std_E, 'ro-', markersize=8, label='Simulation Results (PaiNN + Verlet)')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time Step $dt$ (ps)', fontsize=14)
plt.ylabel('Std($E_{tot}$) (kJ/mol)', fontsize=14)
plt.title('Energy Conservation Scaling (NVE Ensemble)\nMLCG_Framework_v2', fontsize=16)

# Set xticks to exact dt values for clarity
plt.xticks(dt, [f"{x:.3f}" for x in dt], fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()

out_path = '/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT/scaling_plot_1ps.png'
plt.savefig(out_path, dpi=300)
print(f"Plot saved to {out_path}")
