import matplotlib.pyplot as plt
import numpy as np

dt_values = np.array([0.004, 0.002, 0.001, 0.0005, 0.00025, 0.000125]) * 1000 # fs
std_values = np.array([2.769062e-06, 6.403560e-07, 1.533036e-07, 3.745524e-08, 9.255850e-09, 2.300501e-09])

plt.figure(figsize=(8, 6))
plt.loglog(dt_values, std_values, 'o-', markersize=8, label="Dati Misurati (PaiNN in Float64)")

# Plot della retta di riferimento quadratica (pendenza = 2) passante per il primo punto
ref_dE = std_values[0] * (dt_values / dt_values[0])**2
plt.loglog(dt_values, ref_dE, '--', color='red', label="Scaling Teorico O(dt$^2$)")

plt.xlabel("Timestep dt (fs)")
plt.ylabel(r"Fluttuazione E Totale, $\sigma(E)$ (kJ/mol)")
plt.title(f"Perfect Quadratic Scaling (ESPResSo + PaiNN)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("/Users/demichel/.gemini/antigravity/brain/d54d813a-ae54-4ca6-9922-6179a054f737/energy_conservation_perfect_scaling.png", dpi=300)
