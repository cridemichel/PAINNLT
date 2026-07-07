import numpy as np
import matplotlib.pyplot as plt

dt_list = [0.001, 0.002, 0.004, 0.006, 0.008, 0.010]

plt.figure(figsize=(10, 6))

for dt in dt_list:
    data = np.load(f"nve_dt_{dt:.4f}.npz")
    times = data['times']
    e_tots = data['e_tots']
    
    # Energia Totale Assoluta
    plt.plot(times, e_tots, label=f"dt = {dt*1000:.1f} fs", linewidth=1.5)

plt.xlabel("Tempo fisico (ps)", fontsize=12)
plt.ylabel(r"Energia Totale Assoluta (kJ/mol)", fontsize=12)
plt.title("Energia Totale nel tempo (Sistema 9-atomi denso WCA)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.5, linestyle='--')
plt.tight_layout()
plt.savefig("/Users/demichel/.gemini/antigravity/brain/d54d813a-ae54-4ca6-9922-6179a054f737/absolute_total_energy_dense.png", dpi=300)
