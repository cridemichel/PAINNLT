import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("energy.csv", delimiter=",", skiprows=1)
steps = data[:, 0]
e_tot = data[:, 1]
e_kin = data[:, 2]

plt.figure(figsize=(10,6))
plt.plot(steps, e_tot, label="E_tot", color='blue')
plt.plot(steps, e_kin, label="E_kin", color='red')
plt.xlabel("Step")
plt.ylabel("Energy (kJ/mol)")
plt.title("NVT Simulation Energy Profile (Step 0 to 310)")
plt.legend()
plt.grid()
plt.savefig("nvt_explosion.png", dpi=150)
