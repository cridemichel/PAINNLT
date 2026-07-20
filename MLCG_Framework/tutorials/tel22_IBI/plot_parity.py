import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

csv_file = "parity_forces.csv"

if not os.path.exists(csv_file):
    print(f"Errore: File {csv_file} non trovato. L'eseguibile C++ non lo ha generato correttamente.")
    exit(1)

print(f"Caricamento dati da {csv_file}...")
df = pd.read_csv(csv_file)

# I nomi delle colonne nel CSV sono: F_target_x, F_target_y, F_target_z, F_pred_x, F_pred_y, F_pred_z
f_target_x = df['F_target_x'].values
f_target_y = df['F_target_y'].values
f_target_z = df['F_target_z'].values

f_pred_x = df['F_pred_x'].values
f_pred_y = df['F_pred_y'].values
f_pred_z = df['F_pred_z'].values

# Raggruppiamo tutto per fare le statistiche globali su tutte le componenti (3N punti)
f_target_all = np.concatenate([f_target_x, f_target_y, f_target_z])
f_pred_all = np.concatenate([f_pred_x, f_pred_y, f_pred_z])

# Calcolo MAE
mae = np.mean(np.abs(f_pred_all - f_target_all))

# Calcolo RMSE
rmse = np.sqrt(np.mean((f_pred_all - f_target_all)**2))

# Calcolo R^2 (Coefficiente di Determinazione)
ss_res = np.sum((f_target_all - f_pred_all)**2)
ss_tot = np.sum((f_target_all - np.mean(f_target_all))**2)
r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

# Calcolo R (Correlazione di Pearson)
cov = np.cov(f_target_all, f_pred_all)[0, 1]
std_target = np.std(f_target_all)
std_pred = np.std(f_pred_all)
pearson_r = cov / (std_target * std_pred) if (std_target * std_pred) != 0 else 0

print(f"--- RISULTATI FORZE SUL VALIDATION SET ---")
print(f"Punti analizzati (Componenti 3N) : {len(f_target_all)}")
print(f"MAE                              : {mae:.3f} kJ/mol/nm")
print(f"RMSE                             : {rmse:.3f} kJ/mol/nm")
print(f"R^2 (Determinazione)             : {r2:.4f}")
print(f"R (Correlazione Pearson)         : {pearson_r:.4f}")
print(f"------------------------------------------")

plt.figure(figsize=(9, 8))

# Usiamo hexbin perché ci saranno decine di migliaia di punti e lo scatter esploderebbe
hb = plt.hexbin(f_target_all, f_pred_all, gridsize=100, cmap='inferno', mincnt=1, bins='log')

# Linea ideale
min_val = min(np.min(f_target_all), np.min(f_pred_all))
max_val = max(np.max(f_target_all), np.max(f_pred_all))
plt.plot([min_val, max_val], [min_val, max_val], 'w--', lw=2, label='Ideal (Pred = Target)')

cb = plt.colorbar(hb, label='log10(N) points')
cb.ax.tick_params(labelsize=12)

plt.xlabel("F Target (IBI Residual) [kJ/mol/nm]", fontsize=14)
plt.ylabel("F Predicted (PaiNN) [kJ/mol/nm]", fontsize=14)
plt.title(f"Parity Plot: Forze Residue PaiNN\nMAE = {mae:.2f} | RMSE = {rmse:.2f} | R = {pearson_r:.3f}", fontsize=15)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig("parity_plot.png", dpi=300)
print("\\nGrafico salvato con successo come 'parity_plot.png'!")
