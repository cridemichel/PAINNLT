import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = 1.98720e-3  # kcal/mol/K
# Buscaglia, Gray, Chaires 2013, Supp. Table 2: 2JPZ, 25 mM KCl. dH di FOLDING (negativo), Tm in C
P = {"dH": [-7.8, -36.4, -39.9], "sdH": [1.8, 4.7, 4.8], "Tm": [37.5, 43.7, 54.3], "sTm": [22.5, 2.9, 1.7]}

def pops(T, dH, Tm):
    T = np.asarray(T, float)[:, None]
    dHu = -np.asarray(dH)[None, :]                       # unfolding enthalpy > 0
    TmK = np.asarray(Tm)[None, :] + 273.15
    lnK = -(dHu / R) * (1 / T - 1 / TmK)                 # K_i = [next]/[prev], unfolding
    cum = np.concatenate([np.zeros((T.shape[0], 1)), np.cumsum(lnK, 1)], 1)
    w = np.exp(cum - cum.max(1, keepdims=True))
    return w / w.sum(1, keepdims=True)                   # F, I1, I2, U

T = np.linspace(273.15, 410, 2000)
p = pops(T, P["dH"], P["Tm"])
def cross(y, lev=0.5):
    i = np.where(np.diff(np.sign(y - lev)))[0][0]; return T[i] + (lev - y[i]) * (T[i+1]-T[i]) / (y[i+1]-y[i])
res = {"T_F50_C": cross(p[:, 0]) - 273.15, "T_U50_C": cross(p[:, 3]) - 273.15,
       "dG_tot_20C_kcal": float(sum(h * (1 - 293.15 / (t + 273.15)) for h, t in zip(P["dH"], P["Tm"]))),
       "dH_tot_unf_kcal": -sum(P["dH"]), "dH_tot_unf_kJ": -sum(P["dH"]) * 4.184}
for TT in (293.15, 300, 310.15, 330, 350, 400):
    q = pops([TT], P["dH"], P["Tm"])[0]; res[f"pop_{TT:.0f}K"] = [round(x, 4) for x in q]
# incertezza: Monte Carlo sui parametri (gaussiane indipendenti; Tm1 molto incerta)
rng = np.random.default_rng(1); mc = []
for _ in range(4000):
    dH = rng.normal(P["dH"], P["sdH"]); Tm = rng.normal(P["Tm"], P["sTm"])
    if np.any(dH >= 0): continue
    mc.append(pops(T, dH, Tm))
mc = np.array(mc)
lo, hi = np.percentile(mc, [16, 84], axis=0)
u50 = [cross(m[:, 3]) - 273.15 for m in mc]; f50 = [cross(m[:, 0]) - 273.15 for m in mc if m[0,0] > 0.5 and m[-1,0] < 0.5]
res["T_U50_C_MC_16_50_84"] = [round(x, 1) for x in np.percentile(u50, [16, 50, 84])]
res["T_F50_C_MC_16_50_84"] = [round(x, 1) for x in np.percentile(f50, [16, 50, 84])]
print(json.dumps(res, indent=1))
json.dump(res, open("tm_2jpz.json", "w"), indent=1)

cols = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834"]; lab = ["F (ibrido-2)", "I1", "I2 (tripla elica)", "U"]
fig, ax = plt.subplots(figsize=(7.2, 4))
Tc = T - 273.15
for k in range(4):
    ax.fill_between(Tc, lo[:, k], hi[:, k], color=cols[k], alpha=0.15, lw=0)
    ax.plot(Tc, p[:, k], color=cols[k], lw=2, label=lab[k])
ax.axvline(126.85, color="#52514e", ls="--", lw=1); ax.text(125, 0.85, "corse AA\n400 K", ha="right", fontsize=8, color="#52514e")
ax.axvline(res["T_U50_C"], color=cols[3], ls=":", lw=1)
ax.set(xlabel="T (°C)", ylabel="frazione", xlim=(0, 135), ylim=(0, 1.02),
       title="wtTel26 (2JPZ), 25 mM KCl: popolazioni a 4 stati (Buscaglia et al. 2013)")
ax.title.set_fontsize(10); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.3)
ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.72, 0.45))
fig.tight_layout(); fig.savefig("/mnt/user-data/outputs/popolazioni_2jpz.png", dpi=170)
