#!/usr/bin/env python3
"""Figure del report AA contro CG-ML da report_data.npz (script 53)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AA_C, CG_C, HALF_C, PRI_C = "#2a78d6", "#eb6834", "#8a8984", "#3a9a4a"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6, "savefig.dpi": 220, "figure.dpi": 110,
})

d = np.load("report_data_v2.npz")
meta = json.loads(str(d["meta_json"]))
r = d["r_nm"]
CH = meta["channels"]
OV = meta["overlaps"]
AA_NSDAY, CG_NSDAY = 137.0, 54.2
T = int(meta["aa_frames"])


def pr_panel(kind, fname, ylab, xlim=None):
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6))
    for ax, ch in zip(axs.ravel(), CH):
        ax.plot(r, d[f"{kind}_AA meta' 1_{ch}"], color=HALF_C, lw=0.9, ls="--", label="AA, metà 1 / metà 2")
        ax.plot(r, d[f"{kind}_AA meta' 2_{ch}"], color=HALF_C, lw=0.9, ls=":")
        ax.plot(r, d[f"{kind}_AA_{ch}"], color=AA_C, label="AA (136 ns)")
        ax.plot(r, d[f"{kind}_lp2_{ch}"], color=PRI_C, lw=1.2, label="soli prior lp2 (2 × 20 ns)")
        ax.plot(r, d[f"{kind}_CG-ML_{ch}"], color=CG_C, label="CG-ML (2 × 20 ns)")
        key = "intra" if kind == "P" else "inter"
        o_cg, o_h, o_p = OV[ch]["CG-ML"][key], OV[ch]["AA meta' 2"][key], OV[ch]["lp2"][key]
        ax.set_title(f"{ch}\nCG-ML {o_cg:.3f} · soli prior {o_p:.3f} · tetto {o_h:.3f}",
                     loc="left", color=INK, fontsize=8.3)
        ax.set_xlabel("r (nm)")
        ax.set_ylabel(ylab)
        if xlim:
            ax.set_xlim(*xlim)
    axs[0, 1].legend(loc="center right")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


pr_panel("P", "fig_Pr_intra.png", "P(r) (nm$^{-1}$)", (0.2, 2.0))
pr_panel("g", "fig_gr_inter.png", "g(r)", (0.3, 2.0))


# ── P(r) di tutti i siti: soli prior contro lp2 + PaiNN, con lo scarto dall'AA ──
ch0 = CH[0]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
pA, pP, pC = d[f"P_AA_{ch0}"], d[f"P_lp2_{ch0}"], d[f"P_CG-ML_{ch0}"]
a1.plot(r, pA, color=AA_C, lw=1.8, label="AA (136 ns)")
a1.plot(r, pP, color=PRI_C, lw=1.3, label="soli prior lp2")
a1.plot(r, pC, color=CG_C, lw=1.3, label="CG-ML (lp2 + PaiNN)")
a1.set_ylabel("P(r) (nm$^{-1}$)"); a1.set_title("tutti i siti: P(r)", loc="left"); a1.legend(loc="upper right")
dh = d[f"P_AA meta' 2_{ch0}"] - d[f"P_AA meta' 1_{ch0}"]
a2.fill_between(r, -np.abs(dh), np.abs(dh), color=HALF_C, alpha=0.25, lw=0, label="metà 2 − metà 1 AA")
a2.plot(r, pP - pA, color=PRI_C, lw=1.2)
a2.plot(r, pC - pA, color=CG_C, lw=1.2)
a2.axhline(0, color=INK2, lw=0.6)
a2.set_ylabel("ΔP (nm$^{-1}$)"); a2.set_xlabel("r (nm)"); a2.set_title("scarto dall'AA, P(r) − P$_{AA}$(r)", loc="left"); a2.legend(loc="lower right")
a2.set_xlim(0.2, 2.0)
fig.tight_layout()
fig.savefig("fig_Pr_all.png")
plt.close(fig)


# ── Rg ──
A = d["cv_AA_Rg"]
C = d["cv_CG-ML_Rg"]
tA = d["t_AA_ps"] / 1000.0
fig = plt.figure(figsize=(7.2, 5.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.35])
ax = fig.add_subplot(gs[:, 0])
e = np.linspace(0.76, 0.84, 81)
c = 0.5 * (e[1:] + e[:-1])
for k in range(A.shape[1]):
    h, _ = np.histogram(A[:, k], e, density=True)
    ax.plot(c, h, color=AA_C, lw=0.5, alpha=0.35, label="AA, singole copie" if k == 0 else None)
h, _ = np.histogram(A.ravel(), e, density=True)
ax.plot(c, h, color=AA_C, lw=2, label="AA, tutte le copie")
h, _ = np.histogram(C, e, density=True)
ax.plot(c, h, color=CG_C, lw=2, label="CG-ML")
h, _ = np.histogram(d["cv_lp2_Rg"], e, density=True)
ax.plot(c, h, color=PRI_C, lw=1.4, label="soli prior lp2")
ax.set(xlabel="R$_g$ (nm)", ylabel="densità di probabilità", title="Distribuzione di R$_g$")
ax.title.set_ha("left"); ax.title.set_position((0, 1))
ax.legend(loc="upper left", fontsize=7.5)
ax.text(0.98, 0.70, f"media AA {A.mean():.3f} ± {A.std():.3f}\nmedia CG {C.mean():.3f} ± {C.std():.3f}",
        transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)

ax = fig.add_subplot(gs[0, 1])
w = 50  # 1 ns di media mobile
ker = np.ones(w) / w
for k in range(A.shape[1]):
    ax.plot(tA[w - 1:], np.convolve(A[:, k], ker, mode="valid"), lw=0.8, color=AA_C, alpha=0.25 + 0.06 * k)
ax.set(xlabel="t (ns)", ylabel="R$_g$ (nm)", ylim=(0.775, 0.82))
ax.set_title("AA: 10 copie, media mobile 1 ns", loc="left")
ax = fig.add_subplot(gs[1, 1])
# replica r0: tratti 0, 2, 4, 6 (s01..s04), 251 frame x 10 copie ciascuno, frame ogni 20 ps
nf = d["cv_CG-ML_Rg_trace0"].shape[0]
parts = C.reshape(-1, nf, 10)
Cr0 = np.concatenate([parts[i] for i in (0, 2, 4, 6)], axis=0)
tc = np.arange(Cr0.shape[0]) * 0.020
for k in range(10):
    ax.plot(tc[w - 1:], np.convolve(Cr0[:, k], ker, mode="valid"), lw=0.8, color=CG_C, alpha=0.25 + 0.06 * k)
ax.set(xlabel="t (ns)", ylabel="R$_g$ (nm)", ylim=(0.775, 0.82))
ax.set_title("CG-ML: replica r0, 10 copie, media mobile 1 ns", loc="left")
fig.tight_layout()
fig.savefig("fig_Rg.png")
plt.close(fig)


# ── altre coordinate collettive ──
def ovl(a, b, lo, hi, n=80):
    e = np.linspace(lo, hi, n + 1)
    ha, _ = np.histogram(a, e, density=True)
    hb, _ = np.histogram(b, e, density=True)
    return float(np.minimum(ha, hb).sum() * (e[1] - e[0]))


cv_stats = {}
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
labels = {"rmsd_core": "rmsd nucleo (nm)", "rmsd_loops": "rmsd loop (nm)", "Q": "Q (contatti nativi)"}
for v in ["Rg", "rmsd_core", "rmsd_loops", "Q"]:
    Av, Cv = d[f"cv_AA_{v}"], d[f"cv_CG-ML_{v}"]
    lo, hi = min(Av.min(), Cv.min()), max(Av.max(), Cv.max())
    cv_stats[v] = {"aa_mean": float(Av.mean()), "aa_std": float(Av.std()), "cg_mean": float(Cv.mean()),
                   "cg_std": float(Cv.std()), "ov": ovl(Av.ravel(), Cv.ravel(), lo, hi),
                   "ceil": ovl(Av[:T // 2].ravel(), Av[T // 2:].ravel(), lo, hi),
                   "aa_copy_min": float(Av.mean(0).min()), "aa_copy_max": float(Av.mean(0).max()),
                   "pri_mean": float(d[f"cv_lp2_{v}"].mean()), "pri_std": float(d[f"cv_lp2_{v}"].std()),
                   "pri_ov": ovl(Av.ravel(), d[f"cv_lp2_{v}"].ravel(), min(lo, d[f"cv_lp2_{v}"].min()), max(hi, d[f"cv_lp2_{v}"].max()))}
for ax, v in zip(axs, ["rmsd_core", "rmsd_loops", "Q"]):
    Av, Cv = d[f"cv_AA_{v}"], d[f"cv_CG-ML_{v}"]
    lo, hi = np.percentile(np.concatenate([Av.ravel(), Cv, d[f"cv_lp2_{v}"].ravel()]), [0.2, 99.8])
    e = np.linspace(lo, hi, 61)
    c = 0.5 * (e[1:] + e[:-1])
    ax.plot(c, np.histogram(Av[:T // 2], e, density=True)[0], color=HALF_C, lw=0.9, ls="--")
    ax.plot(c, np.histogram(Av[T // 2:], e, density=True)[0], color=HALF_C, lw=0.9, ls=":")
    ax.plot(c, np.histogram(Av, e, density=True)[0], color=AA_C, label="AA")
    ax.plot(c, np.histogram(Cv, e, density=True)[0], color=CG_C, label="CG-ML")
    ax.plot(c, np.histogram(d[f"cv_lp2_{v}"], e, density=True)[0], color=PRI_C, lw=1.2, label="soli prior lp2")
    s = cv_stats[v]
    ax.set_title(f"{labels[v]}\nsovr. {s['ov']:.2f} (tetto {s['ceil']:.2f})", loc="left", fontsize=8.5)
    ax.set_xlabel(labels[v].split(" (")[0])
    ax.set_yticklabels([])
axs[0].set_ylabel("densità")
axs[0].legend(loc="upper right")
fig.tight_layout()
fig.savefig("fig_cv.png")
plt.close(fig)

# FES (rmsd_core, Q)
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
xe = np.linspace(0.02, 0.13, 56)
ye = np.linspace(0.86, 0.945, 46)
kT = 2.494
for ax, lab, col in zip(axs, ["AA", "lp2", "CG-ML"], [AA_C, PRI_C, CG_C]):
    H, _, _ = np.histogram2d(d[f"cv_{lab}_rmsd_core"].ravel(), d[f"cv_{lab}_Q"].ravel(), [xe, ye], density=True)
    with np.errstate(divide="ignore"):
        F = -kT * np.log(H.T)
    F -= np.nanmin(F[np.isfinite(F)])
    F[~np.isfinite(F)] = np.nan
    im = ax.contourf(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), F, levels=np.arange(0, 12.1, 1.0),
                     cmap={"AA": "Blues_r", "lp2": "Greens_r", "CG-ML": "Oranges_r"}[lab], extend="max")
    ax.set_title("soli prior lp2" if lab == "lp2" else lab, loc="left", fontsize=8.5)
    ax.set_xlabel("rmsd nucleo (nm)")
    ax.grid(False)
axs[0].set_ylabel("Q")
cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.02)
cb.set_label("F (kJ/mol)")
fig.savefig("fig_fes.png", bbox_inches="tight")
plt.close(fig)


# ── MSD ──
msd = {e["label"]: e for e in meta["msd"]}
aa, cg = msd["AA"], msd["prod"]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
for col, (key, Dk, unit, name) in enumerate([("msd_trans_nm2", "D_trans", "nm$^2$", "traslazione (centro di massa)"),
                                              ("msd_rot_rad2", "D_rot", "rad$^2$", "rotazione (orientazione del nucleo)")]):
    ax = axs[0, col]
    for e, c, lab in [(aa, AA_C, "AA"), (cg, CG_C, "CG-ML")]:
        t = np.asarray(e["lag_ps"]) / 1000.0
        y = np.asarray(e[key])
        m = t <= e["span_ps"] / 4000.0
        t, y = t[m], y[m]
        ax.loglog(t, y, color=c, label=lab)
        ax.loglog(t, 6 * e[Dk] * t, color=c, lw=0.8, ls="--")
    ax.set(xlabel="tempo simulato (ns)", ylabel=f"MSD ({unit})")
    ax.set_title(f"{name}\nin funzione del tempo simulato", loc="left", fontsize=8.5)
    ax = axs[1, col]
    for e, c, lab, nsd in [(aa, AA_C, "AA (137 ns/day)", AA_NSDAY), (cg, CG_C, "CG-ML (54,2 ns/day aggregati)", CG_NSDAY)]:
        t = np.asarray(e["lag_ps"]) / 1000.0
        m = t <= e["span_ps"] / 4000.0
        hrs = t[m] / nsd * 24.0
        ax.loglog(hrs, np.asarray(e[key])[m], color=c, label=lab)
    rate_aa = 6 * aa[Dk] * AA_NSDAY / 24
    rate_cg = 6 * cg[Dk] * CG_NSDAY / 24
    ax.set(xlabel="ore di GPU (A100)", ylabel=f"MSD ({unit})")
    note = "" if col == 0 else " (sovrastima)"
    ax.set_title(f"in funzione delle ore di GPU\nregime diffusivo: {rate_cg / rate_aa:.0f}× l'AA{note}", loc="left", fontsize=8.5)
axs[0, 0].legend(loc="upper left")
axs[1, 0].legend(loc="upper left", fontsize=7.5)
fig.tight_layout()
fig.savefig("fig_msd.png")
plt.close(fig)

json.dump({"cv": cv_stats,
           "msd": {lab: {k: msd[lab][k] for k in ("D_trans", "D_rot", "D_trans_langevin", "D_rot_langevin", "tau_v_ps", "tau_w_ps")}
                   | {"tau_P2": msd[lab]["tau_P2"][0]} for lab in ("AA", "prod")}},
          open("fig_stats.json", "w"), indent=1)
print(json.dumps(cv_stats, indent=1))
