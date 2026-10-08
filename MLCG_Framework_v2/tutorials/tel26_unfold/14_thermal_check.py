#!/usr/bin/env python3
"""Validazione dei prior dipendenti da T: popolazioni CG simulate contro esperimento e griglia.

Legge le cartelle di 13_validate_thermal.sh (<insieme>/<ramo>/melt.json di 08_tetrad_melt.py)
e per ogni insieme stampa, contro T:
  - F/I/U sperimentali nella mappa dell'insieme (f1: F+I1 / I2 / U;  f: F / I1+I2 / U),
  - F/I/U CG mediati sui rami (chiuso, aperto) e isteresi |P_F(ramo) - media|,
  - F/I/U previsti per interpolazione della griglia (fit_thermal*.json di 11, se c'e'),
poi RMS degli scarti, Tm (U = 1/2), T_F (F = 1/2) e dH di van 't Hoff da ln(P_U/P_F) contro 1/T.

Perche' van 't Hoff basta per il dH calorimetrico: con U(x;T) = U_H(x) - T U_S(x)
  d ln(Z_U/Z_F) / d(-1/kT) = <U_H>_U - <U_H>_F
esattamente (la parte entropica dei prior si cancella), quindi la pendenza di ln(P_U/P_F)
contro 1/T e' il dH CG fra gli stati F e U, da confrontare con la stessa pendenza calcolata
sulle popolazioni sperimentali nella stessa mappa (352 kJ/mol in mappa f: dH totale F -> U).

Uso:
  python3 14_thermal_check.py --set f1=$A/thermo_val/f1:f1:$A/sweep2/fit_thermal.json \\
      --set f=$A/thermo_val/f:f:$A/sweep2/fit_thermal_f.json --plot $A/thermo_val/thermal_check
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fit_thermal", HERE / "11_fit_thermal.py")
_fit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fit)
exp_pops = _fit.exp_pops
R_KJ = 8.314462618e-3


def target(T, mp):
    ex = exp_pops(np.atleast_1d(np.asarray(T, float)))
    if mp == "f1":
        return np.stack([ex[:, 0] + ex[:, 1], ex[:, 2], ex[:, 3]], 1)
    return np.stack([ex[:, 0], ex[:, 1] + ex[:, 2], ex[:, 3]], 1)


def load_set(d):
    """-> {ramo: {T: (F, I, U), ...}}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*", "melt.json"))):
        b = os.path.basename(os.path.dirname(f))
        runs = json.load(open(f))["runs"]
        out[b] = {float(k): np.array([v["P_n_intact"][3], v["P_n_intact"][1] + v["P_n_intact"][2],
                                      v["P_n_intact"][0]]) for k, v in runs.items()}
    return out


def crossing(T, y, level=0.5):
    for (a, ya), (b, yb) in zip(zip(T, y), zip(T[1:], y[1:])):
        if (ya - level) * (yb - level) <= 0 and ya != yb:
            return a + (level - ya) * (b - a) / (yb - ya)
    return np.nan


def vant_hoff(T, pF, pU, lo=0.02):
    """dH (kJ/mol) dalla pendenza di ln(P_U/P_F) contro 1/T, punti con P_F, P_U in [lo, 1-lo]."""
    T, pF, pU = map(np.asarray, (T, pF, pU))
    ok = (pF > lo) & (pU > lo) & (pF < 1 - lo) & (pU < 1 - lo)
    if ok.sum() < 3:
        return np.nan, int(ok.sum()), (np.nan, np.nan)
    x, y = 1.0 / T[ok], np.log(pU[ok] / pF[ok])
    c = np.polyfit(x, y, 1)
    return -c[0] * R_KJ, int(ok.sum()), (T[ok].min(), T[ok].max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True, metavar="NOME=CARTELLA:MAPPA[:FIT_JSON]")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    summary, curves = {}, {}
    for spec in args.set:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        d, mp = parts[0], parts[1]
        fj = parts[2] if len(parts) > 2 else None
        br = load_set(d)
        if not br:
            print(f"[WARN] {name}: nessun melt.json in {d}"); continue
        Ts = sorted(set.intersection(*[set(v) for v in br.values()]))
        P = np.array([[br[b][T] for T in Ts] for b in br])          # (rami, T, 3)
        Pm = P.mean(0)
        hy = np.abs(P[:, :, 0] - Pm[None, :, 0]).max(0) if len(br) > 1 else np.zeros(len(Ts))
        tg = target(Ts, mp)
        grid = None
        if fj and os.path.exists(fj):
            fd = json.load(open(fj))
            if fd.get("map") != mp:
                print(f"[WARN] {name}: {fj} ha mappa {fd.get('map')}, non {mp}")
            gT = np.asarray(fd["T"]); gP = np.asarray(fd["pred_FIU"])
            grid = np.stack([np.interp(Ts, gT, gP[:, k], left=np.nan, right=np.nan) for k in range(3)], 1)
        elif fj:
            print(f"[WARN] {name}: {fj} non trovato, niente confronto con la griglia")

        print(f"\n=== {name}  (mappa {mp}: " + ("F+I1 / I2 / U" if mp == "f1" else "F / I1+I2 / U")
              + f";  rami: {', '.join(br)})")
        print(f"  {'T (K)':>6s}   {'F esp':>6s} {'I esp':>6s} {'U esp':>6s}   {'F CG':>6s} {'I CG':>6s} {'U CG':>6s}"
              f"  {'ister.':>6s}" + (f"   {'F grig':>6s} {'U grig':>6s}" if grid is not None else ""))
        for i, T in enumerate(Ts):
            print(f"  {T:6.0f}   " + " ".join(f"{v:6.3f}" for v in tg[i]) + "   "
                  + " ".join(f"{v:6.3f}" for v in Pm[i]) + f"  {hy[i]:6.3f}"
                  + (f"   {grid[i, 0]:6.3f} {grid[i, 2]:6.3f}" if grid is not None else ""))
        rms = np.sqrt(((Pm - tg) ** 2).mean(0))
        rms_g = np.sqrt(np.nanmean((grid - Pm) ** 2, 0)) if grid is not None else None
        tm_cg, tm_ex = crossing(Ts, Pm[:, 2]), crossing(Ts, tg[:, 2])
        tf_cg, tf_ex = crossing(Ts, Pm[:, 0]), crossing(Ts, tg[:, 0])
        dh_cg, n_cg, rg_cg = vant_hoff(Ts, Pm[:, 0], Pm[:, 2])
        dh_ex, n_ex, rg_ex = vant_hoff(Ts, tg[:, 0], tg[:, 2])
        print(f"  RMS CG - esp (F, I, U): " + ", ".join(f"{v:.3f}" for v in rms)
              + (f";   RMS CG - griglia: " + ", ".join(f"{v:.3f}" for v in rms_g) if rms_g is not None else ""))
        print(f"  Tm (U = 1/2):  CG {tm_cg:6.1f} K   esp {tm_ex:6.1f} K")
        print(f"  T_F (F = 1/2): CG {tf_cg:6.1f} K   esp {tf_ex:6.1f} K")
        print(f"  dH van 't Hoff ln(U/F): CG {dh_cg:6.0f} kJ/mol ({n_cg} punti, {rg_cg[0]:.0f}-{rg_cg[1]:.0f} K)"
              f"   esp stessa mappa e stessi T: {dh_ex:6.0f} kJ/mol ({n_ex} punti)")
        print(f"  isteresi max: {hy.max():.3f}")
        summary[name] = {"map": mp, "T": Ts, "branches": list(br), "CG_FIU": Pm.tolist(),
                         "CG_FIU_branches": P.tolist(), "exp_FIU": tg.tolist(),
                         "grid_FIU": grid.tolist() if grid is not None else None,
                         "rms_exp": rms.tolist(), "rms_grid": rms_g.tolist() if rms_g is not None else None,
                         "Tm_CG": tm_cg, "Tm_exp": tm_ex, "TF_CG": tf_cg, "TF_exp": tf_ex,
                         "dH_vH_CG": dh_cg, "dH_vH_exp": dh_ex, "hyst_max": float(hy.max())}
        curves[name] = (Ts, Pm, P, grid, mp)

    if args.json:
        json.dump(summary, open(args.json, "w"), indent=1, default=float)
        print(f"\n  json: {args.json}")
    if args.plot and curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(curves)
        fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.2), squeeze=False)
        cols = [("F", "#2a78d6"), ("I", "#eda100"), ("U", "#eb6834")]
        Tp = np.linspace(285, 375, 200)
        for ax, (name, (Ts, Pm, P, grid, mp)) in zip(axes[0], curves.items()):
            tg = target(Tp, mp)
            Tc = np.asarray(Ts) - 273.15
            for k, (lab, c) in enumerate(cols):
                ax.plot(Tp - 273.15, tg[:, k], color=c, lw=2.2, label=f"{lab} sperimentale")
                ax.errorbar(Tc, Pm[:, k], yerr=np.abs(P[:, :, k] - Pm[None, :, k]).max(0) if len(P) > 1 else None,
                            fmt="o", ms=4, color=c, capsize=2, label=f"{lab} CG simulato")
                if grid is not None:
                    ax.plot(Tc, grid[:, k], color=c, lw=1.0, ls="--", label=f"{lab} griglia interpolata")
            ax.set(xlabel="T (°C)", ylabel="frazione", ylim=(-0.02, 1.02),
                   title=f"{name}: D(T) validato (mappa {mp})")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=3, frameon=False)
        fig.tight_layout()
        fig.savefig(args.plot + ".png", dpi=160)
        print(f"  grafico: {args.plot}.png")


if __name__ == "__main__":
    main()
