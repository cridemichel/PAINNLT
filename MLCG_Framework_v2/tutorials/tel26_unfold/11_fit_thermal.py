#!/usr/bin/env python3
"""Prior dipendenti da T dalla scansione (lam_tet, lam_oth, T): fit sulle popolazioni sperimentali.

IDEA
    Con D(T) = h D0 (1 - T/T0) (09_scale_contacts.py --h-* --t0-*) l'Hamiltoniana a una data T
    e' quella a fattore costante lam(T) = h (1 - T/T0): la torsione segue oth in entrambi i
    casi.  La scansione di 10_contact_sweep.sh da' quindi, per interpolazione, le popolazioni di
    QUALUNQUE scelta di (h, T0) per le due classi, senza nuove corse:
        P_X(T) = P_X^griglia(lam_tet(T), lam_oth(T), T).
    Si cercano (h_tet, T0_tet, h_oth, T0_oth) che riproducono le popolazioni sperimentali di
    2JPZ (Buscaglia, Gray, Chaires 2013, 25 mM KCl, modello a 4 stati, ref/tm_2jpz.py).

STATI
    CG (08_tetrad_melt.py): F = 3 tetradi intatte, I = 1-2, U = nessuna.
    Esperimento: F, I1, I2 (tripla elica), U.  Corrispondenza di default (--map f1):
        F_CG <-> F + I1   (I1: piccolo riarrangiamento, dH -7.8 kcal/mol, tetradi formate),
        I_CG <-> I2,  U_CG <-> U.
    --map f: F_CG <-> F, I_CG <-> I1 + I2.

RAMI
    --branch nome=cartella, ripetibile: la stessa griglia partendo dal nativo (sweep2) e da uno
    stato aperto (sweep2u, START=).  Le popolazioni si mediano sui rami; lo scarto fra i rami
    (isteresi) si stampa per punto della griglia e pesa i residui del fit.

USO
    python3 11_fit_thermal.py --branch chiuso=$A/sweep2 --branch aperto=$A/sweep2u \\
        --plot $A/sweep2/fit_thermal --json $A/sweep2/fit_thermal.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

R_KCAL = 1.98720e-3
# Buscaglia, Gray, Chaires 2013, Supp. Table 2 (2JPZ, 25 mM KCl): dH di folding, Tm in C
EXP = {"dH": [-7.8, -36.4, -39.9], "Tm": [37.5, 43.7, 54.3]}
D0 = {"tet": 1500.0, "oth": 1200.0}          # somme delle D di lp2 per copia (kJ/mol)
DH_EXP_KJ = 351.9


def exp_pops(T):
    T = np.asarray(T, float)[:, None]
    dHu = -np.asarray(EXP["dH"])[None, :]
    TmK = np.asarray(EXP["Tm"])[None, :] + 273.15
    lnK = -(dHu / R_KCAL) * (1 / T - 1 / TmK)
    cum = np.concatenate([np.zeros((T.shape[0], 1)), np.cumsum(lnK, 1)], 1)
    w = np.exp(cum - cum.max(1, keepdims=True))
    return w / w.sum(1, keepdims=True)                   # F, I1, I2, U


def load_branch(path):
    out = {}
    for f in glob.glob(os.path.join(path, "lt*_lo*", "melt.json")):
        m = re.search(r"lt([0-9.]+)_lo([0-9.]+)", f)
        lt, lo = float(m.group(1)), float(m.group(2))
        for k, v in json.load(open(f))["runs"].items():
            p = np.asarray(v["P_n_intact"], float)       # P(0..3 tetradi intatte)
            out[(lt, lo, float(k))] = np.array([p[3], p[1] + p[2], p[0]])   # F, I, U
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", action="append", required=True, metavar="NOME=CARTELLA")
    ap.add_argument("--map", choices=("f1", "f"), default="f1")
    ap.add_argument("--T-fit", default="295:360:5", help="temperature (K) del fit, A:B:passo")
    ap.add_argument("--dh-weight", type=float, default=0.0,
                    help="peso del vincolo sum D_H = dH sperimentale (0 = solo riportato)")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    branches = {}
    for spec in args.branch:
        name, path = spec.split("=", 1)
        branches[name] = load_branch(path)
        print(f"  ramo {name}: {len(branches[name])} punti da {path}")
    keys = sorted(set.intersection(*[set(b) for b in branches.values()]))
    LT = sorted({k[0] for k in keys}); LO = sorted({k[1] for k in keys}); TT = sorted({k[2] for k in keys})
    P = np.full((len(LT), len(LO), len(TT), 3), np.nan)
    H = np.zeros((len(LT), len(LO), len(TT)))
    for (lt, lo, T) in keys:
        v = np.array([b[(lt, lo, T)] for b in branches.values()])
        i, j, k = LT.index(lt), LO.index(lo), TT.index(T)
        P[i, j, k] = v.mean(0)
        H[i, j, k] = np.abs(v[:, 0] - v[:, 0].mean()).max() if len(v) > 1 else 0.0
    if np.isnan(P).any():
        raise SystemExit("[ERROR] griglia incompleta: punti mancanti in qualche ramo (vedi 10_contact_sweep.sh)")
    print(f"  griglia: lam_tet {LT}, lam_oth {LO}, T {TT}")
    if len(branches) > 1:
        print("\n  isteresi |P_F(ramo) - media| per punto (max sulle T), lam_tet x lam_oth:")
        print("           " + "".join(f"{lo:>8.2f}" for lo in LO))
        for i, lt in enumerate(LT):
            print(f"    {lt:5.2f}  " + "".join(f"{H[i, j].max():8.2f}" for j in range(len(LO))))
    interp = RegularGridInterpolator((LT, LO, TT), P, bounds_error=False, fill_value=np.nan)
    hyst = RegularGridInterpolator((LT, LO, TT), H, bounds_error=False, fill_value=np.nan)

    a, b, s = (float(x) for x in args.T_fit.split(":"))
    Tf = np.arange(a, b + 1e-9, s)
    ex = exp_pops(Tf)
    target = (np.stack([ex[:, 0] + ex[:, 1], ex[:, 2], ex[:, 3]], 1) if args.map == "f1"
              else np.stack([ex[:, 0], ex[:, 1] + ex[:, 2], ex[:, 3]], 1))

    def lam(x, T):
        h_t, t0_t, h_o, t0_o = x
        return h_t * (1 - T / t0_t), h_o * (1 - T / t0_o)

    def predict(x, T):
        lt, lo = lam(x, T)
        pts = np.stack([lt, lo, np.clip(T, TT[0], TT[-1])], 1)
        return interp(pts), hyst(pts), lt, lo

    def loss(x):
        if x[1] <= Tf.max() or x[3] <= Tf.max() or x[0] <= 0 or x[2] <= 0:
            return 1e3
        p, hy, lt, lo = predict(x, Tf)
        out = np.isnan(p).any(1)
        pen = 10.0 * out.sum()
        w = 1.0 / (0.03 ** 2 + np.nan_to_num(hy, nan=0.5) ** 2)
        res = np.nan_to_num(p - target, nan=0.0)
        sdh = x[0] * D0["tet"] + x[2] * D0["oth"]
        return float((w[:, None] * res ** 2).sum() / w.sum() + pen
                     + args.dh_weight * ((sdh - DH_EXP_KJ) / DH_EXP_KJ) ** 2)

    # costante (T0 molto grande) e termico, da una griglia di partenze
    best = None
    rng = np.random.default_rng(3)
    starts = [[ht, t0t, ho, t0o] for ht in (0.1, 0.13, 0.2, 0.3) for t0t in (420, 600, 2000)
              for ho in (0.1, 0.2, 0.3) for t0o in (420, 600, 2000)]
    for x0 in starts:
        r = minimize(loss, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    const = None
    for lt0 in LT:
        for lo0 in LO:
            r = minimize(lambda y: loss([y[0], 1e9, y[1], 1e9]), [lt0, lo0], method="Nelder-Mead")
            if const is None or r.fun < const.fun:
                const = r

    x = best.x
    sdh = x[0] * D0["tet"] + x[2] * D0["oth"]
    p, hy, lt, lo = predict(x, Tf)
    pc, _, _, _ = predict([const.x[0], 1e9, const.x[1], 1e9], Tf)
    print(f"\n  mappa {args.map}: CG F/I/U contro " + ("F+I1 / I2 / U" if args.map == "f1" else "F / I1+I2 / U"))
    print(f"  fit termico:   h_tet {x[0]:.4f}  T0_tet {x[1]:.1f} K   h_oth {x[2]:.4f}  T0_oth {x[3]:.1f} K   "
          f"loss {best.fun:.4g}")
    print(f"                 sum D_H = {sdh:.0f} kJ/mol per copia (dH sperimentale {DH_EXP_KJ:.0f})")
    print(f"  fit costante:  lam_tet {const.x[0]:.4f}  lam_oth {const.x[1]:.4f}   loss {const.fun:.4g}")
    print(f"\n  {'T (K)':>6s}  {'lam_tet':>7s} {'lam_oth':>7s}   {'F esp':>6s} {'I esp':>6s} {'U esp':>6s}   "
          f"{'F CG':>6s} {'I CG':>6s} {'U CG':>6s}   {'F cost':>6s} {'U cost':>6s}  isteresi")
    for i, T in enumerate(Tf):
        print(f"  {T:6.0f}  {lt[i]:7.4f} {lo[i]:7.4f}   " + " ".join(f"{v:6.3f}" for v in target[i]) + "   "
              + " ".join(f"{v:6.3f}" for v in p[i]) + f"   {pc[i, 0]:6.3f} {pc[i, 2]:6.3f}  {hy[i]:6.3f}")
    print(f"\n  validazione diretta (soli prior, una copia):\n"
          f"    python3 09_scale_contacts.py --in $A/cg/cg_priors.lp2_1c.json --out $A/cg/cg_priors.lp2_1c.thermo_fit.json \\\n"
          f"        --h-tet {x[0]:.4f} --t0-tet {x[1]:.1f} --h-oth {x[2]:.4f} --t0-oth {x[3]:.1f}")
    res = {"map": args.map, "thermal": {"h_tet": x[0], "T0_tet": x[1], "h_oth": x[2], "T0_oth": x[3],
                                        "loss": best.fun, "sum_DH_kJ": sdh},
           "constant": {"lam_tet": const.x[0], "lam_oth": const.x[1], "loss": const.fun},
           "T": Tf.tolist(), "target_FIU": target.tolist(), "pred_FIU": p.tolist(), "pred_const_FIU": pc.tolist(),
           "hysteresis_F": hy.tolist(), "branches": list(branches)}
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
        print(f"  json: {args.json}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Tp = np.linspace(285, 380, 200)
        exf = exp_pops(Tp)
        tg = (np.stack([exf[:, 0] + exf[:, 1], exf[:, 2], exf[:, 3]], 1) if args.map == "f1"
              else np.stack([exf[:, 0], exf[:, 1] + exf[:, 2], exf[:, 3]], 1))
        pp, _, _, _ = predict(x, Tp)
        ppc, _, _, _ = predict([const.x[0], 1e9, const.x[1], 1e9], Tp)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for k, (lab, c) in enumerate([("F", "#2a78d6"), ("I", "#eda100"), ("U", "#eb6834")]):
            ax.plot(Tp - 273.15, tg[:, k], color=c, lw=2.2, label=f"{lab} sperimentale")
            ax.plot(Tp - 273.15, pp[:, k], color=c, lw=1.4, ls="--", label=f"{lab} CG, D(T)")
            ax.plot(Tp - 273.15, ppc[:, k], color=c, lw=1.0, ls=":", label=f"{lab} CG, lam costante")
        ax.set(xlabel="T (°C)", ylabel="frazione", ylim=(-0.02, 1.02),
               title="TEL26: popolazioni sperimentali e CG interpolato dalla scansione")
        ax.legend(fontsize=7, ncol=3, frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot + ".png", dpi=160)
        print(f"  grafico: {args.plot}.png")


if __name__ == "__main__":
    main()
