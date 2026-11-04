#!/usr/bin/env python3
"""Popolazioni F / U nelle scatole bulk (27_bulk_box.sh): frazioni, transizioni, P_U dalle velocita'.

PERCHE' (4/11)
    Le scatole partono con la miscela F/U data da P_U(T) delle corse a una copia (fit di 22): la frazione
    di copie U nel tempo, da sola, ripete in buona parte quella condizione iniziale.  L'informazione
    nuova viene dalle transizioni: con 300 copie anche pochi eventi per copia sono centinaia di eventi.
    In un sistema a due stati con conteggi N_FU, N_UF e tempi T_F, T_U spesi negli stati,
        k_u = N_FU / T_F,  k_f = N_UF / T_U,  P_U = k_u / (k_u + k_f),
    che non dipende dalla composizione iniziale (stima di massima verosimiglianza per processi di
    Poisson; errore dalla propagazione con sigma(N) = sqrt(N)).  P_U = 1/2 alla Tm del sistema bulk: il
    confronto con la Tm a una copia (323,0 K) dice quanto pesano le interazioni fra molecole.

STATI PER COPIA (come 22/23): c_a = frazione liscia delle 9 coppie B3-B3 native del tratto G a con gli
    altri, media mobile su --smooth-ns (default 0,5 ns); nucleo U: tutti i c_a < --u-cut (0,2); nucleo F:
    tutti i c_a > --f-cut (0,5).  Assegnazione con isteresi: una copia cambia stato solo quando entra
    nell'altro nucleo (i frame intermedi restano allo stato precedente); i frame prima del primo nucleo
    non si contano.  Le escursioni piu' brevi della media mobile si perdono: lo script stampa le permanenze
    medie e avvisa se sono < 5 x --smooth-ns.  Verificato su un bulk sintetico (40 copie, catena di Markov
    nota, 15 ns): con la media mobile a 0 o 0,05 ns gli stati coincidono con quelli veri (99,6-100 %), e su
    400 realizzazioni della catena P_U(vel.) = 0,400 +- 0,029 contro 0,4 vero, errore stimato 0,028.

USO (Leonardo)
    python3 29_bulk_states.py --dirs $A/bulk/N300_c0.5_T323_s1 $A/bulk/N300_c0.5_T323_s2 ... --skip-ns 1 \\
        --out $A/bulk/states_T323
    -> tabella per scatola e totale (frazioni iniziale/finale, transizioni, P_U dalle velocita'),
       states_T323.json/.png (frazione U nel tempo per scatola).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pl = _load("pop_labels", "19_pop_labels.py")
cv = pl.cv


def box_tract_c(d, r_nat, beta=25.0, lam=1.2):
    """Serie c_a(t) di ogni copia di una scatola: (T, ncopy, 4), tempi (ps)."""
    files = [f for f in sorted(glob.glob(str(pathlib.Path(d) / "chunk*.samples.npz"))) if ".tmp." not in f]
    if not files:
        raise SystemExit(f"[ERROR] nessun chunk*.samples.npz in {d}")
    cs, ts, t_off = [], [], 0.0
    for f in files:
        S, L, nc, t = cv.load_samples(f)
        t = np.asarray(t, float)
        dd = pl.melt.b3_pair_distances(S, L, nc)                        # (T*nc, 3, 6)
        sp = 1.0 / (1.0 + np.exp(beta * (dd - lam * r_nat[None])))
        c = np.stack([sp[:, :, pl.TRACT_MASK[a]].mean(axis=(1, 2)) for a in range(4)], 1)
        cs.append(c.reshape(S.shape[0], nc, 4))
        ts.append(t + t_off)
        t_off = ts[-1][-1] + (t[1] - t[0] if t.size > 1 else 0.0)
    return np.concatenate(cs), np.concatenate(ts)


def committed_states(c, u_cut, f_cut):
    """c (T, nc, 4) -> stati (T, nc): 1 = U, 0 = F, -1 = non ancora assegnato; isteresi sui nuclei."""
    inU = (c < u_cut).all(-1)
    inF = (c > f_cut).all(-1)
    st = np.full(inU.shape, -1, dtype=np.int8)
    cur = np.full(inU.shape[1], -1, dtype=np.int8)
    for k in range(inU.shape[0]):
        cur = np.where(inU[k], 1, np.where(inF[k], 0, cur)).astype(np.int8)
        st[k] = cur
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="scatole alla stessa T")
    ap.add_argument("--skip-ns", type=float, default=1.0)
    ap.add_argument("--smooth-ns", type=float, default=0.5)
    ap.add_argument("--u-cut", type=float, default=0.2)
    ap.add_argument("--f-cut", type=float, default=0.5)
    ap.add_argument("--native", default=str(HERE.parent / "tel26" / "tel26_lp2_dataset.bin"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cv.set_nuc(pl.melt.NUC)
    Sn, Ln, ncn = cv.load_reference(args.native)
    r_nat = np.nanmedian(pl.melt.b3_pair_distances(Sn[::5], Ln[::5], ncn), axis=0)

    rows, series = [], {}
    tot = dict(N_FU=0, N_UF=0, T_F=0.0, T_U=0.0)
    print(f"{'scatola':>28s} {'copie':>5s} {'ns':>5s} {'U ini':>6s} {'U fin':>6s} {'F->U':>5s} {'U->F':>5s} "
          f"{'T_F ns':>8s} {'T_U ns':>8s} {'P_U(vel.)':>12s}")
    for d in args.dirs:
        c, t = box_tract_c(d, r_nat)
        T, nc, _ = c.shape
        c = pl.smooth2(c.reshape(T, nc * 4), t, args.smooth_ns).reshape(T, nc, 4)
        keep = t >= t[0] + args.skip_ns * 1000.0
        st = committed_states(c[keep], args.u_cut, args.f_cut)
        tk = t[keep]
        if tk.size < 2:
            print(f"{pathlib.Path(d).name:>28s}  troppo pochi frame dopo --skip-ns")
            continue
        dt = (tk[1] - tk[0]) / 1000.0                                        # ns per frame
        prev, nxt = st[:-1], st[1:]
        n_fu = int(((prev == 0) & (nxt == 1)).sum())
        n_uf = int(((prev == 1) & (nxt == 0)).sum())
        t_f = float((st[:-1] == 0).sum()) * dt
        t_u = float((st[:-1] == 1).sum()) * dt
        assigned = (st >= 0)
        fu = np.where(assigned.any(1), (st == 1).sum(1) / np.maximum(assigned.sum(1), 1), np.nan)
        series[d] = (tk / 1000.0, fu)
        pu, epu = rate_pu(n_fu, n_uf, t_f, t_u)
        rows.append({"dir": d, "copies": nc, "ns": float(tk[-1] - tk[0]) / 1000.0, "U_first": float(fu[0]),
                     "U_last": float(fu[-1]), "N_FU": n_fu, "N_UF": n_uf, "T_F_ns": t_f, "T_U_ns": t_u,
                     "P_U_rates": pu, "P_U_rates_err": epu, "unassigned_frac": float(1 - assigned.mean())})
        for k, v in (("N_FU", n_fu), ("N_UF", n_uf), ("T_F", t_f), ("T_U", t_u)):
            tot[k] += v
        print(f"{pathlib.Path(d).name:>28s} {nc:5d} {rows[-1]['ns']:5.1f} {fu[0]:6.3f} {fu[-1]:6.3f} {n_fu:5d} {n_uf:5d} "
              f"{t_f:8.1f} {t_u:8.1f} {fmt(pu, epu):>12s}")
    pu, epu = rate_pu(tot["N_FU"], tot["N_UF"], tot["T_F"], tot["T_U"])
    frac = np.nanmean([r["U_last"] for r in rows]) if rows else np.nan
    print(f"{'totale':>28s} {'':5s} {'':5s} {'':6s} {frac:6.3f} {tot['N_FU']:5d} {tot['N_UF']:5d} "
          f"{tot['T_F']:8.1f} {tot['T_U']:8.1f} {fmt(pu, epu):>12s}")
    print("\n[states] P_U(vel.) = k_u / (k_u + k_f), k_u = N_FU / T_F, k_f = N_UF / T_U: non dipende dalla miscela "
          "iniziale; P_U = 1/2 alla Tm.  'U fin' e' la frazione di copie U all'ultimo frame.")
    dwell = [tot["T_F"] / tot["N_FU"] if tot["N_FU"] else np.inf, tot["T_U"] / tot["N_UF"] if tot["N_UF"] else np.inf]
    print(f"[states] permanenza media: F {dwell[0]:.2f} ns, U {dwell[1]:.2f} ns (media mobile {args.smooth_ns} ns)")
    if min(dwell) < 5 * args.smooth_ns:
        print("[WARN] permanenze brevi rispetto alla media mobile: le escursioni piu' corte di ~--smooth-ns si perdono "
              "e le velocita' sono sottostimate; prova --smooth-ns piu' piccolo (stessa definizione di 22 solo con 0,5)")
    json.dump({"boxes": rows, "total": {**tot, "P_U_rates": pu, "P_U_rates_err": epu, "dwell_F_ns": dwell[0],
                                         "dwell_U_ns": dwell[1]},
               "smooth_ns": args.smooth_ns, "u_cut": args.u_cut, "f_cut": args.f_cut, "skip_ns": args.skip_ns},
              open(args.out + ".json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for d, (tt, fu) in series.items():
            ax.plot(tt, fu, label=pathlib.Path(d).name)
        if np.isfinite(pu):
            ax.axhline(pu, color="k", ls="--", lw=1, label=f"P_U dalle velocita' {pu:.2f}")
        ax.set(xlabel="t (ns)", ylabel="frazione di copie U", ylim=(-0.02, 1.02))
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150)
    except Exception as exc:  # pragma: no cover
        print(f"[states] figura non scritta: {exc}")
    print(f"[states] scritti {args.out}.json/.png")


def rate_pu(n_fu, n_uf, t_f, t_u):
    if t_f <= 0 or t_u <= 0 or (n_fu == 0 and n_uf == 0):
        return float("nan"), float("nan")
    ku, kf = n_fu / t_f, n_uf / t_u
    pu = ku / (ku + kf)
    # d pu / d ku = kf / (ku + kf)^2, d pu / d kf = -ku / (ku + kf)^2; sigma(k) = sqrt(N) / T
    eku, ekf = np.sqrt(max(n_fu, 1)) / t_f, np.sqrt(max(n_uf, 1)) / t_u
    epu = np.hypot(kf * eku, ku * ekf) / (ku + kf) ** 2
    return float(pu), float(epu)


def fmt(pu, epu):
    return "-" if not np.isfinite(pu) else f"{pu:.3f}+-{epu:.3f}"


if __name__ == "__main__":
    main()
