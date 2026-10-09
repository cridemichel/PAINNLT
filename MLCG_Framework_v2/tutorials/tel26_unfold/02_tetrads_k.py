#!/usr/bin/env python3
"""Tetradi e K+ del canale lungo le corse AA (DNA + K+, da 01_extract_dna_k.sh).

PERCHE'
    Alla fine dell'equilibratura a 400 K di Giulia la tetrade 3 (G6, G10, G18,
    G24) e' aperta e nel canale c'e' un solo K+ (fra le tetradi 1 e 2).  Prima
    del pilota senza K+ nel canale serve la storia: quando si apre la tetrade
    3, se si richiude, quali K+ entrano ed escono dal canale e in quale sito.

DEFINIZIONI (per frame)
    - legami di Hoogsteen di una coppia adiacente nella tetrade (i, j):
      N1-H...O6 presente se min(d(N1_i,O6_j), d(N1_j,O6_i)) < --hb (0,35 nm),
      N2-H...N7 presente se min(d(N2_i,N7_j), d(N2_j,N7_i)) < --hb;
      Q_tetrade = legami presenti / 8;
    - raggio della tetrade: media delle distanze O6-baricentro O6;
    - asse del canale: dal baricentro O6 della tetrade 1 a quello della 3;
      un K+ e' nel canale se la distanza dall'asse e' < --rho (0,25 nm) e la
      coordinata lungo l'asse cade fra la tetrade 1 - 0,3 nm e la 3 + 0,3 nm;
      sito: sopra 1, fra 1 e 2, fra 2 e 3, sotto 3;
    - Rg del DNA (atomi pesanti) e distanza fra i P/O5' estremi.
    Distanze K+ - DNA con immagine minima (il DNA e' intero, -pbc mol).

USO (in tutorials/tel26_unfold, con la cartella dnak/ copiata da Leonardo)
    python3 02_tetrads_k.py dnak/equil_dna_k.gro dnak/equil_dna_k.xtc \\
        dnak/prod-1_dna_k.xtc dnak/prod-2_dna_k.xtc dnak/prod-3_dna_k.xtc \\
        --out tetrads_k_400K
    -> tetrads_k_400K.npz, tetrads_k_400K.png e un riassunto a schermo.
    Il primo xtc e' l'equilibratura (tempo negativo nel grafico: -20..0 ns).
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

TETRADS = [(4, 12, 16, 22), (5, 11, 17, 23), (6, 10, 18, 24)]   # ordine ciclico (1-based)
SITES = ["sopra 1", "fra 1 e 2", "fra 2 e 3", "sotto 3"]


def smooth(x, w):
    """media mobile centrata su w frame, corretta ai bordi (len invariata)"""
    w = max(1, min(int(w), len(x)))
    k = np.ones(w)
    return np.convolve(x, k, mode="same") / np.convolve(np.ones(len(x)), k, mode="same")


def mic(d, box):
    return d - box * np.round(d / box)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("top")
    ap.add_argument("xtc", nargs="+", help="equilibratura per prima, poi le parti di produzione")
    ap.add_argument("--hb", type=float, default=0.35)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="tetrads_k")
    ap.add_argument("--ion-resname", default="K", help="nome di residuo dei cationi (K, LI)")
    ap.add_argument("--excl-r", type=float, default=0.4,
                    help="soglia (nm) del canale vietato (03 --exclude-r): si riporta quanto spesso un "
                         "catione scende sotto, rispetto agli O6 del core")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    import MDAnalysis as mda

    u = mda.Universe(args.top, *args.xtc, continuous=False)
    dna = u.select_atoms("resname DT5 DT DA DG DT3")
    heavy = dna.select_atoms("not name H*")
    K = u.select_atoms(f"resname {args.ion_resname}")

    def at(res, name):
        sel = dna.select_atoms(f"resid {res} and name {name}")
        assert len(sel) == 1, (res, name, len(sel))
        return sel[0].index

    idx = {n: [[at(r, n) for r in t] for t in TETRADS] for n in ("O6", "N1", "N2", "N7")}
    p5 = dna.select_atoms("resid 1 and name O5'")[0].index
    p3 = dna.select_atoms(f"resid {dna.resids.max()} and name O3'")[0].index
    nk = len(K)
    print(f"[INFO] {len(dna)} atomi DNA, {nk} K+, {len(u.trajectory)} frame in {len(args.xtc)} file")

    # confini fra file: il primo xtc e' l'equilibratura
    starts = []
    off = 0
    for f in args.xtc:
        n = len(mda.Universe(args.top, f).trajectory) if f.endswith(".xtc") else 1
        starts.append(off)
        off += n
    T, Q, Rt, kin, ksite, rg, ee, part, ko6 = [], [], [], [], [], [], [], [], []
    o6_all = np.array(idx["O6"]).ravel()
    t_shift = 0.0
    last_t = None
    for fi, ts in enumerate(u.trajectory[::args.stride]):
        frame = ts.frame
        p = int(np.searchsorted(starts, frame, side="right") - 1)
        box = ts.dimensions[:3] / 10.0
        X = ts.positions / 10.0
        t = ts.time / 1000.0
        if p == 0:
            t_rel = t
        T.append(t)
        part.append(p)
        q, rr, cent = [], [], []
        for k in range(3):
            O = X[idx["O6"][k]]; N1 = X[idx["N1"][k]]; N2 = X[idx["N2"][k]]; N7 = X[idx["N7"][k]]
            c = O.mean(0); cent.append(c)
            rr.append(np.linalg.norm(O - c, axis=1).mean())
            nb = 0
            for a in range(4):
                b = (a + 1) % 4
                d1 = min(np.linalg.norm(N1[a] - O[b]), np.linalg.norm(N1[b] - O[a]))
                d2 = min(np.linalg.norm(N2[a] - N7[b]), np.linalg.norm(N2[b] - N7[a]))
                nb += int(d1 < args.hb) + int(d2 < args.hb)
            q.append(nb / 8)
        Q.append(q); Rt.append(rr)
        cent = np.array(cent)
        ax = cent[2] - cent[0]
        L = np.linalg.norm(ax); ax /= L
        z_t = (cent - cent[0]) @ ax                       # 0, ~0,33, ~0,66
        d = mic(X[K.indices] - cent[0], box)
        z = d @ ax
        rho = np.linalg.norm(d - np.outer(z, ax), axis=1)
        inside = (rho < args.rho) & (z > -0.3) & (z < z_t[2] + 0.3)
        site = np.full(nk, -1)
        site[inside & (z < 0)] = 0
        site[inside & (z >= 0) & (z < z_t[1])] = 1
        site[inside & (z >= z_t[1]) & (z <= z_t[2])] = 2
        site[inside & (z > z_t[2])] = 3
        kin.append(inside.sum()); ksite.append(site)
        dko = mic(X[K.indices][:, None, :] - X[o6_all][None, :, :], box)
        ko6.append(np.sqrt((dko ** 2).sum(-1)).min())          # catione-O6 del core piu' vicini
        H = X[heavy.indices]
        rg.append(np.sqrt(((H - H.mean(0)) ** 2).sum(1).mean()))
        ee.append(np.linalg.norm(X[p5] - X[p3]))
    T = np.array(T); part = np.array(part)
    # tempo: equilibratura in [-durata, 0], produzione continua dopo
    teq = T[part == 0]
    tt = T.copy()
    if (part == 0).any():
        tt[part == 0] = teq - teq.max()
    Q = np.array(Q); Rt = np.array(Rt); kin = np.array(kin); ksite = np.array(ksite); ko6 = np.array(ko6)
    rg = np.array(rg); ee = np.array(ee)
    np.savez_compressed(args.out + ".npz", t_ns=tt, part=part, Q=Q, r_tetrad=Rt, k_in=kin, k_site=ksite,
                        rg=rg, ee=ee, k_o6_min=ko6)

    print(f"\n  {'':<16s}{'Q t1':>7s}{'Q t2':>7s}{'Q t3':>7s}{'r t1':>7s}{'r t2':>7s}{'r t3':>7s}{'K nel canale':>14s}{'Rg':>7s}")
    for lab, m in [("equilibratura", part == 0), ("produzione", part > 0)]:
        if m.any():
            print(f"  {lab:<16s}" + "".join(f"{x:7.2f}" for x in Q[m].mean(0)) + "".join(f"{x:7.3f}" for x in Rt[m].mean(0))
                  + f"{kin[m].mean():14.2f}{rg[m].mean():7.3f}")
    print("\n  K+ nel canale: frazione del tempo per sito (produzione)")
    m = part > 0 if (part > 0).any() else part >= 0
    for s, name in enumerate(SITES):
        occ = (ksite[m] == s).any(1).mean()
        who = sorted(set(np.where(ksite[m] == s)[1]))
        print(f"    {name:<10s} occupato {occ:5.2f}   K+ (indice) {[int(K.resids[w]) for w in who][:10]}")
    # apertura della tetrade 3: primo istante in cui Q (media mobile 1 ns) scende sotto 0,5
    dtf = np.median(np.diff(tt)) if len(tt) > 1 else 0.01
    w = max(1, int(round(1.0 / dtf)))
    for k in range(3):
        qs = smooth(Q[:, k], w)
        below = np.where(qs < 0.5)[0]
        print(f"  tetrade {k + 1}: Q < 0,5 (media 1 ns) per il {np.mean(qs < 0.5):.2f} del tempo"
              + (f", la prima volta a t = {tt[below[0]]:.2f} ns" if below.size else ""))
    # stati come nel CG: tetrade intatta se Q (media 1 ns) >= 0,5; F = 3, I = 1-2, U = 0
    nint = sum((smooth(Q[:, k], w) >= 0.5).astype(int) for k in range(3))
    mp = part > 0 if (part > 0).any() else part >= 0
    pF, pU = np.mean(nint[mp] == 3), np.mean(nint[mp] == 0)
    pI = 1.0 - pF - pU
    tp = tt[mp]
    first = lambda cond: (f"{tp[np.where(cond)[0][0]]:.1f}" if cond.any() else "-")
    t_le1, t_u = first(nint[mp] <= 1), first(nint[mp] == 0)
    print(f"  stati (Q >= 0,5 = intatta): F {pF:.2f}  I {pI:.2f}  U {pU:.2f};  primo n_int <= 1 a {t_le1} ns, "
          f"primo U a {t_u} ns")
    kp = ko6[mp]
    below_r = np.mean(kp < args.excl_r)
    print(f"  catione-O6 del core, distanza minima: min {kp.min():.3f} nm, 1% {np.percentile(kp, 1):.3f}, "
          f"mediana {np.median(kp):.3f};  sotto {args.excl_r:g} nm nel {below_r:.3f} dei frame")
    print(f"[riassunto] {args.out}  t_max {tp.max():.1f} ns  Q {' '.join(f'{x:.2f}' for x in Q[mp].mean(0))}  "
          f"canale {kin[mp].mean():.2f}  F {pF:.2f} I {pI:.2f} U {pU:.2f}  n<=1 {t_le1}  U {t_u}  "
          f"dKO6min {kp.min():.3f}  <{args.excl_r:g} {below_r:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(5, 1, figsize=(8.5, 10.2), sharex=True)
    cols = ["#2a78d6", "#1baf7a", "#eb6834"]
    for k in range(3):
        axs[0].plot(tt, smooth(Q[:, k], w), color=cols[k], lw=1, label=f"tetrade {k + 1}")
        axs[1].plot(tt, smooth(Rt[:, k], w), color=cols[k], lw=1)
    axs[0].set_ylabel("Q Hoogsteen"); axs[0].legend(frameon=False, ncol=3, loc="lower left")
    axs[1].set_ylabel("raggio O6 (nm)")
    for s in range(4):
        occ = (ksite == s).any(1).astype(float)
        axs[2].fill_between(tt, s - 0.4 * occ, s + 0.4 * occ, color="#4a3aa7", lw=0, step="mid")
    axs[2].set_yticks(range(4)); axs[2].set_yticklabels(SITES); axs[2].invert_yaxis()
    axs[2].set_ylabel("K+ nel canale")
    axs[3].plot(tt, ko6, color="#4a3aa7", lw=0.5)
    axs[3].axhline(args.excl_r, color="#eb6834", lw=0.8, ls=":")
    axs[3].set_ylabel("min d(K+, O6 core) (nm)"); axs[3].set_ylim(0, max(1.0, float(np.percentile(ko6, 99))))
    axs[4].plot(tt, rg, color="#52514e", lw=0.6); axs[4].set_ylabel("Rg DNA (nm)")
    axs[4].set_xlabel("t (ns; equilibratura < 0)")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.3)
        a.axvline(0, color="grey", lw=0.8, ls="--")
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150)
    print(f"\n[DONE] {args.out}.npz, {args.out}.png")


if __name__ == "__main__":
    main()
