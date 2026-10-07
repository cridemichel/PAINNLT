#!/usr/bin/env python3
"""TEL26: tetradi intatte in funzione del tempo e della temperatura (CG e AA mappato).

Per ogni (frame, copia) e ogni tetrade k si calcola la frazione liscia di contatti
B3-B3 nativi sulle 6 coppie della tetrade,
    q_k = < 1 / (1 + exp(beta (r - lam r_nat))) >,   beta 25 nm^-1, lam 1.2
con r_nat le mediane del riferimento NATIVO a 300 K (tel26_lp2_dataset.bin, 10 copie,
tetrade 3 intatta).  La tetrade e' intatta se q_k > QCUT (0.5).  Stati:
    n_int = numero di tetradi intatte (0..3);  F = 3 intatte, U = nessuna intatta.
Per ogni corsa: q_k medio e frazione intatta per tetrade, P(n_int), Rg, tempo del primo
evento con n_int <= 1, tutto sulla finestra --window (default seconda meta').

Uso (Leonardo, dalla cartella tel26_unfold):
  python3 08_tetrad_melt.py 300=$C/pri_T300_long_r0.samples.npz+$C/pri_T300_long_r1.samples.npz \\
      400=... aa400=@bin:$A/cg/tel26_lp2_1c_t400_dataset.bin --plot $C/melt_pri --json $C/melt_pri.json
Etichette numeriche = temperatura in K (vanno nel grafico P(F), P(U) contro T); le altre
etichette compaiono solo nelle tabelle e nelle serie temporali.  '@bin:' legge un dataset
CG mappato (tempo = indice x --bin-dt ps).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
T26 = HERE.parent / "tel26"
sys.path.insert(0, str(HERE.parent / "tel22" / "diagnostics" / "scripts"))
import _tel22_cv as cv  # noqa: E402
sys.path.insert(0, str(HERE.parent))
from system_config import load_system  # noqa: E402

SYS = load_system(T26)
NUC = SYS.nuc
TETRADS = [[r - 1 for r in t] for t in SYS.tetrads]
B3 = SYS.hoogsteen_index
PAIRS = [[(t[a], t[b]) for a in range(4) for b in range(a + 1, 4)] for t in TETRADS]


def mi(d, L):
    return d - L * np.round(d / L)


def b3_pair_distances(S, L, ncopy):
    """S (T, M, 6, 3) -> (T*ncopy, 3, 6) distanze B3-B3 per tetrade, con immagine minima."""
    T = S.shape[0]
    X = S.reshape(T, ncopy, NUC, S.shape[2], 3)[:, :, :, B3, :]          # (T, C, NUC, 3)
    Lb = np.asarray(L, float)
    Lb = Lb.reshape(T, 1, 1, 3) if Lb.ndim == 2 else Lb.reshape(1, 1, 1, 3)
    out = np.empty((T, ncopy, len(PAIRS), 6))
    for k, pk in enumerate(PAIRS):
        i = np.array([a for a, _ in pk]); j = np.array([b for _, b in pk])
        out[:, :, k, :] = np.linalg.norm(mi(X[:, :, i] - X[:, :, j], Lb), axis=-1)
    return out.reshape(T * ncopy, len(PAIRS), 6)


def radius_of_gyration(S, L, ncopy):
    """Rg della copia (siti non pesati), copie ricomposte lungo la catena."""
    T = S.shape[0]
    X = S.reshape(T, ncopy, NUC, S.shape[2], 3)
    Lb = np.asarray(L, float)
    Lb = Lb.reshape(T, 1, 1, 3) if Lb.ndim == 2 else Lb.reshape(1, 1, 1, 3)
    anc = X[:, :, :, 0, :]
    steps = mi(np.diff(anc, axis=2), Lb)
    anc_u = np.concatenate([anc[:, :, :1], anc[:, :, :1] + np.cumsum(steps, axis=2)], axis=2)
    Xu = anc_u[:, :, :, None, :] + mi(X - anc[:, :, :, None, :], Lb[..., None, :])
    Xu = Xu.reshape(T * ncopy, -1, 3)
    ok = np.isfinite(Xu).all(axis=-1)
    c = np.nansum(np.where(ok[..., None], Xu, 0), axis=1) / ok.sum(axis=1)[:, None]
    d2 = np.where(ok, ((Xu - c[:, None]) ** 2).sum(-1), 0.0)
    return np.sqrt(d2.sum(axis=1) / ok.sum(axis=1))


def load(spec, bin_dt):
    """-> lista di (S, L, ncopy, t_ps) per replica ('+' separa le repliche)."""
    out = []
    for part in spec.split("+"):
        if part.startswith("@bin:"):
            S, L, nc = cv.load_reference(part[5:])
            out.append((S, L, nc, np.arange(S.shape[0]) * bin_dt))
        else:
            S, L, nc, t = cv.load_samples(part)
            out.append((S, L, nc, np.asarray(t, float)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="etichetta=file.npz[+file2.npz] oppure etichetta=@bin:dataset.bin")
    ap.add_argument("--native", default=str(T26 / "tel26_lp2_dataset.bin"),
                    help="dataset nativo a 300 K per le distanze r_nat")
    ap.add_argument("--native-stride", type=int, default=5)
    ap.add_argument("--window", default="0.5:1", help="frazione della corsa su cui fare le medie (A:B)")
    ap.add_argument("--qcut", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=25.0)
    ap.add_argument("--lam", type=float, default=1.2)
    ap.add_argument("--bin-dt", type=float, default=10.0, help="ps fra frame di un '@bin:'")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cv.set_nuc(NUC)

    Sn, Ln, ncn = cv.load_reference(args.native)
    dn = b3_pair_distances(Sn[::args.native_stride], Ln[::args.native_stride], ncn)
    r_nat = np.nanmedian(dn, axis=0)                                            # (3, 6)
    print(f"  r_nat da {args.native} ({dn.shape[0]} configurazioni): "
          + "  ".join(f"t{k + 1} {r_nat[k].min():.3f}-{r_nat[k].max():.3f}" for k in range(3)) + " nm")
    w0, w1 = (float(x) for x in args.window.split(":"))

    res, series = {}, {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        qs, ns, rgs, first_unf, ts = [], [], [], [], []
        for S, L, nc, t in load(path, args.bin_dt):
            d = b3_pair_distances(S, L, nc)
            q = (1.0 / (1.0 + np.exp(args.beta * (d - args.lam * r_nat[None])))).mean(axis=2)  # (T*C, 3)
            q = q.reshape(S.shape[0], nc, 3)
            n_int = (q > args.qcut).sum(axis=2)                                   # (T, C)
            rg = radius_of_gyration(S, L, nc).reshape(S.shape[0], nc)
            a, b = int(round(w0 * S.shape[0])), int(round(w1 * S.shape[0]))
            qs.append(q[a:b].reshape(-1, 3)); ns.append(n_int[a:b].ravel()); rgs.append(rg[a:b].ravel())
            for c in range(nc):
                hit = np.flatnonzero(n_int[:, c] <= 1)
                first_unf.append(float(t[hit[0]]) if hit.size else None)
            ts.append((t, q.mean(axis=1), n_int.mean(axis=1)))
        q = np.concatenate(qs); n = np.concatenate(ns); rg = np.concatenate(rgs)
        p_n = [float((n == k).mean()) for k in range(4)]
        res[label] = {"q_mean": q.mean(axis=0).tolist(), "intact_frac": (q > args.qcut).mean(axis=0).tolist(),
                      "P_n_intact": p_n, "P_F": p_n[3], "P_U": p_n[0], "Rg_mean": float(rg.mean()),
                      "Rg_std": float(rg.std()), "first_n_le_1_ps": first_unf,
                      "n_conf": int(q.shape[0]), "t_max_ps": [float(s[0][-1]) for s in ts]}
        series[label] = ts

    print(f"\n  medie su {args.window} di ogni replica; intatta se q_k > {args.qcut}")
    print(f"  {'corsa':>8s} {'t_max(ns)':>9s} {'q1':>6s} {'q2':>6s} {'q3':>6s}   {'int1':>5s} {'int2':>5s} {'int3':>5s}"
          f"   {'P(0)':>5s} {'P(1)':>5s} {'P(2)':>5s} {'P(3)':>5s}   {'Rg':>6s}  primo n<=1 (ns)")
    for lab, r in res.items():
        fu = ", ".join("-" if x is None else f"{x / 1000:.1f}" for x in r["first_n_le_1_ps"])
        print(f"  {lab:>8s} {max(r['t_max_ps']) / 1000:9.1f} " + " ".join(f"{x:6.3f}" for x in r["q_mean"]) + "   "
              + " ".join(f"{x:5.2f}" for x in r["intact_frac"]) + "   "
              + " ".join(f"{x:5.2f}" for x in r["P_n_intact"]) + f"   {r['Rg_mean']:6.3f}  {fu}")

    if args.json:
        json.dump({"native": args.native, "window": args.window, "qcut": args.qcut,
                   "r_nat_nm": r_nat.tolist(), "runs": res}, open(args.json, "w"), indent=1)
        print(f"  json: {args.json}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labs = list(series)
        fig, axes = plt.subplots(len(labs) + 1, 1, figsize=(10, 2.1 * (len(labs) + 1)), squeeze=False)
        for ax, lab in zip(axes[:-1, 0], labs):
            for i, (t, qm, nm) in enumerate(series[lab]):
                for k in range(3):
                    ax.plot(t / 1000, qm[:, k], lw=0.6, color=f"C{k}", alpha=0.8 if i == 0 else 0.45,
                            label=f"tetrade {k + 1}" if i == 0 else None)
            ax.axhline(args.qcut, color="k", lw=0.5, ls=":")
            ax.set_ylim(0, 1.02); ax.set_ylabel(f"q_k  [{lab}]")
            ax.legend(loc="lower left", fontsize=7, ncol=3)
        axes[-2, 0].set_xlabel("t (ns)")
        temps = sorted((float(l), l) for l in labs if l.replace(".", "", 1).isdigit())
        ax = axes[-1, 0]
        if temps:
            x = [T for T, _ in temps]
            ax.plot(x, [res[l]["P_F"] for _, l in temps], "o-", label="P(F): 3 tetradi")
            ax.plot(x, [res[l]["P_U"] for _, l in temps], "s-", label="P(U): nessuna tetrade")
            for k in range(3):
                ax.plot(x, [res[l]["intact_frac"][k] for _, l in temps], ":", marker=".", color=f"C{k + 2}",
                        label=f"tetrade {k + 1} intatta")
            ax.axvline(273.15 + 55.3, color="gray", ls="--", lw=0.8, label="Tm sper. (U 50%)")
            ax.set_xlabel("T (K)"); ax.set_ylabel("frazione"); ax.set_ylim(-0.02, 1.02)
            ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(args.plot + ".png", dpi=130)
        print(f"  grafico: {args.plot}.png")


if __name__ == "__main__":
    main()
