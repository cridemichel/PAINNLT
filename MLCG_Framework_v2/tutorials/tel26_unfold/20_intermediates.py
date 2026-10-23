#!/usr/bin/env python3
"""Gli intermedi della fusione CG sono stati popolati (metastabili) o solo di passaggio?

PERCHE' (23/10)
    La curva di fusione di pop1_it45 (17) mostra, fra F (3 tetradi) e U (0), due livelli
    intermedi: 2 tetradi (~320 K, quasi sempre senza la tetrade 1) e solo la tetrade 2
    centrale (324-328 K).  Con la soglia unica q > 0.5 su q mediato su 1 ns, pero', uno
    sfilacciamento graduale in cui la tetrade centrale cede per ultima darebbe le stesse
    popolazioni.  Qui si distingue con tre criteri, corsa per corsa:
      1. distribuzione di q_k (bacini separati o una sola campana che si sposta) e mappa
         -ln P(q2, (q1+q3)/2) con q mediato su --smooth-ns (default 0.1 ns);
      2. assegnazione a isteresi (core set): la tetrade k diventa intatta quando q_k > --hi
         (0.6) e rotta quando q_k < --lo (0.4), altrimenti tiene lo stato precedente.  Ogni
         frame ha un livello n = 0..3 e una configurazione (quali tetradi); si misurano
         numero di visite e tempi di permanenza (media, mediana, massimo);
      3. frazione "pulita" di ogni livello: frame del livello con tutti i q_k fuori da
         [lo, hi].  Uno stato popolato vive soprattutto nel suo bacino (frazione alta,
         permanenze lunghe rispetto al tempo di passaggio); uno stato di transizione vive
         nella zona ambigua e dura poco.

USO (Leonardo, con l'ambiente caricato, dalla cartella tel26_unfold)
    MP=$A/melt_painn/pop1_it45
    python3 20_intermediates.py 320=$MP/chiuso/T320.samples.npz+$MP/aperto/T320.samples.npz \\
        324=$MP/chiuso/T324.samples.npz+$MP/aperto/T324.samples.npz ... \\
        --skip-ns 2 --plot $MP/intermediates --json $MP/intermediates.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("tetrad_melt", HERE / "08_tetrad_melt.py")
melt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(melt)
cv = melt.cv
NAMES = {(1, 1, 1): "123", (0, 1, 1): "23", (1, 1, 0): "12", (1, 0, 1): "13",
         (0, 1, 0): "2", (1, 0, 0): "1", (0, 0, 1): "3", (0, 0, 0): "-"}


def smooth(q, w):
    """media mobile centrata su w frame lungo l'asse 0, corretta ai bordi"""
    if w <= 1:
        return q
    k = np.ones(w) / w
    norm = np.convolve(np.ones(len(q)), k, mode="same")
    return np.stack([np.convolve(q[:, j], k, mode="same") / norm for j in range(q.shape[1])], 1)


def core_assign(q, lo, hi):
    st = q[0] > 0.5 * (lo + hi)
    out = np.empty(q.shape, bool)
    for t in range(len(q)):
        st = np.where(q[t] > hi, True, np.where(q[t] < lo, False, st))
        out[t] = st
    return out


def segments(lab):
    """etichette intere -> lista di (etichetta, lunghezza in frame)"""
    cut = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(lab)]
    return [(int(lab[s]), int(e - s)) for s, e in zip(starts, ends)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="etichetta=file.npz[+file2.npz] (repliche/rami separati da '+')")
    ap.add_argument("--native", default=str(HERE.parent / "tel26" / "tel26_lp2_dataset.bin"))
    ap.add_argument("--native-stride", type=int, default=5)
    ap.add_argument("--smooth-ns", type=float, default=0.1)
    ap.add_argument("--lo", type=float, default=0.4)
    ap.add_argument("--hi", type=float, default=0.6)
    ap.add_argument("--skip-ns", type=float, default=2.0, help="scarta l'inizio di ogni corsa")
    ap.add_argument("--beta", type=float, default=25.0)
    ap.add_argument("--lam", type=float, default=1.2)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cv.set_nuc(melt.NUC)

    Sn, Ln, ncn = cv.load_reference(args.native)
    dn = melt.b3_pair_distances(Sn[::args.native_stride], Ln[::args.native_stride], ncn)
    r_nat = np.nanmedian(dn, axis=0)

    res, maps = {}, {}
    for spec in args.runs:
        label, paths = spec.split("=", 1)
        qs, levels, pats, clean, seglist = [], [], [], [], []
        dt_ns = None
        for path in paths.split("+"):
            S, L, nc, t = cv.load_samples(path)
            t = np.asarray(t, float)
            if nc != 1:
                raise SystemExit(f"[ERROR] {path}: {nc} copie, serve una copia per corsa")
            d = melt.b3_pair_distances(S, L, nc)
            q = (1.0 / (1.0 + np.exp(args.beta * (d - args.lam * r_nat[None])))).mean(axis=2)   # (T, 3)
            dt = float(np.median(np.diff(t)))
            dt_ns = dt / 1000.0
            q = smooth(q, max(1, int(round(args.smooth_ns * 1000.0 / dt))))
            keep = t >= t[0] + args.skip_ns * 1000.0
            q = q[keep]
            intact = core_assign(q, args.lo, args.hi)
            lev = intact.sum(1)
            pat = np.array([NAMES[tuple(int(x) for x in row)] for row in intact])
            qs.append(q); levels.append(lev); pats.append(pat)
            clean.append(((q < args.lo) | (q > args.hi)).all(1))
            seglist.append(segments(lev))
        q = np.concatenate(qs); lev = np.concatenate(levels); pat = np.concatenate(pats)
        cln = np.concatenate(clean)
        r = {"frames": int(len(q)), "dt_ns": dt_ns, "levels": {}, "patterns": {}}
        for n in range(4):
            m = lev == n
            segs = [ln * dt_ns for sl in seglist for (lv, ln) in sl if lv == n]
            r["levels"][n] = {"P": float(m.mean()), "clean_frac": float(cln[m].mean()) if m.any() else None,
                              "visits": len(segs), "dwell_mean_ns": float(np.mean(segs)) if segs else None,
                              "dwell_median_ns": float(np.median(segs)) if segs else None,
                              "dwell_max_ns": float(np.max(segs)) if segs else None}
        for name in sorted(set(pat), key=lambda s: (-len(s.strip("-")), s)):
            r["patterns"][name] = float((pat == name).mean())
        # transizioni fra livelli (conteggi), concatenando le corse senza unire i bordi
        tm = np.zeros((4, 4), int)
        for sl in seglist:
            for (a, _), (b, _) in zip(sl, sl[1:]):
                tm[a, b] += 1
        r["transitions"] = tm.tolist()
        r["q_hist"] = [np.histogram(q[:, k], bins=40, range=(0, 1))[0].tolist() for k in range(3)]
        res[label] = r
        maps[label] = q

        print(f"\n=== {label}: {len(q)} frame ({len(q) * dt_ns:.1f} ns), q mediato su {args.smooth_ns} ns, "
              f"core set q < {args.lo} rotta / q > {args.hi} intatta")
        print(f"  {'livello':>8s} {'P':>6s} {'pulita':>7s} {'visite':>7s} {'perm. media':>12s} {'mediana':>8s} {'max':>7s}  (ns)")
        for n in (3, 2, 1, 0):
            v = r["levels"][n]
            if v["visits"] == 0:
                print(f"  {n:>8d} {0:6.3f}       -       0")
                continue
            print(f"  {n:>8d} {v['P']:6.3f} {v['clean_frac']:7.2f} {v['visits']:7d} {v['dwell_mean_ns']:12.2f} "
                  f"{v['dwell_median_ns']:8.2f} {v['dwell_max_ns']:7.2f}")
        print("  configurazioni (tetradi intatte): " + "  ".join(f"{k} {v:.3f}" for k, v in r["patterns"].items() if v >= 0.005))
        print("  transizioni n -> n' (righe 0..3, colonne 0..3): " + "  ".join(" ".join(str(x) for x in row) for row in tm))

    if args.json:
        json.dump({"smooth_ns": args.smooth_ns, "lo": args.lo, "hi": args.hi, "skip_ns": args.skip_ns,
                   "runs": res}, open(args.json, "w"), indent=1)
        print(f"\n  json: {args.json}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labs = list(maps)
        fig, axes = plt.subplots(2, len(labs), figsize=(3.3 * len(labs), 6.4), squeeze=False)
        cols = ["#2a78d6", "#1baf7a", "#eb6834"]
        for j, lab in enumerate(labs):
            q = maps[lab]
            ax = axes[0, j]
            for k in range(3):
                h, e = np.histogram(q[:, k], bins=40, range=(0, 1), density=True)
                ax.plot(0.5 * (e[1:] + e[:-1]), h, color=cols[k], lw=1.4, label=f"tetrade {k + 1}")
            ax.axvspan(args.lo, args.hi, color="0.9", zorder=0)
            ax.set(title=f"{lab} K", xlabel="q_k", ylabel="densita'" if j == 0 else None)
            if j == 0:
                ax.legend(fontsize=7, frameon=False)
            ax = axes[1, j]
            H, xe, ye = np.histogram2d(q[:, 1], 0.5 * (q[:, 0] + q[:, 2]), bins=30, range=[[0, 1], [0, 1]])
            with np.errstate(divide="ignore"):
                F = -np.log(H.T / H.sum())
            F -= np.nanmin(F[np.isfinite(F)])
            im = ax.imshow(np.where(np.isfinite(F), F, np.nan), origin="lower", extent=[0, 1, 0, 1],
                           vmin=0, vmax=8, cmap="viridis", aspect="auto")
            ax.set(xlabel="q2 (centrale)", ylabel="(q1+q3)/2" if j == 0 else None)
        fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8, label="-ln P (kT)")
        fig.savefig(args.plot + ".png", dpi=150, bbox_inches="tight")
        print(f"  grafico: {args.plot}.png")


if __name__ == "__main__":
    main()
