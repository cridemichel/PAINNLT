#!/usr/bin/env python3
"""Stati di Buscaglia nel CG: quadruplex, tripla elica (un tratto G fuori) o allentamento uniforme?

PERCHE' (23/10)
    Buscaglia et al. 2013 (2JPZ, 25 mM KCl): F <-> I1 <-> I2 <-> U, con I2 tripla elica G-rica
    (un tratto G esce, gli altri tre formano tripletti) e transizioni cooperative (dH ~150-170
    kJ/mol).  Nella fusione CG di pop1_it45 a 324-328 K le tre tetradi stanno a q ~0.5 (20):
    e' quello che darebbe una tripla elica (ogni tetrade diventa una triade: 3 coppie su 6),
    ma anche un allentamento uniforme.  Il parametro per tetrade non li distingue; quello per
    tratto si'.

PARAMETRI (per frame, contatti B3-B3 nativi lisci come in 08, mediati su --smooth-ns)
    Tratti G: A = G4-6, B = G10-12, C = G16-18, D = G22-24 (la posizione j di ogni tetrade).
    c_a     = frazione formata delle 9 coppie fra il tratto a e gli altri (3 tetradi x 3 partner);
    Qr_a    = frazione formata delle 9 coppie fra gli altri tre tratti;
    a*      = argmin c_a (il tratto piu' "fuori"); si usa (c_a*, Qr_a*).
    Regioni: F  c_a* > --hi;  T (tripla elica, I2) c_a* < --lo e Qr_a* > --hi;
             U  Qr_a* < --lo (nessun tratto appaiato);  altrimenti zona ambigua.
    Assegnazione a isteresi (core set): nella zona ambigua il frame tiene lo stato precedente.
    Per stato: popolazione, frazione "pulita" (frame dentro la regione), visite e permanenze;
    nella tripla elica, quale tratto e' fuori.  Mappa -ln P(c_a*, Qr_a*) per T: F in alto a
    destra, tripla elica in alto a sinistra, U in basso a sinistra, allentamento al centro.

USO (Leonardo, ambiente caricato)
    python3 21_gtracts.py 320=$MP/chiuso/T320.samples.npz+$MP/aperto/T320.samples.npz ... \\
        --skip-ns 2 --plot $MP/gtracts --json $MP/gtracts.json
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
POS_PAIRS = [(a, b) for a in range(4) for b in range(a + 1, 4)]     # ordine delle 6 coppie di 08
TRACTS = "ABCD"
STATES = ["F", "T", "U"]


def smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    norm = np.convolve(np.ones(len(x)), k, mode="same")
    flat = x.reshape(len(x), -1)
    out = np.stack([np.convolve(flat[:, j], k, mode="same") / norm for j in range(flat.shape[1])], 1)
    return out.reshape(x.shape)


def segments(lab):
    cut = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(lab)]
    return [(int(lab[s]), int(e - s)) for s, e in zip(starts, ends)]


def tract_params(s):
    """s (T, 3, 6) contatti per tetrade e coppia -> c (T, 4), Qr (T, 4), Qall (T,)"""
    inv = np.array([[a in p for p in POS_PAIRS] for a in range(4)])            # (4, 6)
    c = np.stack([s[:, :, inv[a]].mean(axis=(1, 2)) for a in range(4)], 1)
    qr = np.stack([s[:, :, ~inv[a]].mean(axis=(1, 2)) for a in range(4)], 1)
    return c, qr, s.mean(axis=(1, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="etichetta=file.npz[+file2.npz]")
    ap.add_argument("--native", default=str(HERE.parent / "tel26" / "tel26_lp2_dataset.bin"))
    ap.add_argument("--native-stride", type=int, default=5)
    ap.add_argument("--smooth-ns", type=float, default=0.1)
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--hi", type=float, default=0.6)
    ap.add_argument("--skip-ns", type=float, default=2.0)
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
        X, Y, labs, cln, outs, seglist, cs = [], [], [], [], [], [], []
        dt_ns = None
        for path in paths.split("+"):
            S, L, nc, t = cv.load_samples(path)
            t = np.asarray(t, float)
            if nc != 1:
                raise SystemExit(f"[ERROR] {path}: {nc} copie, serve una copia per corsa")
            d = melt.b3_pair_distances(S, L, nc)                                  # (T, 3, 6)
            s = 1.0 / (1.0 + np.exp(args.beta * (d - args.lam * r_nat[None])))
            dt = float(np.median(np.diff(t)))
            dt_ns = dt / 1000.0
            s = smooth(s, max(1, int(round(args.smooth_ns * 1000.0 / dt))))
            s = s[t >= t[0] + args.skip_ns * 1000.0]
            c, qr, _ = tract_params(s)
            a = c.argmin(1)
            x = c[np.arange(len(c)), a]
            y = qr[np.arange(len(c)), a]
            region = np.full(len(x), -1)
            region[x > args.hi] = 0
            region[(x < args.lo) & (y > args.hi)] = 1
            region[(y < args.lo)] = 2
            st = np.empty(len(x), int)
            cur = region[0] if region[0] >= 0 else (0 if x[0] > 0.5 else 2)
            for i, r in enumerate(region):
                if r >= 0:
                    cur = r
                st[i] = cur
            X.append(x); Y.append(y); labs.append(st); cln.append(region == st); outs.append(a)
            seglist.append(segments(st)); cs.append(c)
        x = np.concatenate(X); y = np.concatenate(Y); st = np.concatenate(labs)
        clean = np.concatenate(cln); out = np.concatenate(outs); c = np.concatenate(cs)
        r = {"frames": int(len(x)), "states": {}, "c_mean": c.mean(0).tolist()}
        for k, name in enumerate(STATES):
            m = st == k
            segs = [ln * dt_ns for sl in seglist for (lv, ln) in sl if lv == k]
            r["states"][name] = {"P": float(m.mean()), "clean_frac": float(clean[m].mean()) if m.any() else None,
                                 "visits": len(segs), "dwell_mean_ns": float(np.mean(segs)) if segs else None,
                                 "dwell_median_ns": float(np.median(segs)) if segs else None,
                                 "dwell_max_ns": float(np.max(segs)) if segs else None}
        mt = st == 1
        r["triplex_tract_out"] = {TRACTS[j]: float((out[mt] == j).mean()) if mt.any() else 0.0 for j in range(4)}
        r["ambiguous_frac"] = float((~clean).mean())
        tm = np.zeros((3, 3), int)
        for sl in seglist:
            for (p, _), (q, _) in zip(sl, sl[1:]):
                tm[p, q] += 1
        r["transitions"] = tm.tolist()
        res[label] = r
        maps[label] = (x, y)

        print(f"\n=== {label}: {len(x)} frame, contatti mediati su {args.smooth_ns} ns; "
              f"F: c* > {args.hi}  T: c* < {args.lo} e Qr* > {args.hi}  U: Qr* < {args.lo}")
        print(f"  c medio per tratto (A G4-6, B G10-12, C G16-18, D G22-24): "
              + "  ".join(f"{TRACTS[j]} {v:.2f}" for j, v in enumerate(r['c_mean']))
              + f";  frame in zona ambigua {r['ambiguous_frac']:.2f}")
        print(f"  {'stato':>6s} {'P':>6s} {'pulita':>7s} {'visite':>7s} {'perm. media':>12s} {'mediana':>8s} {'max':>7s}  (ns)")
        for name in STATES:
            v = r["states"][name]
            if not v["visits"]:
                print(f"  {name:>6s} {0:6.3f}       -       0")
                continue
            print(f"  {name:>6s} {v['P']:6.3f} {v['clean_frac']:7.2f} {v['visits']:7d} {v['dwell_mean_ns']:12.2f} "
                  f"{v['dwell_median_ns']:8.2f} {v['dwell_max_ns']:7.2f}")
        if mt.any():
            print("  tripla elica, tratto fuori: " + "  ".join(f"{k} {v:.2f}" for k, v in r["triplex_tract_out"].items()))
        print("  transizioni (righe/colonne F T U): " + "  ".join(" ".join(str(v) for v in row) for row in tm))

    if args.json:
        json.dump({"smooth_ns": args.smooth_ns, "lo": args.lo, "hi": args.hi, "skip_ns": args.skip_ns,
                   "runs": res}, open(args.json, "w"), indent=1)
        print(f"\n  json: {args.json}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labs = list(maps)
        n = len(labs)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
        for j, lab in enumerate(labs):
            x, y = maps[lab]
            H, _, _ = np.histogram2d(x, y, bins=30, range=[[0, 1], [0, 1]])
            with np.errstate(divide="ignore"):
                F = -np.log(H.T / H.sum())
            F -= np.nanmin(F[np.isfinite(F)])
            ax = axes[0, j]
            im = ax.imshow(np.where(np.isfinite(F), F, np.nan), origin="lower", extent=[0, 1, 0, 1],
                           vmin=0, vmax=8, cmap="viridis", aspect="equal")
            kw = dict(fill=False, lw=0.9, ls="--")
            ax.add_patch(plt.Rectangle((args.hi, 0), 1 - args.hi, 1, ec="#2a78d6", **kw))
            ax.add_patch(plt.Rectangle((0, args.hi), args.lo, 1 - args.hi, ec="#eda100", **kw))
            ax.add_patch(plt.Rectangle((0, 0), 1, args.lo, ec="#eb6834", **kw))
            ax.text(0.98, 0.03, "F", color="#2a78d6", ha="right", fontsize=8)
            ax.text(0.02, 0.97, "tripla", color="#eda100", va="top", fontsize=8)
            ax.text(0.02, 0.03, "U", color="#eb6834", fontsize=8)
            ax.set(title=f"{lab} K", xlabel="c* (tratto meno appaiato)", ylabel="Qr* (altri tre)" if j == 0 else None)
        fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.85, label="-ln P (kT)")
        fig.savefig(args.plot + ".png", dpi=150, bbox_inches="tight")
        print(f"  grafico: {args.plot}.png")


if __name__ == "__main__":
    main()
