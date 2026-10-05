#!/usr/bin/env python3
"""TEL26: distanze dei contatti dei loop, AA contro CG.

PERCHE'
    Gli script 50-51 indicano che al CG mancano pochi contatti specifici a
    lungo raggio: T1 con A15 e con A21 (coda 5' agganciata ai loop 2 e 3) e il
    loop 3 ripiegato nel solco (T20 con G18/G22, A21 con G16/G17).  Prima di
    aggiungerli come Morse pair-specific nei prior servono numeri: distanza di
    contatto, frazione di tempo in contatto nell'AA con la sua incertezza
    (dispersione fra copie, che nell'AA non sono ergodiche), e quanto il CG ne
    e' lontano.  In piu' una scansione di tutte le coppie non adiacenti con
    almeno un residuo di loop, ordinate per differenza AA-CG nella frazione di
    contatto, per non dipendere dalla lista scelta a mano.

DEFINIZIONI
    Posizione di un residuo: baricentro dei siti di base (B1-B5) per le G,
    unico sito per T e A.  Contatto: distanza < --rc (0,70 nm).  Per ogni
    coppia: percentili 5/50/95, picco dell'istogramma sotto 1 nm (stima di r0),
    frazione di contatto totale e per copia, errore = std fra copie / sqrt(copie).

USO (in tutorials/tel26)
    python3 52_tel26_loop_contacts.py tel26_lp1_dataset.bin \\
        prod=r0_s01.npz+r0_s02.npz,r1_s01.npz+r1_s02.npz \\
        [--pairs 1-15,1-21,20-18,20-22,21-16,21-17] [--rc 0.7] [--top 15] \\
        [--png loop_contacts.png] [--json loop_contacts.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("s47", HERE / "47_tel26_timescales.py")
s47 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s47)
s46 = s47.s46
cv = s47.cv

DEFAULT_PAIRS = "1-15,1-21,20-18,20-22,21-16,21-17"


def residue_positions(parts, nuc):
    """parts -> (T, C, nuc, 3): baricentro dei siti di base (G) o unico sito (T, A)."""
    T = min(p[0].shape[0] for p in parts)
    out = []
    for S, L, nc, _t in parts:
        L = L[:T] if np.ndim(L) == 2 else L
        X = s46.unwrap_copies(S[:T], L, nc, nuc)                   # (T, C, R, 6, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            B = np.nanmean(X[:, :, :, 1:, :], axis=3)
        miss = ~np.isfinite(B).all(-1)
        B[miss] = X[:, :, :, 0, :][miss]
        out.append(B.astype(np.float32))
    return np.concatenate(out, axis=1)


def pair_stats(d, rc):
    """d (T, C) distanze -> statistiche."""
    flat = d.ravel()
    per_copy = (d < rc).mean(axis=0)
    h, e = np.histogram(flat, bins=np.arange(0.3, 2.51, 0.02))
    c = 0.5 * (e[1:] + e[:-1])
    below = c < 1.0
    peak = float(c[below][np.argmax(h[below])]) if h[below].sum() > 0.01 * h.sum() else float("nan")
    return {"p5": float(np.percentile(flat, 5)), "p50": float(np.percentile(flat, 50)),
            "p95": float(np.percentile(flat, 95)), "peak_lt1nm": peak,
            "f_contact": float((flat < rc).mean()), "f_per_copy": per_copy.tolist(),
            "f_err": float(per_copy.std(ddof=1) / np.sqrt(len(per_copy))) if len(per_copy) > 1 else float("nan"),
            "hist": h.tolist(), "edges": e.tolist()}


def name(r1):
    return f"{s46.SEQUENCE[r1 - 1]}{r1}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("runs", nargs="*", help="etichetta=a.npz[+b.npz][,r1.npz...]")
    ap.add_argument("--nuc", type=int, default=26)
    ap.add_argument("--aa-dt", type=float, default=20.0)
    ap.add_argument("--aa-stride", type=int, default=1)
    ap.add_argument("--cg-stride", type=int, default=5)
    ap.add_argument("--pairs", default=DEFAULT_PAIRS, help="coppie di residui 1-based, 'i-j,...'")
    ap.add_argument("--rc", type=float, default=0.70, help="nm, soglia di contatto")
    ap.add_argument("--top", type=int, default=15, help="coppie mostrate nella scansione")
    ap.add_argument("--png", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cv.set_nuc(args.nuc)
    nuc = args.nuc
    pairs = [tuple(int(x) for x in p.split("-")) for p in args.pairs.split(",") if p]

    S, L, nc, t = s47.load_aa(args.dataset, args.aa_dt, args.aa_stride)
    runs = {"AA": residue_positions([(S, L, nc, t)], nuc)}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        runs[label] = residue_positions([s47.load_cg(rep, args.cg_stride) for rep in path.split(",")], nuc)
    labels = list(runs)
    for lab, B in runs.items():
        print(f"  {lab}: {B.shape[0]} frame x {B.shape[1]} copie")

    # ── coppie scelte ──
    res = {"rc": args.rc, "pairs": {}, "scan": []}
    print(f"\n  contatti scelti (contatto: distanza < {args.rc:.2f} nm; distanze in nm)")
    hdr = f"  {'coppia':<10s}{'corsa':<8s}{'p5':>7s}{'p50':>7s}{'p95':>7s}{'picco<1':>9s}{'f contatto':>14s}   per copia"
    print(hdr)
    for i, j in pairs:
        key = f"{name(i)}-{name(j)}"
        res["pairs"][key] = {}
        for lab in labels:
            B = runs[lab]
            d = np.linalg.norm(B[:, :, i - 1] - B[:, :, j - 1], axis=-1)
            st = pair_stats(d, args.rc)
            res["pairs"][key][lab] = st
            pc = " ".join(f"{v:.2f}" for v in st["f_per_copy"]) if lab == "AA" else \
                f"(min {min(st['f_per_copy']):.2f}, max {max(st['f_per_copy']):.2f})"
            err = f"±{st['f_err']:.3f}" if np.isfinite(st["f_err"]) else ""
            print(f"  {key if lab == labels[0] else '':<10s}{lab:<8s}{st['p5']:7.2f}{st['p50']:7.2f}{st['p95']:7.2f}"
                  f"{st['peak_lt1nm']:9.2f}{st['f_contact']:8.3f}{err:>6s}   {pc}")

    # ── scansione di tutte le coppie con almeno un residuo di loop ──
    loop = [r for r in range(1, nuc + 1) if s46.SEQUENCE[r - 1] != "G"]
    cand = [(i, j) for i in range(1, nuc + 1) for j in range(i + 2, nuc + 1) if i in loop or j in loop]
    if len(labels) > 1:
        cg = labels[1]
        rows = []
        for i, j in cand:
            fa = (np.linalg.norm(runs["AA"][:, :, i - 1] - runs["AA"][:, :, j - 1], axis=-1) < args.rc)
            fc = (np.linalg.norm(runs[cg][:, :, i - 1] - runs[cg][:, :, j - 1], axis=-1) < args.rc)
            pa = fa.mean(axis=0)
            rows.append((abs(pa.mean() - fc.mean()), i, j, float(pa.mean()), float(pa.std(ddof=1) / np.sqrt(len(pa))),
                         int((pa > 0.5).sum()), float(fc.mean())))
        rows.sort(reverse=True)
        print(f"\n  scansione: coppie non adiacenti con un residuo di loop, ordinate per |f_AA - f_{cg}| (rc {args.rc:.2f} nm)")
        print(f"  {'coppia':<10s}{'f AA':>8s}{'err':>8s}{'copie AA >0,5':>15s}{'f ' + cg:>10s}{'differenza':>12s}")
        for diff, i, j, fa, ea, nmaj, fc in rows[:args.top]:
            print(f"  {name(i) + '-' + name(j):<10s}{fa:8.3f}{ea:8.3f}{nmaj:>15d}{fc:10.3f}{fa - fc:+12.3f}")
            res["scan"].append({"pair": f"{name(i)}-{name(j)}", "f_aa": fa, "f_aa_err": ea,
                                "copies_aa_majority": nmaj, "f_cg": fc})

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(pairs)
        cols = min(3, n)
        rows_ = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows_, cols, figsize=(4.2 * cols, 3.0 * rows_), squeeze=False)
        for ax, (i, j) in zip(axs.ravel(), pairs):
            key = f"{name(i)}-{name(j)}"
            for lab in labels:
                st = res["pairs"][key][lab]
                e = np.asarray(st["edges"]); h = np.asarray(st["hist"], float)
                if h.sum() == 0:
                    continue
                ax.plot(0.5 * (e[1:] + e[:-1]), h / h.sum() / np.diff(e), label=lab)
            ax.axvline(args.rc, color="grey", lw=0.8, ls="--")
            ax.set(title=key, xlabel="distanza (nm)", ylabel="densita'")
            ax.legend(fontsize=8)
        for ax in axs.ravel()[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.png, dpi=140)
        print(f"\n  grafico: {args.png}")

    print("\n  Lettura:")
    print("  - f contatto AA con errore grande e per copia tutto-o-niente: popolazione incerta (copie non ergodiche).")
    print("  - picco < 1 nm nell'AA ma non nel CG: contatto mancante; il picco AA stima r0 per un Morse pair-specific.")
    print("  - 'copie AA >0,5': quante copie hanno il contatto per piu' di meta' del tempo.")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
