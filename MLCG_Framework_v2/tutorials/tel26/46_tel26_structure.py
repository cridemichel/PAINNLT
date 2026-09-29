#!/usr/bin/env python3
"""TEL26: RMSD, Q delle tetradi e Rg, CG contro all-atom mappato.

La g(r) dice se le distanze di coppia sono giuste; qui si guarda la
STRUTTURA della singola copia con le coordinate della letteratura sui CG
di biomolecole (RMSD dalla struttura di riferimento, frazione di contatti
nativi, raggio di girazione), cosi' il confronto e' nello stesso linguaggio
dei lavori su CGnet/CGSchNet.

Per (frame, copia), da riferimento e corse:

  rmsd_core   RMSD dei 72 siti delle 12 guanine delle tetradi, dopo
              sovrapposizione ottimale (Kabsch) sulla struttura media AA
  rmsd_all    lo stesso su tutti gli 86 siti della copia (loop compresi)
  rmsd_loops  RMSD dei 14 siti di A/T dopo sovrapposizione sul NUCLEO:
              quanto i loop si discostano dalla media a nucleo fermo
  Q           frazione liscia dei contatti B3-B3 nativi dentro le tre
              tetradi (tutte le 6 coppie per tetrade, 18 in tutto),
              Q = <1/(1+exp(beta (r - lam r_nat)))>, beta 25 nm^-1, lam 1.2
  Rg          raggio di girazione della copia (siti non pesati)

La struttura di riferimento e' la media AA, ottenuta allineando iterativamente
tutti i (frame, copia) del riferimento; le distanze native di Q sono le
mediane AA.  Per ogni coordinata: media, deviazione standard, sovrapposizione
degli istogrammi (integrale del minimo) e divergenza di Jensen-Shannon (base
2, fra 0 e 1).  In piu' la JSD della superficie 2D (rmsd_core, Q).

Uso:
  python3 46_tel26_structure.py tel26_lp1_dataset.bin \\
      lp1=samples_priors_lp1_100ps.npz#50:100 it30=samples_..._it30_100ps.npz#50:100 \\
      [--ref-range 0:0.5] [--stride 5] [--run-stride 5] [--plot struct_tel26]

Corse: file .npz di run_cg_md (con finestra opzionale #T0:T1 in ps),
'@bin:altro_dataset.bin' (un altro blocco AA), '@ref:A:B' (fetta del
riferimento, per il tetto AA contro AA).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tel22" / "diagnostics" / "scripts"))
import _tel22_cv as cv  # noqa: E402
from _tel22_cv import load_reference, load_samples  # noqa: E402

SEQUENCE = "TTAGGGTTAGGGTTAGGGTTAGGGTT"
# Tetradi (residui 1-based): la piega ibrida 3+1 del 2JPZ.
TETRADS_1B = [(4, 12, 16, 22), (5, 11, 17, 23), (6, 10, 18, 24)]
B3 = 3                     # indice del sito B3 nella guanina (S, B1..B5)


# ── geometria ────────────────────────────────────────────────────────────────

def mi(d, L):
    return d - L * np.round(d / L)


def unwrap_copies(S, L, ncopy, nuc):
    """(T, M, 6, 3) -> (T, ncopy, nuc, 6, 3) con ogni copia intera.

    Gli ancoraggi (sito 0 di ogni residuo) si srotolano lungo la catena,
    poi ogni sito rispetto al suo ancoraggio: una copia a cavallo del bordo
    periodico resta contigua.
    """
    T = S.shape[0]
    X = S.reshape(T, ncopy, nuc, S.shape[2], 3)
    Lb = np.asarray(L, float)
    Lb = Lb.reshape(T, 1, 1, 3) if Lb.ndim == 2 else Lb.reshape(1, 1, 1, 3)
    anc = X[:, :, :, 0, :]                                   # (T, C, R, 3)
    steps = mi(np.diff(anc, axis=2), Lb)
    anc_u = np.concatenate([anc[:, :, :1], anc[:, :, :1] + np.cumsum(steps, axis=2)], axis=2)
    rel = mi(X - anc[:, :, :, None, :], Lb[..., None, :])
    return anc_u[:, :, :, None, :] + rel


def kabsch_rmsd(X, Y, return_aligned=False):
    """X (B, n, 3) sovrapposti su Y (n, 3); RMSD per configurazione."""
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    H = np.einsum("bni,nj->bij", Xc, Yc)
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("bij,bjk->bik", U, Vt)))
    D = np.ones((X.shape[0], 3))
    D[:, 2] = d
    R = np.einsum("bij,bj,bjk->bik", U, D, Vt)
    Xa = np.einsum("bni,bij->bnj", Xc, R)
    rmsd = np.sqrt(((Xa - Yc[None]) ** 2).sum(axis=2).mean(axis=1))
    return (rmsd, Xa + Y.mean(axis=0)) if return_aligned else rmsd


def mean_structure(X, iterations=4):
    """Struttura media per allineamento iterativo (Procrustes generalizzato)."""
    ref = X[0] - X[0].mean(axis=0)
    for _ in range(iterations):
        _r, Xa = kabsch_rmsd(X, ref, return_aligned=True)
        ref = Xa.mean(axis=0)
        ref -= ref.mean(axis=0)
    return ref


class Selection:
    def __init__(self, nuc, template):
        """template: (nuc, 6, 3) di una copia, per sapere quali siti esistono."""
        present = np.isfinite(template).all(axis=-1)          # (nuc, 6)
        g_core = sorted({r - 1 for t in TETRADS_1B for r in t})
        for r in g_core:
            if SEQUENCE[r] != "G" or present[r].sum() != 6:
                raise SystemExit(f"[ERROR] residuo {r + 1} non e' una guanina a 6 siti")
        self.core = [(r, s) for r in g_core for s in range(6)]
        self.all = [(r, s) for r in range(nuc) for s in range(6) if present[r, s]]
        self.loops = [(r, 0) for r in range(nuc) if SEQUENCE[r] != "G"]
        pairs = []
        for tet in TETRADS_1B:
            idx = [r - 1 for r in tet]
            pairs += [(idx[a], idx[b]) for a in range(4) for b in range(a + 1, 4)]
        self.q_pairs = pairs


def take(Xu, sel):
    """(N, nuc, 6, 3) -> (N, n, 3) sui siti della selezione."""
    r = np.array([a for a, _ in sel])
    s = np.array([b for _, b in sel])
    return Xu[:, r, s, :]


# ── coordinate collettive ────────────────────────────────────────────────────

class Reference:
    def __init__(self, Xu, sel):
        self.sel = sel
        self.core = mean_structure(take(Xu, sel.core))
        self.all = mean_structure(take(Xu, sel.all))
        i = np.array([a for a, _ in sel.q_pairs])
        j = np.array([b for _, b in sel.q_pairs])
        d = np.linalg.norm(Xu[:, i, B3] - Xu[:, j, B3], axis=-1)
        self.q_native = np.nanmedian(d, axis=0)


def collective(Xu, ref, beta=25.0, lam=1.2):
    sel = ref.sel
    core = take(Xu, sel.core)
    out = {"rmsd_core": kabsch_rmsd(core, ref.core),
           "rmsd_all": kabsch_rmsd(take(Xu, sel.all), ref.all)}
    # Loop sovrapposti sul nucleo: stessa rotazione del nucleo applicata ai loop.
    Xc = core - core.mean(axis=1, keepdims=True)
    H = np.einsum("bni,nj->bij", Xc, ref.core)
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("bij,bjk->bik", U, Vt)))
    D = np.ones((core.shape[0], 3)); D[:, 2] = d
    R = np.einsum("bij,bj,bjk->bik", U, D, Vt)
    loops = take(Xu, sel.loops) - core.mean(axis=1, keepdims=True)
    loops_a = np.einsum("bni,bij->bnj", loops, R)
    # riferimento dei loop: media dei loop allineati sul nucleo nel riferimento
    if not hasattr(ref, "loops"):
        ref.loops = loops_a.mean(axis=0)
    out["rmsd_loops"] = np.sqrt(((loops_a - ref.loops[None]) ** 2).sum(axis=2).mean(axis=1))
    i = np.array([a for a, _ in sel.q_pairs])
    j = np.array([b for _, b in sel.q_pairs])
    r = np.linalg.norm(Xu[:, i, B3] - Xu[:, j, B3], axis=-1)
    out["Q"] = (1.0 / (1.0 + np.exp(beta * (r - lam * ref.q_native[None])))).mean(axis=1)
    allx = take(Xu, sel.all)
    out["Rg"] = np.sqrt(((allx - allx.mean(axis=1, keepdims=True)) ** 2).sum(axis=2).mean(axis=1))
    return out


def rmsf_on_core(Xu, ref):
    """RMSF per residuo (media sui siti), dopo sovrapposizione sul nucleo
    AA, e posizione media di ogni sito nel sistema del nucleo.
    """
    sel = ref.sel
    core = take(Xu, sel.core)
    Xc = core - core.mean(axis=1, keepdims=True)
    H = np.einsum("bni,nj->bij", Xc, ref.core)
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("bij,bjk->bik", U, Vt)))
    D = np.ones((core.shape[0], 3)); D[:, 2] = d
    R = np.einsum("bij,bj,bjk->bik", U, D, Vt)
    allx = take(Xu, sel.all) - core.mean(axis=1, keepdims=True)
    A = np.einsum("bni,bij->bnj", allx, R)                     # (N, 86, 3)
    mean = A.mean(axis=0)
    msf = ((A - mean[None]) ** 2).sum(axis=2).mean(axis=0)    # (86,)
    res = np.array([r for r, _ in sel.all])
    rmsf = np.array([np.sqrt(msf[res == r].mean()) for r in range(res.max() + 1)])
    return rmsf, mean


# ── confronto fra distribuzioni ─────────────────────────────────────────────

def hist_compare(a, b, lo, hi, bins=60):
    pa, e = np.histogram(a, bins=bins, range=(lo, hi))
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi))
    pa = pa / max(1, pa.sum()); pb = pb / max(1, pb.sum())
    overlap = float(np.minimum(pa, pb).sum())
    return overlap, jsd(pa, pb), e, pa, pb


def jsd(p, q):
    p = np.asarray(p, float).ravel(); q = np.asarray(q, float).ravel()
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(x, y):
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / y[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def jsd_2d(a1, a2, b1, b2, r1, r2, bins=40):
    ha, _, _ = np.histogram2d(a1, a2, bins=bins, range=(r1, r2))
    hb, _, _ = np.histogram2d(b1, b2, bins=bins, range=(r1, r2))
    return jsd(ha, hb) if ha.sum() and hb.sum() else float("nan")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("runs", nargs="*", help="etichetta=percorso")
    ap.add_argument("--nuc", type=int, default=26)
    ap.add_argument("--ref-range", default=None, metavar="A:B")
    ap.add_argument("--stride", type=int, default=5, help="un frame AA ogni N")
    ap.add_argument("--run-stride", type=int, default=5, help="un frame CG ogni N")
    ap.add_argument("--plot", default=None, help="prefisso dei grafici PNG")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cv.set_nuc(args.nuc)
    nuc = args.nuc

    print(f"  riferimento: {args.dataset}")
    S_full, L_full, ncr = load_reference(args.dataset)
    T = S_full.shape[0]

    def frac(spec):
        lo, hi = (float(x) for x in spec.split(":"))
        return int(round(lo * T)), int(round(hi * T))

    i0, i1 = frac(args.ref_range) if args.ref_range else (0, T)
    st = max(1, args.stride)
    Sr, Lr = S_full[i0:i1:st], L_full[i0:i1:st]
    Xr = unwrap_copies(Sr, Lr, ncr, nuc).reshape(-1, nuc, S_full.shape[2], 3)
    sel = Selection(nuc, Xr[0])
    ref = Reference(Xr, sel)
    cv_ref = collective(Xr, ref)
    print(f"  {Sr.shape[0]} frame x {ncr} copie = {Xr.shape[0]} configurazioni di riferimento "
          f"(frame {i0}-{i1}, stride {st})")
    print(f"  siti: nucleo {len(sel.core)}, copia {len(sel.all)}, loop {len(sel.loops)}; "
          f"coppie Q {len(sel.q_pairs)} (r_nat {ref.q_native.min():.3f}-{ref.q_native.max():.3f} nm)")

    def load_run(spec):
        if spec.startswith("@ref:"):
            a, b = frac(spec[5:])
            return S_full[a:b:st], L_full[a:b:st], ncr
        if spec.startswith("@bin:"):
            S, L, nc = load_reference(spec[5:])
            return S[::st], L[::st], nc
        path, win = (spec.split("#", 1) + [None])[:2]
        S, L, nc, t = load_samples(path)
        t = np.asarray(t, float)
        keep = np.arange(t.size)
        if win:
            t0, t1 = (float(x) for x in win.split(":"))
            keep = np.flatnonzero((t >= t0) & (t < t1))
        keep = keep[::max(1, args.run_stride)]
        L = np.asarray(L)
        if L.ndim == 2 and L.shape[0] == S.shape[0]:
            L = L[keep]
        return S[keep], L, nc

    names = ["rmsd_core", "rmsd_all", "rmsd_loops", "Q", "Rg"]
    units = {"rmsd_core": "nm", "rmsd_all": "nm", "rmsd_loops": "nm", "Q": "", "Rg": "nm"}
    results = {"reference": args.dataset, "ref_range": args.ref_range, "runs": {}}
    stats = {k: (float(cv_ref[k].mean()), float(cv_ref[k].std())) for k in names}
    results["runs"]["AA"] = {k: {"mean": stats[k][0], "std": stats[k][1]} for k in names}
    run_cvs = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        S, L, nc = load_run(path)
        X = unwrap_copies(S, L, nc, nuc).reshape(-1, nuc, S.shape[2], 3)
        run_cvs[label] = collective(X, ref)
        print(f"  {label}: {X.shape[0]} configurazioni ({path})")

    # Intervalli comuni per gli istogrammi.
    ranges = {}
    for k in names:
        vals = np.concatenate([cv_ref[k]] + [c[k] for c in run_cvs.values()])
        lo, hi = np.percentile(vals, [0.2, 99.8])
        pad = 0.05 * (hi - lo)
        ranges[k] = (lo - pad, hi + pad)

    print("\n  coordinata        AA media±std         " +
          "".join(f"{lab:>26s}" for lab in run_cvs))
    for k in names:
        row = f"  {k:<14s} {stats[k][0]:8.4f} ± {stats[k][1]:6.4f}   "
        for lab, c in run_cvs.items():
            row += f"   {c[k].mean():7.4f}±{c[k].std():6.4f}"
            row += " " * 3
        print(row)
    print("\n  sovrapposizione degli istogrammi (1 = identici) / JSD (0 = identici)")
    print("  coordinata    " + "".join(f"{lab:>20s}" for lab in run_cvs))
    for k in names:
        row = f"  {k:<12s}  "
        for lab, c in run_cvs.items():
            ov, js, *_ = hist_compare(cv_ref[k], c[k], *ranges[k])
            results["runs"].setdefault(lab, {})[k] = {
                "mean": float(c[k].mean()), "std": float(c[k].std()), "overlap": ov, "jsd": js}
            row += f"      {ov:6.3f} / {js:6.4f}"
        print(row)
    row = "  FES (rmsd_core,Q) JSD "
    for lab, c in run_cvs.items():
        j2 = jsd_2d(cv_ref["rmsd_core"], cv_ref["Q"], c["rmsd_core"], c["Q"],
                    ranges["rmsd_core"], ranges["Q"])
        results["runs"][lab]["fes2d_jsd"] = j2
        row += f"   {lab} {j2:.4f}"
    print(row)
    # Fluttuazioni contro spostamento medio: una RMSD piu' grande puo' venire
    # da fluttuazioni piu' ampie o da una struttura media diversa.
    rmsf_ref, mean_ref = rmsf_on_core(Xr, ref)
    print("\n  struttura media contro AA (nel sistema del nucleo AA) e RMSF per regione")
    groups = {"tetrade 1": [3, 11, 15, 21], "tetrade 2": [4, 10, 16, 22],
              "tetrade 3": [5, 9, 17, 23],
              "loop/code": [r for r in range(nuc) if SEQUENCE[r] != "G"],
              "G fuori tetradi": [r for r in range(nuc) if SEQUENCE[r] == "G"
                                  and r not in {x - 1 for t in TETRADS_1B for x in t}]}
    print("  regione            RMSF AA " + "".join(f"{lab:>14s}" for lab in run_cvs))
    run_rmsf = {}
    for lab in run_cvs:
        S, L, nc = load_run(dict(x.split("=", 1) for x in args.runs)[lab])
        X = unwrap_copies(S, L, nc, nuc).reshape(-1, nuc, S.shape[2], 3)
        run_rmsf[lab] = rmsf_on_core(X, ref)
    for g, idx in groups.items():
        if not idx:
            continue
        row = f"  {g:<18s} {rmsf_ref[idx].mean():8.4f}"
        for lab, (rf, _m) in run_rmsf.items():
            row += f"   {rf[idx].mean():7.4f} ({rf[idx].mean() / rmsf_ref[idx].mean():4.2f}x)"
        print(row)
    row = "  scarto medio (nm)          "
    for lab, (_rf, m) in run_rmsf.items():
        off = np.sqrt(((m - mean_ref) ** 2).sum(axis=1).mean())
        results["runs"][lab]["mean_structure_offset_nm"] = float(off)
        row += f"   {off:13.4f}"
    print(row)
    print("  (scarto medio: RMSD fra la struttura media della corsa e quella AA, a nucleo allineato)")
    print("\n  RMSD in nm dalla struttura media AA dopo sovrapposizione (Kabsch).")
    print("  Il tetto e' un riferimento contro se' stesso: '@ref:0.5:1' con --ref-range 0:0.5.")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=2))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        for ax, k in zip(axes.ravel(), names):
            _o, _j, e, pa, _ = hist_compare(cv_ref[k], cv_ref[k], *ranges[k])
            c = 0.5 * (e[1:] + e[:-1])
            ax.plot(c, pa / (e[1] - e[0]), "k-", lw=2.2, label="all-atom (mappato su CG)")
            for lab, cvs in run_cvs.items():
                ov, js, e, _pa, pb = hist_compare(cv_ref[k], cvs[k], *ranges[k])
                ax.plot(c, pb / (e[1] - e[0]), "--", lw=1.6, label=f"{lab} (sovr. {ov:.3f})")
            ax.set_xlabel(f"{k} ({units[k]})" if units[k] else k)
            ax.set_ylabel("densita'")
            ax.set_title(k)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        ax = axes.ravel()[-1]
        ax.scatter(cv_ref["rmsd_core"], cv_ref["Q"], s=1, c="k", alpha=0.15, label="all-atom")
        for lab, cvs in run_cvs.items():
            ax.scatter(cvs["rmsd_core"], cvs["Q"], s=1, alpha=0.15, label=lab)
        ax.set_xlabel("rmsd_core (nm)"); ax.set_ylabel("Q tetradi")
        ax.set_title("rmsd_core contro Q"); ax.legend(fontsize=8, markerscale=8)
        fig.suptitle("TEL26 - struttura della copia: CG contro all-atom")
        fig.tight_layout()
        out = f"{args.plot}.png"
        fig.savefig(out, dpi=130)
        print(f"  grafico: {out}")


if __name__ == "__main__":
    main()
