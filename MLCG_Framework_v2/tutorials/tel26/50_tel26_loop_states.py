#!/usr/bin/env python3
"""TEL26: stati metastabili dei loop nell'AA e loro presenza nel CG.

PERCHE'
    Lo script 49 mostra che nell'AA il 70-95 % della varianza dei loop rilassa
    in 4-10 ns e meta' della varianza totale e' fra copie (copie bloccate in
    stati diversi), mentre nel CG tutto decade in ps in un solo bacino.  Qui si
    identificano gli stati dei loop nell'AA e si guarda dove cade il CG:
    riproduce uno degli stati, sta in mezzo, o fuori da tutti?

COME
    Per ogni loop (code e loop 1-3, come nello script 49): posizioni dei siti
    del loop a nucleo allineato sul nucleo AA medio; PCA sull'AA (componenti
    fino al 90 % della varianza, al massimo 8); k-means sull'AA per k = 2..6,
    scelta di k con la silhouette (su un sottoinsieme); il CG si proietta sulla
    stessa PCA e si assegna al centro AA piu' vicino.
    Per ogni stato: popolazione AA e CG, popolazione AA copia per copia,
    tempo medio di permanenza (assegnazioni filtrate a maggioranza su 100 ps).
    Per il CG: frazione di frame "fuori dagli stati AA" (distanza dal centro
    piu' vicino oltre il 95-esimo percentile delle distanze AA) e distanza
    della posizione media CG dai centri, confrontata con la distanza fra i
    centri (un CG "in mezzo" ha la media lontana da tutti i centri).

USO (in tutorials/tel26)
    python3 50_tel26_loop_states.py tel26_lp1_dataset.bin \\
        prod=r0_s01.npz+r0_s02.npz,r1_s01.npz+r1_s02.npz [--cg-stride 5] [--json loop_states.json]
    '+' unisce tratti della stessa corsa, ',' separa replicas (trattate come copie).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("s47", HERE / "47_tel26_timescales.py")
s47 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s47)
s46 = s47.s46
cv = s47.cv

GROUPS = [("coda 5' T1-A3", (1, 2, 3)), ("loop 1 T7-A9", (7, 8, 9)),
          ("loop 2 T13-A15", (13, 14, 15)), ("loop 3 T19-A21", (19, 20, 21)),
          ("coda 3' T25-T26", (25, 26))]


# ── dati ─────────────────────────────────────────────────────────────────────

def aligned(parts, ref, nuc):
    """parts -> dt, A (T, C, n_all, 3) nel sistema del nucleo AA, residui 1-based dei siti."""
    T = min(p[0].shape[0] for p in parts)
    dt = float(np.median(np.diff(parts[0][3])))
    out = []
    for S, L, nc, _t in parts:
        L = L[:T] if np.ndim(L) == 2 else L
        X = s46.unwrap_copies(S[:T], L, nc, nuc)
        out.append(s47.aligned_on_core(X.reshape(-1, nuc, S.shape[2], 3), ref).reshape(T, nc, -1, 3))
    res = np.array([r for r, _ in ref.sel.all]) + 1
    return dt, np.concatenate(out, axis=1), res


# ── k-means e silhouette ─────────────────────────────────────────────────────

def kmeans(x, k, rng, restarts=8, iters=100):
    best = None
    for _ in range(restarts):
        c = [x[rng.integers(len(x))]]
        for _j in range(1, k):                                   # k-means++
            d2 = np.min(((x[:, None] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            c.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        c = np.array(c)
        for _it in range(iters):
            lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), axis=1)
            new = np.array([x[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = float(((x - c[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, c, lab)
    return best[1], best[2]


def silhouette(x, lab, rng, n=3000):
    idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
    xs, ls = x[idx], lab[idx]
    sq = (xs ** 2).sum(-1)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2.0 * xs @ xs.T, 0.0))
    ks = np.unique(ls)
    if ks.size < 2:
        return float("nan")
    m = np.stack([d[:, ls == j].mean(1) for j in ks], axis=1)    # distanza media da ogni cluster
    own = np.searchsorted(ks, ls)
    cnt = np.array([(ls == j).sum() for j in ks])
    a = m[np.arange(len(ls)), own] * cnt[own] / np.maximum(cnt[own] - 1, 1)
    m[np.arange(len(ls)), own] = np.inf
    b = m.min(1)
    return float(np.mean((b - a) / np.maximum(a, b)))


# ── tempi di permanenza ──────────────────────────────────────────────────────

def mode_filter(lab, k, w):
    """lab (T, C) -> moda su una finestra centrata di w frame."""
    if w <= 1:
        return lab
    T = lab.shape[0]
    oh = np.zeros((T + 1,) + lab.shape[1:] + (k,))
    oh[1:] = np.eye(k)[lab]
    cs = np.cumsum(oh, axis=0)
    h = w // 2
    lo = np.clip(np.arange(T) - h, 0, T)
    hi = np.clip(np.arange(T) + h + 1, 0, T)
    return np.argmax(cs[hi] - cs[lo], axis=-1)


def dwell(lab, dt):
    """Tempo medio di permanenza per stato (ps): segmenti consecutivi, per copia."""
    runs = {}
    for c in range(lab.shape[1]):
        s = lab[:, c]
        cut = np.flatnonzero(np.diff(s)) + 1
        starts = np.concatenate([[0], cut])
        ends = np.concatenate([cut, [len(s)]])
        for a, b in zip(starts, ends):
            runs.setdefault(int(s[a]), []).append((b - a) * dt)
    return runs


# ── un loop ──────────────────────────────────────────────────────────────────

def analyse_group(name, A_aa, dt_aa, cg, rng, kmax=6, window_ps=100.0):
    T, C, n, _ = A_aa.shape
    xa = A_aa.reshape(T * C, n * 3)
    mu = xa.mean(0)
    U, s, Vt = np.linalg.svd(xa - mu, full_matrices=False)
    ev = s ** 2 / np.sum(s ** 2)
    m = int(min(8, np.searchsorted(np.cumsum(ev), 0.9) + 1))
    P = Vt[:m].T
    za = (xa - mu) @ P
    sil, fits = {}, {}
    for k in range(2, kmax + 1):
        c, lab = kmeans(za, k, rng)
        sil[k] = silhouette(za, lab, rng)
        fits[k] = (c, lab)
    k = max(sil, key=lambda q: sil[q] if np.isfinite(sil[q]) else -1)
    c, lab = fits[k]
    order = np.argsort(-np.bincount(lab, minlength=k))           # stati per popolazione decrescente
    remap = np.empty(k, int); remap[order] = np.arange(k)
    c = c[order]; lab = remap[lab]
    lab_aa = lab.reshape(T, C)
    d_aa = np.sqrt(((za - c[lab]) ** 2).sum(-1))
    q95 = float(np.percentile(d_aa, 95))
    sep = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
    out = {"name": name, "pca_components": m, "pca_var": float(np.cumsum(ev)[m - 1]),
           "silhouette": sil, "k": k, "center_sep_min": float(sep[np.triu_indices(k, 1)].min()),
           "aa": {"pop": np.bincount(lab, minlength=k) / lab.size,
                  "pop_per_copy": np.stack([np.bincount(lab_aa[:, j], minlength=k) / T for j in range(C)]),
                  "q95": q95, "spread_rms": float(np.sqrt((d_aa ** 2).mean()))}}
    w = max(1, int(round(window_ps / dt_aa)) | 1)
    runs = dwell(mode_filter(lab_aa, k, w), dt_aa)
    out["aa"]["dwell_ps"] = [float(np.mean(runs.get(j, [np.nan]))) for j in range(k)]
    out["cg"] = {}
    for label, (dt_cg, A_cg) in cg.items():
        Tc, Cc = A_cg.shape[:2]
        zc = (A_cg.reshape(Tc * Cc, n * 3) - mu) @ P
        dc = np.sqrt(((zc[:, None] - c[None]) ** 2).sum(-1))
        lc = np.argmin(dc, axis=1)
        dmin = dc[np.arange(len(lc)), lc]
        wc = max(1, int(round(window_ps / dt_cg)) | 1)
        runs_c = dwell(mode_filter(lc.reshape(Tc, Cc), k, wc), dt_cg)
        mean_c = zc.mean(0)
        out["cg"][label] = {"pop": np.bincount(lc, minlength=k) / lc.size,
                            "outside": float(np.mean(dmin > q95)),
                            "spread_rms": float(np.sqrt((((zc - mean_c) ** 2).sum(-1)).mean())),
                            "mean_to_centers": np.sqrt(((mean_c[None] - c) ** 2).sum(-1)),
                            "dwell_ps": [float(np.mean(runs_c.get(j, [np.nan]))) for j in range(k)]}
    return out


def report(r, cg_labels):
    k = r["k"]
    print(f"\n  == {r['name']} ==   PCA: {r['pca_components']} componenti ({100 * r['pca_var']:.0f} % della varianza AA)")
    print("  silhouette per k: " + "  ".join(f"k={q}: {v:.3f}" for q, v in r["silhouette"].items()) + f"   -> k = {k}")
    hdr = f"  {'stato':<8s}{'pop AA':>9s}{'permanenza AA':>15s}"
    for lab in cg_labels:
        hdr += f"{'pop ' + lab:>12s}{'permanenza':>12s}"
    print(hdr)
    for j in range(k):
        row = f"  {j:<8d}{r['aa']['pop'][j]:9.3f}{r['aa']['dwell_ps'][j] / 1000:12.2f} ns"
        for lab in cg_labels:
            g = r["cg"][lab]
            row += f"{g['pop'][j]:12.3f}{g['dwell_ps'][j]:9.1f} ps"
        print(row)
    ppc = r["aa"]["pop_per_copy"]
    print("  popolazioni AA per copia (righe = copie):")
    for i, p in enumerate(ppc):
        print(f"    copia {i:2d}: " + " ".join(f"{v:5.2f}" for v in p))
    print(f"  distanza minima fra centri {r['center_sep_min']:.3f} nm;  dispersione rms AA attorno al proprio centro "
          f"{r['aa']['spread_rms']:.3f} nm;  95-esimo percentile AA {r['aa']['q95']:.3f} nm")
    for lab in cg_labels:
        g = r["cg"][lab]
        print(f"  {lab}: fuori dagli stati AA {100 * g['outside']:.1f} %;  dispersione rms attorno alla propria media "
              f"{g['spread_rms']:.3f} nm;  media CG dai centri: " + " ".join(f"{v:.3f}" for v in g["mean_to_centers"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("runs", nargs="*", help="etichetta=a.npz[+b.npz][,r1.npz...]")
    ap.add_argument("--nuc", type=int, default=26)
    ap.add_argument("--aa-dt", type=float, default=20.0)
    ap.add_argument("--aa-stride", type=int, default=1)
    ap.add_argument("--cg-stride", type=int, default=5)
    ap.add_argument("--ref-stride", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cv.set_nuc(args.nuc)
    nuc = args.nuc
    rng = np.random.default_rng(args.seed)

    S, L, nc, t = s47.load_aa(args.dataset, args.aa_dt, args.aa_stride)
    k = max(1, args.ref_stride // args.aa_stride)
    Xr = s46.unwrap_copies(S[::k], L[::k], nc, nuc).reshape(-1, nuc, S.shape[2], 3)
    ref = s46.Reference(Xr, s46.Selection(nuc, Xr[0]))
    dt_aa, A_aa, res = aligned([(S, L, nc, t)], ref, nuc)

    cg = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        parts = [s47.load_cg(rep, args.cg_stride) for rep in path.split(",")]
        cg[label] = aligned(parts, ref, nuc)[:2]

    print(f"\n  AA: {A_aa.shape[0]} frame x {A_aa.shape[1]} copie, dt {dt_aa:g} ps")
    for lab, (dtc, Ac) in cg.items():
        print(f"  {lab}: {Ac.shape[0]} frame x {Ac.shape[1]} copie, dt {dtc:g} ps")

    results = []
    for name, residues in GROUPS:
        msk = np.isin(res, residues)
        r = analyse_group(name, A_aa[:, :, msk], dt_aa,
                          {lab: (dtc, Ac[:, :, msk]) for lab, (dtc, Ac) in cg.items()}, rng, args.kmax)
        report(r, list(cg))
        results.append(r)

    print("\n  Lettura:")
    print("  - silhouette > ~0,5: stati ben separati; < ~0,25: struttura debole, k poco significativo.")
    print("  - popolazioni AA per copia molto diverse fra copie: stati piu' lenti della corsa AA,")
    print("    popolazioni medie incerte.")
    print("  - CG 'fuori' alto e media CG lontana da tutti i centri (oltre la dispersione AA): il CG")
    print("    occupa una regione fra gli stati AA invece degli stati stessi.")
    print("  - CG concentrato in uno stato: riproduce quello stato ma non gli altri.")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=1, default=lambda o: np.asarray(o).tolist()))


if __name__ == "__main__":
    main()
