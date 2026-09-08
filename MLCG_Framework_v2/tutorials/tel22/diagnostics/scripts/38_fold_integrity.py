#!/usr/bin/env python3
"""Integrita' TOPOLOGICA del fold G-quadruplex: le guanine sono ancora
appaiate con i partner di registro corretti?

Complementare a 36/37, che misurano le DISTANZE. Un modello puo' tenere la
struttura compatta e della taglia giusta (JS buono) mentre le specifiche
coppie di Hoogsteen si rimescolano: questa metrica lo rivela.

Criterio: per ogni G di tract 1, la G piu' vicina in ciascun altro tract
coincide con il partner previsto da TEL22_TETRADS?

ATTENZIONE alla scala: non e' 0-100. Il riferimento all-atom stesso segna
~71% (il criterio e' grezzo e la tetrade terminale sfrangia anche
nell'all-atom); l'assegnazione casuale sta a ~33% (1 su 3 per tract).
Leggere sempre rispetto a quei due estremi.

USO
  python3 38_fold_integrity.py <dataset.bin> <label>=<samples.npz> [...]
"""
import sys, pathlib
import numpy as np

here = pathlib.Path(__file__).parent
src = (here / "36_compare_tetrads.py").read_text().split("ref_path, models")[0]
ns = {}; exec(src, ns)
load_samples, load_reference, mi = ns["load_samples"], ns["load_reference"], ns["mi"]

NUC = 22
TRACTS = [[2, 3, 4], [8, 9, 10], [14, 15, 16], [20, 21, 22]]
REGISTRY = {2: [2, 10, 14, 22], 3: [3, 9, 15, 21], 4: [4, 8, 16, 20]}

def integrity(G, L, ncopy, frames):
    ok = tot = 0
    for t in frames:
        Lt = L[t] if (hasattr(L, "ndim") and L.ndim == 2) else L
        for c in range(ncopy):
            for g1, quad in REGISTRY.items():
                a = G[t, c * NUC + (g1 - 1)]
                if np.isnan(a).any():
                    continue
                for ti, other in enumerate(TRACTS[1:]):
                    best, bestr = 1e9, None
                    for r in other:
                        b = G[t, c * NUC + (r - 1)]
                        if np.isnan(b).any():
                            continue
                        d = np.linalg.norm(mi(b - a, Lt))
                        if d < best:
                            best, bestr = d, r
                    if bestr is not None:
                        tot += 1
                        ok += (bestr == quad[ti + 1])
    return 100.0 * ok / max(tot, 1), tot

ref_path = sys.argv[1]
runs = dict(a.split("=", 1) for a in sys.argv[2:])

Gr, Lr, nc = load_reference(ref_path)
ref_val, ref_n = integrity(Gr, Lr, nc, range(0, Gr.shape[0], 15))
print()
print(f"  RIFERIMENTO all-atom : {ref_val:5.1f}%   <- massimo raggiungibile")
print(f"  casuale (1 su 3)     : ~33.3%   <- minimo informativo")
print(f"  (n={ref_n} confronti sul riferimento)")
print()

data = {}
for label, p in runs.items():
    G, L, n = load_samples(p)
    data[label] = (G, L, n, np.load(p)["time_ps"])

tmax = max(d[3][-1] for d in data.values()) if data else 0
w = 5.0 if tmax > 12 else 2.5
edges = [(a, a + w) for a in np.arange(0, tmax, w)]

wl = max([len(l) for l in data] + [8])
print(f"  {'finestra (ps)':>16} | " + " | ".join(f"{l:>{wl}}" for l in data))
print("  " + "-" * (19 + (wl + 3) * len(data)))
for a, b in edges:
    row = []
    for label in data:
        G, L, n, t = data[label]
        m = np.flatnonzero((t >= a) & (t < b))
        if m.size > 3:
            v, _ = integrity(G, L, n, m[::2])
            row.append(f"{v:{wl-1}.1f}%")
        else:
            row.append(" " * wl)
    if any(x.strip() for x in row):
        print(f"  {a:6.1f} - {b:5.1f}  | " + " | ".join(row))
print()
print("  Vicino al riferimento -> appaiamento di Hoogsteen preservato.")
print("  Vicino a 33% -> la struttura puo' essere compatta ma mal appaiata:")
print("  confrontare sempre con 36/37, che da soli non lo rivelerebbero.")
