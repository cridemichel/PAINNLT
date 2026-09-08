#!/usr/bin/env python3
"""Confronto termodinamico: distribuzione degli osservabili G-quadruplex
fra riferimento all-atom mappato e una o piu' traiettorie CG.

Statistica: divergenza di Jensen-Shannon in nats, con CI 95% via bootstrap.
Stesso protocollo di official_cgnet_vs_exact_48bins su Ala2.

USO
  python3 36_compare_tetrads.py <dataset.bin> <label>=<samples.npz> [...]
Esempio
  python3 36_compare_tetrads.py .../tel22_dataset.bin \
      priors=.../thermo/priors_10kstep/samples.npz \
      ml=.../thermo/ml_10kstep/samples.npz
"""
import struct, sys
import numpy as np

NUC, NBINS, NBOOT = 22, 48, 400
TRACTS = [[2,3,4],[8,9,10],[14,15,16],[20,21,22]]
REGISTRY = {2:[2,10,14,22], 3:[3,9,15,21], 4:[4,8,16,20]}
# Registro da TEL22_TETRADS in PaiNN_Architecture.hpp (0-based {1,9,13,21},
# {2,8,14,20}, {3,7,15,19}). Concorda con la misura dello script 34 sulle due
# tetradi ben formate; sulla prima il codice dice 22 e la misura diceva 21,
# artefatto della tetrade terminale sfrangiata (sigma 5x, G21 in due tetradi).

def mi(d, L): return d - L*np.round(d/L)

def obs_from_gcen(gcen, L, ncopy):
    """gcen: (T, M, 3) centri delle basi. Ritorna dict osservabile -> array."""
    out = {}
    for g1, quad in REGISTRY.items():
        d_all = [[] for _ in range(6)]
        for t in range(gcen.shape[0]):
            Lt = L[t] if L.ndim == 2 else L
            for c in range(ncopy):
                P = [gcen[t, c*NUC + (r-1)] for r in quad]
                if any(np.isnan(p).any() for p in P): continue
                P = np.array([P[0]] + [P[0] + mi(p-P[0], Lt) for p in P[1:]])
                D = np.linalg.norm(P[:,None,:]-P[None,:,:], axis=2)
                for k,(i,j) in enumerate([(0,1),(1,2),(2,3),(3,0),(0,2),(1,3)]):
                    d_all[k].append(D[i,j])
        for k in range(6):
            out[f"tet{g1}_d{k}"] = np.array(d_all[k])
    # rise fra tetradi consecutive
    keys = sorted(REGISTRY)
    for i in range(len(keys)-1):
        q1, q2 = REGISTRY[keys[i]], REGISTRY[keys[i+1]]
        rr = []
        for t in range(gcen.shape[0]):
            Lt = L[t] if L.ndim == 2 else L
            for c in range(ncopy):
                A = [gcen[t, c*NUC + (r-1)] for r in q1]
                B = [gcen[t, c*NUC + (r-1)] for r in q2]
                if any(np.isnan(p).any() for p in A+B): continue
                ca = np.mean([A[0] + mi(p-A[0], Lt) for p in A], axis=0)
                cb = np.mean([A[0] + mi(p-A[0], Lt) for p in B], axis=0)
                rr.append(np.linalg.norm(cb-ca))
        out[f"rise{keys[i]}_{keys[i+1]}"] = np.array(rr)
    return out

def load_reference(path):
    buf = open(path,"rb").read(); off = [0]
    def take(f):
        v = struct.unpack_from(f,buf,off[0]); off[0]+=struct.calcsize(f); return v
    (T,) = take("i")
    box, G = [], []
    for _ in range(T):
        nm,_n = take("ii"); box.append(take("3f"))
        row = np.full((nm,3), np.nan)
        for m in range(nm):
            _mid, ns = take("ii")
            take("3f"); take("3f"); take("3f")
            blk = np.frombuffer(buf,dtype=np.int32,count=ns*4,offset=off[0]).reshape(ns,4)
            off[0] += ns*16
            if ns == 6:      # guanina: siti 1..5 sono le basi
                row[m] = blk[1:,1:].copy().view(np.float32).astype(np.float64).mean(0)
        G.append(row)
    return np.asarray(G), np.asarray(box,np.float64), nm//NUC

def load_samples(path):
    z = np.load(path)
    sites, smol, sidx = z["sites"], z["site_molecule"], z["site_index"]
    box = np.asarray(z["box"], np.float64)
    T, M = sites.shape[0], int(smol.max())+1
    nsites = np.bincount(smol, minlength=M)
    G = np.full((T, M, 3), np.nan)
    for m in np.flatnonzero(nsites == 6):          # guanine
        sel = np.flatnonzero((smol == m) & (sidx >= 1) & (sidx <= 5))
        if sel.size == 5:
            G[:, m] = sites[:, sel, :].mean(1)
    return G, box, M//NUC

def js(a, b, edges, rng=None):
    p,_ = np.histogram(a, bins=edges); q,_ = np.histogram(b, bins=edges)
    p = p/max(p.sum(),1); q = q/max(q.sum(),1)
    m = 0.5*(p+q)
    def kl(x,y):
        s = x > 0
        return float((x[s]*np.log(x[s]/np.maximum(y[s],1e-300))).sum())
    return 0.5*kl(p,m) + 0.5*kl(q,m)

def js_ci(a, b, edges, nboot=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    vals = [js(rng.choice(a,a.size,True), rng.choice(b,b.size,True), edges)
            for _ in range(nboot)]
    return js(a,b,edges), float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))

ref_path, models = sys.argv[1], {}
for arg in sys.argv[2:]:
    k,v = arg.split("=",1); models[k] = v

Gr, Lr, nc = load_reference(ref_path)
print(f"  riferimento: {Gr.shape[0]} frame, {nc} copie")
ref = obs_from_gcen(Gr, Lr, nc)

runs = {}
for label, p in models.items():
    Gm, Lm, ncm = load_samples(p)
    print(f"  {label}: {Gm.shape[0]} frame, {ncm} copie")
    runs[label] = obs_from_gcen(Gm, Lm, ncm)

names = [k for k in ref if k.startswith("rise")] + \
        [k for k in ref if k.startswith("tet")]
print()
w = max(len(l) for l in runs) if runs else 6
hdr = f"  {'osservabile':>14} | {'rif media±sd':>16} | {'modello':>{w}} | {'media±sd':>16} | {'JS (nats)':>20}"
print(hdr); print("  " + "-"*(len(hdr)-2))
summary = {l: [] for l in runs}
for nm_ in names:
    r = ref[nm_]
    lo = min([r.min()] + [runs[l][nm_].min() for l in runs if runs[l][nm_].size])
    hi = max([r.max()] + [runs[l][nm_].max() for l in runs if runs[l][nm_].size])
    edges = np.linspace(lo, hi, NBINS+1)
    first = True
    for l in runs:
        m = runs[l][nm_]
        if m.size == 0: continue
        v, a, b = js_ci(r, m, edges)
        summary[l].append(v)
        rr = f"{r.mean():7.4f}±{r.std():.4f}" if first else ""
        nn = nm_ if first else ""
        print(f"  {nn:>14} | {rr:>16} | {l:>{w}} | {m.mean():7.4f}±{m.std():.4f} | "
              f"{v:6.4f} [{a:.4f},{b:.4f}]")
        first = False
print()
print("  === sintesi (JS media su tutti gli osservabili; piu' basso = meglio) ===")
for l, v in summary.items():
    print(f"    {l:>{w}} : {np.mean(v):.4f}")
if len(summary) == 2:
    (l1,v1),(l2,v2) = list(summary.items())
    d = np.mean(v1) - np.mean(v2)
    better = l2 if d > 0 else l1
    print(f"\n    differenza {l1} - {l2} = {d:+.4f}  ->  migliore: {better}")
    print("    Se la rete non migliora il solo-prior, su TEL22 non sta aggiungendo nulla.")
