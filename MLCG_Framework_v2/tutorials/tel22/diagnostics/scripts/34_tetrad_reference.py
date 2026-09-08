#!/usr/bin/env python3
"""Distribuzione di RIFERIMENTO degli osservabili G-quadruplex di TEL22,
calcolata dai frame all-atom gia' mappati a CG dentro tel22_dataset.bin.

TEL22 = AGGG(TTAGGG)3, 22 residui per copia, 10 copie.
G-tract: [2,3,4] [8,9,10] [14,15,16] [20,21,22]  (1-based)
Siti guanina: CG_DG_S=2, B1=3 (N9,C4), B2=4 (N3,C2,N2),
              B3=5 (N1,C6,O6), B4=6 (C5,N7), B5=7 (C8)

Il registro delle tetradi NON viene assunto: viene misurato cercando, per
ogni G del primo tract, la G piu' vicina in ciascun altro tract.
"""
import struct, sys
import numpy as np

path = sys.argv[1]
NUC = 22
TRACTS = [[2,3,4],[8,9,10],[14,15,16],[20,21,22]]
BASE_TYPES = {3,4,5,6,7}

buf = open(path,"rb").read(); off = 0
def take(fmt):
    global off
    v = struct.unpack_from(fmt,buf,off); off += struct.calcsize(fmt); return v

(T,) = take("i")
box, molsites, molcent = [], [], []
for _ in range(T):
    nm,_n = take("ii"); box.append(take("3f"))
    ms, mc = [], []
    for _m in range(nm):
        _mid, ns = take("ii")
        mc.append(take("3f")); take("3f"); take("3f")
        blk = np.frombuffer(buf,dtype=np.int32,count=ns*4,offset=off).reshape(ns,4)
        off += ns*16
        ms.append((blk[:,0].copy(), blk[:,1:].copy().view(np.float32)))
    molsites.append(ms); molcent.append(mc)

L = np.asarray(box,np.float64)
M = len(molsites[0])
ncopy = M // NUC
print(f"  frame={T}  molecole={M}  copie={ncopy}")

# centro della base per ogni (frame, molecola) guanina
gcen = np.full((T, M, 3), np.nan)
for t in range(T):
    for m in range(M):
        types, pos = molsites[t][m]
        sel = np.isin(types, list(BASE_TYPES))
        if sel.sum() >= 3:
            gcen[t,m] = pos[sel].astype(np.float64).mean(0)

def mi(d, t):   # minimum image
    return d - L[t]*np.round(d/L[t])

# --- registro delle tetradi, misurato ---
print("\n  === registro tetradi (misurato sul riferimento) ===")
print(f"  {'G tract1':>9} | piu' vicina in tract2/3/4 (mediana su copie e frame)")
print("  " + "-"*58)
registry = {}
for g1 in TRACTS[0]:
    picks = []
    for other in TRACTS[1:]:
        cnt = {r:0 for r in other}
        for t in range(0, T, 10):
            for c in range(ncopy):
                a = gcen[t, c*NUC + (g1-1)]
                if np.isnan(a).any(): continue
                best, bestr = 1e9, None
                for r in other:
                    b = gcen[t, c*NUC + (r-1)]
                    if np.isnan(b).any(): continue
                    dd = np.linalg.norm(mi(b-a, t))
                    if dd < best: best, bestr = dd, r
                if bestr: cnt[bestr] += 1
        picks.append(max(cnt, key=cnt.get))
    registry[g1] = [g1] + picks
    print(f"  {g1:9d} | {picks}")

# --- osservabili sulle tetradi cosi' definite ---
print("\n  === geometria delle tetradi (riferimento all-atom mappato) ===")
print(f"  {'tetrade':>16} | {'d G-G adiacenti':>17} | {'d diagonale':>13} | {'n':>7}")
print("  " + "-"*66)
obs = {}
for g1, quad in registry.items():
    adj, diag = [], []
    for t in range(T):
        for c in range(ncopy):
            P = [gcen[t, c*NUC + (r-1)] for r in quad]
            if any(np.isnan(p).any() for p in P): continue
            P = [P[0]] + [P[0] + mi(p - P[0], t) for p in P[1:]]
            P = np.array(P)
            D = np.linalg.norm(P[:,None,:]-P[None,:,:], axis=2)
            ring = [D[0,1], D[1,2], D[2,3], D[3,0]]
            dia  = [D[0,2], D[1,3]]
            adj.extend(ring); diag.extend(dia)
    adj, diag = np.array(adj), np.array(diag)
    obs[g1] = (adj, diag)
    print(f"  {str(quad):>16} | {adj.mean():7.4f} +- {adj.std():5.4f} | "
          f"{diag.mean():6.4f} +- {diag.std():5.4f} | {adj.size:7d}")

# --- rise di stacking fra tetradi consecutive ---
print("\n  === rise fra tetradi impilate ===")
keys = sorted(registry)
for i in range(len(keys)-1):
    q1, q2 = registry[keys[i]], registry[keys[i+1]]
    rr = []
    for t in range(T):
        for c in range(ncopy):
            A = [gcen[t, c*NUC + (r-1)] for r in q1]
            B = [gcen[t, c*NUC + (r-1)] for r in q2]
            if any(np.isnan(p).any() for p in A+B): continue
            ca = np.mean([A[0] + mi(p-A[0], t) for p in A], axis=0)
            cb = np.mean([A[0] + mi(p-A[0], t) for p in B], axis=0)
            rr.append(np.linalg.norm(cb-ca))
    rr = np.array(rr)
    print(f"  {str(q1):>16} -> {str(q2):<16} : {rr.mean():6.4f} +- {rr.std():5.4f} nm  (n={rr.size})")

np.savez("tetrad_reference.npz",
         **{f"adj_{k}": v[0] for k,v in obs.items()},
         **{f"diag_{k}": v[1] for k,v in obs.items()},
         registry=np.array([registry[k] for k in sorted(registry)]))
print("\n  [OK] riferimento salvato in tetrad_reference.npz")
print("  Una G-tetrade canonica ha d(G-G) adiacenti ~0.9-1.0 nm fra centri")
print("  di base e rise di stacking ~0.34 nm.")
