#!/usr/bin/env python3
"""Integrita' dei legami H di Hoogsteen nelle G-tetradi di TEL22.

Criterio fisico, scala 0-100 pulita (a differenza dello script 38, in cui
il riferimento stesso segna ~71%): un legame H e' INTATTO se la distanza
fra i siti B3 (N1,C6,O6) delle due guanine sta in 0.35 < d < 0.60 nm (sotto 0.35 e' sovra-compressione
spuria, non un legame H: cfr. l'esplosione di ep40). Opzionalmente
anche il contatto orientato B2/B4 (N2-H...N7) < 0.65 nm (criterio stretto).

Il ciclo di H-bond di ogni tetrade e' fissato dalla topologia del basket
antiparallelo (PDB 143D) e confermato dalle distribuzioni di riferimento:
    tet0  2->10->22->14     tet1  3->9->21->15     tet2  4->8->20->16
Le due coppie rimanenti per tetrade sono diagonali (nessun legame H).

USO
  python3 40_hbond_integrity.py <dataset.bin> <label>=<samples.npz> [...]
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _hb_common import (CONTACTS, TETRAD_OF, load_reference_sites,
                        load_sample_sites, contact_distances)

T33_LO, T33, T24 = 0.35, 0.60, 0.65   # finestra fisica: sotto 0.35 nm e' sovra-compressione, non legame H
CYCLE_1B = {0: [(2,10),(10,22),(22,14),(14,2)],
            1: [(3,9),(9,21),(21,15),(15,3)],
            2: [(4,8),(8,20),(20,16),(16,4)]}
# indice dei contatti H-bonded in CONTACTS (coppie non ordinate, 0-based)
hb_idx = []
for k,(ri,rj,_,_) in enumerate(CONTACTS):
    pair = {ri+1, rj+1}
    if any(pair == set(p) for p in CYCLE_1B[TETRAD_OF[k]]): hb_idx.append(k)
assert len(hb_idx) == 12, hb_idx
hb_tet = np.array([TETRAD_OF[k] for k in hb_idx])

def intact(d33, d24, strict=False):
    a = d33[:, hb_idx]; b = d24[:, hb_idx]
    ok = (a > T33_LO) & (a < T33)
    if strict: ok &= (b < T24)
    ok = np.where(np.isnan(a), np.nan, ok.astype(float))
    return ok  # (n_samples, 12)

def summarize(ok):
    tot = 100*np.nanmean(ok)
    per_tet = [100*np.nanmean(ok[:, hb_tet==t]) for t in range(3)]
    return tot, per_tet

ref_path = sys.argv[1]; runs = dict(a.split("=",1) for a in sys.argv[2:])
S,L,nc = load_reference_sites(ref_path)
d33,d24 = contact_distances(S,L,nc,range(0,S.shape[0],5))
r_l, r_lt = summarize(intact(d33,d24)); r_s, r_st = summarize(intact(d33,d24,True))
print()
print(f"  Soglie: {T33_LO} < B3-B3 < {T33} nm (largo)  |  + B2/B4 < {T24} nm (stretto)")
print(f"  Il limite inferiore esclude contatti sovra-compressi da pozzi spurii (cfr. ep40).")
print(f"  RIFERIMENTO all-atom   largo {r_l:5.1f}%  [tet0 {r_lt[0]:4.1f}  tet1 {r_lt[1]:4.1f}  tet2 {r_lt[2]:4.1f}]"
      f"   stretto {r_s:5.1f}%")
print(f"  (scala: 100 = tutti i 12 legami H intatti; ~0 = quadruplex sfaldato)")
print()
data = {l:load_sample_sites(p) for l,p in runs.items()}
tmax = max(d[3][-1] for d in data.values()) if data else 0
w = 5.0 if tmax > 12 else 2.5
wl = max([len(l) for l in data]+[6])
print(f"  {'finestra (ps)':>14} | " + " | ".join(f"{l:>{wl}}" for l in data) + "   (criterio largo; per tetrade 0/1/2)")
print("  " + "-"*(17+(wl+3)*len(data)+38))
for a in np.arange(0,tmax,w):
    b=a+w; cells=[]; detail=[]
    for l,(Sm,Lm,ncm,t) in data.items():
        m=np.flatnonzero((t>=a)&(t<b))
        if m.size<3: cells.append(" "*wl); continue
        x33,x24 = contact_distances(Sm,Lm,ncm,m[::2])
        tot,pt = summarize(intact(x33,x24))
        cells.append(f"{tot:{wl-1}.1f}%"); detail.append(f"{l}:{pt[0]:.0f}/{pt[1]:.0f}/{pt[2]:.0f}")
    if any(c.strip() for c in cells):
        print(f"  {a:5.1f} - {b:5.1f}   | " + " | ".join(cells) + "   " + "  ".join(detail))
print()
print("  Leggere insieme a 37 (distanze) e 38 (partner corretti). Questo criterio")
print("  misura direttamente se i legami H di Hoogsteen esistono ancora.")
