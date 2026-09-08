#!/usr/bin/env python3
"""Noise floor alla risoluzione dell'ambiente LOCALE (cio' che la rete vede).

La rete predice la forza residua di una molecola dal suo ambiente entro
cutoff. Quindi il noise floor va misurato su coppie di istanze
(frame, molecola) con ambiente locale ~identico, non su frame interi.

Per due istanze con ambiente identico:  E[||F_i-F_j||^2] = 2*Var(eps)
  =>  R2_max = 1 - E[||F_i-F_j||^2] / (2*Var(F))

Descrittore invariante per traslazione/rotazione/permutazione:
distanze minimum-image alle K molecole piu' vicine, ordinate.
Le coppie sono ristrette alla stessa posizione nucleotidica (mol_id % 22)
cosi' si confrontano molecole dello stesso tipo chimico.
"""
import struct, sys
import numpy as np

path = sys.argv[1]
K = 12
NUC_PER_COPY = 22

buf = open(path, "rb").read()
off = 0
def take(fmt):
    global off
    v = struct.unpack_from(fmt, buf, off); off += struct.calcsize(fmt); return v

(T,) = take("i")
C_all, F_all, box_all = [], [], []
for _ in range(T):
    num_mol, _nts = take("ii")
    box_all.append(take("3f"))
    fc, ff = [], []
    for _m in range(num_mol):
        _mid, ns = take("ii")
        fc.append(take("3f")); ff.append(take("3f")); take("3f")
        off += ns * 16
    C_all.append(fc); F_all.append(ff)

C = np.asarray(C_all, dtype=np.float32)      # (T,M,3)
F = np.asarray(F_all, dtype=np.float64)
L = np.asarray(box_all, dtype=np.float32)    # (T,3)
T, M, _ = C.shape
varF = F.var(axis=(0, 1)).sum()
print(f"  frame={T} molecole={M}  Var(F)={varF:.1f}  RMS={np.sqrt(varF):.1f}")

# --- descrittori: K distanze minimum-image piu' vicine, per ogni (t,m) ---
desc = np.empty((T, M, K), dtype=np.float32)
for t in range(T):
    d = C[t][:, None, :] - C[t][None, :, :]
    d -= L[t] * np.round(d / L[t])                    # minimum image
    r = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    r.sort(axis=1)
    desc[t] = r[:, :K]

D = desc.reshape(-1, K)
Ff = F.reshape(-1, 3)
nuc = np.tile(np.arange(M) % NUC_PER_COPY, T)
print(f"  istanze totali: {D.shape[0]}")

rng = np.random.default_rng(0)
rows = []
for nid in range(NUC_PER_COPY):
    idx = np.flatnonzero(nuc == nid)
    if idx.size < 200: continue
    q = rng.choice(idx, size=min(400, idx.size), replace=False)
    Dq, Dc = D[q], D[idx]
    # distanza euclidea nello spazio descrittore, a blocchi
    best_d = np.full(q.size, np.inf, dtype=np.float32)
    best_j = np.zeros(q.size, dtype=np.int64)
    for s in range(0, q.size, 100):
        e = min(s + 100, q.size)
        dd = ((Dq[s:e, None, :] - Dc[None, :, :]) ** 2).sum(-1)
        # esclude se stesso
        same = (idx[None, :] == q[s:e, None])
        dd[same] = np.inf
        j = dd.argmin(1)
        best_d[s:e] = np.sqrt(dd[np.arange(e - s), j])
        best_j[s:e] = idx[j]
    rows.append(np.column_stack([best_d, ((Ff[q] - Ff[best_j]) ** 2).sum(1)]))

R = np.vstack(rows)
print(f"  coppie nearest-neighbour analizzate: {R.shape[0]}")
print()
hdr = f"  {'quantile dist':>14} | {'dist descr (nm)':>15} | {'E||dF||^2/2':>12} | {'R2_max':>7} | {'n':>5}"
print(hdr); print("  " + "-" * (len(hdr) - 2))
qs = [1, 2, 5, 10, 25, 50, 100]
prev = 0.0
for qq in qs:
    thr = np.percentile(R[:, 0], qq)
    sel = R[(R[:, 0] > prev) & (R[:, 0] <= thr)]
    if sel.shape[0] < 5: prev = thr; continue
    nF = sel[:, 1].mean() / 2.0
    print(f"  {'<= p'+str(qq):>14} | {thr:15.4f} | {nF:12.1f} | {1-nF/varF:7.3f} | {sel.shape[0]:5d}")
    prev = thr
print()
print("  Se R2_max resta ~0 anche per le coppie con descrittore piu' simile,")
print("  il target non e' funzione dell'ambiente locale -> rumore irriducibile.")
print("  Se R2_max cresce al restringersi della distanza, c'e' segnale")
print("  apprendibile e il limite e' la quantita' di dati.")
