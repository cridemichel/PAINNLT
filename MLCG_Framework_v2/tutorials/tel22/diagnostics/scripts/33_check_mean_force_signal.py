#!/usr/bin/env python3
"""Verifica che forze residue e configurazioni siano allineate, e misura
l'ampiezza del SEGNALE di forza media (cio' che il force matching apprende).

Metodo: per ogni coppia di molecole (m,n) entro cutoff, proietta la forza
residua di m sull'asse m->n. Mediando su molte coppie, la parte sistematica
sopravvive e il rumore si media via.

  - se allineato e c'e' segnale: a corto raggio la proiezione media e'
    chiaramente negativa (repulsione: m spinto via da n)
  - se piatta a zero come il controllo shuffled: o i prior hanno catturato
    tutto, o forze e configurazioni sono disallineate (bug)
"""
import struct, sys
import numpy as np

path = sys.argv[1]
CUT = 1.2616
NFRAMES = 300

buf = open(path, "rb").read(); off = 0
def take(fmt):
    global off
    v = struct.unpack_from(fmt, buf, off); off += struct.calcsize(fmt); return v
(T,) = take("i")
C_all, F_all, box_all = [], [], []
for _ in range(T):
    nm, _n = take("ii"); box_all.append(take("3f"))
    fc, ff = [], []
    for _m in range(nm):
        _mid, ns = take("ii")
        fc.append(take("3f")); ff.append(take("3f")); take("3f"); off += ns * 16
    C_all.append(fc); F_all.append(ff)
C = np.asarray(C_all, np.float64); F = np.asarray(F_all, np.float64)
L = np.asarray(box_all, np.float64); T, M, _ = C.shape
rms = np.sqrt(F.var(axis=(0,1)).sum() / 3)
print(f"  frame={T} molecole={M}  RMS forza residua (per componente) = {rms:.1f} kJ/mol/nm")

edges = np.linspace(0.2, CUT, 18)
def analyse(Fuse, label):
    num = np.zeros(len(edges)-1); den = np.zeros(len(edges)-1); sq = np.zeros(len(edges)-1)
    step = max(1, T // NFRAMES)
    for t in range(0, T, step):
        d = C[t][None,:,:] - C[t][:,None,:]
        d -= L[t] * np.round(d / L[t])
        r = np.sqrt((d**2).sum(-1))
        np.fill_diagonal(r, np.inf)
        u = np.where(r[...,None] > 0, d / np.maximum(r,1e-12)[...,None], 0.0)
        proj = (Fuse[t][:,None,:] * u).sum(-1)          # F_m . u_{m->n}
        b = np.digitize(r, edges) - 1
        ok = (b >= 0) & (b < len(edges)-1) & np.isfinite(r)
        np.add.at(num, b[ok], proj[ok]); np.add.at(den, b[ok], 1.0)
        np.add.at(sq,  b[ok], proj[ok]**2)
    mean = num / np.maximum(den,1)
    sd   = np.sqrt(np.maximum(sq/np.maximum(den,1) - mean**2, 0))
    sem  = sd / np.sqrt(np.maximum(den,1))
    print(f"\n  === {label} ===")
    print(f"  {'r (nm)':>12} | {'<F.u> medio':>12} | {'SEM':>8} | {'z':>7} | {'n':>9}")
    print("  " + "-"*60)
    for i in range(len(edges)-1):
        if den[i] < 50: continue
        z = mean[i]/sem[i] if sem[i] > 0 else 0
        flag = "  <<<" if abs(z) > 5 else ""
        print(f"  {edges[i]:5.2f}-{edges[i+1]:5.2f} | {mean[i]:12.2f} | {sem[i]:8.2f} | {z:7.1f} | {int(den[i]):9d}{flag}")
    return mean, sem

m1, s1 = analyse(F, "DATI REALI (forze allineate alle configurazioni)")
rng = np.random.default_rng(0)
Fsh = F.copy()
for t in range(T):                     # controllo: forze rimescolate nel frame
    Fsh[t] = Fsh[t][rng.permutation(M)]
m2, s2 = analyse(Fsh, "CONTROLLO shuffled (atteso: nessun segnale)")

zr = np.nanmax(np.abs(m1/np.where(s1>0,s1,np.nan)))
zs = np.nanmax(np.abs(m2/np.where(s2>0,s2,np.nan)))
print()
print(f"  |z| massimo dati reali : {zr:6.1f}")
print(f"  |z| massimo shuffled   : {zs:6.1f}")
print()
if zr > 5 and zr > 3*zs:
    print("  -> ALLINEAMENTO OK: esiste un segnale di forza media significativo.")
    print("     Il force matching ha qualcosa da apprendere; la MSE istantanea")
    print("     non lo misura perche' sommersa dal rumore.")
    i = np.nanargmax(np.abs(m1/np.where(s1>0,s1,np.nan)))
    print(f"     Ampiezza segnale ~{abs(m1[i]):.1f} vs RMS istantaneo {rms:.1f}"
          f"  -> tetto teorico R2 ~ {(m1[i]/rms)**2:.4f}")
else:
    print("  -> ATTENZIONE: nessun segnale distinguibile dal controllo shuffled.")
    print("     Sospetto disallineamento indici forze/configurazioni, oppure")
    print("     i prior catturano gia' tutta la parte sistematica.")
