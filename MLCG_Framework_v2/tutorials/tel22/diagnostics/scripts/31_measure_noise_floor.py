#!/usr/bin/env python3
"""Stima il noise floor irriducibile del target di force matching CG.

IDEA
  Il target e' F = f(config) + eps, con eps rumore per-frame dovuto alle
  fluttuazioni all-atom non rappresentabili nella descrizione CG.
  Per due frame con configurazione CG ~identica:
        E[||F_i - F_j||^2] = 2 * Var(eps)
  quindi la frazione massima di varianza spiegabile da QUALSIASI modello e'
        R2_max = 1 - Var(eps)/Var(F) = 1 - E[||F_i-F_j||^2] / (2*Var(F))

  Su una traiettoria MD i frame consecutivi hanno configurazione quasi
  identica, quindi il lag 1 da' la stima piu' stretta.
"""
import struct, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "tel22_dataset.bin"
buf = open(path, "rb").read()
off = 0

def take(fmt):
    global off
    n = struct.calcsize(fmt)
    v = struct.unpack_from(fmt, buf, off)
    off += n
    return v

(num_frames,) = take("i")
centers, forces, torques, sites = [], [], [], []

for _ in range(num_frames):
    num_mol, num_tot_sites = take("ii")
    take("3f")  # box
    fc, ff, ft, fs = [], [], [], []
    for _m in range(num_mol):
        _mol_id, num_sites = take("ii")
        fc.append(take("3f")); ff.append(take("3f")); ft.append(take("3f"))
        blk = np.frombuffer(buf, dtype=np.int32, count=num_sites * 4, offset=off)
        off += num_sites * 16
        fs.append(blk.reshape(num_sites, 4)[:, 1:].copy().view(np.float32))
    centers.append(fc); forces.append(ff); torques.append(ft); sites.append(np.concatenate(fs))

C = np.asarray(centers, dtype=np.float64)   # (T, M, 3)
F = np.asarray(forces,  dtype=np.float64)
Tq = np.asarray(torques, dtype=np.float64)
S = np.asarray(sites,   dtype=np.float64)   # (T, N_sites, 3)
T, M, _ = F.shape

print(f"Dataset: {path}")
print(f"  frame={T}  molecole/frame={M}  siti/frame={S.shape[1]}")
print()

varF = F.var(axis=(0, 1)).sum()      # somma delle varianze delle 3 componenti
varT = Tq.var(axis=(0, 1)).sum()
print(f"  Var(F) totale = {varF:11.1f} (kJ/mol/nm)^2   -> RMS {np.sqrt(varF):8.1f}")
print(f"  Var(T) totale = {varT:11.1f} (kJ/mol)^2      -> RMS {np.sqrt(varT):8.1f}")
print()

hdr = f"  {'lag':>4} | {'RMSD siti':>10} | {'E||dF||^2/2':>12} {'R2_max F':>9} | {'E||dT||^2/2':>12} {'R2_max T':>9}"
print(hdr); print("  " + "-" * (len(hdr) - 2))
for lag in (1, 2, 5, 10, 20, 50, 100):
    if lag >= T: break
    rmsd = np.sqrt(((S[lag:] - S[:-lag]) ** 2).sum(axis=2).mean())
    nF = ((F[lag:] - F[:-lag]) ** 2).sum(axis=2).mean() / 2.0
    nT = ((Tq[lag:] - Tq[:-lag]) ** 2).sum(axis=2).mean() / 2.0
    print(f"  {lag:4d} | {rmsd:10.4f} | {nF:12.1f} {1-nF/varF:9.3f} | {nT:12.1f} {1-nT/varT:9.3f}")

print()
print("  Interpretazione:")
print("    R2_max ~ 0  -> il target e' rumore: nessun modello puo' generalizzare")
print("                   sulle forze istantanee. Usare metriche termodinamiche.")
print("    R2_max ~ 1  -> il target e' una funzione liscia della configurazione:")
print("                   il problema e' il modello o l'ottimizzazione.")
print()
print("  Nota: il RMSD dei siti a lag 1 dice quanto cambia la configurazione CG")
print("  tra frame consecutivi. Se e' piccolo ma R2_max e' basso, il target")
print("  cambia senza che la configurazione cambi -> rumore per-frame.")
