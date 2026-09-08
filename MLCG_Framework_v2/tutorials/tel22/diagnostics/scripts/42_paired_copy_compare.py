#!/usr/bin/env python3
"""Confronto appaiato fra due run CG, sulle 10 copie di TEL22.

PERCHE' ESISTE
  Lo script 40 stampa un numero per run e nessuna barra d'errore, quindi
  non permette di distinguere un effetto da una fluttuazione. Ma ogni run
  contiene 10 copie indipendenti che partono dallo STESSO equilibrated.npz:
  la copia c del run A e la copia c del run B sono appaiate. Questo da'
  10 differenze appaiate e un test esatto di permutazione (2^10 = 1024
  assegnazioni di segno, enumerabili per intero).

  Serve inoltre un PAVIMENTO DI RUMORE. ml_10kstep e ml_50kstep usano lo
  stesso identico file di pesi (sha ab8aaec0) con gli stessi input e lo
  stesso seed: ogni loro differenza e' non-determinismo float32 su MPS.
  Un effetto piu' piccolo di quel pavimento non e' un effetto.

DUE MISURE, NON UNA
  'largo'  = frazione di legami H nella finestra 0.35 < d33 < 0.60 nm.
             E' un CONTEGGIO IN BANDA: quando la maggior parte delle coppie
             e' gia' oltre 0.60 nm, misura la coda sinistra di una
             distribuzione che collassa, e una distribuzione piu' larga e
             piu' calda puo' segnare piu' alto pur essendo piu' lontana
             dal riferimento. Da solo e' fuorviante.
  'W1'     = distanza di Wasserstein-1 fra la distribuzione delle d33 del
             run e quella del riferimento all-atom. Piu' BASSA = piu'
             fedele. Non premia la dispersione. E' la misura da leggere
             quando le due disagree.

USO
  python3 42_paired_copy_compare.py <dataset.bin> <A>=<samples.npz> <B>=<samples.npz> [t0 t1]
"""
import sys, pathlib, itertools
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _hb_common import CONTACTS, TETRAD_OF, load_reference_sites, load_sample_sites, contact_distances

T33_LO, T33, T24 = 0.35, 0.60, 0.65
CYCLE_1B = {0: [(2,10),(10,22),(22,14),(14,2)],
            1: [(3,9),(9,21),(21,15),(15,3)],
            2: [(4,8),(8,20),(20,16),(16,4)]}
hb_idx = [k for k, (ri, rj, _, _) in enumerate(CONTACTS)
          if any({ri+1, rj+1} == set(p) for p in CYCLE_1B[TETRAD_OF[k]])]
assert len(hb_idx) == 12


def w1(a, b):
    """Wasserstein-1 fra due campioni 1-D, via quantili comuni."""
    a = np.sort(a[np.isfinite(a)]); b = np.sort(b[np.isfinite(b)])
    if a.size == 0 or b.size == 0:
        return np.nan
    q = np.linspace(0, 1, 512)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def per_copy(path, t0, t1, ref_d33):
    """-> (largo%, stretto%, W1) per ciascuna delle ncopy copie."""
    S, L, ncopy, t = load_sample_sites(path)
    frames = np.flatnonzero((t >= t0) & (t < t1))
    d33, d24 = contact_distances(S, L, ncopy, frames)
    # righe ordinate (frame, copia) -> separo le copie
    d33 = d33.reshape(len(frames), ncopy, 18)[:, :, hb_idx]
    d24 = d24.reshape(len(frames), ncopy, 18)[:, :, hb_idx]
    out = []
    for c in range(ncopy):
        a, b = d33[:, c, :], d24[:, c, :]
        largo = (a > T33_LO) & (a < T33)
        stretto = largo & (b < T24)
        out.append((100 * np.nanmean(np.where(np.isnan(a), np.nan, largo.astype(float))),
                    100 * np.nanmean(np.where(np.isnan(a), np.nan, stretto.astype(float))),
                    w1(a.ravel(), ref_d33)))
    return np.array(out)


def paired_perm(d):
    """p a due code, enumerazione esatta dei 2^n segni (n<=20)."""
    d = d[np.isfinite(d)]; n = d.size
    if n == 0:
        return np.nan, np.nan, (np.nan, np.nan)
    obs = d.mean()
    means = np.array([np.dot(s, d) / n for s in itertools.product([1, -1], repeat=n)])
    p = float(np.mean(np.abs(means) >= abs(obs) - 1e-12))
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return obs, p, (obs - 1.96 * se, obs + 1.96 * se)


ref_path, t0, t1 = sys.argv[1], 5.0, 10.0
runs = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
tail = [a for a in sys.argv[2:] if "=" not in a]
if len(tail) == 2:
    t0, t1 = float(tail[0]), float(tail[1])

Sr, Lr, ncr = load_reference_sites(ref_path)
rd33, _ = contact_distances(Sr, Lr, ncr, range(0, Sr.shape[0], 5))
ref_d33 = rd33[:, hb_idx].ravel()
ref_largo = 100 * np.nanmean((rd33[:, hb_idx] > T33_LO) & (rd33[:, hb_idx] < T33))

print(f"  finestra {t0}-{t1} ps   |   riferimento all-atom: largo {ref_largo:.1f}%  (W1 = 0 per definizione)")
print()
res = {l: per_copy(p, t0, t1, ref_d33) for l, p in runs.items()}
names = list(res)

print(f"  {'run':<14} {'largo %':>16} {'stretto %':>16} {'W1 (nm), piu basso = meglio':>30}")
for l in names:
    r = res[l]
    for j, lab in enumerate(["largo", "stretto", "w1"]):
        pass
    print(f"  {l:<14} {r[:,0].mean():>8.1f} +- {r[:,0].std(ddof=1)/np.sqrt(len(r)):<4.1f} "
          f"{r[:,1].mean():>9.1f} +- {r[:,1].std(ddof=1)/np.sqrt(len(r)):<4.1f} "
          f"{r[:,2].mean():>20.4f} +- {r[:,2].std(ddof=1)/np.sqrt(len(r)):.4f}")
print("  (+- = errore standard sulle 10 copie)")

if len(names) == 2:
    A, B = names
    print(f"\n  TEST APPAIATO SULLE COPIE:  {B} - {A}")
    for j, lab in enumerate(["largo %", "stretto %", "W1 (nm)"]):
        d = res[B][:, j] - res[A][:, j]
        obs, p, (lo, hi) = paired_perm(d)
        better = "meglio" if (j < 2 and obs > 0) or (j == 2 and obs < 0) else "peggio"
        sig = "SIGNIFICATIVO" if p < 0.05 else "non significativo"
        print(f"    {lab:<10} {obs:+8.3f}  IC95 [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}  "
              f"({sig}; {B} {better})  [copie migliori: {int(np.sum(d>0) if j<2 else np.sum(d<0))}/{len(d)}]")
