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

QUELLO CHE QUESTO NUMERO NON E'
    L'R2 stampato in fondo NON e' un tetto per la rete: e' un PAVIMENTO, e
    per un canale solo.  Misura E[F.u | r], cioe' la frazione di varianza
    spiegabile da una funzione puramente radiale, additiva a coppie e
    isotropa della distanza intermolecolare.

    PaiNN non e' vincolato a quella forma: vede l'intorno locale completo,
    gli orientamenti dei corpi rigidi, l'identita' dei siti, la geometria a
    molti corpi.  Per la legge della varianza totale, condizionare su piu'
    informazione spiega almeno altrettanto, quindi E[F | configurazione] non
    puo' fare peggio di E[F.u | r].

    Misura reale sul TEL26: questo script da' 0.0050, il trainer arriva a
    0.087 di R2 (la sua "skill" e' esattamente 100*R2) gia' alla seconda
    epoca.  Il rapporto fra i due numeri dice quanto del segnale NON sta nel
    canale radiale di coppia, ed e' un'informazione utile -- ma chiamarlo
    tetto era sbagliato e ha prodotto previsioni fuori di un fattore venti.

A COSA SERVE DAVVERO
    1. Verificare l'ALLINEAMENTO fra forze e configurazioni.  E' l'uso
       principale: il confronto col controllo shuffled e' decisivo, e su una
       pipeline che appaia .xtc e .trr per tempo e' l'unico modo di accorgersi
       di uno sfasamento di un frame.
    2. Vedere la FORMA del segnale rimasto dopo i prior: dove sono i bin
       significativi dice quali distanze i prior non descrivono.

    NON serve a decidere se allenare, e non dice nulla su quale checkpoint
    tenere: quella scelta si fa campionando la dinamica del modello, perche'
    nessuna metrica calcolata sull'ensemble di RIFERIMENTO vede una deriva
    dell'ensemble del modello (vedi il commento in training/train_painn.cpp,
    con la tabella dei quattro modelli che le metriche ordinano al contrario).
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
          f"  -> R2 del canale radiale di coppia ~ {(m1[i]/rms)**2:.4f}")
    print()
    print("     ATTENZIONE: questo R2 e' un PAVIMENTO, non un tetto.  Vale per")
    print("     la sola parte radiale e additiva a coppie del segnale; la rete")
    print("     vede l'intorno completo e arriva molto piu' in alto (sul TEL26")
    print("     0.0050 qui contro 0.087 di skill al trainer).  Non usarlo per")
    print("     prevedere la skill ne' per decidere se allenare.")
else:
    print("  -> ATTENZIONE: nessun segnale distinguibile dal controllo shuffled.")
    print("     Sospetto disallineamento indici forze/configurazioni, oppure")
    print("     i prior catturano gia' tutta la parte sistematica.")
