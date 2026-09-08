#!/usr/bin/env python3
"""Deriva temporale degli osservabili G-quadruplex: quanto il modello CG
allontana la struttura dal riferimento all-atom, in funzione del tempo.

E' la vista piu' informativa quando la run e' corta e non equilibrata:
misura la STABILITA' strutturale, non una distribuzione di equilibrio.

USO
  python3 37_drift_over_time.py <dataset.bin> <label>=<samples.npz> [...]
"""
import sys, importlib.util, pathlib
import numpy as np

here = pathlib.Path(__file__).parent
src = (here / "36_compare_tetrads.py").read_text().split("ref_path, models")[0]
ns = {}; exec(src, ns)
obs_from_gcen, load_samples, load_reference = (
    ns["obs_from_gcen"], ns["load_samples"], ns["load_reference"])

ref_path = sys.argv[1]
runs = dict(a.split("=", 1) for a in sys.argv[2:])

Gr, Lr, nc = load_reference(ref_path)
ref = obs_from_gcen(Gr, Lr, nc)
KEYS = ["rise2_3", "rise3_4", "tet3_d4", "tet4_d4", "tet4_d0"]

print(f"  riferimento: {Gr.shape[0]} frame all-atom mappati\n")
for label, path in runs.items():
    Gm, Lm, ncm = load_samples(path)
    z = np.load(path); tps = z["time_ps"]
    T = Gm.shape[0]
    print(f"  ===== {label}  ({T} frame, {tps[0]:.1f}-{tps[-1]:.1f} ps) =====")
    print(f"  {'t (ps)':>8} | " + " | ".join(f"{k:>13}" for k in KEYS) + " | dev.rel media")
    print("  " + "-" * (11 + 16 * len(KEYS) + 16))
    print(f"  {'RIF':>8} | " + " | ".join(f"{ref[k].mean():6.3f}±{ref[k].std():.3f}" for k in KEYS) + " |         --")
    nseg = 8
    for s in range(nseg):
        a, b = T * s // nseg, max(T * (s + 1) // nseg, T * s // nseg + 1)
        o = obs_from_gcen(Gm[a:b], Lm, ncm)
        devs = [abs(o[k].mean() - ref[k].mean()) / ref[k].mean() for k in KEYS]
        print(f"  {tps[min(b-1,T-1)]:8.2f} | " +
              " | ".join(f"{o[k].mean():6.3f}±{o[k].std():.3f}" for k in KEYS) +
              f" |    {100*np.mean(devs):6.1f}%")
    print()
print("  dev.rel media = scostamento relativo medio dalle medie di riferimento.")
print("  Stabile e piatta -> il modello preserva la struttura.")
print("  In crescita monotona -> il modello la destabilizza.")
