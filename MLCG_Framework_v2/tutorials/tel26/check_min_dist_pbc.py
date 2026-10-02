#!/usr/bin/env python3
"""Controllo a posteriori dei contatti fra copie con l'immagine minima.

Fino al commit MLCG_COMMIT_MSG_mindist_pbc il guardrail su min_dist di
run_cg_md.py usava le posizioni non ripiegate senza immagine minima: un
contatto fra due copie attraverso il bordo della box non era visto.  Le forze e
la dinamica non ne dipendevano; poteva mancare solo un abort.  Questo script
rilegge le traiettorie --sample_npz gia' prodotte e misura, frame per frame, la
distanza minima fra siti di copie diverse con l'immagine minima.

Uso:
  python3 check_min_dist_pbc.py samples_a.npz samples_b.npz ...
  python3 check_min_dist_pbc.py --residues-per-copy 26 --threshold 0.15 $(ls *samples*.npz)

Per ogni file: frame, distanza minima fra copie (nm) e frame in cui si
raggiunge, numero di frame sotto la soglia del guardrail e sotto 0.3 nm.
I file senza 'sites'/'site_molecule'/'box' sono saltati.
"""
import argparse
import sys

import numpy as np
from scipy.spatial import cKDTree


def min_intercopy_distance(sites, copy_of_site, box, probe):
    """Distanza minima fra siti di copie diverse per ogni frame (inf se > probe)."""
    out = np.full(len(sites), np.inf)
    for t, pos in enumerate(sites):
        ok = np.isfinite(pos).all(axis=1)
        p = np.mod(pos[ok], box)
        p[p >= box] = 0.0          # np.mod puo' restituire esattamente box
        c = copy_of_site[ok]
        tree = cKDTree(p, boxsize=box)
        pairs = tree.query_pairs(probe, output_type="ndarray")
        if len(pairs) == 0:
            continue
        pairs = pairs[c[pairs[:, 0]] != c[pairs[:, 1]]]
        if len(pairs) == 0:
            continue
        d = p[pairs[:, 0]] - p[pairs[:, 1]]
        d -= box * np.round(d / box)
        out[t] = float(np.sqrt((d * d).sum(axis=1)).min())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--residues-per-copy", type=int, default=26)
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="soglia del guardrail di run_cg_md.py (nm)")
    ap.add_argument("--stride", type=int, default=1, help="un frame ogni STRIDE")
    ap.add_argument("--probe", type=float, default=1.0,
                    help="raggio di ricerca delle coppie (nm); oltre si riporta 'oltre'")
    args = ap.parse_args()

    print(f"{'file':60s} {'frame':>6s} {'min fra copie':>14s} {'al frame':>9s} "
          f"{'<' + format(args.threshold, 'g'):>7s} {'<0.3':>6s}")
    worst = np.inf
    for path in args.files:
        try:
            data = np.load(path, allow_pickle=False)
            sites = np.asarray(data["sites"], dtype=float)
            site_molecule = np.asarray(data["site_molecule"], dtype=np.int64)
            box = np.asarray(data["box"], dtype=float).reshape(-1)[:3]
        except (KeyError, ValueError, OSError) as exc:
            print(f"{path:60s} saltato ({exc.__class__.__name__}: {exc})")
            continue
        if sites.ndim != 3 or sites.shape[1] != len(site_molecule):
            print(f"{path:60s} saltato (forma di sites {sites.shape} incoerente)")
            continue
        sites = sites[::max(1, args.stride)]
        copy_of_site = site_molecule // args.residues_per_copy
        if len(np.unique(copy_of_site)) < 2:
            print(f"{path:60s} saltato (una sola copia)")
            continue
        dmin = min_intercopy_distance(sites, copy_of_site, box, args.probe)
        t = int(np.argmin(dmin))
        m = dmin[t]
        worst = min(worst, m)
        shown = f"{m:14.4f}" if np.isfinite(m) else f"{'oltre ' + format(args.probe, 'g'):>14s}"
        print(f"{path:60s} {len(sites):6d} {shown} {t * max(1, args.stride):9d} "
              f"{int((dmin < args.threshold).sum()):7d} {int((dmin < 0.3).sum()):6d}")
    if np.isfinite(worst):
        verdict = "SOTTO la soglia" if worst < args.threshold else "sopra la soglia"
        print(f"\nminimo su tutti i file: {worst:.4f} nm ({verdict} di {args.threshold:g} nm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
