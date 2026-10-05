#!/usr/bin/env python3
"""Contatti dei loop DIPENDENTI DALLO STATO, deboli, da aggiungere a una topologia.

PERCHE'
    find_loop_contacts.py ammette solo contatti persistenti in TUTTE le copie
    (--copy-tol): i contatti presenti in una parte delle copie o del tempo --
    T1 con A15 e con A21, il loop 3 vicino al solco, A9 su G18 -- restano
    fuori per costruzione, e i loro residui nel CG hanno solo WCA e residuo ML
    (script 49-52: distribuzioni larghe e unimodali dove l'AA ha picchi di
    contatto).  Un contatto bistabile con il D dei contatti persistenti
    (50 kJ/mol, ~20 kT) sarebbe sempre formato: qui il contatto e' un Morse
    STRETTO (r0 e larghezza dal picco di contatto AA) con D piccolo, da
    regolare poi sulla popolazione AA del contatto (tune_state_contacts.py).

COME, per ogni coppia di residui (1-based, stessa copia)
    - coppia di siti: quella con il 10-esimo percentile della distanza piu'
      basso, sommando copie e frame (T e A hanno un sito solo, le G sei);
    - bacino di contatto: istogramma della distanza (passo 0,01 nm), primo
      picco sotto --r-max e primo minimo dopo di esso (r_basin);
    - r0 = mediana dei campioni nel bacino, sigma = 1,4826 MAD;
      a = sqrt(kT / (2 D_shape sigma^2)) con D_shape = --D, r_cut = r0 + --cut/a;
      a resta fisso quando D viene regolato (la forma e' quella del picco AA
      alla profondita' iniziale);
    - popolazione AA del contatto: frazione dei campioni con d < r_basin,
      totale e per copia, errore = std fra copie / sqrt(copie).
    I contatti (ruolo "state") si scrivono per tutte le copie; i dati del
    bacino e la popolazione AA vanno in g4_topology.state_contacts, dove li
    legge tune_state_contacts.py.

USO (in tutorials/tel26)
    python3 add_state_contacts.py --dataset tel26_lp2_dataset.bin \\
        --topology tel26_topology.lp2.json --pairs 1-15,1-21,20-18,9-18 --D 5 \\
        --out tel26_topology.lp2c.json
    python3 derive_prior_set.py --base lp2 --set lp2c
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_tetrad_site_morse import Reference  # noqa: E402

ROLE = "state"


def load_sites(ref):
    """xyz[mol][site] -> (F, 3), types[mol] -> lista dei tipi dei siti."""
    base = ref.frames[:, None] * ref.length
    xyz, types = [], []
    for w, ns in ref.mols:
        xyz.append([np.stack([ref.words_f[base[:, 0] + w + 4 * s + 1 + k] for k in range(3)],
                             axis=-1).astype(np.float64) for s in range(ns)])
        types.append([int(ref.words_i[w + 4 * s]) for s in range(ns)])
    return xyz, types


def contact_basin(d, r_max, step=0.01):
    """Primo picco sotto r_max e primo minimo dopo; None se non c'e' un picco."""
    edges = np.arange(0.2, max(r_max + 0.6, 2.0), step)
    h, e = np.histogram(d, bins=edges)
    c = 0.5 * (e[1:] + e[:-1])
    k = np.convolve(h, np.ones(5) / 5, mode="same")          # smussamento leggero
    inside = np.flatnonzero(c <= r_max)
    if inside.size == 0 or k[inside].max() <= 0:
        return None
    # primo massimo locale significativo sotto r_max
    peaks = [i for i in inside[1:-1] if k[i] >= k[i - 1] and k[i] >= k[i + 1] and k[i] > 0.05 * k.max()]
    if not peaks:
        return None
    p = peaks[0]
    j = p
    # discesa fino al primo minimo locale (sotto l'80 % del picco), al piu' 0,4 nm oltre r_max
    while j + 1 < k.size and c[j] < r_max + 0.4:
        if k[j + 1] > k[j] and k[j] < 0.8 * k[p]:
            break
        j += 1
    return float(c[p]), float(c[j])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="riferimento AA mappato (posizioni)")
    ap.add_argument("--topology", required=True)
    ap.add_argument("--pairs", required=True, help="coppie di residui 1-based: '1-15,1-21,...'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nuc", type=int, default=None)
    ap.add_argument("--D", type=float, default=5.0, help="profondita' iniziale (kJ/mol), anche D_shape per a")
    ap.add_argument("--kT", type=float, default=2.49)
    ap.add_argument("--cut", type=float, default=7.0, help="r_cut = r0 + cut / a")
    ap.add_argument("--r-max", type=float, default=1.0, help="nm: il picco di contatto va cercato sotto")
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    topo = json.loads(Path(args.topology).read_text())
    nuc = args.nuc or topo.get("g4_topology", {}).get("residues_per_copy")
    if not nuc:
        sys.exit("[ERROR] residui per copia ignoti: passa --nuc")
    if any(b.get("role") == ROLE for b in topo["bonds"]):
        sys.exit(f"[ERROR] {args.topology} ha gia' contatti di ruolo '{ROLE}'")
    names = {int(v): k for k, v in topo["mapping"]["site_types"].items()}
    pairs = [tuple(int(x) - 1 for x in p.split("-")) for p in args.pairs.split(",") if p.strip()]

    ref = Reference(args.dataset, args.stride)
    nmol = len(ref.mols)
    if nmol % nuc:
        sys.exit(f"[ERROR] {nmol} molecole non sono un multiplo di {nuc}")
    ncopy = nmol // nuc
    xyz, types = load_sites(ref)
    existing = {(min(int(b["mol_i"]), int(b["mol_j"])) % nuc, max(int(b["mol_i"]), int(b["mol_j"])) % nuc)
                for b in topo["bonds"] if str(b.get("type", "")).lower() in ("morse", "lj")}

    def dist(m1, s1, m2, s2):
        d = xyz[m1][s1] - xyz[m2][s2]
        d -= ref.box * np.round(d / ref.box)
        return np.linalg.norm(d, axis=1)

    def label(p, s):
        return f"{p + 1}:{names[types[p][s]].replace('CG_', '')}"

    print(f"[INFO] {ref.frames.size} frame, {ncopy} copie, D iniziale {args.D} kJ/mol, kT {args.kT}")
    out = json.loads(json.dumps(topo))
    meta = []
    for r, q in pairs:
        if (min(r, q), max(r, q)) in existing:
            print(f"[WARN] {r + 1}-{q + 1}: la topologia ha gia' un contatto fra questi residui; lo aggiungo comunque")
        best = None
        for sr in range(len(types[r])):
            for sq in range(len(types[q])):
                per_copy = [dist(c * nuc + r, sr, c * nuc + q, sq) for c in range(ncopy)]
                allr = np.concatenate(per_copy)
                p10 = float(np.percentile(allr, 10))
                if best is None or p10 < best[0]:
                    best = (p10, sr, sq, per_copy, allr)
        p10, sr, sq, per_copy, allr = best
        basin = contact_basin(allr, args.r_max)
        if basin is None:
            print(f"[WARN] {label(r, sr)} - {label(q, sq)}: nessun picco di contatto sotto {args.r_max} nm; salto")
            continue
        r_peak, r_basin = basin
        inb = allr[allr < r_basin]
        r0 = float(np.median(inb))
        sig = 1.4826 * float(np.median(np.abs(inb - r0)))
        sig = max(sig, 0.01)
        a = math.sqrt(args.kT / (2.0 * args.D * sig * sig))
        r_cut = r0 + args.cut / a
        f_copy = np.array([(x < r_basin).mean() for x in per_copy])
        f_aa = float((allr < r_basin).mean())
        f_err = float(f_copy.std(ddof=1) / math.sqrt(ncopy))
        for k in range(ncopy):
            out["bonds"].append({"mol_i": k * nuc + r, "mol_j": k * nuc + q, "site_i": sr, "site_j": sq,
                                 "exclude_wca": False, "role": ROLE, "type": "morse",
                                 "D": args.D, "a": a, "r0": r0, "r_cut": r_cut})
        meta.append({"pair": f"{r + 1}-{q + 1}", "res_i": r + 1, "res_j": q + 1, "site_i": sr, "site_j": sq,
                     "label": f"{label(r, sr)}-{label(q, sq)}", "r_peak": r_peak, "r_basin": r_basin,
                     "r0": r0, "sigma": sig, "a": a, "r_cut": r_cut, "D": args.D, "D_shape": args.D,
                     "f_aa": f_aa, "f_aa_err": f_err, "f_aa_per_copy": f_copy.tolist()})
        print(f"  {label(r, sr) + ' - ' + label(q, sq):>18s}  picco {r_peak:.3f}  bacino < {r_basin:.3f} nm  "
              f"r0 {r0:.3f}  sigma {sig:.3f}  a {a:.2f}  r_cut {r_cut:.3f}  "
              f"f_AA {f_aa:.3f} ± {f_err:.3f}  per copia " + " ".join(f"{v:.2f}" for v in f_copy))
    if not meta:
        sys.exit("[ERROR] nessun contatto aggiunto")
    g4 = out.setdefault("g4_topology", {})
    g4["state_contacts"] = meta
    g4["state_contacts_note"] = (f"contatti di stato (ruolo '{ROLE}') da add_state_contacts.py su {args.dataset}: "
                                 f"Morse stretto dal picco di contatto AA, D iniziale {args.D} kJ/mol, "
                                 f"a fisso (D_shape), da regolare con tune_state_contacts.py")
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[DONE] {args.out}: {len(meta)} contatti x {ncopy} copie di ruolo '{ROLE}'")


if __name__ == "__main__":
    main()
