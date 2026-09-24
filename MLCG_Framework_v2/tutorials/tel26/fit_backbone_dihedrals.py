#!/usr/bin/env python3
"""Diedri di backbone lungo il filamento, stimati dal riferimento all-atom.

PERCHE'
    Con contatti di Hoogsteen, impilamento e torsione a convergenza il nucleo
    di guanine e' quasi a posto, ma la P(r) intra resta lontana dove stanno i
    loop: le coppie con DA e DT sono le peggiori, l'S-S a 1,2-2 nm e' piatto e
    la copia e' troppo estesa oltre 1,4 nm.  Per il backbone ci sono solo
    legami e angoli fra nucleotidi consecutivi: niente fissa la conformazione
    di un loop.  Il termine fisico che manca e' la torsione del backbone, come
    nei modelli CG di acidi nucleici (3SPN e simili).

IL TERMINE
    Un diedro a coseno sul sito di backbone (indice --site-index, default 0:
    il sito unico di DA/DT, il sito S di DG) di quattro nucleotidi
    consecutivi m, m+1, m+2, m+3 della stessa copia: 23 per copia con 26
    residui.  phi0 = media circolare, K = kappa kT dal von Mises equivalente,
    come per la torsione fra guanine.  Una distribuzione multimodale (loop
    che visita piu' conformazioni) da' un kappa piccolo: il termine resta
    debole invece di forzare un solo stato.  --k-max limita K.

USO (in tutorials/tel26)
    python3 fit_backbone_dihedrals.py --dataset tel26_dataset.bin \\
        --topology tel26_topology.twA2.json --out tel26_topology.bbd0.json
    python3 derive_prior_set.py --base b3stack --set bbd0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_tetrad_site_morse import Reference, dihedral_angles, fit_twist  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nuc", type=int, default=None)
    ap.add_argument("--site-index", type=int, default=0, help="indice del sito di backbone")
    ap.add_argument("--kT", type=float, default=2.49)
    ap.add_argument("--k-max", type=float, default=100.0, help="K massimo, in kT")
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    topo = json.loads(Path(args.topology).read_text())
    nuc = args.nuc or topo.get("g4_topology", {}).get("residues_per_copy")
    if not nuc:
        sys.exit("[ERROR] residui per copia ignoti: passa --nuc")
    ref = Reference(args.dataset, args.stride)
    nmol = len(ref.mols)
    if nmol % nuc:
        sys.exit(f"[ERROR] {nmol} molecole non sono un multiplo di {nuc}")
    ncopy = nmol // nuc

    # posizioni del sito di backbone di ogni molecola, qualunque sia il tipo
    xyz = np.empty((ref.frames.size, nmol, 3))
    base = ref.frames[:, None] * ref.length
    for m, (w, ns) in enumerate(ref.mols):
        if args.site_index >= ns:
            sys.exit(f"[ERROR] la molecola {m} ha {ns} siti, manca l'indice {args.site_index}")
        word = w + 4 * args.site_index + 1
        xyz[:, m] = np.stack([ref.words_f[base[:, 0] + word + k] for k in range(3)], axis=-1)

    samples = defaultdict(list)
    quads = []
    for c in range(ncopy):
        for r in range(nuc - 3):
            q = [c * nuc + r + k for k in range(4)]
            quads.append(q)
            samples[r + 1].append(dihedral_angles(*(xyz[:, m] for m in q), ref.box))
    classes = {}
    for r, v in sorted(samples.items()):
        c = fit_twist(np.concatenate(v), args.kT)
        c["k"] = min(c["k"], args.k_max * args.kT)
        classes[r] = c

    print(f"[INFO] {len(quads)} diedri di backbone ({len(quads) // ncopy} per copia), "
          f"sito di indice {args.site_index}, {ref.frames.size} frame")
    print(f"\n  {'residui':>9} {'phi0 (gradi)':>13} {'dev. circ.':>11} {'K (kT)':>8}")
    for r, c in classes.items():
        print(f"  {f'{r}-{r + 3}':>9} {math.degrees(c['phi0']):13.1f} {c['circ_std_deg']:10.1f}° "
              f"{c['k'] / args.kT:8.2f}")

    out = json.loads(json.dumps(topo))
    out["dihedrals"] = [d for d in out.get("dihedrals", []) if d.get("role") != "backbone"]
    s = args.site_index
    for q in quads:
        c = classes[q[0] % nuc + 1]
        out["dihedrals"].append({
            "mol_i": q[0], "site_i": s, "mol_j": q[1], "site_j": s,
            "mol_k": q[2], "site_k": s, "mol_l": q[3], "site_l": s,
            "type": "cosine", "k": c["k"], "n": 1, "phi0": c["phi0"], "role": "backbone"})
    out.setdefault("g4_topology", {})["backbone_dihedral_representation"] = (
        f"cosine dihedral on backbone site {s} of four consecutive nucleotides; phi0 = circular "
        f"mean, K = kappa kT (von Mises) from the mapped all-atom reference, K <= {args.k_max} kT")
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    Path(args.out).with_suffix(".backbone.json").write_text(json.dumps(
        {str(r): c for r, c in classes.items()}, indent=2) + "\n")
    print(f"\n[DONE] {args.out}")


if __name__ == "__main__":
    main()
