#!/usr/bin/env python3
"""Secondo legame idrogeno di Hoogsteen, N2-H...N7, come contatto Morse sito-sito.

PERCHE'
    Con contatti B3-B3, impilamento, torsione e diedri di backbone (insieme
    bbd0) la P(r) intra arriva a 0,958 e i loop sono a posto.  Lo scarto piu'
    grande che resta sta fra i siti di base delle guanine (B1-B2, B2-B3,
    B1-B4, B3-B4 ~ 0,82-0,85) e in B3-B3 (0,78): l'orientazione relativa
    delle guanine nel piano della tetrade.  Un solo contatto B3-B3 per coppia
    e' isotropo: lascia ruotare ogni guanina attorno al suo B3.

    Nella coppia di Hoogsteen la guanina i dona due legami idrogeno alla
    vicina j: N1-H...O6 e N2-H...N7.  Nella mappatura CG
        B2 = N3, C2, N2      B3 = N1, C6, O6      B4 = C5, N7
    il primo e' il contatto B3-B3 che c'e' gia'; il secondo e' B2(i)-B4(j).
    Due legami fra punti diversi delle due basi fissano l'orientazione
    relativa: e' la direzionalita' del legame di Hoogsteen, con un termine che
    ha un significato fisico preciso.

COME
    Per ogni tetrade (i quattro G uniti dai contatti di ruolo "tetrad") e per
    ogni coppia (i, j) si misura nel riferimento la mediana di B2(i)-B4(j) e di
    B2(j)-B4(i).  Il verso di donazione e' quello piu' corto; i quattro lati
    della tetrade sono le quattro coppie con il legame piu' corto, e devono
    formare un ciclo (ogni G dona una volta e riceve una volta): e' il
    controllo che la geometria sia quella di Hoogsteen.  I parametri Morse si
    stimano come per i contatti B3-B3 (r0 = mediana, a da sigma = 1,4826 MAD),
    per classe (residuo donatore, residuo accettore) sommando le copie.

USO (in tutorials/tel26)
    python3 fit_hoogsteen_hbond.py --dataset tel26_dataset.bin \\
        --topology tel26_topology.bbd0.json --out tel26_topology.hb0.json
    python3 derive_prior_set.py --base b3stack --set hb0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_tetrad_site_morse import Reference, fit_class, morse_entry, site_type_of  # noqa: E402

ROLE = "hoogsteen_n2n7"


def tetrads_from_contacts(bonds):
    adj = defaultdict(set)
    for b in bonds:
        # le topologie piu' vecchie non scrivono il ruolo dei contatti di
        # tetrade: senza ruolo vale "tetrad", come in fit_tetrad_site_morse.py
        if str(b.get("type", "")).lower() in ("morse", "lj") and b.get("role", "tetrad") == "tetrad":
            i, j = int(b["mol_i"]), int(b["mol_j"])
            adj[i].add(j)
            adj[j].add(i)
    seen, groups = set(), []
    for m in sorted(adj):
        if m in seen:
            continue
        stack, comp = [m], set()
        while stack:
            x = stack.pop()
            if x not in comp:
                comp.add(x)
                stack.extend(adj[x] - comp)
        seen |= comp
        groups.append(sorted(comp))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--donor", default="CG_DG_B2", help="sito con N2 (donatore)")
    ap.add_argument("--acceptor", default="CG_DG_B4", help="sito con N7 (accettore)")
    ap.add_argument("--nuc", type=int, default=None)
    ap.add_argument("--D", type=float, default=50.0, help="profondita' in kJ/mol")
    ap.add_argument("--kT", type=float, default=2.49)
    ap.add_argument("--cut", type=float, default=7.0, help="r_cut = r0 + cut/a")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--width", choices=("mad", "std"), default="mad")
    args = ap.parse_args()
    fit_args = SimpleNamespace(form="morse", width=args.width, kT=args.kT, D=args.D,
                               cut=args.cut, lj_cut=3.5)

    topo = json.loads(Path(args.topology).read_text())
    nuc = args.nuc or topo.get("g4_topology", {}).get("residues_per_copy")
    if not nuc:
        sys.exit("[ERROR] residui per copia ignoti: passa --nuc")
    if any(b.get("role") == ROLE for b in topo["bonds"]):
        sys.exit(f"[ERROR] {args.topology} ha gia' contatti {ROLE}")
    groups = tetrads_from_contacts(topo["bonds"])
    if not groups or any(len(g) != 4 for g in groups):
        sys.exit(f"[ERROR] tetradi non riconosciute dai contatti 'tetrad': {[len(g) for g in groups]}")
    involved = sorted(m for g in groups for m in g)

    ref = Reference(args.dataset, args.stride)
    xd, cd, idx_d = ref.site(involved, site_type_of(topo, args.donor), args.donor)
    xa, ca, idx_a = ref.site(involved, site_type_of(topo, args.acceptor), args.acceptor)
    print(f"[INFO] {len(groups)} tetradi, {ref.frames.size} frame; donatore {args.donor} "
          f"(indice {sorted(set(idx_d.values()))}), accettore {args.acceptor} "
          f"(indice {sorted(set(idx_a.values()))})")

    def dist(i, j):
        d = xd[:, cd[i]] - xa[:, ca[j]]
        d -= ref.box * np.round(d / ref.box)
        return np.linalg.norm(d, axis=1)

    samples, bad = defaultdict(list), 0
    pairs = []
    for g in groups:
        cand = []
        for i, j in combinations(g, 2):
            dij, dji = dist(i, j), dist(j, i)
            mij, mji = float(np.median(dij)), float(np.median(dji))
            cand.append((mij, i, j, dij, mji) if mij <= mji else (mji, j, i, dji, mij))
        cand.sort(key=lambda c: c[0])
        sides = cand[:4]
        donors = sorted(c[1] for c in sides)
        acceptors = sorted(c[2] for c in sides)
        if donors != g or acceptors != g:
            bad += 1
            print(f"[WARN] tetrade {[m % nuc + 1 for m in g]} (copia {g[0] // nuc + 1}): "
                  "i quattro legami N2-N7 piu' corti non formano un ciclo di Hoogsteen")
        for m, i, j, d, other in sides:
            samples[(i % nuc + 1, j % nuc + 1)].append(d)
            pairs.append((i, j, other / m))
    if bad:
        sys.exit(f"[ERROR] {bad} tetradi senza ciclo di Hoogsteen: controlla --donor/--acceptor")

    classes = {k: fit_class(np.concatenate(v), fit_args) for k, v in sorted(samples.items())}
    print(f"\n  {'donatore->accettore':>20} {'r0 (nm)':>8} {'sigma':>7} {'a (1/nm)':>9} {'r_cut':>6}"
          f" {'verso opposto / r0':>19}")
    ratio = defaultdict(list)
    for i, j, rr in pairs:
        ratio[(i % nuc + 1, j % nuc + 1)].append(rr)
    for k, c in classes.items():
        print(f"  {f'{k[0]}->{k[1]}':>20} {c['r0']:8.3f} {c['sigma']:7.3f} {c['a']:9.2f} "
              f"{c['r_cut']:6.2f} {np.mean(ratio[k]):19.2f}")

    out = json.loads(json.dumps(topo))
    for i, j, _ in pairs:
        c = classes[(i % nuc + 1, j % nuc + 1)]
        out["bonds"].append(morse_entry(i, j, idx_d[i], idx_a[j], c, fit_args, ROLE))
    out.setdefault("g4_topology", {})["hoogsteen_second_hbond"] = (
        f"Morse {args.donor}(i)-{args.acceptor}(j), N2-H...N7 of each Hoogsteen pair, "
        f"direction and sides from the mapped all-atom reference, D={args.D}, width={args.width}")
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[INFO] {len(pairs)} contatti N2-H...N7 ({len(pairs) // len(groups)} per tetrade), "
          f"{len(classes)} classi")
    print(f"[DONE] {args.out}")


if __name__ == "__main__":
    main()
