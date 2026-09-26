#!/usr/bin/env python3
"""Contatti persistenti delle basi dei loop (DA, DT), cercati nel riferimento.

PERCHE'
    Con l'insieme hb0 (Hoogsteen B3-B3 e N2-N7, impilamento, torsione, diedri
    di backbone) la P(r) intra e' a 0,971.  Lo scarto che resta e' nelle
    coppie fra i siti dei loop e le basi delle guanine (DT-B 0,84-0,87, DA-B
    0,81-0,83): l'orientazione dei loop rispetto al nucleo.  Nei G4 le basi
    dei loop e delle code spesso si impilano sulle tetradi esterne o si
    appaiano fra loro (coppie A-T di "cappuccio"): interazioni fisiche che i
    prior attuali non hanno.

COME
    Per ogni residuo di loop r (sito unico di DA/DT) e ogni altro residuo q
    della stessa copia, con |r - q| >= --min-sep lungo il filamento, si prende
    la coppia di siti (sito di r, sito di q) con la distanza mediana piu'
    corta sommando le copie.  E' un contatto persistente se:
        mediana <= --r-max           (a contatto)
        sigma = 1,4826 MAD <= --s-max (stabile)
        le mediane per copia stanno entro --copy-tol dalla mediana comune
    I candidati si stampano ordinati per sigma.  Con --out si aggiungono come
    contatti Morse sito-sito di ruolo "loop", stimati come i contatti di
    Hoogsteen (r0 = mediana, a da sigma, D = --D).

USO (in tutorials/tel26)
    python3 find_loop_contacts.py --dataset tel26_dataset.bin --topology tel26_topology.hb0.json
    python3 find_loop_contacts.py --dataset tel26_dataset.bin --topology tel26_topology.hb0.json \\
        --out tel26_topology.lp0.json
    python3 derive_prior_set.py --base b3stack --set lp0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_tetrad_site_morse import Reference, fit_class, morse_entry  # noqa: E402

ROLE = "loop"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--out", default=None, help="scrivi la topologia coi contatti trovati")
    ap.add_argument("--loop-sites", default="CG_DA,CG_DT", help="tipi dei siti di loop")
    ap.add_argument("--nuc", type=int, default=None)
    ap.add_argument("--min-sep", type=int, default=3, help="residui di distanza minima lungo il filamento")
    ap.add_argument("--r-max", type=float, default=0.70, help="nm")
    ap.add_argument("--s-max", type=float, default=0.05, help="nm")
    ap.add_argument("--copy-tol", type=float, default=0.10, help="nm")
    ap.add_argument("--D", type=float, default=50.0, help="profondita' in kJ/mol")
    ap.add_argument("--kT", type=float, default=2.49)
    ap.add_argument("--cut", type=float, default=7.0)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()
    fit_args = SimpleNamespace(form="morse", width="mad", kT=args.kT, D=args.D, cut=args.cut, lj_cut=3.5)

    topo = json.loads(Path(args.topology).read_text())
    nuc = args.nuc or topo.get("g4_topology", {}).get("residues_per_copy")
    if not nuc:
        sys.exit("[ERROR] residui per copia ignoti: passa --nuc")
    if any(b.get("role") == ROLE for b in topo["bonds"]):
        sys.exit(f"[ERROR] {args.topology} ha gia' contatti {ROLE}")
    names = {int(v): k for k, v in topo["mapping"]["site_types"].items()}
    loop_types = {int(topo["mapping"]["site_types"][s]) for s in args.loop_sites.split(",")}

    ref = Reference(args.dataset, args.stride)
    nmol = len(ref.mols)
    if nmol % nuc:
        sys.exit(f"[ERROR] {nmol} molecole non sono un multiplo di {nuc}")
    ncopy = nmol // nuc
    base = ref.frames[:, None] * ref.length
    xyz, types = [], []
    for w, ns in ref.mols:
        xyz.append([np.stack([ref.words_f[base[:, 0] + w + 4 * s + 1 + k] for k in range(3)],
                             axis=-1).astype(np.float64) for s in range(ns)])
        types.append([int(ref.words_i[w + 4 * s]) for s in range(ns)])
    for p in range(nuc):
        if any(types[c * nuc + p] != types[p] for c in range(ncopy)):
            sys.exit(f"[ERROR] il residuo {p + 1} non ha gli stessi siti in tutte le copie")
    loops = [p for p in range(nuc) if types[p][0] in loop_types and len(types[p]) == 1]
    label = lambda p, s: f"{p + 1}:{names[types[p][s]].replace('CG_', '')}"  # noqa: E731
    print(f"[INFO] {ref.frames.size} frame, {ncopy} copie; residui di loop "
          f"{[label(p, 0) for p in loops]}")

    existing = {(min(int(b['mol_i']), int(b['mol_j'])) % nuc, max(int(b['mol_i']), int(b['mol_j'])) % nuc)
                for b in topo["bonds"] if str(b.get("type", "")).lower() in ("morse", "lj")}

    def dist(m1, s1, m2, s2):
        d = xyz[m1][s1] - xyz[m2][s2]
        d -= ref.box * np.round(d / ref.box)
        return np.linalg.norm(d, axis=1)

    cands = []
    seen = set()
    for r in loops:
        for q in range(nuc):
            if abs(q - r) < args.min_sep or (min(r, q), max(r, q)) in seen:
                continue
            seen.add((min(r, q), max(r, q)))
            best = None
            for sq in range(len(types[q])):
                per_copy = [dist(c * nuc + r, 0, c * nuc + q, sq) for c in range(ncopy)]
                med = float(np.median(np.concatenate(per_copy)))
                if best is None or med < best[0]:
                    best = (med, sq, per_copy)
            med, sq, per_copy = best
            allr = np.concatenate(per_copy)
            sig = 1.4826 * float(np.median(np.abs(allr - med)))
            spread = max(abs(float(np.median(x)) - med) for x in per_copy)
            if med <= args.r_max:
                cands.append({"r": r, "q": q, "sq": sq, "med": med, "sig": sig, "spread": spread,
                              "ok": sig <= args.s_max and spread <= args.copy_tol,
                              "dup": (min(r, q), max(r, q)) in existing, "samples": per_copy})

    cands.sort(key=lambda c: c["sig"])
    print(f"\n  {'coppia':>16} {'mediana':>8} {'sigma':>7} {'scarto copie':>13}  esito")
    for c in cands:
        tag = "gia' presente" if c["dup"] else ("CONTATTO" if c["ok"] else "-")
        print(f"  {label(c['r'], 0) + ' - ' + label(c['q'], c['sq']):>16} {c['med']:8.3f} "
              f"{c['sig']:7.3f} {c['spread']:13.3f}  {tag}")
    chosen = [c for c in cands if c["ok"] and not c["dup"]]
    print(f"\n[INFO] {len(chosen)} contatti persistenti (mediana <= {args.r_max} nm, sigma <= "
          f"{args.s_max} nm, copie entro {args.copy_tol} nm) su {len(cands)} coppie a contatto")

    if args.out:
        out = json.loads(json.dumps(topo))
        for c in chosen:
            fc = fit_class(np.concatenate(c["samples"]), fit_args)
            for k in range(ncopy):
                out["bonds"].append(morse_entry(k * nuc + c["r"], k * nuc + c["q"], 0, c["sq"],
                                                fc, fit_args, ROLE))
        out.setdefault("g4_topology", {})["loop_contacts"] = (
            f"persistent loop-base contacts from the mapped reference: median <= {args.r_max} nm, "
            f"sigma <= {args.s_max} nm, min_sep {args.min_sep}; Morse D={args.D}")
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"[DONE] {args.out}: {len(chosen) * ncopy} contatti di ruolo '{ROLE}'")


if __name__ == "__main__":
    main()
