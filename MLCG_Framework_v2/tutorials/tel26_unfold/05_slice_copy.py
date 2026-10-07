#!/usr/bin/env python3
"""Ritaglia una copia da topologia CG e prior a N copie (es. lp2, 10 copie -> 1).

PERCHE'
    Le corse AA a 400 K (Giulia, pilota) hanno un solo TEL26.  Per mapparle e
    simularle con le STESSE interazioni del modello di produzione (lp2) serve
    la topologia a una copia: legami, angoli e diedri della copia --copy con
    gli indici di molecola rinumerati da 0, g4_topology.copies = 1, e nei
    prior le esclusioni WCA (direct_pairs, direct_site_pairs, one_three_pairs)
    ritagliate allo stesso modo.  Parametri, mapping, tipi e WCA per coppia di
    tipi restano identici.  Fallisce se trova termini fra copie diverse.

USO (in tutorials/tel26_unfold)
    python3 05_slice_copy.py --topology ref/tel26_topology.lp2.json --priors ref/cg_priors.lp2.json \\
        --out-topology tel26_topology.lp2_1c.json --out-priors cg_priors.lp2_1c.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

MOLK = ("mol_i", "mol_j", "mol_k", "mol_l")


def slice_terms(terms, lo, hi, what):
    out, other = [], 0
    for e in terms:
        ms = [int(e[k]) for k in MOLK if k in e]
        inside = [lo <= m < hi for m in ms]
        if all(inside):
            e = dict(e)
            for k in MOLK:
                if k in e:
                    e[k] = int(e[k]) - lo
            out.append(e)
        elif any(inside):
            raise SystemExit(f"[ERROR] {what}: termine fra copie diverse {ms}")
        else:
            other += 1
    return out, other


def slice_pairs(pairs, lo, hi, nmol):
    out = []
    for p in pairs:
        ms = [int(x) for x in p[:nmol]]
        if all(lo <= m < hi for m in ms):
            out.append([m - lo for m in ms] + list(p[nmol:]))
        elif any(lo <= m < hi for m in ms):
            raise SystemExit(f"[ERROR] coppia di esclusione fra copie diverse {p}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--priors", default=None)
    ap.add_argument("--out-topology", required=True)
    ap.add_argument("--out-priors", default=None)
    ap.add_argument("--copy", type=int, default=0)
    ap.add_argument("--nuc", type=int, default=None, help="residui per copia (default: g4_topology)")
    args = ap.parse_args()

    topo = json.loads(pathlib.Path(args.topology).read_text())
    g4 = topo.get("g4_topology", {})
    nuc = args.nuc or g4.get("residues_per_copy")
    ncop = g4.get("copies")
    if not nuc:
        raise SystemExit("[ERROR] residui per copia ignoti: --nuc")
    lo, hi = args.copy * nuc, (args.copy + 1) * nuc
    print(f"[INFO] {args.topology}: {ncop} copie x {nuc} residui; tengo la copia {args.copy} (molecole {lo}-{hi - 1})")
    out = json.loads(json.dumps(topo))
    for sec in ("bonds", "angles", "dihedrals"):
        if sec in out:
            out[sec], other = slice_terms(out[sec], lo, hi, sec)
            print(f"  {sec:<10s} {len(out[sec]):4d} tenuti, {other:5d} delle altre copie")
    if "g4_topology" in out:
        out["g4_topology"]["copies"] = 1
        out["g4_topology"]["sliced_from"] = {"topology": args.topology, "copy": args.copy, "copies": ncop}
    pathlib.Path(args.out_topology).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[DONE] {args.out_topology}")

    if args.priors:
        pri = json.loads(pathlib.Path(args.priors).read_text())
        outp = json.loads(json.dumps(pri))
        for sec in ("bonds", "angles", "dihedrals"):
            if sec in outp:
                outp[sec], other = slice_terms(outp[sec], lo, hi, "prior " + sec)
                print(f"  prior {sec:<10s} {len(outp[sec]):4d} tenuti, {other:5d} delle altre copie")
        w = outp.get("wca_exclusions")
        if w:
            for key, nm in (("direct_pairs", 2), ("direct_site_pairs", 2), ("one_three_pairs", 2)):
                if key in w:
                    w[key] = slice_pairs(w[key], lo, hi, nm)
                    cnt = key.replace("pairs", "pair_count")
                    if cnt in w:
                        w[cnt] = len(w[key])
                    print(f"  wca_exclusions.{key:<18s} {len(w[key]):4d}")
        if outp.get("morse_type_pairs"):
            print("[WARN] morse_type_pairs non vuoto: copiato com'e' (e' per tipo, non per molecola)")
        pathlib.Path(args.out_priors).write_text(json.dumps(outp, indent=2) + "\n")
        print(f"[DONE] {args.out_priors}")


if __name__ == "__main__":
    main()
