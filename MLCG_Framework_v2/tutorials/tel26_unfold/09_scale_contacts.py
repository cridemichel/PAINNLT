#!/usr/bin/env python3
"""Scala le profondita' dei contatti Morse pair-specific di un insieme di prior.

Due classi, riconosciute dalla geometria (system.json di tel26):
  tet  contatti fra due guanine della STESSA tetrade (B3-B3 e Hoogsteen N2-N7):
       30 per copia in lp2;
  oth  tutti gli altri contatti Morse (stacking fra tetradi, loop): 24 per copia.
D -> lam_tet * D e D -> lam_oth * D; a, r0, r_cut invariati (la coda smorzata a r_cut
resta la stessa forma).  Bond armonici, angoli, diedri e WCA non vengono toccati.

Uso:
  python3 09_scale_contacts.py --in cg_priors.lp2_1c.json --out cg_priors.lp2_1c.lt0.15_lo0.20.json \\
      --lam-tet 0.15 --lam-oth 0.20
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from system_config import load_system  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam-tet", type=float, required=True)
    ap.add_argument("--lam-oth", type=float, required=True)
    args = ap.parse_args()
    if args.lam_tet < 0 or args.lam_oth < 0:
        raise SystemExit("[ERROR] fattori di scala negativi")

    sysc = load_system(HERE.parent / "tel26")
    nuc = sysc.nuc
    tetrad_of = {r - 1: k for k, t in enumerate(sysc.tetrads) for r in t}
    pri = json.load(open(args.inp))
    stats = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for e in pri["bonds"]:
        if e.get("type") != "morse":
            continue
        ri, rj = e["mol_i"] % nuc, e["mol_j"] % nuc
        if e["mol_i"] // nuc != e["mol_j"] // nuc:
            raise SystemExit(f"[ERROR] contatto Morse fra copie diverse: {e}")
        cls = "tet" if (ri in tetrad_of and rj in tetrad_of and tetrad_of[ri] == tetrad_of[rj]) else "oth"
        lam = args.lam_tet if cls == "tet" else args.lam_oth
        s = stats[cls]
        s[0] += 1; s[1] += e["D"]
        e["D"] = float(e["D"]) * lam
        s[2] += e["D"]
    if not stats:
        raise SystemExit("[ERROR] nessun contatto Morse nei bond: insieme di prior inatteso")
    if pri.get("morse_type_pairs"):
        print("[WARN] morse_type_pairs non vuoto: NON scalato (solo i contatti pair-specific)")
    ncopy = max(e["mol_i"] for e in pri["bonds"]) // nuc + 1
    meta = pri.setdefault("derived_prior_set", {})
    meta.setdefault("base", pathlib.Path(args.inp).name)
    meta["contact_scaling"] = {"lam_tet": args.lam_tet, "lam_oth": args.lam_oth,
                               "classes": "tet = stessa tetrade (B3-B3, Hoogsteen); oth = stacking, loop"}
    json.dump(pri, open(args.out, "w"), indent=1)
    for cls in ("tet", "oth"):
        n, d0, d1 = stats.get(cls, [0, 0.0, 0.0])
        print(f"  {cls}: {n // ncopy} contatti per copia, sum D {d0 / ncopy:7.1f} -> {d1 / ncopy:7.1f} kJ/mol per copia")
    tot = sum(s[2] for s in stats.values()) / ncopy
    print(f"  totale {tot:.1f} kJ/mol per copia ({tot / 4.184:.1f} kcal/mol; dH sperimentale 352 kJ/mol)  -> {args.out}")


if __name__ == "__main__":
    main()
