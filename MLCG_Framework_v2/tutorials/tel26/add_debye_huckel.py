#!/usr/bin/env python3
"""Aggiunge a una topologia CG una repulsione di Debye-Hueckel fra i backbone.

PERCHE'
    Nei prior non c'e' elettrostatica: fra copie diverse e fra siti S lontani
    della stessa copia agisce solo la WCA.  Nel riferimento all-atom i fosfati
    si respingono, schermati dai K+: e' la repulsione che tiene separate le
    copie (S-S inter ~0.04 sotto 1 nm) e che da' forma ai loop (S-S intra a
    1.2-2 nm).  Il residuo ML, dove il dataset non ha quasi campioni -- i
    contatti fra copie --, non la impara: estrapola, e sul TEL26 attacca le
    copie fra loro.

IL MODELLO
    U = (e^2 / 4 pi eps0 eps_r) q_i q_j exp(-r / lambda_D) / r,  r < r_cut
    carica q sui siti di backbone: il sito unico di DA/DT, il sito S di DG.
    Agisce su TUTTE le coppie di siti carichi di molecole diverse, legate
    comprese: ESPResSo non applica le esclusioni all'elettrostatica, e il
    builder del dataset sottrae la stessa somma.  Sui primi vicini (0.6 nm)
    la forza e' ~4 kJ/mol/nm, contro costanti di legame di migliaia: sposta la
    lunghezza di equilibrio di ~1e-3 nm.

LA LUNGHEZZA DI DEBYE
    O data direttamente (--debye-length), o dagli ioni del riferimento:
        --ions N --box L    (N ioni monovalenti in una scatola cubica di lato L nm)
    con forza ionica I = c/2 (solo i controioni sono mobili; le cariche del
    backbone sono fisse) e
        lambda_D = sqrt(eps_r eps0 kB T / (2 N_A e^2 I)).
    Per il TEL26: 250 K+ in 11.91 nm -> c = 0.246 M, I = 0.123 M,
    lambda_D ~ 0.87 nm a 300 K.  E' un'approssimazione di campo medio: la
    condensazione dei controioni sul quadruplex riduce la carica efficace,
    e --charge permette di scalarla.

USO
    python3 add_debye_huckel.py --topology tel26_topology.b3stack.json \\
        --out tel26_topology.b3dh.json --ions 250 --box 11.91
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

E = 1.602176634e-19        # C
EPS0 = 8.8541878128e-12    # F/m
KB = 1.380649e-23          # J/K
NA = 6.02214076e23         # 1/mol
COULOMB_KJ_MOL_NM = 138.935458


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", default="CG_DA,CG_DT,CG_DG_S",
                    help="tipi di sito carichi, separati da virgole")
    ap.add_argument("--charge", type=float, default=-1.0, help="carica per sito, in e")
    ap.add_argument("--epsilon-r", type=float, default=78.0)
    ap.add_argument("--temperature", type=float, default=300.0, help="K, per lambda_D")
    ap.add_argument("--debye-length", type=float, default=None, help="nm")
    ap.add_argument("--ions", type=int, default=None, help="ioni monovalenti nella scatola")
    ap.add_argument("--box", type=float, default=None, help="lato della scatola cubica, nm")
    ap.add_argument("--r-cut", type=float, default=None, help="nm (default 4 lambda_D)")
    args = ap.parse_args()

    if args.debye_length is None:
        if not (args.ions and args.box):
            sys.exit("[ERROR] indica --debye-length, oppure --ions e --box")
        volume_l = (args.box * 1e-9) ** 3 * 1e3
        c = args.ions / NA / volume_l                     # mol/L
        ionic = 0.5 * c
        lam = math.sqrt(args.epsilon_r * EPS0 * KB * args.temperature
                        / (2.0 * NA * E * E * ionic * 1e3)) * 1e9
        print(f"[INFO] {args.ions} ioni in {args.box} nm: c = {c:.3f} M, I = {ionic:.3f} M, "
              f"lambda_D = {lam:.3f} nm a {args.temperature:.0f} K")
    else:
        lam = args.debye_length
    r_cut = args.r_cut if args.r_cut is not None else 4.0 * lam

    topo = json.loads(Path(args.topology).read_text())
    known = topo["mapping"]["site_types"]
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    missing = [s for s in sites if s not in known]
    if missing:
        sys.exit(f"[ERROR] tipi di sito sconosciuti {missing}; noti: {sorted(known)}")
    if "debye_huckel" in topo:
        print("[WARNING] la topologia ha gia' un blocco debye_huckel: lo sostituisco")
    topo["debye_huckel"] = {
        "charges": {s: args.charge for s in sites},
        "debye_length_nm": lam,
        "epsilon_r": args.epsilon_r,
        "r_cut_nm": r_cut,
        "note": "add_debye_huckel.py; all charged pairs, no exclusions (ESPResSo semantics)",
    }
    pref = COULOMB_KJ_MOL_NM / args.epsilon_r
    kT = KB * args.temperature * NA / 1e3
    for r in (0.6, 1.0, 1.5, 2.0):
        u = pref * args.charge ** 2 * math.exp(-r / lam) / r
        print(f"  U(r = {r:.1f} nm) = {u:.3f} kJ/mol = {u / kT:.3f} kT")
    print(f"[INFO] siti carichi {sites} (q = {args.charge:+g}), eps_r = {args.epsilon_r}, "
          f"lambda_D = {lam:.3f} nm, r_cut = {r_cut:.3f} nm")
    Path(args.out).write_text(json.dumps(topo, indent=2) + "\n")
    print(f"[DONE] {args.out}")


if __name__ == "__main__":
    main()
