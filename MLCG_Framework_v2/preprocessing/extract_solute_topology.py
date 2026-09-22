#!/usr/bin/env python3
"""Estrae dal .tpr la topologia dei soli atomi presenti in un .xtc ridotto.

PERCHE' SERVE
    Una produzione GROMACS con `compressed-x-grps = non-Water` scrive
    nell'.xtc i soli atomi del soluto, mentre il .tpr descrive l'intero
    sistema solvatato.  MDAnalysis non puo' appaiarli -- il numero di atomi
    non coincide -- quindi serve una topologia ridotta che contenga
    esattamente gli atomi dell'.xtc, nello stesso ordine.

    Questo script la ricava dal .tpr, senza bisogno di GROMACS: MDAnalysis
    legge i .tpr nativamente, comprese le masse del force field, che contano
    perche' il mapping CG e' un centro di massa.

USO
    python3 extract_solute_topology.py --tpr md.tpr --out solute.gro \\
            [--selection 'not resname SOL WAT HOH TIP3 T3P'] \\
            [--check-xtc prod-1.part0001.xtc]

    Con --check-xtc verifica che il conteggio degli atomi corrisponda davvero
    a quello della traiettoria compressa: e' il controllo che evita di
    scoprire il disallineamento a meta' della costruzione del dataset.
"""
import argparse
import sys

import MDAnalysis as mda


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tpr", required=True, help="Topologia completa del sistema (.tpr)")
    ap.add_argument("--out", required=True, help="Topologia ridotta da scrivere (.gro o .pdb)")
    ap.add_argument("--selection", default="not resname SOL WAT HOH TIP3 T3P",
                    help="Selezione MDAnalysis degli atomi salvati nell'.xtc")
    ap.add_argument("--check-xtc", default=None,
                    help="Traiettoria compressa con cui verificare il conteggio degli atomi")
    ap.add_argument("--coordinates", default=None,
                    help="File di coordinate da cui prendere le posizioni (default: il primo frame di --check-xtc)")
    args = ap.parse_args()

    u = mda.Universe(args.tpr)
    sel = u.select_atoms(args.selection)
    if len(sel) == 0:
        sys.exit(f"[ERROR] la selezione '{args.selection}' non seleziona alcun atomo")

    print(f"[INFO] sistema completo: {len(u.atoms)} atomi")
    print(f"[INFO] selezione '{args.selection}': {len(sel)} atomi, "
          f"{len(sel.residues)} residui")
    resnames = sorted({str(r) for r in sel.residues.resnames})
    print(f"[INFO] residui selezionati: {', '.join(resnames)}")

    if args.check_xtc:
        # Un .gro/.pdb ha bisogno di coordinate: si prendono dal primo frame
        # della traiettoria compressa, che e' anche il modo di verificare che
        # il numero di atomi coincida.
        probe = mda.Universe(args.tpr, args.check_xtc)
        n_xtc = probe.trajectory.n_atoms
        print(f"[INFO] atomi nella traiettoria compressa: {n_xtc}")
        if n_xtc != len(sel):
            sys.exit(
                f"[ERROR] la selezione da' {len(sel)} atomi ma l'.xtc ne contiene "
                f"{n_xtc}.\n"
                f"        Aggiusta --selection: deve riprodurre esattamente il gruppo "
                f"usato in compressed-x-grps."
            )
        print("[OK] il conteggio corrisponde")

    src = args.coordinates or args.check_xtc
    if src:
        # Le posizioni del primo frame, per avere un file di topologia valido.
        u_xtc = mda.Universe(args.tpr, src)
        u_xtc.trajectory[0]
        sel_xtc = u_xtc.select_atoms(args.selection)
        sel_xtc.write(args.out)
    else:
        sel.write(args.out)

    print(f"[DONE] scritto {args.out}")
    print()
    print("Usalo come AA_TOPOLOGY nello stadio dataset, con l'.xtc come "
          "AA_TRAJECTORY e il .trr come AA_FORCES_TRAJECTORY "
          "(AA_FORCES_TOPOLOGY resta il .tpr completo).")


if __name__ == "__main__":
    main()
