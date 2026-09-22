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

import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.core import reader as coordinate_reader


def first_frame(path):
    """Posizioni e box del primo frame, senza costruire un Universe.

    Serve proprio perche' il .tpr e l'.xtc hanno conteggi di atomi diversi:
    `mda.Universe(tpr, xtc)` rifiuterebbe la coppia, che e' il caso d'uso
    per cui questo script esiste.  Il reader, da solo, non fa quel controllo.
    """
    with coordinate_reader(path) as trj:
        ts = trj[0]
        return ts.n_atoms, np.array(ts.positions, dtype=np.float32), (
            None if ts.dimensions is None else np.array(ts.dimensions, dtype=np.float32)
        )


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

    src_coords = args.coordinates or args.check_xtc
    positions = None
    box = None

    if src_coords:
        n_src, positions, box = first_frame(src_coords)
        print(f"[INFO] atomi nel file di coordinate ({src_coords}): {n_src}")
        if n_src != len(sel):
            sys.exit(
                f"[ERROR] la selezione da' {len(sel)} atomi ma il file di coordinate "
                f"ne contiene {n_src}.\n"
                f"        Aggiusta --selection: deve riprodurre esattamente il gruppo "
                f"usato in compressed-x-grps."
            )
        print("[OK] il conteggio corrisponde")

    if positions is not None:
        # Il .tpr non porta coordinate: si allega un frame in memoria al sistema
        # completo e vi si scrivono le posizioni del sottoinsieme selezionato,
        # che l'.xtc elenca nello stesso ordine del .tpr.
        u.load_new(np.zeros((len(u.atoms), 3), dtype=np.float32))
        sel = u.select_atoms(args.selection)
        sel.positions = positions
        if box is not None:
            u.dimensions = box
    else:
        print("[WARN] nessuna coordinata fornita: il file scritto avra' posizioni nulle")
        u.load_new(np.zeros((len(u.atoms), 3), dtype=np.float32))
        sel = u.select_atoms(args.selection)

    sel.write(args.out)

    print(f"[DONE] scritto {args.out}")
    print()
    print("Usalo come AA_TOPOLOGY nello stadio dataset, con l'.xtc come "
          "AA_TRAJECTORY e il .trr come AA_FORCES_TRAJECTORY "
          "(AA_FORCES_TOPOLOGY resta il .tpr completo).")


if __name__ == "__main__":
    main()
