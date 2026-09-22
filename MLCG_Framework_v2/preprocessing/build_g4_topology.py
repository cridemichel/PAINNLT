#!/usr/bin/env python3
"""Genera la topologia CG di un G-quadruplex a piu' copie.

PERCHE'
    tel22_topology.json descrive un caso specifico: 22 residui per copia, 10
    copie, piega antiparallela basket UDDU del 143D, con il registro delle
    tetradi scritto a mano.  Per un altro quadruplex -- per esempio il Tel26
    ibrido (3+1) del 2JPZ -- cambiano il numero di residui, la sequenza, il
    registro delle tetradi e quindi bond, angoli e Morse.  Quello che NON
    cambia e' il mapping per residuo (DA/DT un sito, DG sei): e' chimica del
    nucleotide, non della piega, e viene ereditato dal template.

COSA GENERA
    - backbone: un legame harmonic COM-COM fra residui consecutivi della
      stessa copia, con il nome derivato dalla sequenza (bb_G_G, bb_T_A, ...)
    - angoli: harmonic su ogni terna consecutiva
    - Morse: K4 completo per ogni tetrade (sei coppie), COM-COM, con gli
      stessi parametri del template

IL REGISTRO DELLE TETRADI
    E' l'unica parte che non si puo' dedurre dalla sequenza.  Lo script lo
    ricava dalla geometria: calcola il centro della base di ogni guanina della
    prima copia e cerca le quartine mutuamente vicine.  Il risultato va
    SEMPRE guardato: per una piega ibrida l'alternanza dei filamenti e' diversa
    da quella di un basket antiparallelo, e una tetrade sbagliata produce un
    campo di forza che sembra funzionare ma vincola le basi sbagliate.
    In alternativa si passa il registro a mano con --tetrads.

USO
    python3 build_g4_topology.py \\
        --template tutorials/tel22/tel22_topology.json \\
        --structure <md.gro o md.tpr> [--trajectory <xtc>] \\
        --residues-per-copy 26 --copies 10 \\
        --output tel26_topology.json \\
        [--tetrads "2,10,14,22 3,9,15,21 4,8,16,20"] \\
        [--tetrad-cutoff 1.0]
"""
import argparse
import itertools
import json
import sys

import numpy as np

try:
    import MDAnalysis as mda
except ImportError:
    sys.exit("[ERROR] serve MDAnalysis: attiva il venv (source hpc/env_leonardo.sh)")


def unwrap(points, box_nm, reference=None):
    """Immagine minima rispetto a un riferimento, in nm.

    Un frame GROMACS e' ripiegato nella scatola: una molecola a cavallo di una
    faccia esce da un lato e rientra dall'altro.  Chi fa medie di posizioni --
    il centro di una base, il centro di massa di un residuo -- ottiene un punto
    che non sta da nessuna parte, e le distanze fra i centri diventano
    insensate: in un caso reale sono uscite tetradi con lati fra 0.26 e 2.01 nm
    invece dei 0.7-1.1 nm di una tetrade vera.

    Vale finche' l'oggetto e' piu' piccolo di meta' scatola, che per un
    quadruplex (~3 nm) in una scatola da 12 nm e' sempre vero.
    """
    points = np.asarray(points, dtype=float)
    if box_nm is None:
        return points
    ref = points[0] if reference is None else np.asarray(reference, dtype=float)
    delta = points - ref
    delta -= box_nm * np.round(delta / box_nm)
    return ref + delta


def box_from(universe):
    """Lati della scatola in nm, o None se la struttura non li porta."""
    dims = getattr(universe, "dimensions", None)
    if dims is None:
        return None
    box = np.asarray(dims[:3], dtype=float) / 10.0
    if not np.all(np.isfinite(box)) or float(box.min()) <= 0.0:
        return None
    return box


def base_centers(universe, residues, mapping, box_nm=None):
    """Centro geometrico degli atomi di base di ogni guanina, in nm."""
    centers = {}
    for local_index, residue in enumerate(residues, start=1):
        resname = residue.resname.rstrip("53") if residue.resname not in mapping else residue.resname
        if resname.rstrip("53") != "DG":
            continue
        base_atoms = []
        for site_name, atom_names in mapping.get(residue.resname, mapping.get("DG", {})).items():
            if site_name == "CG_DG_S":      # zucchero-fosfato: non e' la base
                continue
            if atom_names == ["*"]:
                continue
            sel = residue.atoms.select_atoms("name " + " ".join(atom_names))
            base_atoms.extend(sel.positions)
        if base_atoms:
            # Due ricuciture, non una.  La prima rende intera la singola base,
            # che puo' essere tagliata in mezzo dal bordo della scatola.
            atoms_nm = unwrap(np.array(base_atoms) / 10.0, box_nm)
            centers[local_index] = atoms_nm.mean(axis=0)
    if centers and box_nm is not None:
        # La seconda rende contigua la copia: basi intere ma ciascuna
        # ripiegata per conto suo darebbero comunque distanze sbagliate.
        labels = sorted(centers)
        stitched = unwrap([centers[l] for l in labels], box_nm)
        centers = {l: stitched[i] for i, l in enumerate(labels)}
    return centers


def find_tetrads(centers, n_tetrads=None, planarity_tol=0.15):
    """Tetradi come gruppi COMPLANARI lungo l'asse del quadruplex.

    Non si possono cercare come "quartine piu' compatte": in un G4 le guanine
    impilate -- stesso filamento, tetradi adiacenti -- distano ~0.34 nm, meno
    di quelle complanari della stessa tetrade (~0.6 nm), quindi un criterio di
    vicinanza trova le pile invece dei piani.

    Le tetradi sono invece perpendicolari all'asse di impilamento: si proietta
    sull'asse e si raggruppa per quota.  L'asse si sceglie fra le componenti
    principali dei centri come quella che separa meglio gruppi da quattro.
    """
    labels = sorted(centers)
    coords = np.array([centers[l] for l in labels])
    if len(labels) % 4 != 0:
        print(f"[ATTENZIONE] {len(labels)} guanine non sono un multiplo di 4: "
              f"il raggruppamento in tetradi sara' parziale.")
    expected = n_tetrads or len(labels) // 4

    centered = coords - coords.mean(axis=0)
    _, _, directions = np.linalg.svd(centered, full_matrices=False)

    best = None
    for axis in directions:
        projection = centered @ axis
        order = np.argsort(projection)
        groups = [order[i * 4:(i + 1) * 4] for i in range(expected)]
        if any(len(g) != 4 for g in groups):
            continue
        spread = max(float(projection[g].max() - projection[g].min()) for g in groups)
        gaps = [float(projection[groups[i + 1]].min() - projection[groups[i]].max())
                for i in range(len(groups) - 1)]
        gap = min(gaps) if gaps else float("inf")
        score = gap - spread          # piani stretti e ben separati
        if best is None or score > best[0]:
            best = (score, axis, groups, projection, spread, gap)

    if best is None:
        return [], None
    _, axis, groups, projection, spread, gap = best
    print(f"[INFO] asse di impilamento: [{axis[0]:+.2f} {axis[1]:+.2f} {axis[2]:+.2f}]  "
          f"spessore massimo di un piano {spread:.3f} nm, "
          f"separazione minima fra piani {gap:.3f} nm")
    if spread > gap:
        print("[ATTENZIONE] i piani non sono ben separati: il registro "
              "ricavato e' inaffidabile, passalo a mano con --tetrads.")

    result = []
    for group in groups:
        members = tuple(sorted(labels[i] for i in group))
        points = np.array([centers[m] for m in members])
        # planarita': distanza dei quattro centri dal loro piano ai minimi quadrati
        local = points - points.mean(axis=0)
        _, singular, _ = np.linalg.svd(local, full_matrices=False)
        out_of_plane = float(singular[-1]) / np.sqrt(len(points))
        pair_distances = [float(np.linalg.norm(points[i] - points[j]))
                          for i, j in itertools.combinations(range(4), 2)]
        result.append({
            "residues": members,
            "out_of_plane_nm": out_of_plane,
            "min_nm": min(pair_distances),
            "max_nm": max(pair_distances),
            "planar": out_of_plane <= planarity_tol,
        })
    return result, axis


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", required=True, help="topologia da cui ereditare mapping e parametri")
    ap.add_argument("--structure", required=True, help="struttura all-atom (.gro, .tpr, .pdb)")
    ap.add_argument("--trajectory", default=None, help="opzionale: si usa il primo frame")
    ap.add_argument("--residues-per-copy", type=int, required=True)
    ap.add_argument("--copies", type=int, default=None, help="default: dedotto dal conteggio dei residui")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tetrads", default=None,
                    help="registro 1-based, es. \"2,10,14,22 3,9,15,21\"; se assente si ricava dalla geometria")
    ap.add_argument("--n-tetrads", type=int, default=None,
                    help="numero di tetradi atteso; default: guanine/4")
    ap.add_argument("--pdb", default=None, help="codice PDB di riferimento, solo per i metadati")
    ap.add_argument("--fold", default=None, help="descrizione della piega, solo per i metadati")
    args = ap.parse_args()

    template = json.load(open(args.template))
    mapping = dict(template["mapping"]["residues"])

    universe = mda.Universe(args.structure, args.trajectory) if args.trajectory \
        else mda.Universe(args.structure)

    # Le coordinate servono: il registro delle tetradi si ricava dalla
    # geometria.  Un .tpr da solo porta la topologia ma non le posizioni, e il
    # risultato sarebbero tetradi con lati di pochi centesimi di nm.
    if not args.tetrads:
        try:
            positions = universe.atoms.positions
        except Exception:
            positions = None
        if positions is None or float(np.abs(positions).max()) < 1e-6:
            sys.exit("[ERROR] la struttura non contiene coordinate: passa --trajectory "
                     "(un .xtc o .gro), oppure fornisci il registro con --tetrads")

    # Terminali: in convenzione AMBER si chiamano DA5, DT3, DG3...  Il mapping
    # del template conosce solo DA, DT, DG, quindi senza alias il primo e
    # l'ultimo nucleotide di ogni copia verrebbero scartati -- e le copie
    # risulterebbero piu' corte, sfalsando tutto quello che segue.
    canonical = set(mapping)
    observed = {str(r) for r in universe.residues.resnames}
    aliases = {}
    for name in sorted(observed):
        if name in canonical:
            continue
        stripped = name.rstrip("53")
        if stripped in canonical and stripped != name:
            aliases[name] = stripped
    if aliases:
        print("[INFO] terminali riconosciuti: "
              + ", ".join(f"{k}->{v}" for k, v in sorted(aliases.items())))
        for name, base in aliases.items():
            mapping[name] = mapping[base]

    known = set(mapping)
    residues = [r for r in universe.residues if r.resname in known]
    n = args.residues_per_copy
    copies = args.copies if args.copies else len(residues) // n
    print(f"[INFO] residui mappabili: {len(residues)}  -> {copies} copie da {n}")
    if copies * n != len(residues):
        sys.exit(
            f"[ERROR] {len(residues)} residui mappabili non sono {copies}x{n}.\n"
            f"        O --residues-per-copy e' sbagliato, o alcuni residui non sono\n"
            f"        riconosciuti dal mapping: residui presenti nella struttura = "
            f"{sorted({str(r) for r in universe.residues.resnames})}"
        )

    first = residues[:n]
    sequence = [aliases.get(r.resname, r.resname) for r in first]
    print(f"[INFO] sequenza prima copia: {' '.join(s[-1] for s in sequence)}")

    # ── registro delle tetradi ──────────────────────────────────────────────
    if args.tetrads:
        tetrads = [tuple(int(x) for x in group.split(","))
                   for group in args.tetrads.split()]
        print(f"[INFO] registro fornito a mano: {tetrads}")
    else:
        box_nm = box_from(universe)
        if box_nm is None:
            print("[ATTENZIONE] la struttura non porta la scatola: se il frame e' "
                  "ripiegato ai bordi le tetradi saranno sbagliate.")
        else:
            print(f"[INFO] scatola: {box_nm[0]:.2f} x {box_nm[1]:.2f} x {box_nm[2]:.2f} nm "
                  f"(le basi vengono ricucite all'immagine minima)")
        centers = base_centers(universe, first, mapping, box_nm)
        print(f"[INFO] guanine nella prima copia: {sorted(centers)}")
        if not centers:
            sys.exit("[ERROR] nessuna guanina trovata: controlla --structure e il mapping")
        found, _axis = find_tetrads(centers, args.n_tetrads)
        if not found:
            sys.exit("[ERROR] non sono riuscito a raggruppare le guanine in tetradi; "
                     "passa il registro a mano con --tetrads")
        print("[INFO] tetradi ricavate dalla geometria "
              "(da verificare contro la struttura di riferimento):")
        for t in found:
            flag = "" if t["planar"] else "   <-- NON PLANARE"
            print(f"         {t['residues']}   fuori piano {t['out_of_plane_nm']:.3f} nm, "
                  f"lati {t['min_nm']:.2f}-{t['max_nm']:.2f} nm{flag}")
        if any(not t["planar"] for t in found):
            print("[ATTENZIONE] almeno una tetrade non e' planare: verifica il registro.")
        tetrads = [t["residues"] for t in found]

    used = sorted(x for t in tetrads for x in t)
    if len(set(used)) != len(used):
        sys.exit("[ERROR] una guanina compare in piu' tetradi: registro incoerente")

    # ── controllo indipendente dalla geometria ──────────────────────────────
    # In un quadruplex intramolecolare ogni tetrade prende UNA guanina da
    # ciascun tratto di G consecutive.  E' un criterio strutturale, non
    # geometrico, e vale anche quando i piani non si separano bene lungo un
    # asse -- cosa che capita nelle pieghe ibride, dove le tetradi non sono
    # equidistanti ne' perpendicolari a un asse unico.
    tracts, current = [], []
    for index, resname in enumerate(sequence, start=1):
        if resname.rstrip("53") == "DG":
            current.append(index)
        elif current:
            tracts.append(current)
            current = []
    if current:
        tracts.append(current)
    print(f"[INFO] tratti di guanine: {tracts}")

    if len(tracts) == 4:
        coherent = True
        for quartet in tetrads:
            origin = [next((t_i for t_i, tract in enumerate(tracts) if g in tract), None)
                      for g in quartet]
            if sorted(x for x in origin if x is not None) != [0, 1, 2, 3]:
                print(f"[ATTENZIONE] la tetrade {quartet} non prende una guanina "
                      f"da ciascun tratto (provenienza: {origin})")
                coherent = False
        if coherent:
            print("[OK] ogni tetrade prende una guanina da ciascuno dei quattro tratti.")
            versi = []
            for tract in tracts:
                positions = [next((k for k, q in enumerate(tetrads) if g in q), None)
                             for g in tract]
                positions = [p for p in positions if p is not None]
                versi.append("+" if positions == sorted(positions) else "-")
            # il verso e' relativo all'ordine in cui elenchiamo le tetradi: conta
            # quanti filamenti sono discordi dalla maggioranza, non quanti sono "-"
            opposti = min(versi.count("+"), versi.count("-"))
            print(f"[INFO] versi dei quattro tratti: {' '.join(versi)}")
            if opposti == 1:
                print("       -> tre filamenti concordi e uno opposto: piega IBRIDA (3+1)")
            elif opposti == 2:
                print("       -> due e due: piega ANTIPARALLELA")
            elif opposti == 0:
                print("       -> tutti concordi: piega PARALLELA")
    else:
        print(f"[ATTENZIONE] trovati {len(tracts)} tratti di guanine invece di 4: "
              f"il controllo di coerenza non si applica.")

    # ── bond di backbone e angoli ───────────────────────────────────────────
    bonds, angles = [], []
    for copy in range(copies):
        offset = copy * n
        for i in range(n - 1):
            bonds.append({
                "mol_i": offset + i, "mol_j": offset + i + 1,
                "site_i": 0, "site_j": 0,
                "type": "harmonic", "k": "auto", "r0": "auto",
                "name": f"bb_{sequence[i][-1]}_{sequence[i+1][-1]}",
            })
        for i in range(n - 2):
            angles.append({
                "mol_i": offset + i, "mol_j": offset + i + 1, "mol_k": offset + i + 2,
                "type": "harmonic", "k": "auto", "theta0": "auto",
                "name": f"ang_{sequence[i][-1]}_{sequence[i+1][-1]}_{sequence[i+2][-1]}",
            })

    # ── Morse: K4 completo per tetrade, con i parametri del template ────────
    morse_template = next((b for b in template["bonds"] if b.get("type") == "morse"), None)
    if morse_template is None:
        sys.exit("[ERROR] il template non contiene legami Morse da cui copiare i parametri")
    for copy in range(copies):
        offset = copy * n
        for quartet in tetrads:
            for a, b in itertools.combinations(sorted(quartet), 2):
                entry = dict(morse_template)
                entry["mol_i"] = offset + a - 1      # il registro e' 1-based
                entry["mol_j"] = offset + b - 1
                bonds.append(entry)

    # ── assemblaggio ────────────────────────────────────────────────────────
    out = {k: v for k, v in template.items()
           if k not in ("bonds", "angles", "tel22_g4_topology")}
    # il mapping esteso ai terminali deve finire nella topologia prodotta,
    # altrimenti build_cg_dataset.py scartera' gli stessi residui
    out["mapping"] = dict(template["mapping"])
    out["mapping"]["residues"] = mapping
    out["bonds"] = bonds
    out["angles"] = angles
    out["dihedrals"] = []
    out["g4_topology"] = {
        "pdb": args.pdb or "?",
        "fold": args.fold or "?",
        "residues_per_copy": n,
        "copies": copies,
        "tetrads_1based": [list(t) for t in tetrads],
        "morse_representation": "complete K4 per tetrad, explicit COM-COM endpoints",
        "generated_from": args.structure,
    }

    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)

    n_harm = sum(1 for b in bonds if b.get("type") == "harmonic")
    n_morse = sum(1 for b in bonds if b.get("type") == "morse")
    print(f"\n[DONE] {args.output}")
    print(f"       {n_harm} legami di backbone, {n_morse} Morse, {len(angles)} angoli")
    print(f"       attesi: {(n-1)*copies}, {6*len(tetrads)*copies}, {(n-2)*copies}")
    print("\nPrima di usarla, verifica il registro delle tetradi contro la struttura "
          "di riferimento: una tetrade sbagliata da' un campo di forza che gira "
          "ma vincola le basi sbagliate.")


if __name__ == "__main__":
    main()
