#!/usr/bin/env python3
"""Morse sito-sito per le tetradi (e per l'impilamento), stimati dal riferimento.

PERCHE'
    I contatti delle tetradi ereditati dal template TEL22 sono Morse COM-COM
    con D = 50 kJ/mol e a = 0.3 nm^-1: una buca larga 1/a ~ 3.3 nm, la cui
    curvatura al minimo (2 D a^2 = 9 kJ/mol/nm^2) lascia fluttuare ogni
    contatto di ~0.5 nm.  Tengono vicine le copie, non la geometria della
    tetrade: da soli, in 100 ps, lasciano srotolare il quadruplex.  E agendo
    sui centri di massa non dicono nulla sull'orientamento delle basi, cioe'
    sul contatto di Hoogsteen.

    Questo script sposta gli estremi sul sito indicato (B3, per default) e
    stima dal riferimento mappato, coppia per coppia, la distanza tipica e la
    larghezza della sua distribuzione.  I Morse restano pair-specific e
    reversibili: una tetrade puo' ancora aprirsi.

L'IMPILAMENTO (--stacking SITO)
    Con i soli Morse di Hoogsteen le tetradi si formano ma non restano
    impilate: sul TEL26 il picco B5-B5 a 0.42 nm (guanine sovrapposte di
    tetradi adiacenti) sparisce, e il B3-B3 intra acquista una coda fino a
    2 nm.  Fra due guanine consecutive dello stesso tratto non c'e' nulla che
    ne orienti le basi: il legame di backbone agisce sui siti S, e la WCA fra
    molecole legate e' esclusa proprio sulla coppia legata.
    --stacking aggiunge un Morse sito-sito fra le guanine sovrapposte, cioe'
    fra le coppie (g, h) con g e h in tetradi diverse e |g - h| = 1: stesso
    tratto, residui consecutivi.  Sono 8 per copia con tre tetradi.  I
    parametri si stimano come per i contatti di Hoogsteen.
    Con --keep-tetrads i Morse delle tetradi gia' presenti restano come sono:
    serve per aggiungere l'impilamento a una topologia gia' adattata.

LE TETRADI
    Si ricavano dai Morse della topologia che non sono di impilamento: in ogni
    copia formano grafi K4, uno per tetrade.  Nessun registro da fornire a
    parte, e lo script vale per qualunque G-quadruplex che usi quella
    rappresentazione.

I PARAMETRI
    Ogni classe (coppia di posizioni nella copia) e' mediata sulle copie.  Per
    la distribuzione della distanza r della coppia:
        r0 = mediana di r
        a  = sqrt(kT / (2 D sigma^2))
    cioe' la larghezza di un Morse la cui curvatura al minimo, da sola,
    darebbe quella sigma.  Per default sigma = 1.4826 * MAD, una stima robusta
    della larghezza del NUCLEO della distribuzione: nel riferimento TEL22 le
    distanze B3-B3 hanno code lunghe verso le grandi r (aperture transitorie,
    p99 fino a 1.3 nm con mediana 0.6), che con la deviazione standard
    allargherebbero la buca fino a 3-4 volte.  --width std usa la deviazione
    standard.  E' un'approssimazione: la larghezza osservata risente anche
    degli altri contatti e di tutto il resto del campo, e sommando piu' Morse
    stimati cosi' la struttura esce piu' rigida del riferimento.  Il residuo
    ML serve anche a questo.
        r_cut = r0 + CUT/a   (a CUT = 7 la buca e' scesa a e^-7 ~ 0.1%)

LA FORMA LJ (--form lj)
    Al posto del Morse un Lennard-Jones 12-6 sugli stessi contatti:
        sigma   = r0 / 2^(1/6)            (minimo in r0)
        epsilon = kT r0^2 / (72 sigma_r^2)  (curvatura al minimo 72 eps / r0^2
                                             = kT / sigma_r^2)
        r_cut   = LJ_CUT * sigma, energia spostata a zero al cutoff
    Nel LJ profondita' e curvatura sono legate: fissata la larghezza osservata
    la profondita' non e' piu' una scelta (D = 50 per i Morse) ma segue dal
    riferimento.  Con r0 = 0.45 nm e sigma_r = 0.02 nm, epsilon ~ 17 kJ/mol.
    E' la forma dei contatti del modello unfoldable.

USO
    python3 fit_tetrad_site_morse.py --dataset tel26_dataset.bin \\
        --topology tel26_topology.json --out tel26_topology.b3morse.json \\
        [--site CG_DG_B3] [--D 50] [--stride 5] [--width mad|std]

    python3 fit_tetrad_site_morse.py --dataset tel26_dataset.bin \\
        --topology tel26_topology.b3morse.json --keep-tetrads \\
        --stacking CG_DG_B5 --out tel26_topology.b3stack.json

    python3 fit_tetrad_site_morse.py --dataset tel26_dataset.bin \\
        --topology tel26_topology.json --form lj --stacking CG_DG_B5 \\
        --out tel26_topology.b3stack_lj.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def frame_layout(words_i, n_frames):
    """Posizione (in parole da 4 byte) del frame 0 e lunghezza di un frame.

    Formato (vedi _tel22_cv.load_reference): T; per frame nm, n, box[3]; per
    molecola mid, ns, centro[3], forza[3], coppia[3], poi ns blocchi da 4
    parole (tipo, x, y, z).  La struttura e' identica in ogni frame.
    """
    w = 1
    start = w
    nm = int(words_i[w]); w += 2
    w += 3
    mols = []
    for _ in range(nm):
        ns = int(words_i[w + 1]); w += 2
        w += 9
        mols.append((w, ns))
        w += 4 * ns
    length = w - start
    expected = 1 + n_frames * length
    if expected > words_i.size:
        sys.exit(f"[ERROR] il dataset e' piu' corto di {n_frames} frame da {length} parole")
    return start, length, mols


class Reference:
    """Coordinate dei siti dal dataset, lette in blocco con numpy."""

    def __init__(self, path, stride):
        buf = Path(path).read_bytes()
        self.words_i = np.frombuffer(buf, dtype=np.int32)
        self.words_f = np.frombuffer(buf, dtype=np.float32)
        self.n_frames = int(self.words_i[0])
        self.start, self.length, self.mols = frame_layout(self.words_i, self.n_frames)
        self.frames = np.arange(0, self.n_frames, max(1, stride))
        self.box = np.stack(
            [self.words_f[self.start + self.frames * self.length + 2 + k] for k in range(3)],
            axis=-1).astype(np.float64)

    def site(self, molecules, site_type, label):
        """(xyz[frame, colonna, 3], colonna per molecola, indice del sito)."""
        words, index = [], {}
        for m in molecules:
            w, ns = self.mols[m]
            types = [int(self.words_i[w + 4 * s]) for s in range(ns)]
            if site_type not in types:
                sys.exit(f"[ERROR] la molecola {m} non ha un sito di tipo {site_type} ({label})")
            s = types.index(site_type)
            words.append(w + 4 * s + 1)
            index[m] = s
        idx = np.array(words)
        # idx sono indici assoluti nel frame 0 (gia' comprendono start)
        base = self.frames[:, None] * self.length
        xyz = np.stack([self.words_f[base + idx + k] for k in range(3)], axis=-1).astype(np.float64)
        return xyz, {m: c for c, m in enumerate(molecules)}, index

    def distances(self, xyz, col, i, j):
        d = xyz[:, col[i]] - xyz[:, col[j]]
        d -= self.box * np.round(d / self.box)
        return np.linalg.norm(d, axis=1)


def fit_class(r, args):
    med, std = float(np.median(r)), float(np.std(r))
    mad = 1.4826 * float(np.median(np.abs(r - med)))
    sig = mad if args.width == "mad" else std
    out = {"r0": med, "sigma": sig, "sigma_std": std, "sigma_mad": mad,
           "p01": float(np.percentile(r, 1)), "p99": float(np.percentile(r, 99)),
           "n": int(r.size)}
    if args.form == "lj":
        lj_sigma = med / 2.0 ** (1.0 / 6.0)
        eps = args.kT * med * med / (72.0 * sig * sig)
        out.update({"lj_sigma": lj_sigma, "epsilon": eps, "r_cut": args.lj_cut * lj_sigma,
                    "a": float("nan"), "strength": eps})
    else:
        a = math.sqrt(args.kT / (2.0 * args.D * sig * sig))
        out.update({"a": a, "r_cut": med + args.cut / a, "strength": a})
    return out


def site_type_of(topo, name):
    site_types = topo["mapping"]["site_types"]
    if name not in site_types:
        sys.exit(f"[ERROR] sito {name} assente: {sorted(site_types)}")
    return int(site_types[name])


def morse_entry(i, j, si, sj, c, args, role):
    base = {"mol_i": int(i), "mol_j": int(j), "site_i": int(si), "site_j": int(sj),
            "exclude_wca": False, "role": role}
    if args.form == "lj":
        return {**base, "type": "lj", "epsilon": c["epsilon"], "sigma": c["lj_sigma"],
                "r_cut": c["r_cut"], "shift": "auto"}
    return {**base, "type": "morse", "D": args.D, "a": c["a"], "r0": c["r0"],
            "r_cut": c["r_cut"]}


CONTACT_TYPES = ("morse", "lj")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--site", default="CG_DG_B3", help="sito dei contatti di Hoogsteen")
    ap.add_argument("--keep-tetrads", action="store_true",
                    help="lascia invariati i Morse delle tetradi gia' presenti")
    ap.add_argument("--stacking", default=None, metavar="SITO",
                    help="aggiungi Morse di impilamento fra guanine sovrapposte su SITO (es. CG_DG_B5)")
    ap.add_argument("--D", type=float, default=50.0, help="profondita' in kJ/mol")
    ap.add_argument("--kT", type=float, default=2.49)
    ap.add_argument("--cut", type=float, default=7.0, help="r_cut = r0 + cut/a")
    ap.add_argument("--nuc", type=int, default=None, help="residui per copia")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--width", choices=("mad", "std"), default="mad",
                    help="stima della larghezza: 1.4826*MAD (robusta) o deviazione standard")
    ap.add_argument("--form", choices=("morse", "lj"), default="morse",
                    help="forma dei contatti: Morse (D fissata) o LJ 12-6 (epsilon dalla larghezza)")
    ap.add_argument("--lj-cut", type=float, default=3.5, help="r_cut dei LJ in unita' di sigma")
    args = ap.parse_args()
    if args.keep_tetrads and not args.stacking:
        sys.exit("[ERROR] --keep-tetrads senza --stacking non cambierebbe nulla")

    topo = json.loads(Path(args.topology).read_text())
    nuc = args.nuc or topo.get("g4_topology", {}).get("residues_per_copy")
    if not nuc:
        sys.exit("[ERROR] residui per copia ignoti: passa --nuc")

    all_morse = [(k, b) for k, b in enumerate(topo["bonds"]) if b.get("type") in CONTACT_TYPES]
    morse = [(k, b) for k, b in all_morse if b.get("role", "tetrad") != "stacking"]
    if not morse:
        sys.exit("[ERROR] nessun Morse di tetrade nella topologia: niente tetradi da cui partire")
    if args.stacking and len(morse) != len(all_morse):
        sys.exit("[ERROR] la topologia ha gia' Morse di impilamento: parti da quella senza")

    # ── tetradi: componenti connesse dei Morse in ogni copia ───────────────
    adj = defaultdict(set)
    for _, b in morse:
        i, j = int(b["mol_i"]), int(b["mol_j"])
        if i // nuc != j // nuc:
            sys.exit(f"[ERROR] Morse fra copie diverse ({i}, {j}): non e' una tetrade")
        adj[i].add(j); adj[j].add(i)
    seen, groups = set(), []
    for v in sorted(adj):
        if v in seen:
            continue
        comp, stack = set(), [v]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u); stack.extend(adj[u] - comp)
        seen |= comp
        if len(comp) != 4:
            sys.exit(f"[ERROR] gruppo di contatti da {len(comp)} molecole, non una tetrade: {sorted(comp)}")
        groups.append(tuple(sorted(comp)))
    local = sorted({tuple(m % nuc + 1 for m in g) for g in groups})
    tetrad_of = {}
    for g in groups:
        key = tuple(m % nuc + 1 for m in g)
        for m in g:
            tetrad_of[m] = local.index(key)
    n_copies = len(groups) // len(local)
    print(f"[INFO] {len(groups)} tetradi in {n_copies} copie; registro {local}")

    ref = Reference(args.dataset, args.stride)
    involved = sorted(tetrad_of)
    print(f"[INFO] {ref.frames.size} frame su {ref.n_frames} (stride {args.stride})")

    out = json.loads(json.dumps(topo))
    report = {"dataset": args.dataset, "topology": args.topology, "form": args.form,
              "lj_cut": args.lj_cut, "D": args.D,
              "kT": args.kT, "cut": args.cut, "stride": args.stride, "width": args.width}

    # ── contatti di Hoogsteen ──────────────────────────────────────────────
    if not args.keep_tetrads:
        site_type = site_type_of(topo, args.site)
        xyz, col, site_index = ref.site(involved, site_type, args.site)
        print(f"[INFO] tetradi: sito {args.site} = indice "
              f"{sorted(set(site_index.values()))} nella guanina")
        samples = defaultdict(list)
        for k, b in morse:
            i, j = int(b["mol_i"]), int(b["mol_j"])
            key = (tetrad_of[i], tuple(sorted((i % nuc + 1, j % nuc + 1))))
            samples[key].append(ref.distances(xyz, col, i, j))
        classes = {key: {**fit_class(np.concatenate(rs), args), "copies": len(rs)}
                   for key, rs in sorted(samples.items())}

        head = "eps (kT)" if args.form == "lj" else "a (1/nm)"
        print(f"\n  {'tetrade':>7} {'coppia':>9} {'r0 (nm)':>8} {'sigma':>7} {head:>9} "
              f"{'r_cut':>6}  ruolo")
        for t in range(len(local)):
            keys = sorted((k for k in classes if k[0] == t), key=lambda k: classes[k]["r0"])
            for rank, key in enumerate(keys):
                c = classes[key]
                # in una tetrade quadrata le due coppie piu' lontane sono le
                # diagonali (rapporto atteso ~ sqrt(2))
                c["role"] = "diagonale" if rank >= 4 else "lato"
                val = c["epsilon"] / args.kT if args.form == "lj" else c["a"]
                print(f"  {t + 1:>7} {str(key[1]):>9} {c['r0']:8.3f} {c['sigma']:7.3f} "
                      f"{val:9.2f} {c['r_cut']:6.2f}  {c['role']}")
            lati = [classes[k]["r0"] for k in keys[:4]]
            diag = [classes[k]["r0"] for k in keys[4:]]
            if lati and diag:
                print(f"          diagonale / lato = {np.mean(diag) / np.mean(lati):.3f}"
                      f"   (quadrato: {math.sqrt(2):.3f})")

        for k, b in morse:
            i, j = int(b["mol_i"]), int(b["mol_j"])
            c = classes[(tetrad_of[i], tuple(sorted((i % nuc + 1, j % nuc + 1))))]
            # sostituzione completa: da Morse a LJ non devono restare D, a, r0
            new = morse_entry(i, j, site_index[i], site_index[j], c, args, "tetrad")
            if "name" in out["bonds"][k]:
                new["name"] = out["bonds"][k]["name"]
            out["bonds"][k] = new
        out.setdefault("g4_topology", {})["morse_representation"] = (
            f"complete K4 per tetrad, site-site {args.site}, {args.form}, fitted from the "
            f"mapped all-atom reference (fit_tetrad_site_morse.py, D={args.D}, width={args.width})")
        print(f"[INFO] {len(morse)} contatti di tetrade sito-sito ({args.form})")
        report["tetrad_site"] = args.site
        report["tetrad_classes"] = [{"tetrad": k[0] + 1, "pair": list(k[1]), **v}
                                    for k, v in classes.items()]
    else:
        print(f"[INFO] {len(morse)} Morse di tetrade lasciati come sono (--keep-tetrads)")

    # ── impilamento fra guanine sovrapposte ────────────────────────────────
    if args.stacking:
        site_type = site_type_of(topo, args.stacking)
        xyz, col, site_index = ref.site(involved, site_type, args.stacking)
        per_copy = defaultdict(list)
        for m in involved:
            per_copy[m // nuc].append(m)
        pairs = []
        for copy_mols in per_copy.values():
            for a_ in copy_mols:
                for b_ in copy_mols:
                    if b_ == a_ + 1 and tetrad_of[a_] != tetrad_of[b_]:
                        pairs.append((a_, b_))
        per_class = len(pairs) // n_copies
        print(f"\n[INFO] impilamento: sito {args.stacking} = indice "
              f"{sorted(set(site_index.values()))}; {len(pairs)} coppie, {per_class} per copia")
        if per_class != 4 * (len(local) - 1):
            print(f"[WARNING] attese {4 * (len(local) - 1)} coppie sovrapposte per copia "
                  f"({len(local)} tetradi, 4 tratti): controlla il registro")
        existing = {tuple(sorted(((int(b["mol_i"]), int(b.get("site_i", -1))),
                                  (int(b["mol_j"]), int(b.get("site_j", -1))))))
                    for _, b in all_morse}
        samples = defaultdict(list)
        for i, j in pairs:
            samples[(i % nuc + 1, j % nuc + 1)].append(ref.distances(xyz, col, i, j))
        sclasses = {key: {**fit_class(np.concatenate(rs), args), "copies": len(rs)}
                    for key, rs in sorted(samples.items())}
        head = "eps (kT)" if args.form == "lj" else "a (1/nm)"
        print(f"\n  {'coppia':>9} {'tetradi':>8} {'r0 (nm)':>8} {'sigma':>7} {head:>9} {'r_cut':>6}")
        for key, c in sclasses.items():
            ti = local.index(next(t for t in local if key[0] in t)) + 1
            tj = local.index(next(t for t in local if key[1] in t)) + 1
            val = c["epsilon"] / args.kT if args.form == "lj" else c["a"]
            print(f"  {str(key):>9} {f'{ti}-{tj}':>8} {c['r0']:8.3f} {c['sigma']:7.3f} "
                  f"{val:9.2f} {c['r_cut']:6.2f}")
        added = 0
        for i, j in pairs:
            endpoint = tuple(sorted(((i, site_index[i]), (j, site_index[j]))))
            if endpoint in existing:
                sys.exit(f"[ERROR] esiste gia' un Morse fra {endpoint}")
            c = sclasses[(i % nuc + 1, j % nuc + 1)]
            out["bonds"].append(morse_entry(i, j, site_index[i], site_index[j], c, args, "stacking"))
            added += 1
        out.setdefault("g4_topology", {})["stacking_representation"] = (
            f"site-site {args.stacking} {args.form} between stacked guanines (same G-tract, "
            f"adjacent tetrads), r0/a fitted from the mapped all-atom reference "
            f"(fit_tetrad_site_morse.py, D={args.D}, width={args.width})")
        print(f"[INFO] {added} contatti di impilamento aggiunti ({args.form})")
        report["stacking_site"] = args.stacking
        report["stacking_classes"] = [{"pair": list(k), **v} for k, v in sclasses.items()]

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[DONE] {args.out}")
    report_path = args.report or str(Path(args.out).with_suffix(".report.json"))
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"[DONE] {report_path}")


if __name__ == "__main__":
    main()
