#!/usr/bin/env python3
"""Morse sito-sito per le tetradi, con parametri stimati dal riferimento all-atom.

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

LE TETRADI
    Si ricavano dagli stessi Morse della topologia: in ogni copia i contatti
    formano grafi K4, uno per tetrade.  Nessun registro da fornire a parte, e
    lo script vale per qualunque G-quadruplex che usi quella rappresentazione.

I PARAMETRI
    Ogni classe (tetrade, coppia di posizioni) e' mediata sulle copie.  Per la
    distribuzione della distanza r della coppia:
        r0 = mediana di r
        a  = sqrt(kT / (2 D sigma^2))
    cioe' la larghezza di un Morse la cui curvatura al minimo, da sola,
    darebbe quella sigma.  Per default sigma = 1.4826 * MAD, una stima robusta
    della larghezza del NUCLEO della distribuzione: nel riferimento TEL22 le
    distanze B3-B3 hanno code lunghe verso le grandi r (aperture transitorie,
    p99 fino a 1.3 nm con mediana 0.6), che con la deviazione standard
    allargherebbero la buca fino a 3-4 volte.  --width std usa la deviazione
    standard.  E' un'approssimazione: la larghezza osservata
    risente anche degli altri cinque contatti e di tutto il resto del campo.
    Il residuo ML serve anche a questo.
        r_cut = r0 + CUT/a   (a CUT = 7 la buca e' scesa a e^-7 ~ 0.1%)

USO
    python3 fit_tetrad_site_morse.py --dataset tel26_dataset.bin \\
        --topology tel26_topology.json --out tel26_topology.b3morse.json \\
        [--site CG_DG_B3] [--D 50] [--stride 5] [--width mad|std]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--site", default="CG_DG_B3")
    ap.add_argument("--D", type=float, default=50.0, help="profondita' in kJ/mol")
    ap.add_argument("--kT", type=float, default=2.49)
    ap.add_argument("--cut", type=float, default=7.0, help="r_cut = r0 + cut/a")
    ap.add_argument("--nuc", type=int, default=None, help="residui per copia")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--width", choices=("mad", "std"), default="mad",
                    help="stima della larghezza: 1.4826*MAD (robusta) o deviazione standard")
    args = ap.parse_args()

    topo = json.loads(Path(args.topology).read_text())
    site_types = topo["mapping"]["site_types"]
    if args.site not in site_types:
        sys.exit(f"[ERROR] sito {args.site} assente: {sorted(site_types)}")
    site_type = int(site_types[args.site])
    nuc = args.nuc or topo.get("g4_topology", {}).get("residues_per_copy")
    if not nuc:
        sys.exit("[ERROR] residui per copia ignoti: passa --nuc")

    morse = [(k, b) for k, b in enumerate(topo["bonds"]) if b.get("type") == "morse"]
    if not morse:
        sys.exit("[ERROR] nessun Morse nella topologia: niente tetradi da cui partire")

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
    print(f"[INFO] {len(groups)} tetradi in {len(groups) // len(local)} copie; "
          f"registro {local}")

    # ── dataset: coordinate del sito scelto, tutti i frame in un colpo ─────
    buf = Path(args.dataset).read_bytes()
    words_i = np.frombuffer(buf, dtype=np.int32)
    words_f = np.frombuffer(buf, dtype=np.float32)
    n_frames = int(words_i[0])
    start, length, mols = frame_layout(words_i, n_frames)
    frames = np.arange(0, n_frames, max(1, args.stride))

    involved = sorted(tetrad_of)
    site_word, site_index = {}, {}
    for m in involved:
        w, ns = mols[m]
        types = [int(words_i[w + 4 * s]) for s in range(ns)]
        if site_type not in types:
            sys.exit(f"[ERROR] la molecola {m} non ha un sito di tipo {site_type} ({args.site})")
        s = types.index(site_type)
        site_word[m], site_index[m] = w + 4 * s + 1, s
    idx = np.array([site_word[m] for m in involved])
    # idx sono indici assoluti nel frame 0 (gia' comprendono start)
    base = frames[:, None] * length
    xyz = np.stack([words_f[base + idx + k] for k in range(3)], axis=-1).astype(np.float64)
    box = np.stack([words_f[start + frames * length + 2 + k] for k in range(3)], axis=-1).astype(np.float64)
    col = {m: c for c, m in enumerate(involved)}
    print(f"[INFO] {frames.size} frame su {n_frames} (stride {args.stride}), "
          f"sito {args.site} = indice {sorted(set(site_index.values()))} nella guanina")

    # ── distribuzioni per classe (tetrade, coppia locale) ──────────────────
    samples = defaultdict(list)
    for k, b in morse:
        i, j = int(b["mol_i"]), int(b["mol_j"])
        d = xyz[:, col[i]] - xyz[:, col[j]]
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=1)
        key = (tetrad_of[i], tuple(sorted((i % nuc + 1, j % nuc + 1))))
        samples[key].append(r)

    classes = {}
    for key, rs in sorted(samples.items()):
        r = np.concatenate(rs)
        med, std = float(np.median(r)), float(np.std(r))
        mad = 1.4826 * float(np.median(np.abs(r - med)))
        sig = mad if args.width == "mad" else std
        a = math.sqrt(args.kT / (2.0 * args.D * sig * sig))
        classes[key] = {"r0": med, "sigma": sig, "sigma_std": std, "sigma_mad": mad, "a": a,
                        "r_cut": med + args.cut / a,
                        "p01": float(np.percentile(r, 1)), "p99": float(np.percentile(r, 99)),
                        "copies": len(rs), "n": int(r.size)}

    # lati e diagonali: in una tetrade quadrata le due coppie piu' lontane
    # sono le diagonali (rapporto atteso ~ sqrt(2))
    print(f"\n  {'tetrade':>7} {'coppia':>9} {'r0 (nm)':>8} {'sigma':>7} {'a (1/nm)':>9} "
          f"{'r_cut':>6}  ruolo")
    for t in range(len(local)):
        keys = sorted((k for k in classes if k[0] == t), key=lambda k: classes[k]["r0"])
        for rank, key in enumerate(keys):
            c = classes[key]
            c["role"] = "diagonale" if rank >= 4 else "lato"
            print(f"  {t + 1:>7} {str(key[1]):>9} {c['r0']:8.3f} {c['sigma']:7.3f} "
                  f"{c['a']:9.2f} {c['r_cut']:6.2f}  {c['role']}")
        lati = [classes[k]["r0"] for k in keys[:4]]
        diag = [classes[k]["r0"] for k in keys[4:]]
        if lati and diag:
            print(f"          diagonale / lato = {np.mean(diag) / np.mean(lati):.3f}"
                  f"   (quadrato: {math.sqrt(2):.3f})")

    # ── topologia nuova ────────────────────────────────────────────────────
    out = json.loads(json.dumps(topo))
    for k, b in morse:
        i, j = int(b["mol_i"]), int(b["mol_j"])
        c = classes[(tetrad_of[i], tuple(sorted((i % nuc + 1, j % nuc + 1))))]
        e = out["bonds"][k]
        e.update({"site_i": site_index[i], "site_j": site_index[j],
                  "D": args.D, "a": c["a"], "r0": c["r0"], "r_cut": c["r_cut"],
                  "exclude_wca": False})
        e.pop("r_switch", None)
    out.setdefault("g4_topology", {})["morse_representation"] = (
        f"complete K4 per tetrad, site-site {args.site}, r0/a fitted from the "
        f"mapped all-atom reference (fit_tetrad_site_morse.py, D={args.D}, width={args.width})")
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[DONE] {args.out}: {len(morse)} Morse sito-sito")

    report = args.report or str(Path(args.out).with_suffix(".report.json"))
    Path(report).write_text(json.dumps({
        "dataset": args.dataset, "topology": args.topology, "site": args.site,
        "D": args.D, "kT": args.kT, "cut": args.cut, "stride": args.stride,
        "width": args.width,
        "classes": [{"tetrad": k[0] + 1, "pair": list(k[1]), **v} for k, v in classes.items()],
    }, indent=2) + "\n")
    print(f"[DONE] {report}")


if __name__ == "__main__":
    main()
