#!/usr/bin/env python3
"""Unisce al dataset di un insieme di prior i blocchi costruiti con PART=...

PERCHE'
    Con forze salvate ogni 20 ps il residuo da imparare e' per il 99 % rumore:
    la forza media si estrae solo con molti campioni.  La continuazione della
    produzione all-atom (prod-2, ~3 200 frame) aggiunge il 50 % di dati.
    02_build_dataset.sh con PART=prod2 la costruisce con gli STESSI prior
    dell'insieme (tel26_<set>_dataset.prod2.bin); qui si uniscono i frame in
    un insieme nuovo, cosi' i modelli allenati sul dataset piu' grande hanno
    nomi e manifest loro e non si confondono con quelli vecchi.

CONTROLLI (fallisce se non passano)
    - stessa struttura di frame (molecole, siti, tipi) in tutti i blocchi;
    - prior identici (il blocco deve averli letti, non ristimati);
    - geometria dei corpi rigidi entro --rb-tol nm (e' stimata dai dati di
      ciascun blocco: differenze minime sono attese, grandi no).

USO (in tutorials/tel26)
    python3 concat_datasets.py --set lp0 --parts prod2 --out-set lp0x
        legge  tel26_lp0_dataset.bin, tel26_lp0_dataset.prod2.bin, ...
        scrive tel26_lp0x_dataset.bin e copia topologia, cg_priors e
               rigid_bodies_info di lp0 come quelli di lp0x
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_tetrad_site_morse import frame_layout  # noqa: E402


def names(s):
    return (f"tel26_topology.{s}.json", f"tel26_{s}_dataset.bin",
            f"cg_priors.{s}.json", f"rigid_bodies_info.{s}.json")


def read(path):
    words = np.fromfile(path, dtype=np.int32)
    T = int(words[0])
    start, length, mols = frame_layout(words, T)
    if 1 + T * length != words.size:
        sys.exit(f"[ERROR] {path}: {words.size} parole, attese {1 + T * length}")
    types = np.array([words[start + w + 4 * s] for w, ns in mols for s in range(ns)])
    return words, T, length, mols, types


def strip(d):
    """Prior senza i campi descrittivi che possono differire fra i blocchi."""
    d = dict(d)
    for k in ("derived_prior_set", "source", "provenance", "created", "dataset"):
        d.pop(k, None)
    return d


def rb_sites(rb):
    out = {}
    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, path + (k,))
        elif isinstance(x, list) and x and all(isinstance(v, (int, float)) for v in x):
            out[path] = np.asarray(x, dtype=float)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, path + (i,))
    walk(rb, ())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="insieme di partenza (es. lp0)")
    ap.add_argument("--parts", required=True, help="blocchi da aggiungere, separati da virgole (es. prod2)")
    ap.add_argument("--out-set", required=True, help="nome del nuovo insieme (es. lp0x)")
    ap.add_argument("--rb-tol", type=float, default=0.01, help="nm")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    topo, data, priors, rb = names(args.set)
    otopo, odata, opriors, orb = names(args.out_set)
    for f in (topo, data, priors, rb):
        if not Path(f).is_file():
            sys.exit(f"[ERROR] manca {f}")
    if Path(odata).exists() and not args.force:
        sys.exit(f"[ERROR] {odata} esiste gia' (usa --force)")

    words0, T0, length, mols, types = read(data)
    p0 = strip(json.loads(Path(priors).read_text()))
    rb0 = rb_sites(json.loads(Path(rb).read_text()))
    blocks = [(data, words0, T0)]
    for part in args.parts.split(","):
        pdata = data.replace(".bin", f".{part}.bin")
        ppriors = priors.replace(".json", f".{part}.json")
        prb = rb.replace(".json", f".{part}.json")
        for f in (pdata, ppriors, prb):
            if not Path(f).is_file():
                sys.exit(f"[ERROR] manca {f} (02_build_dataset.sh con PART={part})")
        w, T, L, m, ty = read(pdata)
        if L != length or m != mols or not np.array_equal(ty, types):
            sys.exit(f"[ERROR] {pdata}: struttura di frame diversa da {data}")
        if strip(json.loads(Path(ppriors).read_text())) != p0:
            sys.exit(f"[ERROR] {ppriors} diverso da {priors}: il blocco non ha usato gli stessi prior")
        rbp = rb_sites(json.loads(Path(prb).read_text()))
        common = set(rb0) & set(rbp)
        dmax = max((float(np.max(np.abs(rb0[k] - rbp[k]))) for k in common
                    if rb0[k].shape == rbp[k].shape), default=0.0)
        print(f"[INFO] {pdata}: {T} frame; corpi rigidi, scarto massimo {dmax:.4f} "
              f"(su {len(common)} campi numerici)")
        if dmax > args.rb_tol:
            sys.exit(f"[ERROR] geometria dei corpi rigidi diversa oltre {args.rb_tol}: "
                     "controlla che il blocco venga dalla stessa produzione")
        blocks.append((pdata, w, T))

    total = sum(T for _, _, T in blocks)
    with open(odata, "wb") as fh:
        np.asarray([total], dtype=np.int32).tofile(fh)
        for _, w, _ in blocks:
            w[1:].tofile(fh)
    for src, dst in ((topo, otopo), (priors, opriors), (rb, orb)):
        shutil.copyfile(src, dst)
    check = np.fromfile(odata, dtype=np.int32, count=1)[0]
    print(f"[DONE] {odata}: {total} frame ({' + '.join(str(T) for _, _, T in blocks)}); "
          f"{otopo}, {opriors}, {orb} copiati da {args.set}")
    print(f"       validazione 'tail' = ultimo 20 % = frame del blocco aggiunto; controllo {check}")


if __name__ == "__main__":
    main()
