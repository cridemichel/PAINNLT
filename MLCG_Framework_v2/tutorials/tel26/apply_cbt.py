#!/usr/bin/env python3
"""Passa i diedri di un ruolo alla forma a flessione-torsione combinate (CBT).

PERCHE'
    Un diedro a coseno ha forze ~ K / sin(t) quando uno dei due angoli di
    legame t si avvicina a 0 o 180 gradi.  Nel TEL26 col residuo ML acceso i
    loop si raddrizzano: l'angolo 14T-15A-16G e' arrivato a sin t = 0,009 e i
    due diedri di backbone che ci poggiano hanno dato forze ~10^4 (tutte le
    produzioni ML fermate dal controllo di sicurezza).  Qui il diedro si
    moltiplica per g(t1) g(t2), con g = 1 finche' sin t >= 0,3 e un raccordo C1
    a zero sotto (l'idea della flessione-torsione combinate di Bulacu et al.,
    JCTC 2013, ma senza il loro sin^3, che sul backbone del TEL26 -- angoli di
    140-150 gradi -- ridurrebbe il diedro di 20-100 volte ovunque).

LA RICALIBRAZIONE
    K era stimato dalla larghezza della distribuzione di phi (von Mises) per
    V = K (1 - cos).  Con V = K g1 g2 (1 - cos) la rigidita' media
    in phi e' K <g1 g2>: per conservarla K diventa K / <g1 g2>, con la media sul
    riferimento mappato, classe per classe (ruolo, residuo iniziale e finale
    nella copia).  --k-max limita K.

USO (in tutorials/tel26)
    python3 apply_cbt.py --dataset tel26_dataset.bin --topology tel26_topology.lp0.json \\
        --out tel26_topology.lp1.json
    python3 derive_prior_set.py --base b3stack --set lp1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_tetrad_site_morse import Reference  # noqa: E402


S0 = 0.3  # come mlcg_cbt_s0 in install_dihedral_cbt.py


def sin3(u, w):
    """(fattore di smorzamento g, sin t) per ogni frame."""
    c = -np.einsum("ij,ij->i", u, w) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1))
    c = np.clip(c, -1.0, 1.0)
    x = (1.0 - c * c) / (S0 * S0)
    return np.where(x >= 1.0, 1.0, x * (2.0 - x)), np.sqrt(1.0 - c * c)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roles", default="backbone", help="ruoli dei diedri da convertire")
    ap.add_argument("--nuc", type=int, default=None)
    ap.add_argument("--kT", type=float, default=2.49)
    ap.add_argument("--k-max", type=float, default=100.0, help="K massimo dopo la ricalibrazione, in kT")
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    topo = json.loads(Path(args.topology).read_text())
    nuc = args.nuc or topo.get("g4_topology", {}).get("residues_per_copy")
    if not nuc:
        sys.exit("[ERROR] residui per copia ignoti: passa --nuc")
    roles = set(args.roles.split(","))
    sel = [(k, d) for k, d in enumerate(topo.get("dihedrals", [])) if d.get("role") in roles]
    if not sel:
        sys.exit(f"[ERROR] nessun diedro con ruolo {sorted(roles)}")
    if any(d.get("cbt") for _, d in sel):
        sys.exit("[ERROR] alcuni diedri sono gia' CBT: parti dalla topologia senza CBT")
    if any(str(d.get("type", "cosine")).lower() != "cosine" for _, d in sel):
        sys.exit("[ERROR] la CBT vale solo per i diedri a coseno")

    ref = Reference(args.dataset, args.stride)
    base = ref.frames[:, None] * ref.length
    cache = {}

    def xyz(m, s):
        if (m, s) not in cache:
            w, ns = ref.mols[m]
            if s >= ns:
                sys.exit(f"[ERROR] la molecola {m} non ha il sito {s}")
            cache[(m, s)] = np.stack([ref.words_f[base[:, 0] + w + 4 * s + 1 + k] for k in range(3)],
                                     axis=-1).astype(np.float64)
        return cache[(m, s)]

    def mic(v):
        return v - ref.box * np.round(v / ref.box)

    g12 = defaultdict(list)
    smin = defaultdict(list)
    for _, d in sel:
        p = [xyz(int(d[f"mol_{c}"]), int(d[f"site_{c}"])) for c in "ijkl"]
        b1, b2, b3 = mic(p[1] - p[0]), mic(p[2] - p[1]), mic(p[3] - p[2])
        g1, s1 = sin3(b1, b2)
        g2, s2 = sin3(b2, b3)
        key = (d.get("role"), int(d["mol_i"]) % nuc + 1, int(d["mol_l"]) % nuc + 1)
        g12[key].append(g1 * g2)
        smin[key].append(np.minimum(s1, s2))

    factor = {k: float(np.mean(np.concatenate(v))) for k, v in g12.items()}
    print(f"[INFO] {len(sel)} diedri {sorted(roles)} in {len(factor)} classi, {ref.frames.size} frame")
    print(f"\n  {'classe':>12} {'<g1 g2>':>8} {'sin min 0,1%':>13} {'K prima':>8} {'K dopo':>8}  (kT)")
    out = json.loads(json.dumps(topo))
    shown = set()
    for k, d in sel:
        key = (d.get("role"), int(d["mol_i"]) % nuc + 1, int(d["mol_l"]) % nuc + 1)
        f = factor[key]
        k_old = float(d["k"])
        k_new = min(k_old / max(f, 1e-6), args.k_max * args.kT)
        e = out["dihedrals"][k]
        e["k"] = k_new
        e["cbt"] = True
        if key not in shown:
            shown.add(key)
            q = float(np.percentile(np.concatenate(smin[key]), 0.1))
            print(f"  {key[0][:4]} {key[1]:>2}-{key[2]:<2}  {f:8.3f} {q:13.3f} {k_old / args.kT:8.2f} "
                  f"{k_new / args.kT:8.2f}")
    out.setdefault("g4_topology", {})["cbt_dihedrals"] = (
        f"roles {sorted(roles)}: angle-damped dihedral V = K g(t1) g(t2) (1 - cos), g = 1 for "
        f"sin t >= {S0}, x(2-x) below (x = sin^2 t / S0^2); K rescaled by 1/<g1 g2> over the mapped "
        f"reference, K <= {args.k_max} kT")
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[DONE] {args.out}")


if __name__ == "__main__":
    main()
