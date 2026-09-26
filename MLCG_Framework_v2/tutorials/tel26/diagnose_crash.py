#!/usr/bin/env python3
"""Dove nasce la forza che ha fatto scattare il controllo di sicurezza.

run_cg_md.py, quando max_f supera la soglia, scrive crash_checkpoint.npz con
posizioni e forze di ogni particella.  Qui si guardano:
  - le particelle con la forza piu' grande, tradotte in residuo:sito e copia;
  - i diedri (torsione fra guanine, backbone) con un angolo di legame vicino a
    0 o 180 gradi: la forza di un diedro a coseno cresce come 1/sin(theta), e
    con K di centinaia di kT basta un angolo a 15 gradi dall'allineamento per
    superare la soglia.  Se le forze massime cadono su un diedro cosi', il
    colpevole e' il prior, non la rete.

Ordine delle particelle come in run_cg_md.py: per ogni molecola il COM e poi i
suoi siti virtuali (6 per DG, 1 per DA/DT); i marker dei contatti vengono dopo.

USO (in tutorials/tel26)
    python3 diagnose_crash.py tel26_topology.lp0.json [--checkpoint crash_checkpoint.npz]
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topology")
    ap.add_argument("--checkpoint", default="crash_checkpoint.npz")
    ap.add_argument("--seq", default="TTAGGGTTAGGGTTAGGGTTAGGGTT")
    ap.add_argument("--copies", type=int, default=10)
    ap.add_argument("--box", type=float, default=11.91)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    z = np.load(args.checkpoint)
    pos, f = np.asarray(z["positions"], float), np.asarray(z["forces"], float)
    topo = json.load(open(args.topology))
    nuc, seq = len(args.seq), args.seq
    pid, who, n = {}, {}, 0
    for c in range(args.copies):
        for r, ch in enumerate(seq):
            m = c * nuc + r
            who[n] = (m, -1); n += 1
            for s in range(6 if ch == "G" else 1):
                pid[(m, s)] = n; who[n] = (m, s); n += 1
    print(f"{len(pos)} particelle: {n} fra COM e siti, {len(pos) - n} marker")
    box = np.full(3, args.box)

    def lab(i):
        if i not in who:
            return "marker"
        m, s = who[i]
        return f"{m % nuc + 1}{seq[m % nuc]}:{'COM' if s < 0 else s} (copia {m // nuc + 1})"

    fn = np.linalg.norm(f, axis=1)
    print(f"\nforze piu' grandi:")
    for i in np.argsort(-fn)[:args.top]:
        print(f"  {int(i):5d}  {lab(int(i)):<24} |f| = {fn[i]:10.1f}")

    def sin_angle(a, b, c):
        u, v = pos[a] - pos[b], pos[c] - pos[b]
        u -= box * np.round(u / box); v -= box * np.round(v / box)
        return float(np.linalg.norm(np.cross(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v)))

    rows = []
    hot = set(int(i) for i in np.argsort(-fn)[:args.top])
    for d in topo.get("dihedrals", []):
        q = [pid[(d[f"mol_{k}"], d[f"site_{k}"])] for k in "ijkl"]
        s = min(sin_angle(q[0], q[1], q[2]), sin_angle(q[1], q[2], q[3]))
        rows.append((s, d.get("role", "?"), d["k"] / 2.49, q, bool(hot & set(q))))
    rows.sort()
    print(f"\ndiedri con gli angoli di legame piu' vicini a 0/180 gradi:")
    print(f"  {'sin':>7} {'ruolo':<9} {'K (kT)':>7}  particelle  (* = fra le forze piu' grandi)")
    for s, role, k, q, h in rows[:args.top]:
        print(f"  {s:7.4f} {role:<9} {k:7.1f}  {' '.join(lab(i).split(' ')[0] for i in q)}{'  *' if h else ''}")
    on_hot = [r for r in rows if r[4]]
    if on_hot:
        s, role, k, q, _ = on_hot[0]
        print(f"\n[INFO] fra i diedri che toccano le forze massime il piu' allineato ha sin {s:.3f} "
              f"({role}, K {k:.0f} kT)")
    else:
        print("\n[INFO] nessun diedro tocca le particelle con le forze massime")


if __name__ == "__main__":
    main()
