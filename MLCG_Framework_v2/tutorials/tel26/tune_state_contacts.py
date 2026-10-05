#!/usr/bin/env python3
"""Regola le profondita' D dei contatti di stato sulle popolazioni AA.

PERCHE'
    add_state_contacts.py aggiunge contatti Morse stretti (ruolo "state") con
    D iniziale piccolo; la popolazione del contatto nel CG va portata a quella
    AA.  Per un contatto a due stati (dentro / fuori dal bacino) la popolazione
    e' f = 1 / (1 + exp(dG/kT)) e una variazione della profondita' del pozzo
    sposta dG di circa -dD, quindi
        D_nuovo = D + lambda * kT * [logit(f_AA) - logit(f_CG)],
    logit(f) = ln(f / (1 - f)).  E' l'inversione di Boltzmann iterativa
    applicata alla sola popolazione del contatto; le frazioni si tagliano in
    [--f-min, 1 - --f-min] e il passo in +-(--max-step) kT.  Il bacino
    (r_basin), i siti e f_AA vengono da g4_topology.state_contacts.

COME
    Legge i campioni CG (stesso formato dello script 47: '+' tratti della
    stessa corsa, ',' replicas), calcola per ogni contatto la distanza fra i
    due siti in tutte le copie (immagine minima), f_CG = frazione con
    d < r_basin, e scrive una nuova topologia con i D aggiornati (bonds e
    state_contacts).  Con --drop si tolgono contatti (per esempio quelli
    sostenuti da una sola copia AA).  Poi derive_prior_set.py --base lp2.

USO (in tutorials/tel26)
    python3 tune_state_contacts.py --topology tel26_topology.lp2c.json \\
        --cg r0.npz,r1.npz --skip-ps 50 --out tel26_topology.lp2c1.json [--drop 20-18] [--dry-run]
    python3 derive_prior_set.py --base lp2 --set lp2c1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("s47", HERE / "47_tel26_timescales.py")
s47 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s47)
cv = s47.cv


def logit(f, fmin):
    f = min(max(f, fmin), 1.0 - fmin)
    return math.log(f / (1.0 - f))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology", required=True, help="topologia con g4_topology.state_contacts")
    ap.add_argument("--cg", default=None, help="campioni CG: a.npz[+b.npz][,r1.npz...]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nuc", type=int, default=26)
    ap.add_argument("--kT", type=float, default=2.49)
    ap.add_argument("--lam", type=float, default=1.0, help="frazione del passo di Boltzmann")
    ap.add_argument("--max-step", type=float, default=2.0, help="passo massimo in kT")
    ap.add_argument("--f-min", type=float, default=0.01)
    ap.add_argument("--D-min", type=float, default=0.1)
    ap.add_argument("--skip-ps", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--drop", default="", help="coppie da togliere, 'i-j,...' (1-based)")
    ap.add_argument("--dry-run", action="store_true", help="solo la tabella, nessun file")
    args = ap.parse_args()
    cv.set_nuc(args.nuc)
    nuc = args.nuc

    topo = json.loads(pathlib.Path(args.topology).read_text())
    meta = topo.get("g4_topology", {}).get("state_contacts", [])
    if not meta:
        raise SystemExit(f"[ERROR] {args.topology} non ha g4_topology.state_contacts")
    drop = {tuple(sorted(int(x) for x in p.split("-"))) for p in args.drop.split(",") if p.strip()}

    S = L = None
    if args.cg:
        parts = []
        for rep in args.cg.split(","):
            Sc, Lc, ncc, tc = s47.load_cg(rep, args.stride)
            keep = (tc - tc[0]) >= args.skip_ps
            Lc = np.asarray(Lc, float)
            parts.append((Sc[keep], Lc[keep] if Lc.ndim == 2 else np.tile(Lc, (keep.sum(), 1))))
        T = min(p[0].shape[0] for p in parts)
        print(f"[INFO] CG: {len(parts)} replicas, {T} frame ciascuna (dopo {args.skip_ps:g} ps)")

    rows, new_D = [], {}
    for m in meta:
        key = tuple(sorted((m["res_i"], m["res_j"])))
        if key in drop:
            rows.append((m, None, None, "tolto"))
            continue
        f_cg = None
        if args.cg:
            hits, tot = 0, 0
            for Sc, Lc in parts:
                ncopy = Sc.shape[1] // nuc
                for c in range(ncopy):
                    a = Sc[:T, c * nuc + m["res_i"] - 1, m["site_i"]]
                    b = Sc[:T, c * nuc + m["res_j"] - 1, m["site_j"]]
                    d = a - b
                    d -= Lc[:T] * np.round(d / Lc[:T])
                    r = np.linalg.norm(d, axis=-1)
                    hits += int((r < m["r_basin"]).sum()); tot += r.size
            f_cg = hits / tot
            step = args.lam * (logit(m["f_aa"], args.f_min) - logit(f_cg, args.f_min))
            step = max(-args.max_step, min(args.max_step, step))
            D_new = max(args.D_min, m["D"] + step * args.kT)
        else:
            D_new = m["D"]
        new_D[key] = D_new
        ok = f_cg is not None and abs(f_cg - m["f_aa"]) <= max(m["f_aa_err"], 0.02)
        rows.append((m, f_cg, D_new, "entro l'errore AA" if ok else ""))

    print(f"\n  {'contatto':<18s}{'f_AA':>14s}{'f_CG':>8s}{'D (kJ/mol)':>12s}{'D nuovo':>10s}   nota")
    for m, f_cg, D_new, note in rows:
        fcg = f"{f_cg:8.3f}" if f_cg is not None else f"{'-':>8s}"
        dn = f"{D_new:10.2f}" if D_new is not None else f"{'-':>10s}"
        print(f"  {m['label']:<18s}{m['f_aa']:8.3f}±{m['f_aa_err']:.3f}{fcg}{m['D']:12.2f}{dn}   {note}")

    if args.dry_run or not args.out:
        return
    out = json.loads(json.dumps(topo))
    bonds = []
    for b in out["bonds"]:
        if b.get("role") != "state":
            bonds.append(b); continue
        key = tuple(sorted(((int(b["mol_i"]) % nuc) + 1, (int(b["mol_j"]) % nuc) + 1)))
        if key in drop:
            continue
        b["D"] = new_D[key]
        bonds.append(b)
    out["bonds"] = bonds
    newmeta = []
    for m in out["g4_topology"]["state_contacts"]:
        key = tuple(sorted((m["res_i"], m["res_j"])))
        if key in drop:
            continue
        m.setdefault("history", []).append({"D": m["D"], "f_cg": next((r[1] for r in rows if r[0]["pair"] == m["pair"]), None),
                                            "from": args.topology, "cg": args.cg})
        m["D"] = new_D[key]
        newmeta.append(m)
    out["g4_topology"]["state_contacts"] = newmeta
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[DONE] {args.out}: {len(newmeta)} contatti di stato, {sum(b.get('role') == 'state' for b in bonds)} legami")


if __name__ == "__main__":
    main()
