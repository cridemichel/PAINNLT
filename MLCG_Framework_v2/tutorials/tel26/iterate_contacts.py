#!/usr/bin/env python3
"""Aggiustamento iterativo dei contatti pair-specific sulle distribuzioni del riferimento.

PERCHE'
    fit_tetrad_site_morse.py stima ogni contatto dalla distribuzione di UNA
    distanza, come se fosse l'unica forza in gioco.  Ma quella distribuzione
    risente anche degli altri contatti, degli angoli, della WCA: sommando i
    termini la struttura esce piu' rigida (o piu' molle) del riferimento, e il
    difetto non si corregge ritoccando a mano un termine alla volta.

    Qui si parte da una corsa con i soli prior e si confronta, classe per
    classe, la distanza nel CG con quella del riferimento mappato:
        Morse   r0 <- r0 + f (mediana_rif - mediana_CG)
                a  <- a  (sigma_CG / sigma_rif)^f
        LJ      sigma   <- sigma + f (mediana_rif - mediana_CG) / 2^(1/6)
                epsilon <- epsilon (sigma_CG / sigma_rif)^(2 f)
    (f = --damp; lo spostamento di r0 e' limitato a --max-dr per iterazione,
    0.05 nm).  Tutte le classi si aggiornano insieme, da una sola corsa:
    e' lo spirito dell'Iterative Boltzmann Inversion, ma con la forma
    analitica dei contatti, che restano reversibili.  La profondita' D dei
    Morse non cambia.  Il fattore per iterazione su a/epsilon e' limitato a
    [1/--max-step, --max-step] per non oscillare.

CRITERIO DI ARRESTO (stampato a ogni iterazione)
    |mediana_rif - mediana_CG| < --tol-r (0.01 nm) e
    |sigma_CG / sigma_rif - 1| < --tol-s (10%) per tutte le classi.

USO (in tutorials/tel26)
    python3 iterate_contacts.py --topology tel26_topology.b3stack.json \\
        --run samples_priors_b3stack_100ps.npz --out tel26_topology.b3it1.json
    python3 derive_prior_set.py --base b3stack --set b3it1
    ... corsa con i soli prior PRIOR_SET=b3it1, poi di nuovo con
        --topology tel26_topology.b3it1.json --run samples_priors_b3it1_100ps.npz
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_tetrad_site_morse import Reference  # noqa: E402

CONTACT_TYPES = ("morse", "lj")


def width(r, how):
    med = float(np.median(r))
    if how == "mad":
        return med, 1.4826 * float(np.median(np.abs(r - med)))
    return med, float(np.std(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology", required=True, help="topologia con i contatti attuali")
    ap.add_argument("--run", required=True, help="samples_*.npz della corsa con quei contatti")
    ap.add_argument("--dataset", default="tel26_dataset.bin", help="riferimento mappato")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--nuc", type=int, default=None)
    ap.add_argument("--skip-ps", type=float, default=50.0, help="scarta il transiente iniziale")
    ap.add_argument("--run-stride", type=int, default=5)
    ap.add_argument("--stride", type=int, default=5, help="stride sul riferimento")
    ap.add_argument("--width", choices=("mad", "std"), default="mad")
    ap.add_argument("--damp", type=float, default=1.0, help="frazione della correzione applicata")
    ap.add_argument("--max-step", type=float, default=2.0)
    ap.add_argument("--max-dr", type=float, default=0.05,
                    help="spostamento massimo di r0 per iterazione, nm")
    ap.add_argument("--cut", type=float, default=7.0, help="Morse: r_cut = r0 + cut/a")
    ap.add_argument("--lj-cut", type=float, default=3.5, help="LJ: r_cut = lj_cut * sigma")
    ap.add_argument("--tol-r", type=float, default=0.01)
    ap.add_argument("--tol-s", type=float, default=0.10)
    args = ap.parse_args()

    topo = json.loads(Path(args.topology).read_text())
    nuc = args.nuc or topo.get("g4_topology", {}).get("residues_per_copy")
    if not nuc:
        sys.exit("[ERROR] residui per copia ignoti: passa --nuc")
    contacts = [(k, b) for k, b in enumerate(topo["bonds"])
                if str(b.get("type", "")).lower() in CONTACT_TYPES]
    if not contacts:
        sys.exit("[ERROR] nessun contatto pair-specific nella topologia")
    for _, b in contacts:
        if int(b.get("site_i", -1)) < 0 or int(b.get("site_j", -1)) < 0:
            sys.exit("[ERROR] contatto sui centri di massa: servono contatti sito-sito")

    # classi: ruolo, coppia di posizioni nella copia, siti
    def key(b):
        i, j = int(b["mol_i"]), int(b["mol_j"])
        a_, b_ = (i, int(b["site_i"])), (j, int(b["site_j"]))
        if (a_[0] % nuc) > (b_[0] % nuc):
            a_, b_ = b_, a_
        return (b.get("role", "?"), a_[0] % nuc + 1, a_[1], b_[0] % nuc + 1, b_[1])

    # ── riferimento ────────────────────────────────────────────────────────
    ref = Reference(args.dataset, args.stride)
    ref_samples = defaultdict(list)
    by_site = defaultdict(set)
    for _, b in contacts:
        by_site[int(b["site_i"])].add(int(b["mol_i"]))
        by_site[int(b["site_j"])].add(int(b["mol_j"]))
    ref_xyz = {}
    for s, mols in by_site.items():
        mols = sorted(mols)
        w_type = None
        # il tipo del sito s: dalla prima molecola, dal dataset
        w, ns = ref.mols[mols[0]]
        w_type = int(ref.words_i[w + 4 * s])
        xyz, col, idx = ref.site(mols, w_type, f"indice {s}")
        if any(v != s for v in idx.values()):
            sys.exit(f"[ERROR] sito {s}: il tipo {w_type} non sta sempre allo stesso indice")
        ref_xyz[s] = (xyz, col)
    for _, b in contacts:
        i, j, si, sj = int(b["mol_i"]), int(b["mol_j"]), int(b["site_i"]), int(b["site_j"])
        xi, ci = ref_xyz[si]
        xj, cj = ref_xyz[sj]
        d = xi[:, ci[i]] - xj[:, cj[j]]
        d -= ref.box * np.round(d / ref.box)
        ref_samples[key(b)].append(np.linalg.norm(d, axis=1))

    # ── corsa CG ───────────────────────────────────────────────────────────
    z = np.load(args.run)
    t = np.asarray(z["time_ps"], dtype=float)
    keep = np.flatnonzero(t >= args.skip_ps)[::max(1, args.run_stride)]
    if keep.size == 0:
        sys.exit(f"[ERROR] {args.run}: nessun frame dopo {args.skip_ps} ps")
    sites = np.asarray(z["sites"][keep], dtype=np.float64)
    smol, sidx = np.asarray(z["site_molecule"]), np.asarray(z["site_index"])
    box = np.asarray(z["box"], dtype=np.float64)
    box = box[keep] if box.ndim == 2 and box.shape[0] == t.size else np.broadcast_to(box, (keep.size, 3))
    lookup = {(int(m), int(s)): n for n, (m, s) in enumerate(zip(smol, sidx))}
    print(f"[INFO] riferimento: {ref.frames.size} frame; CG {args.run}: {keep.size} frame "
          f"da {t[keep[0]]:.1f} a {t[keep[-1]]:.1f} ps")
    cg_samples = defaultdict(list)
    for _, b in contacts:
        a_ = lookup[(int(b["mol_i"]), int(b["site_i"]))]
        b_ = lookup[(int(b["mol_j"]), int(b["site_j"]))]
        d = sites[:, a_] - sites[:, b_]
        d -= box * np.round(d / box)
        cg_samples[key(b)].append(np.linalg.norm(d, axis=1))

    # ── correzione per classe ──────────────────────────────────────────────
    stats = {}
    for k in ref_samples:
        mr, sr = width(np.concatenate(ref_samples[k]), args.width)
        mc, sc = width(np.concatenate(cg_samples[k]), args.width)
        stats[k] = {"med_ref": mr, "sig_ref": sr, "med_cg": mc, "sig_cg": sc}

    out = json.loads(json.dumps(topo))
    lo, hi = 1.0 / args.max_step, args.max_step
    for kb, b in contacts:
        st = stats[key(b)]
        # spostamento limitato: con una corsa lontana dal riferimento (tetradi
        # aperte) la differenza delle mediane non e' una correzione lineare
        dmed = float(np.clip(args.damp * (st["med_ref"] - st["med_cg"]), -args.max_dr, args.max_dr))
        ratio = (st["sig_cg"] / st["sig_ref"]) if st["sig_ref"] > 0 else 1.0
        e = out["bonds"][kb]
        if e["type"] == "morse":
            e["r0"] = max(0.1, float(e["r0"]) + dmed)
            e["a"] = float(e["a"]) * min(hi, max(lo, ratio ** args.damp))
            e["r_cut"] = e["r0"] + args.cut / e["a"]
        else:
            e["sigma"] = max(0.1, float(e["sigma"]) + dmed / 2.0 ** (1.0 / 6.0))
            e["epsilon"] = float(e["epsilon"]) * min(hi, max(lo, ratio ** (2.0 * args.damp)))
            e["r_cut"] = args.lj_cut * e["sigma"]
            e["shift"] = "auto"
        e.pop("r_switch", None)

    # ── riepilogo e criterio di arresto ────────────────────────────────────
    by_role = defaultdict(list)
    for k, st in stats.items():
        by_role[k[0]].append(st)
    ok_all = True
    print(f"\n  {'ruolo':<10}{'classi':>7}{'|dmed| max':>12}{'|dmed| med':>12}"
          f"{'sCG/sRif':>18}{'entro tol.':>12}")
    for role, sts in sorted(by_role.items()):
        dm = np.array([abs(s["med_ref"] - s["med_cg"]) for s in sts])
        rs = np.array([s["sig_cg"] / s["sig_ref"] for s in sts if s["sig_ref"] > 0])
        ok = (dm < args.tol_r) & (np.abs(rs - 1.0) < args.tol_s)
        ok_all &= bool(ok.all())
        print(f"  {role:<10}{len(sts):7d}{dm.max():12.4f}{np.median(dm):12.4f}"
              f"{rs.min():8.2f}-{rs.max():<8.2f}{int(ok.sum()):>6d}/{len(sts)}")
    worst = sorted(stats.items(), key=lambda kv: -abs(kv[1]["med_ref"] - kv[1]["med_cg"]))[:5]
    print("\n  classi piu' lontane (ruolo, res_i, sito_i, res_j, sito_j): mediana rif / CG, sigma rif / CG")
    for k, st in worst:
        print(f"    {str(k):<34} {st['med_ref']:.3f} / {st['med_cg']:.3f}   "
              f"{st['sig_ref']:.3f} / {st['sig_cg']:.3f}")
    print(f"\n[{'CONVERGIUTO' if ok_all else 'NON ANCORA'}] criterio: |dmed| < {args.tol_r} nm e "
          f"|sCG/sRif - 1| < {args.tol_s:.0%} per tutte le classi")

    out.setdefault("g4_topology", {}).setdefault("contact_iterations", []).append({
        "from": args.topology, "run": args.run, "damp": args.damp, "converged": ok_all})
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    report = args.report or str(Path(args.out).with_suffix(".iter.json"))
    Path(report).write_text(json.dumps({
        "topology": args.topology, "run": args.run, "converged": ok_all,
        "classes": [{"key": list(k), **v} for k, v in stats.items()]}, indent=2) + "\n")
    print(f"[DONE] {args.out}\n[DONE] {report}")


if __name__ == "__main__":
    main()
