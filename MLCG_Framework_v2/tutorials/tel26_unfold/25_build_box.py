#!/usr/bin/env python3
"""Scatola con molti TEL26: prior a N copie + configurazione iniziale da corse a una copia.

PERCHE' (2/11)
    La S(q) totale da confrontare con lo SAXS richiede centinaia di molecole alla concentrazione
    sperimentale (Rosi 2023: 0,5 mM; TEL26 a 20 o 150 mM KCl, cariche trascurate).  Il modello e'
    quello a una copia (prior termici f, residuo PaiNN), replicato: interazioni interne di ogni copia
    identiche, fra copie diverse solo WCA e PaiNN entro il suo raggio.  I contatti nativi vanno usati
    con run_cg_md.py --pair_specific_morse_runtime bonded-switched (i marcatori costerebbero un tipo
    di particella per estremo e per copia).

PRIOR (--priors -> --out-priors)
    Ogni termine di bonds / angles / dihedrals (chiavi mol_i, mol_j, mol_k, mol_l) e ogni coppia di
    wca_exclusions (direct_pairs, direct_site_pairs, one_three_pairs) si ripete per copia con gli
    indici di molecola spostati di c * nuc; i contatori si aggiornano.  Il resto (tabelle per tipo,
    parametri termici D_H/D_S, politica WCA) resta com'e'.  rigid_bodies_info.json non cambia: e'
    per tipo di residuo.

CONFIGURAZIONE (--samples -> --out-dataset, un frame)
    Si scelgono frame dalle corse a una copia (17 o 19, T<T>.samples.npz), dopo --skip-ps, e si
    classificano per tratto (come 22/23: U se tutti i c_a < --tract-cut).  Con --pu (o --pu-json
    e --T: P_U dal fit a due stati di 22) la scatola contiene round(pU N) copie U e le altre dal
    resto, cosi' la miscela parte gia' all'equilibrio delle popolazioni senza aspettare transizioni.
    Ogni copia: rotazione uniforme attorno al proprio centro, traslazione casuale, rifiutata se un
    sito dista meno di --min-dist da un sito di un'altra copia (immagine minima).  Lato della
    scatola da --conc-mM (L = (N / (c N_A))^(1/3)) o --box.

USO (Leonardo)
    python3 25_build_box.py --n-copies 300 --conc-mM 0.5 --pu-json $V/validation.json --T 330 \\
        --samples $V/chiuso/T330.samples.npz $V/aperto/T330.samples.npz --skip-ps 5000 \\
        --priors $A/thermo_val/f/cg_priors.json --template $A/cg/tel26_lp2_1c_t400_dataset.bin \\
        --out-priors box300/cg_priors.json --out-dataset box300/dataset.bin --seed 1
    -> prior, dataset a un frame e box300/dataset.json (composizione, L, k_min).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import pathlib
import struct

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
N_A = 6.02214076e23
MOLK = ("mol_i", "mol_j", "mol_k", "mol_l")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def replicate_priors(pri, n, nuc):
    out = copy.deepcopy(pri)
    for sec in ("bonds", "angles", "dihedrals"):
        terms = pri.get(sec) or []
        rep = []
        for c in range(n):
            for e in terms:
                f = dict(e)
                for k in MOLK:
                    if k in f:
                        if not (0 <= int(f[k]) < nuc):
                            raise SystemExit(f"[ERROR] {sec}: indice di molecola {f[k]} fuori dalla copia (gia' a piu' copie?)")
                        f[k] = int(f[k]) + c * nuc
                rep.append(f)
        if sec in out:
            out[sec] = rep
    w = out.get("wca_exclusions")
    if w:
        for key, nm in (("direct_pairs", 2), ("direct_site_pairs", 2), ("one_three_pairs", 2)):
            if key in w:
                w[key] = [[int(m) + c * nuc for m in p[:nm]] + list(p[nm:])
                          for c in range(n) for p in pri["wca_exclusions"][key]]
                cnt = key.replace("pairs", "pair_count")
                if cnt in w:
                    w[cnt] = len(w[key])
    if out.get("morse_type_pairs"):
        print("[WARN] morse_type_pairs non vuoto: copiato com'e' (per tipo, agisce anche fra copie)")
    out["multicopy"] = {"copies": n, "residues_per_copy": nuc, "source": "25_build_box.py"}
    return out


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-copies", type=int, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--conc-mM", type=float, help="concentrazione di TEL26 (mM) -> lato della scatola")
    g.add_argument("--box", type=float, help="lato della scatola (nm)")
    ap.add_argument("--samples", nargs="+", required=True, help="corse a una copia da cui prendere le conformazioni")
    ap.add_argument("--skip-ps", type=float, default=5000.0)
    ap.add_argument("--pu", type=float, help="frazione di copie nello stato U")
    ap.add_argument("--pu-json", help="json di 22_anchor_L0.py (fit a due stati) per P_U(T)")
    ap.add_argument("--T", type=float, help="temperatura (K) per --pu-json")
    ap.add_argument("--tract-cut", type=float, default=0.2)
    ap.add_argument("--smooth-ns", type=float, default=1.0)
    ap.add_argument("--min-dist", type=float, default=0.8, help="distanza minima fra siti di copie diverse (nm)")
    ap.add_argument("--priors", required=True, help="cg_priors.json a una copia")
    ap.add_argument("--template", required=True, help="dataset a una copia (tipi dei siti)")
    ap.add_argument("--out-priors", required=True)
    ap.add_argument("--out-dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    an = _load("anchor_L0", "22_anchor_L0.py")
    pl, cv = an.pl, an.pl.cv
    cv.set_nuc(pl.melt.NUC)
    nuc = pl.melt.NUC
    rng = np.random.default_rng(args.seed)
    N = args.n_copies

    if args.pu is not None:
        pU = args.pu
    elif args.pu_json and args.T:
        pj = json.load(open(args.pu_json))
        pU = 1.0 / (1.0 + np.exp((float(pj["dH_kJ"]) / an.R_KJ) * (1.0 / args.T - 1.0 / float(pj["Tm_K"]))))
    else:
        raise SystemExit("[ERROR] serve --pu oppure --pu-json con --T")
    nU = int(round(pU * N))

    # serbatoio di conformazioni: (siti (M, 6, 3) ricomposti, com (M, 3), stato)
    Sn, Ln, ncn = cv.load_reference(str(HERE.parent / "tel26" / "tel26_lp2_dataset.bin"))
    r_nat = np.nanmedian(pl.melt.b3_pair_distances(Sn[::5], Ln[::5], ncn), axis=0)
    pool = {"U": [], "N": []}
    for f in args.samples:
        z = np.load(f)
        S, L, nc, t = cv.load_samples(f)
        if nc != 1:
            raise SystemExit(f"[ERROR] {f}: serve una copia per corsa")
        t = np.asarray(t, float)
        keep = np.flatnonzero(t >= t[0] + args.skip_ps)
        _, c = an.tract_c(f, r_nat, 25.0, 1.2, args.smooth_ns)
        isU = (c[keep] < args.tract_cut).all(1)
        Lb = np.asarray(L, float)
        com = np.asarray(z["com"], float)
        for k, u in zip(keep, isU):
            box = Lb[k] if Lb.ndim == 2 else Lb
            X = S[k]                                            # (M, 6, 3) con NaN
            ref = com[k][:, None, :]
            Xw = ref + (X - ref) - box * np.round((X - ref) / box)          # siti vicini al proprio corpo
            # catena ricomposta: corpi rigidi vicini al precedente
            cm = com[k].copy()
            for m in range(1, cm.shape[0]):
                cm[m] = cm[m - 1] + (cm[m] - cm[m - 1]) - box * np.round((cm[m] - cm[m - 1]) / box)
            Xw = Xw - com[k][:, None, :] + cm[:, None, :]
            pool["U" if u else "N"].append((Xw, cm, f, int(k)))
    print(f"[box] serbatoio: {len(pool['U'])} frame U, {len(pool['N'])} frame non U; P_U {pU:.3f} -> {nU} copie U su {N}")
    for st, need in (("U", nU), ("N", N - nU)):
        if need and not pool[st]:
            raise SystemExit(f"[ERROR] nessun frame nello stato {st} nelle corse date")

    if args.conc_mM:
        Lbox = (N / (args.conc_mM * 1e-3 * N_A * 1e-24)) ** (1.0 / 3.0)       # mol/L -> nm^-3: 1 L = 1e24 nm^3
    else:
        Lbox = args.box
    print(f"[box] {N} copie, lato {Lbox:.2f} nm, k_min = 2 pi / L = {2 * np.pi / Lbox:.4f} nm^-1 "
          f"({0.2 * np.pi / Lbox:.5f} A^-1)")

    types = cv.load_site_types(args.template)              # (M, 6), -1 nei posti vuoti
    if types.shape[0] != nuc:
        raise SystemExit(f"[ERROR] il template ha {types.shape[0]} molecole, attese {nuc}")

    order = ["U"] * nU + ["N"] * (N - nU)
    rng.shuffle(order)
    placed_sites, placed, comp = [], [], []
    for ci, st in enumerate(order):
        X, cm, src, frame = pool[st][rng.integers(len(pool[st]))]
        ok_sites = np.isfinite(X[..., 0])
        c0 = X[ok_sites].mean(axis=0)
        for attempt in range(10000):
            R = random_rotation(rng)
            t = rng.uniform(0, Lbox, 3)
            Y = (X - c0) @ R.T + t
            Ycm = (cm - c0) @ R.T + t
            pts = Y[ok_sites]
            rad = np.sqrt(((pts - t) ** 2).sum(-1).max())
            good = True
            for (P, tc, rc) in placed_sites:
                dc = t - tc
                dc -= Lbox * np.round(dc / Lbox)
                if np.sqrt((dc * dc).sum()) > rad + rc + args.min_dist:      # sfere che non si toccano
                    continue
                d = pts[:, None, :] - P[None, :, :]
                d -= Lbox * np.round(d / Lbox)
                if (d * d).sum(-1).min() < args.min_dist ** 2:
                    good = False
                    break
            if good:
                break
        else:
            raise SystemExit(f"[ERROR] copia {ci}: nessuna posizione libera dopo 10000 tentativi (scatola troppo piena?)")
        placed_sites.append((pts, t, rad))
        placed.append((Y, Ycm))
        comp.append({"copy": ci, "state": st, "source": src, "frame": frame, "attempts": attempt + 1})

    out_ds = pathlib.Path(args.out_dataset)
    out_ds.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ds, "wb") as fh:
        fh.write(struct.pack("i", 1))
        nsites = int(sum((types[m] >= 0).sum() for m in range(nuc)) * N)
        fh.write(struct.pack("ii", N * nuc, nsites))
        fh.write(struct.pack("3f", Lbox, Lbox, Lbox))
        for ci, (Y, Ycm) in enumerate(placed):
            for m in range(nuc):
                ns = int((types[m] >= 0).sum())
                fh.write(struct.pack("ii", ci * nuc + m, ns))
                fh.write(struct.pack("3f", *Ycm[m]))
                fh.write(struct.pack("6f", *([0.0] * 6)))
                for s in range(ns):
                    if not np.isfinite(Y[m, s]).all():
                        raise SystemExit(f"[ERROR] sito mancante nella conformazione (molecola {m}, sito {s})")
                    fh.write(struct.pack("i", int(types[m, s])))
                    fh.write(struct.pack("3f", *Y[m, s]))
    pri = json.load(open(args.priors))
    outp = replicate_priors(pri, N, nuc)
    pathlib.Path(args.out_priors).parent.mkdir(parents=True, exist_ok=True)
    json.dump(outp, open(args.out_priors, "w"), indent=1)
    meta = {"n_copies": N, "box_nm": Lbox, "conc_mM": args.conc_mM, "k_min_nm": 2 * np.pi / Lbox,
            "P_U": pU, "n_U": nU, "min_dist_nm": args.min_dist, "seed": args.seed,
            "priors": args.out_priors, "dataset": str(out_ds), "copies": comp}
    json.dump(meta, open(out_ds.with_suffix(".json"), "w"), indent=1)
    nterms = {sec: len(outp.get(sec) or []) for sec in ("bonds", "angles", "dihedrals")}
    print(f"[box] scritti {out_ds} ({N * nuc} corpi, {nsites} siti), {args.out_priors} {nterms}, "
          f"{out_ds.with_suffix('.json')}")


if __name__ == "__main__":
    main()
