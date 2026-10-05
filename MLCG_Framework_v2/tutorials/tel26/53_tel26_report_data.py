#!/usr/bin/env python3
"""Dati per il report AA contro CG-ML: P(r)/g(r) per canale, Rg e coordinate collettive, MSD.

PERCHE'
    Il report PDF si costruisce fuori da Leonardo; qui si calcola tutto cio' che
    serve dalle traiettorie (pesanti) e si salva in un solo file .npz compatto:
      - P(r) intra e g(r) inter per canale (tutti i siti, B3-B3, B5-B5, S-S),
        con gli stessi istogrammi e la stessa normalizzazione dello script 44;
        per l'AA anche le due meta' della traiettoria (tetto: meta' contro meta');
      - per (frame, copia): Rg, rmsd_core, rmsd_loops, Q (script 46), con il
        tempo di ogni frame, per distribuzioni e serie temporali;
      - curve MSD (traslazione, rotazione) e tempi di rilassamento gia' calcolati
        dagli script 48 e 47 (--msd-json, --ts-json), copiati cosi' come sono;
      - sovrapposizioni (integrale del minimo, script 44) di ogni corsa e della
        seconda meta' AA contro la prima, per canale.

USO (in tutorials/<sistema>, con system.json)
    python3 53_tel26_report_data.py tel26_lp1_dataset.bin \\
        --run "CG-ML=prod_re1_s01_r0.samples.npz,prod_re1_s01_r1.samples.npz,..." \\
        [--run "soli prior=..."] --msd-json msd_wall_prod.json --ts-json ts_prod.json \\
        --out report_data.npz
    Nella lista di una corsa ogni file e' un tratto (replicas e segmenti si sommano).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("s47", HERE / "47_tel26_timescales.py")
s47 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s47)
s46, cv, SYS = s47.s46, s47.cv, s47.s46.SYS
_spec = importlib.util.spec_from_file_location(
    "s44", HERE.parent / "tel22" / "diagnostics" / "scripts" / "44_tel22_rdf.py")
s44 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s44)


def channel_hist(parts, types, rmax, nbins, stride):
    """Somma degli istogrammi per canale su piu' tratti."""
    tot = None
    for S, L, nc in parts:
        acc, nint, nf, vol, edges = s44.accumulate_channels(S, L, nc, types, rmax, nbins, stride)
        if tot is None:
            tot = {"acc": acc, "nint": nint, "nf": nf, "volnf": vol * nf, "edges": edges}
        else:
            for k in acc:
                tot["acc"][k][0] += acc[k][0]
                tot["acc"][k][1] += acc[k][1]
                tot["nint"][k] += nint[k]
            tot["nf"] += nf
            tot["volnf"] += vol * nf
    out = {}
    vol = tot["volnf"] / max(tot["nf"], 1)
    for name, _ in s44.CHANNELS:
        r, p, g = s44.normalize(tot["acc"][name][0], tot["acc"][name][1], tot["nint"][name],
                                tot["nf"], vol, tot["edges"], 0)
        out[name] = (p, g)
    return r, out, tot["nf"]


def collective(parts, ref, nuc, times):
    """Rg, rmsd_core, rmsd_loops, Q per (frame, copia), tratti concatenati come copie in piu'."""
    cols = {k: [] for k in ("Rg", "rmsd_core", "rmsd_loops", "Q")}
    tt = []
    for (S, L, nc), t in zip(parts, times):
        X = s46.unwrap_copies(S, L, nc, nuc).reshape(-1, nuc, S.shape[2], 3)
        cvs = s46.collective(X, ref)
        for k in cols:
            cols[k].append(np.asarray(cvs[k], np.float32).reshape(S.shape[0], nc))
        tt.append(np.asarray(t, np.float32))
    return cols, tt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", help="riferimento AA mappato")
    ap.add_argument("--run", action="append", default=[], metavar="ETICHETTA=a.npz,b.npz,...")
    ap.add_argument("--nuc", type=int, default=SYS.nuc)
    ap.add_argument("--aa-dt", type=float, default=SYS.aa_frame_dt_ps)
    ap.add_argument("--rmax", type=float, default=2.0)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--aa-stride", type=int, default=2, help="frame AA per le P(r)")
    ap.add_argument("--cg-stride", type=int, default=20, help="frame CG per le P(r) e le coordinate")
    ap.add_argument("--skip-ps", type=float, default=0.0, help="ps iniziali scartati in ogni tratto CG")
    ap.add_argument("--msd-json", default=None)
    ap.add_argument("--ts-json", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cv.set_nuc(args.nuc)
    nuc = args.nuc

    S, L, nc = s44.load_reference(args.dataset)
    T = S.shape[0]
    types = s44.load_site_types(args.dataset)
    print(f"[INFO] AA: {T} frame x {nc} copie ({SYS.name})")
    data = {"system": SYS.name, "aa_frames": T, "aa_dt_ps": args.aa_dt, "copies": nc}

    # P(r) e g(r) per canale: AA intera, prima e seconda meta'
    r, aa, nf = channel_hist([(S, L, nc)], types, args.rmax, args.bins, args.aa_stride)
    _, aa1, _ = channel_hist([(S[:T // 2], L[:T // 2] if np.ndim(L) == 2 else L, nc)], types, args.rmax, args.bins, args.aa_stride)
    _, aa2, _ = channel_hist([(S[T // 2:], L[T // 2:] if np.ndim(L) == 2 else L, nc)], types, args.rmax, args.bins, args.aa_stride)
    curves = {"AA": aa, "AA meta' 1": aa1, "AA meta' 2": aa2}
    print(f"[INFO] P(r) AA: {nf} frame")

    # coordinate collettive AA (riferimento della struttura media: prima meta', come lo script 46)
    k = 5
    Xr = s46.unwrap_copies(S[:T // 2:k], L[:T // 2:k] if np.ndim(L) == 2 else L, nc, nuc).reshape(-1, nuc, S.shape[2], 3)
    ref = s46.Reference(Xr, s46.Selection(nuc, Xr[0]))
    cols_aa, _ = collective([(S, L, nc)], ref, nuc, [np.arange(T) * args.aa_dt])
    for kname, v in cols_aa.items():
        data[f"cv_AA_{kname}"] = v[0]
    data["t_AA_ps"] = (np.arange(T) * args.aa_dt).astype(np.float32)

    runs_meta = []
    for spec in args.run:
        label, files = spec.split("=", 1)
        parts, times = [], []
        for f in files.split(","):
            Sc, Lc, ncc, tc = s44.load_samples(f)
            tc = np.asarray(tc, float)
            keep = np.flatnonzero(tc - tc[0] >= args.skip_ps)[::args.cg_stride]
            Lc = np.asarray(Lc)
            parts.append((Sc[keep], Lc[keep] if Lc.ndim == 2 and Lc.shape[0] == Sc.shape[0] else Lc, ncc))
            times.append(tc[keep])
        rr, cg, nfc = channel_hist(parts, types, args.rmax, args.bins, 1)
        curves[label] = cg
        cols, tt = collective(parts, ref, nuc, times)
        for kname, v in cols.items():
            data[f"cv_{label}_{kname}"] = np.concatenate([x.ravel() for x in v])
        data[f"cv_{label}_Rg_trace0"] = cols["Rg"][0]          # primo tratto: serie temporale per copia
        data[f"t_{label}_trace0_ps"] = tt[0]
        runs_meta.append({"label": label, "files": files.split(","), "frames_used": int(nfc)})
        print(f"[INFO] {label}: {len(parts)} tratti, {nfc} frame per le P(r)")

    # sovrapposizioni per canale
    overlaps = {}
    for name, _ in s44.CHANNELS:
        overlaps[name] = {}
        for lab, cur in curves.items():
            if lab == "AA":
                continue
            refc = aa1 if lab == "AA meta' 2" else aa
            overlaps[name][lab] = {
                "intra": s44.overlap_metrics(refc[name][0], cur[name][0], r)["integral_overlap"],
                "inter": s44.overlap_metrics(refc[name][1], cur[name][1], r)["integral_overlap"]}
    data["r_nm"] = r
    for lab, cur in curves.items():
        for name, _ in s44.CHANNELS:
            data[f"P_{lab}_{name}"] = cur[name][0]
            data[f"g_{lab}_{name}"] = cur[name][1]
    meta = {"system": SYS.name, "channels": [n for n, _ in s44.CHANNELS], "curves": list(curves),
            "runs": runs_meta, "overlaps": overlaps, "aa_frames": T, "aa_dt_ps": args.aa_dt,
            "cg_stride": args.cg_stride, "aa_stride": args.aa_stride, "skip_ps": args.skip_ps}
    if args.msd_json and pathlib.Path(args.msd_json).is_file():
        meta["msd"] = json.loads(pathlib.Path(args.msd_json).read_text())
    if args.ts_json and pathlib.Path(args.ts_json).is_file():
        meta["timescales"] = json.loads(pathlib.Path(args.ts_json).read_text())
    data["meta_json"] = json.dumps(meta, default=float)
    np.savez_compressed(args.out, **{k: v for k, v in data.items()})
    print(f"[DONE] {args.out}")
    for name in overlaps:
        print(f"  {name:<20s} " + "  ".join(f"{lab}: intra {v['intra']:.4f}, inter {v['inter']:.4f}"
                                          for lab, v in overlaps[name].items()))


if __name__ == "__main__":
    main()
