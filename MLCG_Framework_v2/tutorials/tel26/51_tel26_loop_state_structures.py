#!/usr/bin/env python3
"""TEL26: che cosa sono gli stati dei loop?  Strutture rappresentative e descrittori.

PERCHE'
    Lo script 50 trova gli stati dei loop nell'AA e mostra che il CG non entra
    nello stato 1 del loop 3 (26 % nell'AA, con scambi veri nella copia 1) ne'
    nello stato minore del loop 2 (una sola copia), e che la coda 5' nel CG e'
    troppo diffusa.  Qui si guarda la fisica degli stati, per decidere se
    servono termini nei prior (stacking, contatti) o target RE:
      - per ogni stato AA e per il CG, posizione della base di ogni residuo
        del loop rispetto alla pila: quota z lungo l'asse (dal baricentro del
        nucleo; le tetradi 1 e 3 sono ai valori riportati) e distanza
        laterale rho dall'asse.  Una base impilata su una tetrade esterna ha
        |z| ~ |z_tetrade| + 0,35 nm e rho piccolo;
      - i due residui piu' vicini alla base (escluso il residuo stesso e i
        vicini di catena), con la distanza media: contatti e appaiamenti
        caratteristici dello stato;
      - un file PDB per loop con un MODEL per stato AA (il frame piu' vicino al
        centro dello stato, copia e tempo nei REMARK) e un MODEL per il CG (il
        frame piu' vicino alla media CG), tutto nel sistema del nucleo AA medio.
    Gli stati si ricostruiscono come nello script 50 (stessa PCA, k-means con
    k dato o scelto dalla silhouette, stati ordinati per popolazione): le
    popolazioni stampate vanno confrontate con quelle dello script 50.

USO (in tutorials/tel26)
    python3 51_tel26_loop_state_structures.py tel26_lp1_dataset.bin \\
        prod=r0_s01.npz+r0_s02.npz,r1_s01.npz+r1_s02.npz \\
        [--k "coda 5'=5" --k "loop 1=5" --k "loop 2=2" --k "loop 3=2" --k "coda 3'=2"] \\
        [--pdb-prefix loop_states] [--json loop_state_structures.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("s50", HERE / "50_tel26_loop_states.py")
s50 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s50)
s47, s46, cv = s50.s47, s50.s46, s50.cv

SITE_NAMES = ["S", "B1", "B2", "B3", "B4", "B5"]
RESNAME = {"G": "DG", "T": "DT", "A": "DA", "C": "DC"}


def stack_frame(ref):
    """Asse della pila e quote delle tetradi 1 e 3 nel sistema del nucleo AA medio."""
    rows = {rs: i for i, rs in enumerate(ref.sel.core)}
    def centroid(tetrad):
        idx = [rows[(r - 1, s)] for r in tetrad for s in range(6)]
        return ref.core[idx].mean(0)
    c1, c3 = centroid(s46.TETRADS_1B[0]), centroid(s46.TETRADS_1B[2])
    n = (c1 - c3) / np.linalg.norm(c1 - c3)
    return n, float(c1 @ n), float(c3 @ n)


def base_centroids(A, sites):
    """A (N, n_all, 3), sites = lista (r, s) -> (N, nuc, 3): media dei siti s >= 1 di ogni residuo."""
    nuc = max(r for r, _ in sites) + 1
    out = np.full((A.shape[0], nuc, 3), np.nan)
    for r in range(nuc):
        idx = [i for i, (rr, s) in enumerate(sites) if rr == r and s >= 1]
        if not idx:
            idx = [i for i, (rr, s) in enumerate(sites) if rr == r]
        out[:, r] = A[:, idx].mean(1)
    return out


def describe(B, residues, n):
    """B (N, nuc, 3) centroidi di base -> per residuo: z, rho medi e std; due residui piu' vicini."""
    out = {}
    nuc = B.shape[1]
    for r1 in residues:
        r = r1 - 1
        b = B[:, r]
        z = b @ n
        rho = np.linalg.norm(b - z[:, None] * n[None], axis=1)
        d = np.linalg.norm(B - b[:, None], axis=-1).mean(0)        # (nuc,)
        cand = [j for j in range(nuc) if abs(j - r) > 1 and np.isfinite(d[j])]
        near = sorted(cand, key=lambda j: d[j])[:2]
        out[r1] = {"z": (float(z.mean()), float(z.std())), "rho": (float(rho.mean()), float(rho.std())),
                   "near": [(j + 1, float(d[j])) for j in near]}
    return out


def pdb_model(f, model, coords, sites, remarks):
    f.write(f"MODEL     {model:4d}\n")
    for line in remarks:
        f.write(f"REMARK   1 {line}\n")
    for i, ((r, s), x) in enumerate(zip(sites, coords)):
        res = s46.SEQUENCE[r]
        f.write("ATOM  {:5d} {:<4s} {:>3s} A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           C\n".format(
            i + 1, SITE_NAMES[s], RESNAME.get(res, "UNK"), r + 1, *(10.0 * x)))
    f.write("ENDMDL\n")


def fmt_res(r1):
    return f"{r1}{s46.SEQUENCE[r1 - 1]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("runs", nargs="*", help="etichetta=a.npz[+b.npz][,r1.npz...] (si usa la prima)")
    ap.add_argument("--nuc", type=int, default=26)
    ap.add_argument("--aa-dt", type=float, default=20.0)
    ap.add_argument("--aa-stride", type=int, default=1)
    ap.add_argument("--cg-stride", type=int, default=5)
    ap.add_argument("--ref-stride", type=int, default=5)
    ap.add_argument("--k", action="append", default=[], metavar="LOOP=K",
                    help="k per un loop (prefisso del nome, per esempio \"loop 3=2\"); senza, silhouette come lo script 50")
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pdb-prefix", default="loop_states")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cv.set_nuc(args.nuc)
    nuc = args.nuc
    rng = np.random.default_rng(args.seed)
    kfix = {}
    for spec in args.k:
        name, v = spec.rsplit("=", 1)
        kfix[name.strip()] = int(v)

    S, L, nc, t = s47.load_aa(args.dataset, args.aa_dt, args.aa_stride)
    k = max(1, args.ref_stride // args.aa_stride)
    Xr = s46.unwrap_copies(S[::k], L[::k], nc, nuc).reshape(-1, nuc, S.shape[2], 3)
    ref = s46.Reference(Xr, s46.Selection(nuc, Xr[0]))
    sites = ref.sel.all
    dt_aa, A_aa, res = s50.aligned([(S, L, nc, t)], ref, nuc)
    Ta, Ca = A_aa.shape[:2]
    cg_label, A_cg, dt_cg = None, None, None
    if args.runs:
        cg_label, path = args.runs[0].split("=", 1)
        parts = [s47.load_cg(rep, args.cg_stride) for rep in path.split(",")]
        dt_cg, A_cg = s50.aligned(parts, ref, nuc)[:2]
    n, z1, z3 = stack_frame(ref)
    B_aa = base_centroids(A_aa.reshape(Ta * Ca, -1, 3), sites)
    B_cg = base_centroids(A_cg.reshape(-1, A_cg.shape[2], 3), sites) if A_cg is not None else None

    print(f"\n  AA: {Ta} frame x {Ca} copie, dt {dt_aa:g} ps" +
          (f";  {cg_label}: {A_cg.shape[0]} frame x {A_cg.shape[1]} copie, dt {dt_cg:g} ps" if A_cg is not None else ""))
    print(f"  asse della pila: quota della tetrade 1 {z1:+.3f} nm, della tetrade 3 {z3:+.3f} nm (dal baricentro del nucleo)")

    results = []
    for gname, residues in s50.GROUPS:
        msk = np.isin(res, residues)
        xa = A_aa[:, :, msk].reshape(Ta * Ca, -1)
        mu = xa.mean(0)
        _U, sv, Vt = np.linalg.svd(xa - mu, full_matrices=False)
        ev = sv ** 2 / np.sum(sv ** 2)
        m = int(min(8, np.searchsorted(np.cumsum(ev), 0.9) + 1))
        P = Vt[:m].T
        za = (xa - mu) @ P
        kk = next((v for key, v in kfix.items() if gname.startswith(key)), None)
        if kk is None:
            sil = {q: s50.silhouette(za, s50.kmeans(za, q, rng)[1], rng) for q in range(2, args.kmax + 1)}
            kk = max(sil, key=lambda q: sil[q])
        c, lab = s50.kmeans(za, kk, rng)
        order = np.argsort(-np.bincount(lab, minlength=kk))
        remap = np.empty(kk, int); remap[order] = np.arange(kk)
        c, lab = c[order], remap[lab]
        pops = np.bincount(lab, minlength=kk) / lab.size
        r = {"name": gname, "k": kk, "pop_aa": pops.tolist(), "states": []}
        tag = gname.split()[0] + gname.split()[1].replace("'", "")
        pdb_path = pathlib.Path(f"{args.pdb_prefix}_{tag}.pdb")
        print(f"\n  == {gname} ==  k = {kk}, popolazioni AA " + " ".join(f"{p:.3f}" for p in pops) +
              f"   (PDB: {pdb_path})")
        with open(pdb_path, "w") as f:
            for j in range(kk):
                members = np.flatnonzero(lab == j)
                rep = members[np.argmin(((za[members] - c[j]) ** 2).sum(-1))]
                frame, copy = divmod(int(rep), Ca)
                desc = describe(B_aa[members], residues, n)
                r["states"].append({"state": j, "pop": float(pops[j]), "rep_copy": copy,
                                    "rep_time_ps": frame * dt_aa, "residues": desc})
                pdb_model(f, j + 1, A_aa[frame, copy], sites,
                          [f"{gname} stato AA {j} pop {pops[j]:.3f} copia {copy} t {frame * dt_aa / 1000:.2f} ns"])
                print(f"  stato {j} (pop {pops[j]:.3f}; rappresentativo: copia {copy}, t {frame * dt_aa / 1000:.2f} ns)")
                for r1, dsc in desc.items():
                    near = ", ".join(f"{fmt_res(q)} {dq:.2f}" for q, dq in dsc["near"])
                    print(f"     {fmt_res(r1):>4s}: z {dsc['z'][0]:+.2f}±{dsc['z'][1]:.2f}  rho {dsc['rho'][0]:.2f}±{dsc['rho'][1]:.2f} nm"
                          f"   vicini: {near}")
            if A_cg is not None:
                Tc, Cc = A_cg.shape[:2]
                xc = A_cg[:, :, msk].reshape(Tc * Cc, -1)
                zc = (xc - mu) @ P
                lc = np.argmin(((zc[:, None] - c[None]) ** 2).sum(-1), axis=1)
                rep = int(np.argmin(((zc - zc.mean(0)) ** 2).sum(-1)))
                frame, copy = divmod(rep, Cc)
                desc = describe(B_cg, residues, n)
                r["cg"] = {"pop": (np.bincount(lc, minlength=kk) / lc.size).tolist(),
                           "rep_copy": copy, "rep_time_ps": frame * dt_cg, "residues": desc}
                pdb_model(f, kk + 1, A_cg[frame, copy], sites,
                          [f"{gname} {cg_label} (frame piu' vicino alla media CG) copia {copy} t {frame * dt_cg / 1000:.2f} ns"])
                print(f"  {cg_label} (tutta la corsa; rappresentativo: copia {copy}, t {frame * dt_cg / 1000:.2f} ns)")
                for r1, dsc in desc.items():
                    near = ", ".join(f"{fmt_res(q)} {dq:.2f}" for q, dq in dsc["near"])
                    print(f"     {fmt_res(r1):>4s}: z {dsc['z'][0]:+.2f}±{dsc['z'][1]:.2f}  rho {dsc['rho'][0]:.2f}±{dsc['rho'][1]:.2f} nm"
                          f"   vicini: {near}")
        results.append(r)

    print("\n  Lettura:")
    print(f"  - z oltre la tetrade 1 ({z1:+.2f}) o la 3 ({z3:+.2f}) di ~0,35 nm con rho < ~0,5 nm: base impilata su quella tetrade.")
    print("  - vicini: distanza media fra centroidi di base; < ~0,6 nm indica un contatto/appaiamento stabile nello stato.")
    print("  - deviazioni std grandi (> ~0,3 nm) in uno stato AA: lo stato raccoglie conformazioni diverse.")
    print("  - i PDB (siti CG come atomi, in Angstrom) si aprono in VMD/PyMOL: un MODEL per stato AA, l'ultimo e' il CG.")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
