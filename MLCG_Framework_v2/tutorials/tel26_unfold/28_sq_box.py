#!/usr/bin/env python3
"""S(q) totale di una scatola con molti TEL26 (27_bulk_box.sh), con le sue parti.

    S(k)      = (1/N_s) < |sum_j exp(-i k . r_j)|^2 >     tutti i siti di tutte le copie
    S_intra   = (1/N_s) < sum_c |sum_{j in c} exp(-i k . r_j)|^2 >   (Debye per copia, media sulle direzioni)
    S_inter   = S - S_intra                                (correlazioni fra copie diverse)
    S_cm(k)   = (1/N_c) < |sum_c exp(-i k . R_c)|^2 >      centri delle copie (media dei siti)

k sul reticolo reciproco della scatola cubica, k = 2 pi n / L, raggruppati in gusci di larghezza 2 pi / L
(al piu' --vectors per guscio, scelti a caso una volta; k e -k equivalenti).  S(k -> 0) = n_s S_cm(0)
(n_s = 86 siti per copia), S(k -> inf) = 1.

k -> 0: oltre al primo guscio (k_min = 2 pi / L), fluttuazioni del numero di copie in sottocubi di lato
l = L/m (m = 2..--max-m, origine casuale per frame):  chi(l) = <dN^2> / <N>.  In una scatola chiusa
chi(l) = S_cm(0) (1 - (l/L)^3) + O(1/l) (superficie): si riporta chi grezzo, chi / (1 - (l/L)^3) e il
fit lineare in 1/l di quest'ultimo per l <= L/4, la cui intercetta stima S_cm(0) = rho k_B T kappa_T.
Verificato su un gas ideale di copie rigide (S_cm = 1, S_inter = 0 entro gli errori; S_intra sul
reticolo uguale a Debye entro 0,3 %).

Stato delle copie per frame (opzionale, --states): frazione U (tutti i c_a < 0.2, istantaneo) nel tempo,
per controllare che la miscela F/U iniziale resti stabile.

USO (Leonardo; GPU consigliata per rho(k): --device cuda)
    python3 28_sq_box.py --dirs $A/bulk/N300_c0.5_T322_s1 $A/bulk/N300_c0.5_T322_s2 --skip-ns 4 \\
        --kmax 6 --out $A/bulk/sq_T322
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sq1 = _load("sq_single", "23_sq.py")          # Hist, whole_copies, pl, cv
pl, cv = sq1.pl, sq1.cv


def lattice_shells(L, kmax, per_shell, rng):
    """Vettori n (meta' spazio) raggruppati per guscio |n| in [s - 1/2, s + 1/2)."""
    nmax = int(np.ceil(kmax * L / (2 * np.pi)))
    r = np.arange(-nmax, nmax + 1)
    n = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    half = (n[:, 0] > 0) | ((n[:, 0] == 0) & (n[:, 1] > 0)) | ((n[:, 0] == 0) & (n[:, 1] == 0) & (n[:, 2] > 0))
    n = n[half]
    nn = np.linalg.norm(n, axis=1)
    n, nn = n[nn * 2 * np.pi / L <= kmax], nn[nn * 2 * np.pi / L <= kmax]
    shell = np.rint(nn).astype(int)
    vecs, sid, kval = [], [], []
    for s in np.unique(shell):
        idx = np.flatnonzero(shell == s)
        if idx.size > per_shell:
            idx = rng.choice(idx, per_shell, replace=False)
        vecs.append(n[idx]); sid.append(np.full(idx.size, len(kval)))
        kval.append(2 * np.pi / L * nn[idx].mean())
    return np.concatenate(vecs) * (2 * np.pi / L), np.concatenate(sid), np.array(kval)


def rho2(X, K, backend, block=4096):
    """|sum_j exp(-i k.r_j)|^2 per ogni vettore di K; X (n, 3), K (m, 3)."""
    if backend is not None:
        torch, dev = backend
        Xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
        Kt = torch.as_tensor(K, dtype=torch.float64, device=dev)
        re = torch.zeros(K.shape[0], dtype=torch.float64, device=dev)
        im = torch.zeros_like(re)
        for a in range(0, X.shape[0], block):
            ph = Xt[a:a + block] @ Kt.T
            re += torch.cos(ph).sum(0); im -= torch.sin(ph).sum(0)
        return (re * re + im * im).cpu().numpy()
    re = np.zeros(K.shape[0]); im = np.zeros(K.shape[0])
    for a in range(0, X.shape[0], block):
        ph = X[a:a + block] @ K.T
        re += np.cos(ph).sum(0); im -= np.sin(ph).sum(0)
    return re * re + im * im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="cartelle di 27_bulk_box.sh (chunk*.samples.npz)")
    ap.add_argument("--skip-ns", type=float, default=4.0, help="equilibrazione scartata per cartella")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--kmax", type=float, default=6.0, help="nm^-1 (S_intra per copia fino a --kmax-intra)")
    ap.add_argument("--kmax-intra", type=float, default=300.0)
    ap.add_argument("--vectors", type=int, default=48, help="vettori k per guscio")
    ap.add_argument("--max-m", type=int, default=10, help="sottocubi di lato L/m, m = 2..max-m")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--states", action="store_true", help="frazione U per frame (tratti G)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (torch) per rho(k)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cv.set_nuc(pl.melt.NUC)
    nuc = pl.melt.NUC
    rng = np.random.default_rng(args.seed)
    backend = None
    if args.device != "cpu":
        import torch
        backend = (torch, torch.device(args.device))

    if args.states:
        Sn, Ln, ncn = cv.load_reference(str(HERE.parent / "tel26" / "tel26_lp2_dataset.bin"))
        r_nat = np.nanmedian(pl.melt.b3_pair_distances(Sn[::5], Ln[::5], ncn), axis=0)

    frames = []                      # (cartella, file, indice)
    Lbox = None
    for d in args.dirs:
        files = sorted(glob.glob(str(pathlib.Path(d) / "chunk*.samples.npz")))
        files = [f for f in files if ".tmp." not in f]
        if not files:
            raise SystemExit(f"[ERROR] nessun chunk*.samples.npz in {d}")
        t_off = 0.0
        for f in files:
            z = np.load(f)
            t = np.asarray(z["time_ps"], float) + t_off
            box = np.asarray(z["box"], float).reshape(-1)[:3]
            if Lbox is None:
                Lbox = float(box[0])
            if not np.allclose(box, Lbox):
                raise SystemExit("[ERROR] scatole diverse o non cubiche fra le cartelle")
            for i in np.flatnonzero(t >= args.skip_ns * 1000.0)[::args.stride]:
                frames.append((d, f, int(i)))
            t_off = t[-1] + (t[1] - t[0] if t.size > 1 else 0.0)
    if not frames:
        raise SystemExit("[ERROR] nessun frame dopo --skip-ns")
    K, sid, kshell = lattice_shells(Lbox, args.kmax, args.vectors, rng)
    nsh = kshell.size
    kin = np.geomspace(2 * np.pi / Lbox, args.kmax_intra, 300)
    print(f"[sqbox] L {Lbox:.2f} nm, {len(frames)} frame da {len(args.dirs)} cartelle, {nsh} gusci fino a {args.kmax} nm^-1 "
          f"({K.shape[0]} vettori), k_min {kshell[0]:.4f} nm^-1")

    nb = args.blocks
    blk_of = np.minimum((np.arange(len(frames)) * nb) // len(frames), nb - 1)
    S_tot = np.zeros((nb, nsh)); S_cm = np.zeros((nb, nsh)); S_in = np.zeros((nb, nsh)); cnt = np.zeros(nb)
    ms = np.arange(2, args.max_m + 1)
    chi_num = np.zeros((nb, ms.size)); chi_den = np.zeros((nb, ms.size)); chi_n = np.zeros((nb, ms.size))
    hist_in = sq1.Hist(0.0005, Lbox)
    hb = [hist_in.new_block() for _ in range(nb)]
    fracU, Ns, Nc = [], None, None
    cache = {}
    for fi, (d, f, i) in enumerate(frames):
        if f not in cache:
            cache.clear()
            S, L, nc, t = cv.load_samples(f)
            cache[f] = (S, L, nc)
        S, L, nc = cache[f]
        b = blk_of[fi]
        copies = sq1.whole_copies(S[i:i + 1], L, nc, nuc)           # lista di (1, n_s, 3)
        Xc = np.stack([c[0] for c in copies])                       # (N_c, n_s, 3), ricomposte
        Nc, ns = Xc.shape[0], Xc.shape[1]
        Ns = Nc * ns
        X = Xc.reshape(-1, 3)
        Rc = Xc.mean(axis=1)
        pk = rho2(X, K, backend) / Ns
        pc = rho2(Rc, K, backend) / Nc
        # parte intramolecolare sugli stessi vettori (stessa media angolare numerica del totale)
        pin = np.zeros(K.shape[0])
        for a in range(0, Nc, 64):
            for c in range(a, min(Nc, a + 64)):
                pin += rho2(Xc[c], K, backend)
        pin /= Ns
        for arr, val in ((S_tot, pk), (S_cm, pc), (S_in, pin)):
            arr[b] += np.bincount(sid, weights=val, minlength=nsh) / np.bincount(sid, minlength=nsh)
        cnt[b] += 1
        hist_in.add(Xc, hb[b])                                      # Debye per copia fino a k grandi
        # fluttuazioni del numero di copie in sottocubi
        Rw = (Rc + rng.uniform(0, Lbox, 3)) % Lbox
        for mi, m in enumerate(ms):
            cell = np.floor(Rw / (Lbox / m)).astype(int).clip(0, m - 1)
            counts = np.bincount((cell[:, 0] * m + cell[:, 1]) * m + cell[:, 2], minlength=m ** 3)
            chi_num[b, mi] += counts.var(); chi_den[b, mi] += counts.mean(); chi_n[b, mi] += 1
        if args.states:
            dd = pl.melt.b3_pair_distances(S[i:i + 1], L, nc)
            sp = 1.0 / (1.0 + np.exp(25.0 * (dd - 1.2 * r_nat[None])))
            c_a = np.stack([sp[:, :, pl.TRACT_MASK[a]].mean(axis=(1, 2)) for a in range(4)], 1)
            fracU.append((frames[fi][0], float((c_a < 0.2).all(1).mean())))
        if (fi + 1) % 100 == 0:
            print(f"[sqbox] {fi + 1}/{len(frames)} frame")

    good = cnt > 0
    def mean_err(arr):
        per = arr[good] / cnt[good][:, None]
        w = cnt[good]
        m = (per * w[:, None]).sum(0) / w.sum()
        e = np.sqrt(((per - m) ** 2 * w[:, None]).sum(0) / w.sum() / max(1, good.sum() - 1))
        return m, e
    St, eSt = mean_err(S_tot); Sc, eSc = mean_err(S_cm); Si, eSi = mean_err(S_in)
    Sinter = St - Si
    chi = (chi_num / np.maximum(chi_den, 1e-12))[good].mean(0)
    ell = Lbox / ms
    chi_corr = chi / (1.0 - (ell / Lbox) ** 3)
    sel = ell <= Lbox / 4 + 1e-9                     # sottocubi piccoli: la correzione (l/L)^3 e' piccola
    coef = np.polyfit(1.0 / ell[sel], chi_corr[sel], 1) if sel.sum() >= 2 else [np.nan, np.nan]
    S_debye = hist_in.sk(kin)
    print(f"\n[sqbox] N_c {Nc} copie, N_s {Ns} siti ({Ns // Nc} per copia); S(k -> 0) = n_s S_cm(0)")
    print(f"{'k (nm^-1)':>10s} {'S':>9s} {'err':>8s} {'S_intra':>9s} {'S_inter':>9s} {'S_cm':>7s} {'err':>6s}")
    for j in range(min(nsh, 12)):
        print(f"{kshell[j]:10.4f} {St[j]:9.3f} {eSt[j]:8.3f} {Si[j]:9.3f} {Sinter[j]:9.3f} {Sc[j]:7.3f} {eSc[j]:6.3f}")
    print(f"\n[sqbox] fluttuazioni del numero di copie (chi = <dN^2>/<N>):")
    for m, l, c0, c1 in zip(ms, ell, chi, chi_corr):
        print(f"  l = L/{m:<2d} = {l:6.2f} nm  chi {c0:.4f}  chi/(1-(l/L)^3) {c1:.4f}")
    print(f"[sqbox] estrapolazione in 1/l (l <= L/4): S_cm(0) = {coef[1]:.4f}  (pendenza {coef[0]:.3f} nm)")
    if fracU:
        fu = np.array([x for _, x in fracU])
        print(f"[sqbox] frazione U per frame: media {fu.mean():.3f}, primo/ultimo quinto {fu[:len(fu) // 5].mean():.3f} / "
              f"{fu[-(len(fu) // 5):].mean():.3f}")
    np.savez(args.out + ".npz", k=kshell, S=St, S_err=eSt, S_intra=Si, S_intra_err=eSi, S_inter=Sinter,
             S_cm=Sc, S_cm_err=eSc, k_intra=kin, S_intra_debye=S_debye, ell=ell, chi=chi, chi_corr=chi_corr,
             fracU=np.array([x for _, x in fracU]) if fracU else np.array([]))
    json.dump({"L_nm": Lbox, "frames": len(frames), "dirs": args.dirs, "N_copies": Nc, "N_sites": Ns,
               "S_cm0_fluct": float(coef[1]), "fluct_slope_nm": float(coef[0]), "kmax": args.kmax,
               "vectors_per_shell": args.vectors, "skip_ns": args.skip_ns,
               "fracU_mean": float(np.mean([x for _, x in fracU])) if fracU else None},
              open(args.out + ".json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        ax[0].errorbar(kshell, St, eSt, fmt="o", ms=3, label="S totale (reticolo)")
        ax[0].loglog(kin, S_debye, "-", color="0.4", label="S_intra (Debye, per copia)")
        ax[0].loglog(kshell, Si, "x", ms=3, color="0.4", label="S_intra (reticolo)")
        ax[0].axhline(Ns // Nc, color="0.7", ls=":"); ax[0].axhline(1.0, color="0.7", ls=":")
        ax[0].set(xscale="log", yscale="log", xlabel="k (nm$^{-1}$)", ylabel="S(k)", title=f"S(k), {Nc} copie, L {Lbox:.0f} nm")
        ax[0].legend(fontsize=8)
        ax[1].errorbar(kshell, Sc, eSc, fmt="o", ms=3)
        ax[1].axhline(1.0, color="0.7", ls=":")
        ax[1].axhline(coef[1], color="C3", ls="--", label=f"fluttuazioni: S_cm(0) = {coef[1]:.3f}")
        ax[1].set(xscale="log", xlabel="k (nm$^{-1}$)", ylabel="S_cm(k)", title="centri delle copie")
        ax[1].legend(fontsize=8)
        ax[2].plot(1.0 / ell, chi, "o-", label="chi")
        ax[2].plot(1.0 / ell, chi_corr, "s-", label="chi / (1-(l/L)$^3$)")
        ax[2].set(xlabel="1/l (nm$^{-1}$)", ylabel="<dN$^2$>/<N>", title="fluttuazioni nei sottocubi")
        ax[2].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150)
    except Exception as exc:  # pragma: no cover
        print(f"[sqbox] figura non scritta: {exc}")
    print(f"[sqbox] scritti {args.out}.npz/.json/.png")


if __name__ == "__main__":
    main()
