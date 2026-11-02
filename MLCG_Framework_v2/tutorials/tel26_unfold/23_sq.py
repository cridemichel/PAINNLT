#!/usr/bin/env python3
"""Fattore di struttura del TEL26 CG prima, durante e dopo la fusione.

    S(k) = (1/N) < rho(k) rho*(k) >,   rho(k) = sum_i exp(-i k . r_i),

somma sui siti CG di una copia (G: S, B1..B5; A, T: S; N = 86 per il TEL26), media sui frame e,
per una molecola isolata, sulle direzioni di k.  La media sulle direzioni e' esatta con Debye:
    S(k) = 1 + (2/N) sum_{i<j} sin(k r_ij) / (k r_ij),
calcolata dall'istogramma delle distanze intramolecolari (bin --dr, errore relativo ~ (k dr)^2/24).
Con --check-vectors si confronta su un sottoinsieme di frame con rho(k) esplicito su direzioni
di Fibonacci (stessa quantita', media numerica sulle direzioni).  S(k -> 0) = N; Guinier:
S(k) ~ N (1 - k^2 Rg^2 / 3) con Rg dei siti a peso uguale.  Siti a peso unitario: e' la struttura
del modello, non un'intensita' SAXS (niente fattori di forma, solvente, ioni).

STATI (come 22_anchor_L0.py): c_a = frazione liscia delle 9 coppie B3-B3 native del tratto G a con
gli altri, media mobile su --smooth-ns.  U: tutti i c_a < --tract-cut (0.2); F: tutti i c_a >
--f-cut (0.5); I: il resto.  S(k) si accumula per T e per stato, sommando i rami (chiuso/aperto).

MEDIA DI EQUILIBRIO ALLE T (--pu-json, opzionale): con i rami bloccati dall'isteresi la media
diretta dei frame dipende da dove si parte.  Con il P_U(T) misurato (22_anchor_L0.py, Tm e dH del
fit a due stati) si scrive anche
    S_eq(k, T) = P_U(T) S_U(k, T) + (1 - P_U(T)) S_nonU(k, T).

USO (Leonardo)
    python3 23_sq.py --dirs $A/melt_painn/pop6_it21/chiuso $A/melt_painn/pop6_it21/aperto \\
        --skip-ps 5000 --pu-json $A/melt_painn/pop6_it21/validation.json \\
        --reference $T26/tel26_lp2_dataset.bin --out $A/melt_painn/pop6_it21/sq
    -> sq.npz (k, S per T e stato, errori a blocchi), sq.json (Rg, frazioni, N), sq.png.
    k in nm^-1 (1 nm^-1 = 0.1 A^-1).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import pathlib
import re

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


an = _load("anchor_L0", "22_anchor_L0.py")
pl = an.pl
cv = pl.cv
R_KJ = an.R_KJ


def mi(d, L):
    return d - L * np.round(d / L)


def whole_copies(S, L, ncopy, nuc):
    """S (T, M, 6, 3) con NaN -> lista per copia di (T, n_siti, 3) ricomposti lungo la catena."""
    T = S.shape[0]
    X = S.reshape(T, ncopy, nuc, S.shape[2], 3)
    Lb = np.asarray(L, float)
    Lb = Lb.reshape(T, 1, 1, 3) if Lb.ndim == 2 else Lb.reshape(1, 1, 1, 3)
    anc = X[:, :, :, 0, :]
    steps = mi(np.diff(anc, axis=2), Lb)
    anc_u = np.concatenate([anc[:, :, :1], anc[:, :, :1] + np.cumsum(steps, axis=2)], axis=2)
    Xu = anc_u[:, :, :, None, :] + mi(X - anc[:, :, :, None, :], Lb[..., None, :])
    ok = np.isfinite(X[0, 0, :, :, 0])                 # stessi siti per ogni frame e copia
    return [Xu[:, c][:, ok] for c in range(ncopy)]     # (T, N, 3)


class Hist:
    """Istogramma delle distanze intramolecolari, un blocco per segmento di corsa (errore a blocchi)."""

    def __init__(self, dr, rmax):
        self.edges = np.arange(0.0, rmax + dr, dr)
        self.r = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.h, self.n, self.rg2 = [], [], []
        self.N = None

    def new_block(self):
        self.h.append(np.zeros(self.r.size)); self.n.append(0); self.rg2.append(0.0)
        return len(self.h) - 1

    def add(self, X, block):
        """X (F, N, 3) ricomposto."""
        if X.shape[0] == 0:
            return
        F, N, _ = X.shape
        if self.N is None:
            self.N = N
        elif self.N != N:
            raise SystemExit(f"[ERROR] numero di siti diverso ({N} contro {self.N})")
        iu, ju = np.triu_indices(N, 1)
        for a in range(0, F, 256):
            Y = X[a:a + 256]
            d = np.linalg.norm(Y[:, iu] - Y[:, ju], axis=-1).ravel()
            if d.max() >= self.edges[-1]:
                raise SystemExit(f"[ERROR] distanza {d.max():.2f} nm oltre --rmax")
            self.h[block] += np.histogram(d, self.edges)[0]
            c = Y.mean(axis=1, keepdims=True)
            self.rg2[block] += ((Y - c) ** 2).sum(-1).mean(-1).sum()
        self.n[block] += F

    def merge(self, other):
        for h, n, r in zip(other.h, other.n, other.rg2):
            if n:
                b = self.new_block(); self.h[b] += h; self.n[b] += n; self.rg2[b] += r
        self.N = self.N or other.N

    def frames(self):
        return int(sum(self.n))

    def sk(self, k, blocks=None):
        idx = range(len(self.h)) if blocks is None else blocks
        h = sum(self.h[b] for b in idx)
        n = sum(self.n[b] for b in idx)
        sinc = np.sinc(np.outer(k, self.r) / np.pi)                 # sin(kr)/(kr)
        return 1.0 + (2.0 / self.N) * (sinc @ h) / n

    def sk_err(self, k):
        good = [b for b in range(len(self.h)) if self.n[b] > 0]
        s = self.sk(k)
        if len(good) < 2:
            return s, np.full(k.size, np.nan)
        per = np.array([self.sk(k, [b]) for b in good])
        w = np.array([self.n[b] for b in good], float)
        var = (w[:, None] * (per - s) ** 2).sum(0) / w.sum() / (len(good) - 1)
        return s, np.sqrt(var)

    def rg(self):
        return float(np.sqrt(sum(self.rg2) / sum(self.n)))


def fib_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def sk_vectors(X, k, ndir):
    """(1/N) <|sum_i exp(-i k.r_i)|^2> su ndir direzioni per |k|; X (F, N, 3)."""
    D = fib_dirs(ndir)
    out = np.zeros(k.size)
    for f in range(X.shape[0]):
        proj = X[f] @ D.T                                           # (N, ndir)
        ph = np.exp(-1j * k[:, None, None] * proj[None])            # (nk, N, ndir)
        rho = ph.sum(axis=1)
        out += (np.abs(rho) ** 2).mean(axis=1)
    return out / (X.shape[0] * X.shape[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="cartelle (rami) con T<T>[_r<k>].samples.npz")
    ap.add_argument("--native", default=str(HERE.parent / "tel26" / "tel26_lp2_dataset.bin"))
    ap.add_argument("--reference", help="dataset binario AA mappato sui siti CG (es. tel26_lp2_dataset.bin): S(k) di riferimento")
    ap.add_argument("--reference-stride", type=int, default=5)
    ap.add_argument("--skip-ps", type=float, default=5000.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=1e9)
    ap.add_argument("--smooth-ns", type=float, default=1.0)
    ap.add_argument("--tract-cut", type=float, default=0.2)
    ap.add_argument("--f-cut", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=25.0)
    ap.add_argument("--lam", type=float, default=1.2)
    ap.add_argument("--kmin", type=float, default=0.1, help="nm^-1")
    ap.add_argument("--kmax", type=float, default=20.0, help="nm^-1")
    ap.add_argument("--nk", type=int, default=200)
    ap.add_argument("--dr", type=float, default=0.002, help="larghezza dei bin delle distanze (nm)")
    ap.add_argument("--rmax", type=float, default=30.0)
    ap.add_argument("--blocks", type=int, default=5, help="blocchi per corsa per l'errore")
    ap.add_argument("--pu-json", help="json di 22_anchor_L0.py: P_U(T) dal fit per la media di equilibrio")
    ap.add_argument("--check-vectors", type=int, default=0, help="frame per il controllo con rho(k) esplicito (0 = no)")
    ap.add_argument("--ndir", type=int, default=300)
    ap.add_argument("--out", required=True, help="prefisso delle uscite")
    args = ap.parse_args()
    cv.set_nuc(pl.melt.NUC)
    nuc = pl.melt.NUC

    k = np.geomspace(args.kmin, args.kmax, args.nk)
    Sn, Ln, ncn = cv.load_reference(args.native)
    r_nat = np.nanmedian(pl.melt.b3_pair_distances(Sn[::5], Ln[::5], ncn), axis=0)

    H = {}            # (T, stato) -> Hist ; stato in all/F/I/U
    frac = {}
    check = []
    for dd in args.dirs:
        for f in sorted(glob.glob(str(pathlib.Path(dd) / "T*.samples.npz"))):
            m = re.match(r"T(\d+(?:\.\d+)?)(?:_r\d+)?\.samples\.npz$", pathlib.Path(f).name)
            if not m:
                continue
            T = float(m.group(1))
            if not (args.tmin <= T <= args.tmax):
                continue
            S, L, nc, t = cv.load_samples(f)
            t = np.asarray(t, float)
            if nc != 1:
                raise SystemExit(f"[ERROR] {f}: {nc} copie, serve una copia per corsa")
            keep = np.flatnonzero(t >= t[0] + args.skip_ps)[::args.stride]
            if keep.size == 0:
                continue
            _, c = an.tract_c(f, r_nat, args.beta, args.lam, args.smooth_ns)
            c = c[keep]
            lab = np.where((c < args.tract_cut).all(1), "U", np.where((c > args.f_cut).all(1), "F", "I"))
            X = whole_copies(S[keep], L, 1, nuc)[0]
            blk = np.minimum((np.arange(keep.size) * args.blocks) // keep.size, args.blocks - 1)
            for st in ("all", "F", "I", "U"):
                h = H.setdefault((T, st), Hist(args.dr, args.rmax))
                for b in range(args.blocks):        # ogni corsa porta blocchi nuovi
                    sel = (blk == b) if st == "all" else ((blk == b) & (lab == st))
                    h.add(X[sel], h.new_block())
            for st in ("F", "I", "U"):
                frac.setdefault((T, pathlib.Path(dd).name), {})[st] = float((lab == st).mean())
            if args.check_vectors and len(check) < 3:
                sub = X[np.linspace(0, X.shape[0] - 1, args.check_vectors).astype(int)]
                hh = Hist(args.dr, args.rmax)
                hh.add(sub, hh.new_block())
                kc = k[::10]
                check.append((f, kc, hh.sk(kc), sk_vectors(sub, kc, args.ndir)))
            print(f"[sq] {f}: {keep.size} frame, F {np.mean(lab == 'F'):.2f} I {np.mean(lab == 'I'):.2f} "
                  f"U {np.mean(lab == 'U'):.2f}")

    if not H:
        raise SystemExit("[ERROR] nessuna corsa trovata")
    Ts = sorted({T for T, _ in H})
    N = next(h.N for h in H.values() if h.N)
    out = {"k_nm": k, "N": N, "T": np.array(Ts)}
    summary = {"N_sites": N, "k_units": "nm^-1", "skip_ps": args.skip_ps, "smooth_ns": args.smooth_ns,
               "tract_cut": args.tract_cut, "f_cut": args.f_cut, "per_T": []}
    print(f"\n[sq] N = {N} siti; S(k->0) = N; k in nm^-1")
    print(f"{'T':>6s} {'frame':>7s} {'F':>5s} {'I':>5s} {'U':>5s} {'Rg':>6s} {'Rg_F':>6s} {'Rg_U':>6s}")
    for T in Ts:
        row = {"T_K": T}
        for st in ("all", "F", "I", "U"):
            h = H[(T, st)]
            if h.frames() == 0:
                continue
            s, e = h.sk_err(k)
            out[f"S_{st}_T{T:g}"] = s
            out[f"err_{st}_T{T:g}"] = e
            row[f"frames_{st}"] = h.frames()
            row[f"Rg_{st}_nm"] = h.rg()
        tot = H[(T, "all")].frames()
        fr = {st: H[(T, st)].frames() / tot for st in ("F", "I", "U")}
        row["frac"] = fr
        row["frac_by_branch"] = {b: v for (TT, b), v in frac.items() if TT == T}
        summary["per_T"].append(row)
        print(f"{T:6.1f} {tot:7d} {fr['F']:5.2f} {fr['I']:5.2f} {fr['U']:5.2f} {row['Rg_all_nm']:6.3f} "
              f"{row.get('Rg_F_nm', float('nan')):6.3f} {row.get('Rg_U_nm', float('nan')):6.3f}")

    if args.pu_json:
        pj = json.load(open(args.pu_json))
        Tm, dH = float(pj["Tm_K"]), float(pj["dH_kJ"])
        summary["pu_fit"] = {"Tm_K": Tm, "dH_kJ": dH, "source": args.pu_json}
        print(f"\n[sq] media di equilibrio con P_U del fit a due stati (Tm {Tm:.2f} K, dH {dH:.1f} kJ/mol)")
        for T in Ts:
            pU = 1.0 / (1.0 + np.exp((dH / R_KJ) * (1.0 / T - 1.0 / Tm)))
            hU = H[(T, "U")]
            hN = Hist(args.dr, args.rmax)
            hN.merge(H[(T, "F")]); hN.merge(H[(T, "I")])
            if hU.frames() == 0 or hN.frames() == 0:
                print(f"  T {T:g}: manca uno dei due stati, niente media di equilibrio")
                continue
            out[f"S_eq_T{T:g}"] = pU * hU.sk(k) + (1 - pU) * hN.sk(k)
            rg_eq = np.sqrt(pU * hU.rg() ** 2 + (1 - pU) * hN.rg() ** 2)
            next(r for r in summary["per_T"] if r["T_K"] == T).update({"P_U_fit": pU, "Rg_eq_nm": float(rg_eq)})
            print(f"  T {T:g}: P_U {pU:.3f}, Rg_eq {rg_eq:.3f} nm")

    if args.reference:
        Sr, Lr, ncr = cv.load_reference(args.reference)
        Sr = Sr[::args.reference_stride]
        hr = Hist(args.dr, args.rmax)
        for X in whole_copies(Sr, Lr, ncr, nuc):
            blk = np.minimum((np.arange(X.shape[0]) * args.blocks) // X.shape[0], args.blocks - 1)
            for b in range(args.blocks):
                hr.add(X[blk == b], hr.new_block())
        s, e = hr.sk_err(k)
        out["S_ref"], out["err_ref"] = s, e
        summary["reference"] = {"path": args.reference, "frames": hr.frames(), "copies": ncr, "Rg_nm": hr.rg()}
        print(f"\n[sq] riferimento {args.reference}: {hr.frames()} frame x copia, Rg {hr.rg():.3f} nm")

    for f, kc, sd, svec in check:
        rel = np.max(np.abs(sd - svec) / sd)
        print(f"[sq] controllo rho(k) esplicito ({args.ndir} direzioni) su {pathlib.Path(f).name}: "
              f"scarto relativo max {rel:.2e}")
        summary.setdefault("check_vectors", []).append({"file": f, "max_rel_diff": float(rel)})

    np.savez(args.out + ".npz", **out)
    json.dump(summary, open(args.out + ".json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        cmap = plt.get_cmap("coolwarm")
        for i, T in enumerate(Ts):
            key = f"S_eq_T{T:g}" if f"S_eq_T{T:g}" in out else f"S_all_T{T:g}"
            col = cmap(i / max(1, len(Ts) - 1))
            ax[0].loglog(k, out[key] / N, color=col, label=f"{T:g} K")
            ax[1].plot(k, k ** 2 * out[key] / N, color=col)
        if "S_ref" in out:
            ax[0].loglog(k, out["S_ref"] / N, "k--", label="AA 300 K")
            ax[1].plot(k, k ** 2 * out["S_ref"] / N, "k--")
        Tmid = Ts[len(Ts) // 2]
        for st, ls in (("F", "-"), ("I", ":"), ("U", "--")):
            key = f"S_{st}_T{Tmid:g}"
            if key in out:
                ax[2].loglog(k, out[key] / N, ls, label=f"{st}, {Tmid:g} K")
        ax[0].set(xlabel="k (nm$^{-1}$)", ylabel="S(k) / N", title="S(k)" + (" di equilibrio" if args.pu_json else ""))
        ax[1].set(xlabel="k (nm$^{-1}$)", ylabel="k$^2$ S(k) / N", title="Kratky")
        ax[2].set(xlabel="k (nm$^{-1}$)", ylabel="S(k) / N", title="per stato")
        ax[0].legend(fontsize=7, ncol=2)
        ax[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out + ".png", dpi=150)
    except Exception as exc:  # pragma: no cover
        print(f"[sq] figura non scritta: {exc}")
    print(f"[sq] scritti {args.out}.npz, {args.out}.json, {args.out}.png")


if __name__ == "__main__":
    main()
