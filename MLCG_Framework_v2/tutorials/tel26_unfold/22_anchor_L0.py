#!/usr/bin/env python3
"""Ancora della catena di perturbazioni: L(T) = ln(P_U/P_N) del modello dei soli prior.

PERCHE' (26/10)
    pop5 (19_relent_pop.sh, d64 da zero) ha alzato la barriera F/U: con il modello fermo da
    it20 a it36 (~50 ns per lignaggio e per T) si sono viste due transizioni in ~500 ns.  La
    frazione di frame U dei lignaggi non stima piu' P_U.  La ripesatura dentro ciascun bacino
    resta pero' esatta: L(theta_k) = L(theta_0) + sum_j [L(theta_j) - L(theta_j-1)], con ogni
    incremento calcolato da train_relent sui campioni dei due bacini (POP_CHAIN=1 in 19).
    Serve solo L(theta_0), cioe' il modello dei soli prior (PaiNN nullo), dove le transizioni
    sono frequenti: lo si misura qui sulle corse di 13_validate_thermal.sh (100 ns per corsa,
    rami chiuso e aperto, due repliche per T).

STATO U (come 19_pop_labels.py --order tract)
    c_a = frazione liscia delle 9 coppie B3-B3 native fra il tratto G a e gli altri; media
    mobile su --smooth-ns; U = c_a < --tract-cut per tutti e quattro i tratti.

STIMA
    Per T: P_U dalla somma dei frame di tutte le corse (dopo --skip-ps), separata anche per
    ramo (isteresi), e numero di transizioni U <-> non U per corsa (con isteresi: si esce da
    U solo se un tratto supera --exit-cut).  Fit a due stati
        L(T) = -(dH / R) (1/T - 1/Tm)
    per massima verosimiglianza binomiale con conteggi efficaci n_eff = 1 + transizioni per
    corsa (frame correlati: conta il numero di visite indipendenti, non i frame).  Le T in cui
    i due rami restano separati senza transizioni pesano ~1 e non spostano il fit.

USO (Leonardo)
    python3 22_anchor_L0.py --dirs $A/thermo_val/f/chiuso $A/thermo_val/f/aperto \\
        --template $A/cg/tel26_lp2_1c_t400_dataset.bin --out $A/relent_pop/anchor_f.json
    -> tabella per T, Tm e dH del fit, file JSON con {"Tm_K", "dH_kJ", "per_T": [...]}.
    19_relent_pop.sh (POP_CHAIN=1, POP_ANCHOR=...) usa L(T) del fit come L assoluto di it00.
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
R_KJ = 0.008314462618


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pl = _load("pop_labels", "19_pop_labels.py")


def tract_c(path, r_nat, beta, lam, smooth_ns):
    S, L, nc, t = pl.cv.load_samples(path)
    t = np.asarray(t, float)
    if nc != 1:
        raise SystemExit(f"[ERROR] {path}: {nc} copie, serve una copia per corsa")
    d = pl.melt.b3_pair_distances(S, L, nc)
    sp = 1.0 / (1.0 + np.exp(beta * (d - lam * r_nat[None])))
    c = np.stack([sp[:, :, pl.TRACT_MASK[a]].mean(axis=(1, 2)) for a in range(4)], 1)
    return t, pl.smooth2(c, t, smooth_ns)


def transitions(cmax, cut, exit_cut):
    """U se tutti i tratti < cut; si esce da U solo con un tratto > exit_cut (isteresi)."""
    state, n = None, 0
    for x in cmax:
        new = state
        if x < cut:
            new = 1
        elif x > exit_cut:
            new = 0
        if state is not None and new != state:
            n += 1
        state = new
    return n


def fit_two_state(Ts, P, n_eff):
    from scipy.optimize import minimize
    Ts, P, n = map(np.asarray, (Ts, P, n_eff))
    P = np.clip(P, 1e-6, 1 - 1e-6)

    def nll(x):
        Tm, dH = x
        Lm = -(dH / R_KJ) * (1.0 / Ts - 1.0 / Tm)
        ls_u = -np.logaddexp(0.0, -Lm)
        ls_n = -np.logaddexp(0.0, Lm)
        return -np.sum(n * (P * ls_u + (1 - P) * ls_n))

    # punto di partenza: T con P_U piu' vicino a 0.5, dH di Buscaglia
    T0 = float(Ts[np.argmin(np.abs(P - 0.5))])
    best = None
    for dH0 in (150.0, 350.0, 700.0):
        r = minimize(nll, [T0, dH0], method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    Tm, dH = best.x
    # errori dall'hessiana numerica
    h = np.array([0.05, 1.0])
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            e_i, e_j = np.eye(2)[i] * h[i], np.eye(2)[j] * h[j]
            H[i, j] = (nll(best.x + e_i + e_j) - nll(best.x + e_i - e_j)
                       - nll(best.x - e_i + e_j) + nll(best.x - e_i - e_j)) / (4 * h[i] * h[j])
    try:
        err = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        err = np.array([np.nan, np.nan])
    return float(Tm), float(dH), [float(e) for e in err]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="cartelle con T<T>_r<k>.samples.npz (un ramo ciascuna)")
    ap.add_argument("--template", required=False, help="(non usato; per simmetria con 19_pop_labels)")
    ap.add_argument("--native", default=str(HERE.parent / "tel26" / "tel26_lp2_dataset.bin"))
    ap.add_argument("--native-stride", type=int, default=5)
    ap.add_argument("--beta", type=float, default=25.0)
    ap.add_argument("--lam", type=float, default=1.2)
    ap.add_argument("--smooth-ns", type=float, default=1.0)
    ap.add_argument("--tract-cut", type=float, default=0.2)
    ap.add_argument("--exit-cut", type=float, default=0.4, help="soglia d'uscita da U per contare le transizioni")
    ap.add_argument("--skip-ps", type=float, default=5000.0)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=1e9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pl.cv.set_nuc(pl.melt.NUC)

    Sn, Ln, ncn = pl.cv.load_reference(args.native)
    r_nat = np.nanmedian(pl.melt.b3_pair_distances(Sn[::args.native_stride], Ln[::args.native_stride], ncn), axis=0)

    runs = {}
    for dd in args.dirs:
        branch = pathlib.Path(dd).name
        for f in sorted(glob.glob(str(pathlib.Path(dd) / "T*_r*.samples.npz"))):
            m = re.match(r"T(\d+(?:\.\d+)?)_r(\d+)\.samples\.npz$", pathlib.Path(f).name)
            if not m:
                continue
            T = float(m.group(1))
            if not (args.tmin <= T <= args.tmax):
                continue
            t, c = tract_c(f, r_nat, args.beta, args.lam, args.smooth_ns)
            keep = t >= t[0] + args.skip_ps
            cm = c[keep].max(axis=1)
            u = cm < args.tract_cut
            runs.setdefault(T, []).append({"file": f, "branch": branch, "frames": int(keep.sum()),
                                           "ns": float((t[keep][-1] - t[keep][0]) / 1000.0) if keep.any() else 0.0,
                                           "P_U": float(u.mean()) if keep.any() else None,
                                           "transitions": transitions(cm, args.tract_cut, args.exit_cut),
                                           "c_mean": c[keep].mean(0).tolist()})
    if not runs:
        raise SystemExit("[ERROR] nessuna corsa trovata")

    per_T, Ts, Ps, ns = [], [], [], []
    print(f"{'T':>6s} {'P_U':>6s} {'chiuso':>7s} {'aperto':>7s} {'trans':>6s} {'n_eff':>6s}  corse")
    for T in sorted(runs):
        rs = [r for r in runs[T] if r["frames"] > 0]
        nfr = sum(r["frames"] for r in rs)
        pu = sum(r["P_U"] * r["frames"] for r in rs) / nfr
        by = {}
        for b in sorted({r["branch"] for r in rs}):
            rb = [r for r in rs if r["branch"] == b]
            by[b] = sum(r["P_U"] * r["frames"] for r in rb) / sum(r["frames"] for r in rb)
        ntr = sum(r["transitions"] for r in rs)
        neff = sum(1 + r["transitions"] for r in rs)
        per_T.append({"T_K": T, "P_U": pu, "P_U_branch": by, "transitions": ntr, "n_eff": neff, "runs": rs})
        Ts.append(T); Ps.append(pu); ns.append(neff)
        print(f"{T:6.1f} {pu:6.3f} {by.get('chiuso', float('nan')):7.3f} {by.get('aperto', float('nan')):7.3f} "
              f"{ntr:6d} {neff:6d}  {len(rs)} x {np.mean([r['ns'] for r in rs]):.0f} ns")

    Tm, dH, err = fit_two_state(Ts, Ps, ns)
    print(f"\n[ANCHOR] fit a due stati: Tm = {Tm:.2f} +- {err[0]:.2f} K, dH = {dH:.1f} +- {err[1]:.1f} kJ/mol "
          f"(Buscaglia: Tm 328.4 K, dH totale 352 kJ/mol)")
    print(f"{'T':>6s} {'P_U oss':>8s} {'P_U fit':>8s} {'L fit':>8s}")
    for row in per_T:
        Lf = -(dH / R_KJ) * (1.0 / row["T_K"] - 1.0 / Tm)
        row["L_fit"] = Lf
        print(f"{row['T_K']:6.1f} {row['P_U']:8.3f} {1 / (1 + np.exp(-Lf)):8.3f} {Lf:+8.3f}")
    out = {"Tm_K": Tm, "dH_kJ": dH, "err": err, "tract_cut": args.tract_cut, "exit_cut": args.exit_cut,
           "smooth_ns": args.smooth_ns, "skip_ps": args.skip_ps, "dirs": args.dirs,
           "formula": "L(T) = -(dH_kJ / R) (1/T - 1/Tm_K), R = 0.008314462618 kJ/mol/K", "per_T": per_T}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[ANCHOR] scritto {args.out}")


if __name__ == "__main__":
    main()
