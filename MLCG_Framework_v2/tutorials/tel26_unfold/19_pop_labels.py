#!/usr/bin/env python3
"""Campioni CG a una temperatura -> dataset + etichette di stato per train_relent --pop.

PERCHE' (19/10)
    Il residuo PaiNN allenato a entropia relativa a 300 K vede solo lo stato nativo e tiene il
    TEL26 piegato fino a 380 K (17); la scansione di kappa (18) mostra che nessun riscalamento
    globale da' insieme Tm e larghezza.  train_relent --pop aggiunge all'obiettivo un vincolo
    sulle popolazioni sperimentali, L = ln(P_U/P_N) a T fissata, stimato per ripesatura dei
    campioni del modello corrente.  Questo script prepara, per una T, i due ingressi:
      - il dataset binario dei frame (samples_to_cg_bin.py, stessi --skip-ps e --stride),
      - la probabilita' p_U di ogni frame di essere nello stato U, nello stesso ordine.

DEFINIZIONE DI STATO (come 08_tetrad_melt.py --smooth-ns e 14_thermal_check.py)
    q_k = frazione liscia dei 6 contatti B3-B3 nativi della tetrade k, mediata su --smooth-ns
    (1 ns, finestra centrata sulla corsa intera); U = nessuna tetrade con q_k > --qcut (0.5).
    --delta 0: etichette 0/1.  --delta > 0 (default 0.05): p_U = prod_k sigmoid((qcut - q_k)/delta),
    che vale ~1 in U e ~0 in F ma non e' mai esattamente zero: anche se U non e' mai visitato,
    il gradiente spinge verso i frame piu' aperti invece di sparire.
    Le etichette dipendono dalla finestra temporale, non dal solo frame: l'approssimazione e'
    buona se gli stati vivono molto piu' della finestra (ns contro 1 ns, vedi 13/17).

BERSAGLIO
    P_U sperimentale (Buscaglia 2013, 2JPZ 25 mM KCl, modello a 4 stati: U e' lo stesso nelle
    mappe f e f1) alla T data, L* = ln(P_U / (1 - P_U)).

USO (Leonardo, dentro 19_relent_pop.sh)
    python3 19_pop_labels.py --T 330 --template $A/cg/tel26_lp2_1c_t400_dataset.bin \\
        --samples F/T330.samples.npz U/T330.samples.npz --skip-ps 500 --out-prefix pop_it01_T330
    -> pop_it01_T330.bin, pop_it01_T330.labels, pop_it01_T330.json; ultima riga
       '[POPSPEC] <bin>,<labels>,<T>,<L*>' da passare a train_relent --pop.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FRAMEWORK = HERE.parents[1]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


melt = _load("tetrad_melt", "08_tetrad_melt.py")
fit = _load("fit_thermal", "11_fit_thermal.py")
cv = melt.cv


def smooth_q(q, t_ps, smooth_ns):
    """media mobile centrata, corretta ai bordi; q (T, C, 3)"""
    if smooth_ns <= 0 or q.shape[0] < 2:
        return q
    w = max(1, int(round(smooth_ns * 1000.0 / float(np.median(np.diff(t_ps))))))
    k = np.ones(w) / w
    norm = np.convolve(np.ones(q.shape[0]), k, mode="same")
    out = np.empty_like(q)
    for c in range(q.shape[1]):
        for j in range(3):
            out[:, c, j] = np.convolve(q[:, c, j], k, mode="same") / norm
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--template", required=True, help="dataset a una copia (topologia dei frame)")
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--skip-ps", type=float, default=500.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--smooth-ns", type=float, default=1.0)
    ap.add_argument("--qcut", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--native", default=str(HERE.parent / "tel26" / "tel26_lp2_dataset.bin"))
    ap.add_argument("--native-stride", type=int, default=5)
    ap.add_argument("--beta", type=float, default=25.0)
    ap.add_argument("--lam", type=float, default=1.2)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    cv.set_nuc(melt.NUC)

    out_bin = pathlib.Path(args.out_prefix + ".bin")
    out_lab = pathlib.Path(args.out_prefix + ".labels")
    out_json = pathlib.Path(args.out_prefix + ".json")
    for f in (out_bin, out_lab):
        if f.exists():
            raise SystemExit(f"[ERROR] esiste gia': {f}")

    Sn, Ln, ncn = cv.load_reference(args.native)
    dn = melt.b3_pair_distances(Sn[::args.native_stride], Ln[::args.native_stride], ncn)
    r_nat = np.nanmedian(dn, axis=0)

    labels, per_run = [], []
    for path in args.samples:
        S, L, nc, t = cv.load_samples(path)
        t = np.asarray(t, float)
        if nc != 1:
            raise SystemExit(f"[ERROR] {path}: {nc} copie, serve una copia per corsa")
        d = melt.b3_pair_distances(S, L, nc)
        q = (1.0 / (1.0 + np.exp(args.beta * (d - args.lam * r_nat[None])))).mean(axis=2)
        q = smooth_q(q.reshape(S.shape[0], nc, 3), t, args.smooth_ns)[:, 0, :]        # (T, 3)
        keep = np.flatnonzero(t >= t[0] + args.skip_ps)[::args.stride]               # come samples_to_cg_bin
        qk = q[keep]
        hard = (qk <= args.qcut).all(axis=1).astype(float)
        if args.delta > 0:
            soft = np.prod(1.0 / (1.0 + np.exp(-(args.qcut - qk) / args.delta)), axis=1)
        else:
            soft = hard
        labels.append(soft)
        n_int = (qk > args.qcut).sum(axis=1)
        per_run.append({"samples": path, "frames": int(keep.size), "t_ps": [float(t[keep[0]]), float(t[keep[-1]])]
                        if keep.size else None, "P_U_hard": float(hard.mean()) if keep.size else None,
                        "p_U_soft_mean": float(soft.mean()) if keep.size else None,
                        "P_n_intact": [float((n_int == k).mean()) for k in range(4)] if keep.size else None,
                        "q_mean": qk.mean(0).tolist() if keep.size else None})
    p = np.concatenate(labels)
    if p.size == 0:
        raise SystemExit("[ERROR] nessun frame dopo --skip-ps")

    cmd = [sys.executable, str(FRAMEWORK / "preprocessing" / "samples_to_cg_bin.py"), "--template", args.template,
           "--samples", *args.samples, "--skip-ps", str(args.skip_ps), "--stride", str(args.stride),
           "--out", str(out_bin)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    tmp = out_lab.with_suffix(".labels.tmp")
    with open(tmp, "w") as f:
        f.write(f"# p_U per frame, T = {args.T} K, smooth {args.smooth_ns} ns, qcut {args.qcut}, delta {args.delta}\n")
        f.writelines(f"{x:.6e}\n" for x in p)
    tmp.rename(out_lab)

    pu_exp = float(fit.exp_pops([args.T])[0, 3])
    target = float(np.log(pu_exp / (1.0 - pu_exp)))
    pu_cg = float(np.clip(p.mean(), 1e-12, 1 - 1e-12))
    summary = {"T_K": args.T, "frames": int(p.size), "P_U_exp": pu_exp, "L_target": target,
               "p_U_mean": pu_cg, "L_uniform": float(np.log(pu_cg / (1 - pu_cg))), "runs": per_run,
               "smooth_ns": args.smooth_ns, "qcut": args.qcut, "delta": args.delta, "skip_ps": args.skip_ps,
               "stride": args.stride, "bin": str(out_bin), "labels": str(out_lab)}
    json.dump(summary, open(out_json, "w"), indent=1)
    for r in per_run:
        print(f"  T {args.T:5.1f}  {pathlib.Path(r['samples']).name.split('.')[0]:>8s}  {r['frames']:5d} frame  "
              f"P_U {r['P_U_hard']:.3f} (morbida {r['p_U_soft_mean']:.3g})  "
              f"P(n) {' '.join(f'{x:.2f}' for x in r['P_n_intact'])}  q {' '.join(f'{x:.2f}' for x in r['q_mean'])}")
    print(f"  T {args.T:5.1f}  bersaglio P_U {pu_exp:.3f}  L* {target:+.3f};  modello <p_U> {pu_cg:.3g}")
    print(f"[POPSPEC] {out_bin},{out_lab},{args.T:g},{target:.6f}")


if __name__ == "__main__":
    main()
