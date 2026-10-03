#!/usr/bin/env python3
"""TEL26: spostamento quadratico medio per ora di GPU, CG contro AA.

PERCHE'
    L'efficienza di una simulazione si puo' misurare senza passare per un
    fattore di scala temporale: basta chiedersi quanto spazio esplora una copia
    per unita' di costo.  Per ogni corsa si misura l'MSD del centro di massa
    di ogni copia (traslazione) e l'MSD angolare della sua orientazione
    (rotazione), e si cambia l'asse dei tempi da tempo simulato a ore di GPU
    con i ns/giorno della corsa.  Nel regime diffusivo
        MSD per ora di GPU = 6 D x (ns simulati per ora di GPU)
    e il rapporto CG/AA coincide con (ns/giorno CG x alpha) / ns/giorno AA,
    alpha = D_CG / D_AA (script 47).

COSA MISURA, per ogni corsa
    D_trasl   MSD del centro di massa (non pesato) di ogni copia, srotolato
              nel tempo; retta sopra il regime balistico e fit di Langevin.
    D_rot     MSD angolare: orientazione del nucleo (36 siti delle tre
              tetradi) di ogni copia per Kabsch sul nucleo AA medio; le
              rotazioni fra frame successivi, nel sistema del laboratorio, si
              convertono in vettori di rotazione e si sommano:
                  phi(t) = sum_k rotvec(A_{k+1} A_k^T),
              <|phi(t+s) - phi(t)|^2> = 6 D_rot s per diffusione rotazionale
              isotropa.  Richiede rotazioni piccole fra due frame (controllato).
    tau_P2    decorrelazione dell'asse della pila (come lo script 47), per
              confronto: con gamma basso la rotazione CG e' inerziale e i due
              indicatori divergono (MSD angolare grande, decorrelazione
              limitata dall'inerzia).

    Poi, con --ns-day ETICHETTA=VALORE (AA compreso), i tassi per ora di GPU
    e l'MSD raggiunto a pari ore di GPU (curve in --png).

REPLICAS
    ETICHETTA=a.npz+b.npz       tratti consecutivi della stessa corsa (come 47)
    ETICHETTA=r0.npz,r1.npz     replicas indipendenti: MSD mediato sulle
                                replicas; --ns-day va dato AGGREGATO (somma
                                delle replicas sulla stessa GPU).

USO (in tutorials/tel26)
    python3 48_tel26_msd_wall.py tel26_lp1_dataset.bin \\
        opt=samples_tel26_opt_1ns.npz \\
        --ns-day AA=137 --ns-day opt=19 [--png msd_wall.png] [--json msd_wall.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("s47", HERE / "47_tel26_timescales.py")
s47 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s47)
s46 = s47.s46
cv = s47.cv


# ── orientazione e MSD angolare ─────────────────────────────────────────────

def core_orientation(X, ref):
    """X (T, C, R, 6, 3) -> A (T, C, 3, 3): x_lab - c = A @ x_ref (colonne)."""
    T, C = X.shape[:2]
    core = s46.take(X.reshape(T * C, *X.shape[2:]), ref.sel.core)      # (TC, n, 3)
    xc = core - core.mean(axis=1, keepdims=True)
    # Kabsch con il nucleo attuale come bersaglio: ref @ M ~ xc (righe)
    H = np.einsum("ni,bnj->bij", ref.core, xc)
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("bij,bjk->bik", U, Vt)))
    D = np.ones((xc.shape[0], 3)); D[:, 2] = d
    M = np.einsum("bij,bj,bjk->bik", U, D, Vt)                         # righe: ref @ M
    return np.swapaxes(M, 1, 2).reshape(T, C, 3, 3)                    # colonne: A = M^T


def rotation_path(A):
    """A (T, C, 3, 3) -> phi (T, C, 3) cumulato, e massimo angolo fra frame."""
    from scipy.spatial.transform import Rotation

    dA = np.einsum("tcij,tckj->tcik", A[1:], A[:-1])                   # A_{k+1} A_k^T
    rv = Rotation.from_matrix(dA.reshape(-1, 3, 3)).as_rotvec().reshape(dA.shape[:2] + (3,))
    step = np.linalg.norm(rv, axis=-1)
    phi = np.concatenate([np.zeros((1,) + rv.shape[1:]), np.cumsum(rv, axis=0)], axis=0)
    return phi, float(np.percentile(step, 99)), float(step.max())


def msd_curve(path, lags):
    return np.asarray([np.mean(np.sum((path[k:] - path[:-k]) ** 2, axis=-1)) for k in lags])


# ── una corsa (eventualmente piu' replicas) ─────────────────────────────────

def analyse(label, parts, ref, nuc, msd_lo):
    """parts: lista di (S, L, nc, t), una per replica indipendente."""
    T = min(p[0].shape[0] for p in parts)
    dt = float(np.median(np.diff(parts[0][3])))
    span = (T - 1) * dt
    lags = s47.lag_grid(T)
    lt = lags * dt
    tmax = span / 4
    m_tr, m_rot, c2, p99, pmax, ncop = [], [], [], [], [], 0
    for S, L, nc, _t in parts:
        S, L = S[:T], (L[:T] if np.ndim(L) == 2 else L)
        X = s46.unwrap_copies(S, L, nc, nuc)
        com = s47.com_unwrapped(X, L)
        m_tr.append(msd_curve(com, lags))
        phi, a99, amax = rotation_path(core_orientation(X, ref))
        m_rot.append(msd_curve(phi, lags))
        c2.append(s47.p2_corr(s47.stack_axis(X), lags))
        p99.append(a99); pmax.append(amax); ncop += nc
    m_tr, m_rot, c2 = (np.mean(v, axis=0) for v in (m_tr, m_rot, c2))
    lo = max(msd_lo, 5 * dt)
    out = {"label": label, "replicas": len(parts), "copies": ncop, "frames": T,
           "dt_ps": dt, "span_ps": span, "msd_window_ps": [lo, tmax],
           "rot_step_rad_p99": max(p99), "rot_step_rad_max": max(pmax),
           "lag_ps": lt.tolist(), "msd_trans_nm2": m_tr.tolist(), "msd_rot_rad2": m_rot.tolist()}
    out["D_trans"] = s47.diffusion(lt, m_tr, lo, tmax) * 1000.0                 # nm^2/ns
    out["D_rot"] = s47.diffusion(lt, m_rot, lo, tmax) * 1000.0                  # rad^2/ns
    Dl, tv, fl = s47.diffusion_langevin(lt, m_tr, 2 * dt, tmax)
    out["D_trans_langevin"], out["tau_v_ps"], out["flag_trans"] = Dl * 1000.0, tv, fl
    Dr, tw, fr = s47.diffusion_langevin(lt, m_rot, 2 * dt, tmax)
    out["D_rot_langevin"], out["tau_w_ps"], out["flag_rot"] = Dr * 1000.0, tw, fr
    out["tau_P2"] = s47.tau_1e(lt, c2, tmax)
    return out


def load_run(spec, stride, skip_ps):
    parts = []
    for rep in spec.split(","):
        S, L, nc, t = s47.load_cg(rep, stride)
        keep = t - t[0] >= skip_ps
        L = np.asarray(L)
        parts.append((S[keep], L[keep] if L.ndim == 2 else L, nc, t[keep]))
    return parts


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", help="dataset AA (riferimento e corsa AA)")
    ap.add_argument("runs", nargs="*", help="etichetta=a.npz[+b.npz][,r1.npz...]")
    ap.add_argument("--nuc", type=int, default=26)
    ap.add_argument("--aa-dt", type=float, default=20.0, help="ps fra due frame del dataset AA")
    ap.add_argument("--aa-stride", type=int, default=1)
    ap.add_argument("--cg-stride", type=int, default=1)
    ap.add_argument("--ref-stride", type=int, default=5)
    ap.add_argument("--skip-ps", type=float, default=0.0, help="ps iniziali scartati nelle corse CG")
    ap.add_argument("--msd-from", type=float, default=50.0,
                    help="ps: inizio della retta (CG); per l'AA almeno 100 ps")
    ap.add_argument("--ns-day", action="append", default=[], metavar="ETICHETTA=NS",
                    help="ns/giorno per GPU della corsa (AA compreso, etichetta AA); aggregati per le replicas")
    ap.add_argument("--png", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cv.set_nuc(args.nuc)
    nuc = args.nuc

    S, L, nc, t = s47.load_aa(args.dataset, args.aa_dt, args.aa_stride)
    k = max(1, args.ref_stride // args.aa_stride)
    Xr = s46.unwrap_copies(S[::k], L[::k], nc, nuc).reshape(-1, nuc, S.shape[2], 3)
    ref = s46.Reference(Xr, s46.Selection(nuc, Xr[0]))

    results = [analyse("AA", [(S, L, nc, t)], ref, nuc, max(args.msd_from, 100.0))]
    for spec in args.runs:
        label, path = spec.split("=", 1)
        results.append(analyse(label, load_run(path, args.cg_stride, args.skip_ps), ref, nuc, args.msd_from))
    nsd = {}
    for spec in args.ns_day:
        label, v = spec.split("=", 1)
        nsd[label] = float(v)

    W = 14
    hdr = "".join(f"{r['label']:>{W}s}" for r in results)
    print(f"\n  {'corsa':<36s}{hdr}")
    print(f"  {'replicas x copie':<36s}" + "".join(f"{str(r['replicas']) + ' x ' + str(r['copies'] // r['replicas']):>{W}s}" for r in results))
    print(f"  {'durata (ns), dt frame (ps)':<36s}" + "".join(f"{format(r['span_ps'] / 1000, '.3g') + ', ' + format(r['dt_ps'], '.3g'):>{W}s}" for r in results))
    print(f"  {'rotazione fra frame, p99/max (rad)':<36s}" + "".join(f"{format(r['rot_step_rad_p99'], '.2f') + '/' + format(r['rot_step_rad_max'], '.2f'):>{W}s}" for r in results))
    print(f"  {'D_trasl, retta (nm^2/ns)':<36s}" + "".join(f"{r['D_trans']:{W}.4g}" for r in results))
    print(f"  {'D_trasl, fit Langevin (nm^2/ns)':<36s}" + "".join(f"{r['flag_trans'] + format(r['D_trans_langevin'], '.4g'):>{W}s}" for r in results))
    print(f"  {'D_rot, retta (rad^2/ns)':<36s}" + "".join(f"{r['D_rot']:{W}.4g}" for r in results))
    print(f"  {'D_rot, fit Langevin (rad^2/ns)':<36s}" + "".join(f"{r['flag_rot'] + format(r['D_rot_langevin'], '.4g'):>{W}s}" for r in results))
    print(f"  {'tau_omega, fit (ps)':<36s}" + "".join(f"{r['flag_rot'] + format(r['tau_w_ps'], '.3g'):>{W}s}" for r in results))
    print(f"  {'tau_P2 asse della pila':<36s}" + "".join(f"{s47.fmt_tau(r['tau_P2']):>{W}s}" for r in results))
    print(f"  {'1/(6 D_rot) (ps), diffusivo':<36s}" + "".join(f"{1e3 / (6 * r['D_rot']):{W}.3g}" for r in results))

    aa = results[0]
    if "AA" in nsd and any(r["label"] in nsd for r in results[1:]):
        print("\n  per ora di GPU: tasso = 6 D x ns/ora;  rapporto rispetto all'AA")
        print(f"  {'':<36s}{hdr}")
        print(f"  {'ns/giorno per GPU':<36s}" + "".join(f"{nsd.get(r['label'], float('nan')):{W}.4g}" for r in results))
        for key, name, unit in (("D_trans", "traslazione, retta", "nm^2"),
                                ("D_trans_langevin", "traslazione, fit Langevin", "nm^2"),
                                ("D_rot", "rotazione, retta", "rad^2"),
                                ("D_rot_langevin", "rotazione, fit Langevin", "rad^2")):
            rate = [6 * r[key] * nsd.get(r["label"], np.nan) / 24 for r in results]
            # per l'AA si usa sempre la retta (il tratto balistico e' sotto i 20 ps)
            base = 6 * aa[key.replace("_langevin", "")] * nsd["AA"] / 24
            print(f"  {name + ' (' + unit + '/ora)':<36s}" + "".join(f"{v:{W}.4g}" for v in rate))
            print(f"  {'   rapporto CG/AA':<36s}" + "".join(f"{v / base:{W}.3g}" for v in rate))
        print(f"  {'rotazione, decorrelazione tau_P2':<36s}" + "".join(
            f"{(aa['tau_P2'][0] / r['tau_P2'][0]) * nsd.get(r['label'], np.nan) / nsd['AA']:{W}.3g}" for r in results))

        # MSD raggiunto a pari ore di GPU (entro un quarto di ciascuna corsa)
        print("\n  MSD raggiunto dopo w ore di GPU (traslazione nm^2 / rotazione rad^2)")
        wmax = min(r["msd_window_ps"][1] / 1000 / (nsd[r["label"]] / 24) for r in results if r["label"] in nsd)
        grid = [w for w in (1 / 60, 5 / 60, 0.25, 0.5, 1, 2, 4, 8, 24) if w <= wmax] or [wmax]
        W2 = 20
        print(f"  {'w (ore)':<36s}" + "".join(f"{r['label']:>{W2}s}" for r in results))
        for w in grid:
            cells = []
            for r in results:
                if r["label"] not in nsd:
                    cells.append(f"{'-':>{W2}s}"); continue
                lag = w * nsd[r["label"]] / 24 * 1000
                tr = np.interp(lag, r["lag_ps"], r["msd_trans_nm2"])
                ro = np.interp(lag, r["lag_ps"], r["msd_rot_rad2"])
                cells.append(f"{format(tr, '.3g') + ' / ' + format(ro, '.3g'):>{W2}s}")
            print(f"  {w:<36.3g}" + "".join(cells))

    print("\n  Note: la retta dell'MSD parte da --msd-from (AA almeno 100 ps).  '?' = tempo di")
    print("  rilassamento della velocita' oltre un terzo della finestra, D estrapolato.  Con rotazioni")
    print("  fra frame oltre ~0,3 rad l'MSD angolare va rifatto con frame piu' fitti.")
    print("  Se tau_P2 >> 1/(6 D_rot) la rotazione e' inerziale: l'MSD angolare sovrastima il")
    print("  guadagno nel campionamento delle orientazioni, che resta quello di tau_P2.")

    if args.png and "AA" in nsd:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for r in results:
            if r["label"] not in nsd:
                continue
            h = np.asarray(r["lag_ps"]) / 1000 / (nsd[r["label"]] / 24)
            sel = np.asarray(r["lag_ps"]) <= r["msd_window_ps"][1]
            ax[0].loglog(h[sel], np.asarray(r["msd_trans_nm2"])[sel], label=r["label"])
            ax[1].loglog(h[sel], np.asarray(r["msd_rot_rad2"])[sel], label=r["label"])
        ax[0].set(xlabel="ore di GPU", ylabel="MSD traslazionale (nm$^2$)")
        ax[1].set(xlabel="ore di GPU", ylabel="MSD angolare (rad$^2$)")
        for a in ax:
            a.legend(); a.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.png, dpi=150)
        print(f"\n  grafico: {args.png}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
