#!/usr/bin/env python3
"""TEL26: fattore di scala temporale fra la dinamica CG e quella all-atom.

PERCHE'
    Il CG con il residuo ML costa ~60 ms per passo (dt 1 fs, ~1,4 ns al giorno
    su una A100); l'AA in acqua esplicita fa ~100-200 ns al giorno.  A parita'
    di picosecondi il CG e' quindi piu' lento.  Ma un picosecondo CG non e' un
    picosecondo AA: il solvente implicito e il paesaggio liscio accelerano i
    moti.  Il confronto giusto e' fra i tempi di rilassamento degli stessi
    osservabili: se un moto decorrela in tau_AA nell'AA e in tau_CG nel CG, il
    CG guadagna alpha = tau_AA / tau_CG, e la velocita' efficiente e'
        (ns/giorno CG) * alpha / (ns/giorno AA).
    alpha non e' uno solo: dipende dal moto (diffusione, rotazione, moti
    interni), e per i moti diffusivi scala come 1/gamma del termostato.

COSA MISURA, per ogni corsa
    D_trasl    coefficiente di diffusione del centro di massa di ogni copia
               (MSD lineare, finestra sopra il regime balistico)
    tau_rot    rotazione della copia: <P2(n(0).n(t))> con n = asse della pila
               (baricentro tetrade 1 - baricentro tetrade 3)
    tau_Rg     autocorrelazione del raggio di girazione della copia
    tau_core   autocorrelazione di rmsd_core (nucleo allineato sull'AA medio)
    tau_loops  autocorrelazione degli spostamenti dei siti dei loop dalla loro
               posizione media, a nucleo allineato (moti interni lenti)
    tau = tempo in cui la funzione di correlazione scende a 1/e (interpolato).
    Un tempo sotto il passo di campionamento si riporta come limite superiore
    ('<'), uno oltre un quarto della corsa come limite inferiore ('>').

ATTENZIONE AL CAMPIONAMENTO
    Il dataset AA ha un frame ogni 20 ps (--aa-dt): i moti piu' rapidi di
    qualche decina di ps non si risolvono nell'AA; restano D, la rotazione,
    Rg e i loop lenti.  Le corse CG da 1 ns (tratti s1..s4 concatenati con
    '+') risolvono i tempi dal decimo di ps a ~250 ps.

USO (in tutorials/tel26)
    python3 47_tel26_timescales.py tel26_lp1_dataset.bin \\
        re0_it30=samples_1ns_it30_s1.npz+samples_1ns_it30_s2.npz+samples_1ns_it30_s3.npz+samples_1ns_it30_s4.npz \\
        re1_it08=samples_tel26_lp2_re1_it08_100ps.npz \\
        [--aa-dt 20] [--aa-stride 1] [--ns-day-aa 150 --ns-day-cg 1.4] [--json ts.json] \\
        [--gamma re0_it30=20 --gamma re1_it08=20 --kT 2.49]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("s46", HERE / "46_tel26_structure.py")
s46 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s46)
cv = s46.cv


# ── caricamento ──────────────────────────────────────────────────────────────

def load_aa(path, dt, stride):
    S, L, nc = cv.load_reference(path)
    S, L = S[::stride], L[::stride]
    return S, L, nc, np.arange(S.shape[0]) * dt * stride


def load_cg(spec, stride):
    """'a.npz+b.npz+...' -> tratti concatenati in ordine, tempo continuo."""
    parts = []
    for k, path in enumerate(spec.split("+")):
        S, L, nc, t = cv.load_samples(path)
        t = np.asarray(t, float)
        L = np.asarray(L, float)
        if L.ndim == 1:
            L = np.tile(L, (S.shape[0], 1))
        if parts:
            Sp, Lp, _n, tp = parts[-1]
            dt = float(np.median(np.diff(tp)))
            # Il primo frame di un tratto puo' ripetere l'ultimo del precedente.
            same = np.nanmax(np.abs(S[0] - Sp[-1])) < 1e-6
            if same:
                S, L, t = S[1:], L[1:], t[1:]
            t = t - t[0] + tp[-1] + dt
        parts.append((S, L, nc, t))
    S = np.concatenate([p[0] for p in parts])
    L = np.concatenate([p[1] for p in parts])
    t = np.concatenate([p[3] for p in parts])
    nc = parts[0][2]
    S, L, t = S[::stride], L[::stride], t[::stride]
    d = np.diff(t)
    if d.size and (d.max() - d.min()) > 1e-3 * np.median(d):
        raise SystemExit(f"[ERROR] {spec}: campionamento non uniforme (dt {d.min():.4g}-{d.max():.4g} ps)")
    return S, L, nc, t


# ── osservabili per (frame, copia) ──────────────────────────────────────────

def com_unwrapped(X, L):
    """X (T, C, R, 6, 3) copie intere -> centro di massa (non pesato) srotolato nel tempo."""
    com = np.nanmean(X.reshape(X.shape[0], X.shape[1], -1, 3), axis=2)      # (T, C, 3)
    Lb = L.reshape(-1, 1, 3) if L.ndim == 2 else L.reshape(1, 1, 3)
    step = np.diff(com, axis=0)
    Ls = Lb[1:] if Lb.shape[0] > 1 else Lb
    step -= Ls * np.round(step / Ls)
    return np.concatenate([com[:1], com[:1] + np.cumsum(step, axis=0)], axis=0)


def stack_axis(X):
    """Asse della pila: baricentro della tetrade 1 meno quello della tetrade 3."""
    t1 = [r - 1 for r in s46.TETRADS_1B[0]]
    t3 = [r - 1 for r in s46.TETRADS_1B[2]]
    a = X[:, :, t1].reshape(X.shape[0], X.shape[1], -1, 3).mean(axis=2)
    b = X[:, :, t3].reshape(X.shape[0], X.shape[1], -1, 3).mean(axis=2)
    n = a - b
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def aligned_on_core(Xf, ref):
    """(N, R, 6, 3) -> (N, n_all, 3) nel sistema del nucleo AA medio."""
    sel = ref.sel
    core = s46.take(Xf, sel.core)
    cc = core.mean(axis=1, keepdims=True)
    H = np.einsum("bni,nj->bij", core - cc, ref.core)
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("bij,bjk->bik", U, Vt)))
    D = np.ones((core.shape[0], 3)); D[:, 2] = d
    R = np.einsum("bij,bj,bjk->bik", U, D, Vt)
    return np.einsum("bni,bij->bnj", s46.take(Xf, sel.all) - cc, R)


# ── funzioni di correlazione ─────────────────────────────────────────────────

def acf_columns(x):
    """x (T, K) a media nulla per colonna -> ACF normalizzata, media sulle colonne."""
    T = x.shape[0]
    f = np.fft.rfft(x, n=2 * T, axis=0)
    c = np.fft.irfft(f * np.conj(f), axis=0)[:T].sum(axis=1)
    c /= (T - np.arange(T))
    return c / c[0]


def acf_scalar(y):
    """y (T, C): ogni copia a media nulla, ACF media sulle copie."""
    return acf_columns(y - y.mean(axis=0, keepdims=True))


def acf_vectors(v):
    """v (T, C, n, 3): spostamenti dalla media per copia e sito."""
    d = v - v.mean(axis=0, keepdims=True)
    return acf_columns(d.reshape(d.shape[0], -1))


def lag_grid(T, nmax=200):
    g = np.unique(np.round(np.geomspace(1, max(2, T - 1), nmax)).astype(int))
    return g[g < T]


def p2_corr(n, lags):
    out = []
    for k in lags:
        c = np.einsum("tci,tci->tc", n[:-k], n[k:])
        out.append(np.mean(1.5 * c * c - 0.5))
    return np.asarray(out)


def msd(com, lags):
    return np.asarray([np.mean(np.sum((com[k:] - com[:-k]) ** 2, axis=-1)) for k in lags])


def tau_1e(lag_t, c, tmax):
    """Primo attraversamento di 1/e, interpolato; (valore, '<' | '' | '>')."""
    target = np.exp(-1.0)
    below = np.flatnonzero(c < target)
    if below.size == 0 or lag_t[below[0]] > tmax:
        return tmax, ">"
    i = below[0]
    if i == 0:
        return lag_t[0], "<"
    t0, t1, c0, c1 = lag_t[i - 1], lag_t[i], c[i - 1], c[i]
    return t0 + (c0 - target) / (c0 - c1) * (t1 - t0), ""


def diffusion(lag_t, m, lo, hi):
    sel = (lag_t >= lo) & (lag_t <= hi)
    if sel.sum() < 3:
        return float("nan")
    slope = np.polyfit(lag_t[sel], m[sel], 1)[0]
    return slope / 6.0                       # nm^2/ps


def diffusion_langevin(lag_t, m, lo, hi):
    """D e tau_v dal fit dell'MSD del moto di Langevin libero.

    MSD(t) = 6 D [t - tau (1 - exp(-t/tau))] + b, con tau = M / sum(gamma) il
    tempo di rilassamento della velocita' del centro di massa e b un piccolo
    offset (fluttuazioni interne veloci).  Con gamma basso il regime balistico
    dura ~tau e una retta su una finestra fissa sottostima D: questo fit usa
    anche il tratto balistico.  Restituisce (D in nm^2/ps, tau in ps, flag):
    flag '?' se tau supera un terzo della finestra (D estrapolato).
    """
    from scipy.optimize import curve_fit

    sel = (lag_t >= lo) & (lag_t <= hi) & (m > 0)
    if sel.sum() < 5:
        return float("nan"), float("nan"), "?"
    t, y = lag_t[sel], m[sel]

    def model(t, D, tau, b):
        return 6.0 * D * (t + tau * np.expm1(-t / tau)) + b

    tail = max(3, len(t) // 3)
    D0 = max(np.polyfit(t[-tail:], y[-tail:], 1)[0] / 6.0, 1e-12)
    p0 = [D0, max(float(t[len(t) // 4]), 1e-3), 0.0]
    try:
        p, _ = curve_fit(model, t, y, p0=p0, sigma=y,
                         bounds=([0.0, 1e-6, -np.inf], [np.inf, 1e7, np.inf]),
                         maxfev=20000)
    except (RuntimeError, ValueError):
        return float("nan"), float("nan"), "?"
    D, tau = float(p[0]), float(p[1])
    return D, tau, ("?" if tau > hi / 3.0 else "")


# ── una corsa ────────────────────────────────────────────────────────────────

def analyse(label, S, L, nc, t, ref, nuc, msd_lo):
    T = S.shape[0]
    dt = float(np.median(np.diff(t)))
    span = t[-1] - t[0]
    X = s46.unwrap_copies(S, L, nc, nuc)                          # (T, C, R, 6, 3)
    Xf = X.reshape(-1, nuc, S.shape[2], 3)
    out = {"label": label, "frames": T, "dt_ps": dt, "span_ps": span, "copies": nc}
    tmax = span / 4

    # traslazione e rotazione
    com = com_unwrapped(X, L)
    lags = lag_grid(T)
    lt = lags * dt
    m = msd(com, lags)
    lo = max(msd_lo, 5 * dt)
    out["D_nm2_per_ns"] = diffusion(lt, m, lo, tmax) * 1000.0
    out["msd_window_ps"] = [lo, tmax]
    D_l, tau_v, flag_l = diffusion_langevin(lt, m, 2 * dt, tmax)
    out["D_langevin_nm2_per_ns"] = D_l * 1000.0
    out["tau_v_ps"] = tau_v
    out["langevin_flag"] = flag_l
    c2 = p2_corr(stack_axis(X), lags)
    out["tau_rot"] = tau_1e(lt, c2, tmax)

    # moti interni, a nucleo allineato
    cvs = s46.collective(Xf, ref)
    full = np.arange(T) * dt
    out["tau_Rg"] = tau_1e(full, acf_scalar(cvs["Rg"].reshape(T, nc)), tmax)
    out["tau_core"] = tau_1e(full, acf_scalar(cvs["rmsd_core"].reshape(T, nc)), tmax)
    A = aligned_on_core(Xf, ref).reshape(T, nc, -1, 3)
    res = np.array([r for r, _ in ref.sel.all])
    loop = np.array([s46.SEQUENCE[r] != "G" for r in res])
    out["tau_loops"] = tau_1e(full, acf_vectors(A[:, :, loop]), tmax)
    out["tau_core_sites"] = tau_1e(full, acf_vectors(A[:, :, ~loop]), tmax)
    return out


def fmt_tau(v):
    val, flag = v
    unit, x = ("ns", val / 1000.0) if val >= 1000 else ("ps", val)
    return f"{flag}{x:.3g} {unit}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", help="dataset AA (riferimento e corsa AA)")
    ap.add_argument("runs", nargs="*", help="etichetta=a.npz[+b.npz...]")
    ap.add_argument("--nuc", type=int, default=26)
    ap.add_argument("--aa-dt", type=float, default=20.0, help="ps fra due frame del dataset AA")
    ap.add_argument("--aa-stride", type=int, default=1)
    ap.add_argument("--cg-stride", type=int, default=1)
    ap.add_argument("--ref-stride", type=int, default=5, help="frame AA per la struttura media")
    ap.add_argument("--msd-from", type=float, default=50.0,
                    help="ps: inizio della finestra lineare dell'MSD (sopra il regime balistico)")
    ap.add_argument("--ns-day-aa", type=float, default=None)
    ap.add_argument("--ns-day-cg", type=float, default=None)
    ap.add_argument("--gamma", action="append", default=[], metavar="ETICHETTA=GAMMA",
                    help="attrito di Langevin di una corsa CG, come --gamma di run_cg_md.py (amu/ps): aggiunge D teorico = kT/(N gamma)")
    ap.add_argument("--kT", type=float, default=2.49, help="kJ/mol, per D teorico")
    ap.add_argument("--bodies-per-copy", type=int, default=None,
                    help="corpi con attrito per copia (default: --nuc, un corpo per residuo)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cv.set_nuc(args.nuc)
    nuc = args.nuc

    S, L, nc, t = load_aa(args.dataset, args.aa_dt, args.aa_stride)
    Sr = S[::max(1, args.ref_stride // args.aa_stride)]
    Lr = L[::max(1, args.ref_stride // args.aa_stride)]
    Xr = s46.unwrap_copies(Sr, Lr, nc, nuc).reshape(-1, nuc, S.shape[2], 3)
    ref = s46.Reference(Xr, s46.Selection(nuc, Xr[0]))

    gammas = {}
    for spec in args.gamma:
        label, value = spec.split("=", 1)
        gammas[label] = float(value)
    n_bodies = args.bodies_per_copy or nuc

    results = [analyse("AA", S, L, nc, t, ref, nuc, max(args.msd_from, 100.0))]
    for spec in args.runs:
        label, path = spec.split("=", 1)
        Sc, Lc, ncc, tc = load_cg(path, args.cg_stride)
        results.append(analyse(label, Sc, Lc, ncc, tc, ref, nuc, args.msd_from))
        if label in gammas:
            # Einstein per un insieme di corpi con attrito gamma ciascuno e forze
            # interne conservative: D_COM = kT / (N gamma), esatto e indipendente
            # dalle masse (le forze interne si cancellano nel centro di massa).
            results[-1]["D_theory_nm2_per_ns"] = 1000.0 * args.kT / (n_bodies * gammas[label])

    keys = [("tau_rot", "rotazione della copia"), ("tau_Rg", "raggio di girazione"),
            ("tau_loops", "spostamenti dei loop"), ("tau_core_sites", "spostamenti del nucleo"),
            ("tau_core", "rmsd_core")]
    print("\n  corsa          frame    dt       durata")
    for r in results:
        print(f"  {r['label']:<12s} {r['frames']:6d}  {r['dt_ps']:6.3g} ps  {r['span_ps'] / 1000:7.3f} ns")
    print("\n  tempo di rilassamento (ACF a 1/e)   " + "".join(f"{r['label']:>14s}" for r in results))
    for k, name in keys:
        print(f"  {name:<34s} " + "".join(f"{fmt_tau(r[k]):>14s}" for r in results))
    print(f"  {'D traslazionale (nm^2/ns)':<34s} " + "".join(f"{r['D_nm2_per_ns']:14.4g}" for r in results))
    print(f"  {'D, fit Langevin (nm^2/ns)':<34s} " + "".join(
        f"{r['langevin_flag'] + format(r['D_langevin_nm2_per_ns'], '.4g'):>14s}" for r in results))
    if gammas:
        print(f"  {'D teorico kT/(N gamma) (nm^2/ns)':<34s} " + "".join(
            f"{r.get('D_theory_nm2_per_ns', float('nan')):14.4g}" for r in results))
    print(f"  {'tau_v del centro di massa (ps)':<34s} " + "".join(
        f"{r['langevin_flag'] + format(r['tau_v_ps'], '.3g'):>14s}" for r in results))

    aa = results[0]
    print("\n  fattore di accelerazione alpha = tau_AA / tau_CG  (D_CG / D_AA per la traslazione)")
    print("  osservabile                         " + "".join(f"{r['label']:>14s}" for r in results[1:]))
    rows = []
    for k, name in keys:
        row, vals = f"  {name:<34s} ", []
        for r in results[1:]:
            (ta, fa), (tc_, fc) = aa[k], r[k]
            a = ta / tc_
            # un limite su uno dei due tempi rende alpha un limite
            flag = ""
            if fa == "<" or fc == ">":
                flag = "<"
            if fa == ">" or fc == "<":
                flag = ">" if flag == "" else "?"
            vals.append((a, flag))
            row += f"{flag + format(a, '.3g'):>14s}"
        rows.append((name, vals))
        print(row)
    row = f"  {'traslazione':<34s} "
    for r in results[1:]:
        row += f"{r['D_nm2_per_ns'] / aa['D_nm2_per_ns']:14.3g}"
    print(row)
    # Per l'AA resta la retta: frame ogni 20 ps, il tratto balistico (sub-ps)
    # non si vede e la retta sopra 100 ps e' gia' nel regime diffusivo.
    row = f"  {'traslazione (fit Langevin)':<34s} "
    trans_l = []
    for r in results[1:]:
        a = r["D_langevin_nm2_per_ns"] / aa["D_nm2_per_ns"]
        trans_l.append((a, r["langevin_flag"]))
        row += f"{r['langevin_flag'] + format(a, '.3g'):>14s}"
    print(row)
    rows.append(("traslazione (fit Langevin)", trans_l))
    if gammas:
        row = f"  {'traslazione (teorico)':<34s} "
        trans_t = []
        for r in results[1:]:
            a = r.get("D_theory_nm2_per_ns", float("nan")) / aa["D_nm2_per_ns"]
            trans_t.append((a, ""))
            row += f"{a:14.3g}"
        print(row)
        rows.append(("traslazione (teorico)", trans_t))

    if args.ns_day_aa and args.ns_day_cg:
        print(f"\n  velocita' efficiente CG / AA = (ns/giorno CG x alpha) / ns/giorno AA "
              f"= {args.ns_day_cg:g} x alpha / {args.ns_day_aa:g}")
        for name, vals in rows:
            print(f"  {name:<34s} " + "".join(
                f"{v[1] + format(args.ns_day_cg * v[0] / args.ns_day_aa, '.3g'):>14s}" for v in vals))
        print("  (> 1: il CG campiona quel moto piu' in fretta dell'AA, a parita' di GPU)")

    print("\n  Note: alpha dipende dal moto; per i moti diffusivi scala come 1/gamma del termostato CG.")
    print("  D, fit Langevin: MSD = 6D[t - tau_v(1 - exp(-t/tau_v))] + b, dal tratto balistico in su;")
    print("  '?' = tau_v oltre un terzo della finestra, D estrapolato.  Per l'AA alpha usa la retta.")
    print("  Un tempo AA sotto --aa-dt (\"<\") non e' risolto dal dataset: alpha e' allora un limite superiore.")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
