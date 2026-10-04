#!/usr/bin/env python3
"""TEL26: coda lunga delle autocorrelazioni dei loop, CG contro AA.

PERCHE'
    Lo script 47 misura il tempo in cui l'ACF scende a 1/e.  Per Rg e loop
    l'AA da' 5-7 ns e il CG 2-3 ps: un alpha nominale di ~2 400 che pero'
    confronta processi diversi.  Nell'AA (un frame ogni 20 ps) la parte veloce
    delle fluttuazioni e' gia' decaduta al primo frame e il 1/e lo decide un
    rilassamento lento (riarrangiamenti dei loop); nel CG il 1/e cade dentro
    le fluttuazioni veloci.  Qui si guarda tutta la curva:
      - ACF a ritardi fissi, da 1 ps a 5 ns (sotto i 20 ps solo il CG);
      - ampiezza lenta A_slow = ACF a 20 ps e a 100 ps: quanta varianza
        sopravvive alle fluttuazioni veloci;
      - tau integrato (integrale dell'ACF fino al primo valore sotto 0,05 o a
        un quarto della corsa) e tau della sola parte lenta (integrale da 20 ps
        diviso per l'ACF a 20 ps);
      - frazione della varianza fra copie: varianza delle medie per copia sulla
        varianza totale.  Se e' grande, le copie restano in stati diversi per
        tutta la corsa (rilassamento piu' lento della corsa stessa) e l'ACF per
        copia, che sottrae la media di ogni copia, non lo vede.
    Se nel CG A_slow ~ 0 e la varianza fra copie e' piccola, il CG non ha il
    processo lento: i loop esplorano un solo bacino liscio.

OSSERVABILI
    Rg della copia; spostamenti dei siti di ogni loop a nucleo allineato sul
    nucleo AA medio (coda 5' T1-A3, loop 1 T7-A9, loop 2 T13-A15, loop 3
    T19-A21, coda 3' T25-T26) e di tutti i loop insieme.

USO (in tutorials/tel26)
    python3 49_tel26_loop_acf.py tel26_lp1_dataset.bin \\
        prod=r0_s01.npz+r0_s02.npz,r1_s01.npz+r1_s02.npz [--json loop_acf.json]
    '+' unisce tratti consecutivi della stessa corsa, ',' separa replicas
    indipendenti (trattate come copie in piu', tagliate alla stessa lunghezza).
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

GROUPS = [("coda 5' T1-A3", (1, 2, 3)), ("loop 1 T7-A9", (7, 8, 9)),
          ("loop 2 T13-A15", (13, 14, 15)), ("loop 3 T19-A21", (19, 20, 21)),
          ("coda 3' T25-T26", (25, 26))]
LAGS_PS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
_trapz = getattr(np, "trapezoid", None) or np.trapz


def observables(parts, ref, nuc):
    """parts: lista di (S, L, nc, t).  -> dt, {nome: (T, C, K)} con K componenti."""
    T = min(p[0].shape[0] for p in parts)
    dt = float(np.median(np.diff(parts[0][3])))
    rg, disp = [], []
    for S, L, nc, _t in parts:
        L = L[:T] if np.ndim(L) == 2 else L
        X = s46.unwrap_copies(S[:T], L, nc, nuc)                       # (T, C, R, 6, 3)
        Xf = X.reshape(-1, nuc, S.shape[2], 3)
        allx = s46.take(Xf, ref.sel.all)
        rg.append(np.sqrt(((allx - allx.mean(axis=1, keepdims=True)) ** 2).sum(-1).mean(-1)).reshape(T, nc, 1))
        disp.append(s47.aligned_on_core(Xf, ref).reshape(T, nc, -1, 3))
    rg = np.concatenate(rg, axis=1)
    disp = np.concatenate(disp, axis=1)                                 # (T, C, n_all, 3)
    res = np.array([r for r, _ in ref.sel.all]) + 1                     # 1-based
    obs = {"Rg": rg}
    loop_mask = np.array([s46.SEQUENCE[r - 1] != "G" for r in res])
    obs["tutti i loop"] = disp[:, :, loop_mask].reshape(T, disp.shape[1], -1)
    for name, residues in GROUPS:
        m = np.isin(res, residues)
        obs[name] = disp[:, :, m].reshape(T, disp.shape[1], -1)
    return dt, obs


def acf_and_split(x):
    """x (T, C, K) -> ACF per copia (media su copie e componenti), frazione fra copie."""
    T, C, K = x.shape
    means = x.mean(axis=0)                                              # (C, K)
    within = ((x - means[None]) ** 2).mean(axis=0).sum(axis=-1).mean()  # media sulle copie
    between = ((means - means.mean(axis=0, keepdims=True)) ** 2).sum(axis=-1).mean()
    c = s47.acf_columns((x - means[None]).reshape(T, C * K))
    return c, float(between / (between + within))


def summarize(c, dt, span):
    tmax = span / 4
    t = np.arange(c.size) * dt
    out = {"acf": {}}
    for lag in LAGS_PS:
        k = int(round(lag / dt))
        if abs(k * dt - lag) > 0.01 * lag or k == 0 or lag > tmax:
            continue
        out["acf"][lag] = float(c[k])
    sel = t <= tmax
    below = np.flatnonzero((c < 0.05) & sel)
    end = below[0] if below.size else np.flatnonzero(sel)[-1]
    out["tau_int_ps"] = float(_trapz(c[:end + 1], t[:end + 1]))
    out["tau_int_cut"] = "" if below.size else ">"
    k20 = int(round(20.0 / dt))
    if 0 < k20 < c.size and 20.0 <= tmax:
        a20 = float(c[k20])
        out["A_slow_20"] = a20
        ok = k20 < end and a20 > 0.02
        out["tau_slow_ps"] = float(_trapz(c[k20:end + 1], t[k20:end + 1]) / a20) if ok else float("nan")
    k100 = int(round(100.0 / dt))
    if 0 < k100 < c.size and 100.0 <= tmax:
        out["A_slow_100"] = float(c[k100])
    out["tau_1e_ps"] = s47.tau_1e(t[sel], c[sel], tmax)[0]
    return out


def fmt(v, w=10, spec=".3f"):
    return f"{'-':>{w}s}" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{format(v, spec):>{w}s}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("runs", nargs="*", help="etichetta=a.npz[+b.npz][,r1.npz...]")
    ap.add_argument("--nuc", type=int, default=26)
    ap.add_argument("--aa-dt", type=float, default=20.0)
    ap.add_argument("--aa-stride", type=int, default=1)
    ap.add_argument("--cg-stride", type=int, default=1)
    ap.add_argument("--ref-stride", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cv.set_nuc(args.nuc)
    nuc = args.nuc

    S, L, nc, t = s47.load_aa(args.dataset, args.aa_dt, args.aa_stride)
    k = max(1, args.ref_stride // args.aa_stride)
    Xr = s46.unwrap_copies(S[::k], L[::k], nc, nuc).reshape(-1, nuc, S.shape[2], 3)
    ref = s46.Reference(Xr, s46.Selection(nuc, Xr[0]))

    runs = [("AA", [(S, L, nc, t)])]
    for spec in args.runs:
        label, path = spec.split("=", 1)
        parts = []
        for rep in path.split(","):
            Sc, Lc, ncc, tc = s47.load_cg(rep, args.cg_stride)
            parts.append((Sc, Lc, ncc, tc))
        runs.append((label, parts))

    results = {}
    for label, parts in runs:
        dt, obs = observables(parts, ref, nuc)
        T = obs["Rg"].shape[0]
        span = (T - 1) * dt
        results[label] = {"dt_ps": dt, "span_ps": span, "copies": obs["Rg"].shape[1], "obs": {}}
        for name, x in obs.items():
            c, fb = acf_and_split(x)
            r = summarize(c, dt, span)
            r["frac_between"] = fb
            results[label]["obs"][name] = r

    labels = [lab for lab, _ in runs]
    print("\n  corsa        copie   dt (ps)   durata (ns)")
    for lab in labels:
        r = results[lab]
        print(f"  {lab:<12s} {r['copies']:5d}   {r['dt_ps']:7.3g}   {r['span_ps'] / 1000:9.3f}")

    names = list(results["AA"]["obs"].keys())
    for name in names:
        print(f"\n  == {name} ==")
        print(f"  {'':<26s}" + "".join(f"{lab:>12s}" for lab in labels))
        for lag in LAGS_PS:
            vals = [results[lab]["obs"][name]["acf"].get(lag) for lab in labels]
            if all(v is None for v in vals):
                continue
            unit = f"{lag} ps" if lag < 1000 else f"{lag / 1000:g} ns"
            print(f"  {'ACF a ' + unit:<26s}" + "".join(fmt(v, 12) for v in vals))
        rows = [("tau a 1/e (ps)", "tau_1e_ps", ".4g"), ("tau integrato (ps)", "tau_int_ps", ".4g"),
                ("A_slow = ACF(20 ps)", "A_slow_20", ".3f"), ("ACF(100 ps)", "A_slow_100", ".3f"),
                ("tau parte lenta (ps)", "tau_slow_ps", ".4g"),
                ("varianza fra copie", "frac_between", ".3f")]
        for title, key, spec in rows:
            cells = []
            for lab in labels:
                o = results[lab]["obs"][name]
                v = o.get(key)
                s = fmt(v, 12, spec)
                if key == "tau_int_ps" and o.get("tau_int_cut"):
                    s = f"{'>' + format(v, spec):>12s}"
                cells.append(s)
            print(f"  {title:<26s}" + "".join(cells))

    print("\n  Lettura:")
    print("  - A_slow: frazione della varianza (entro copia) che sopravvive dopo 20 ps.  AA grande e CG ~0:")
    print("    il CG non ha il rilassamento lento dei loop; i tempi a 1/e misurano processi diversi.")
    print("  - tau parte lenta: integrale dell'ACF da 20 ps diviso per A_slow (solo se A_slow > 0,02).")
    print("  - varianza fra copie: quota della varianza totale dovuta a copie in stati medi diversi.")
    print("    Grande nell'AA: stati dei loop piu' lenti della corsa, non visti dall'ACF per copia.")
    print("  - '>' sul tau integrato: l'ACF non scende sotto 0,05 entro un quarto della corsa.")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
