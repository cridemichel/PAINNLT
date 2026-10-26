#!/usr/bin/env python3
"""Catena di perturbazioni per L(T) = ln(P_U/P_N) lungo l'allenamento (19_relent_pop.sh, POP_CHAIN=1).

    L(theta_n, T) = L(theta_0, T) + sum_{j<=n} dL_j(T),
    dL_j(T) = L(theta_j) - L(theta_j-1) dalla ripesatura dentro i bacini (train_relent --pop,
              campioni dei due lignaggi a T generati da theta_j-1; report: pop_final L - L0).

L(theta_0, T) viene da 22_anchor_L0.py (soli prior: fit a due stati sulle corse di 13).  Il file
della catena tiene L su una griglia (300-360 K ogni K, piu' le T della scala): alle T senza
entrambi i bacini in un'iterazione dL si interpola dalle T misurate con dL = a + b/T (una sola
T misurata: costante).  Iterazione senza allenamento (modello copiato): dL = 0.

    19_pop_chain.py init   --chain C.json --anchor A.json --ts "315 318 ..."
    19_pop_chain.py get    --chain C.json --it N --T T          -> stampa L(theta_N, T)
    19_pop_chain.py update --chain C.json --prev P --it N [--report R.json] --ts "..."
    19_pop_chain.py show   --chain C.json --it N --ts "..."
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

R_KJ = 0.008314462618


def grid(ts):
    g = {float(t) for t in range(300, 361)}
    g |= {float(t) for t in ts}
    return sorted(g)


def key(T):
    return f"{float(T):.2f}"


def load(path):
    with open(path) as f:
        return json.load(f)


def save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def interp(table, T):
    pts = sorted((float(k), v) for k, v in table.items())
    if key(T) in table:
        return table[key(T)]
    xs = [p[0] for p in pts]
    if T <= xs[0]:
        return pts[0][1]
    if T >= xs[-1]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= T <= x1:
            return y0 + (y1 - y0) * (T - x0) / (x1 - x0)
    raise RuntimeError("interpolazione fallita")


def exp_L(T):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("fit_thermal", pathlib.Path(__file__).resolve().parent / "11_fit_thermal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = float(mod.exp_pops([T])[0][3])
    return math.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("init", "get", "update", "show"))
    ap.add_argument("--chain", required=True)
    ap.add_argument("--anchor")
    ap.add_argument("--ts", default="")
    ap.add_argument("--it", type=int)
    ap.add_argument("--prev", type=int)
    ap.add_argument("--T", type=float)
    ap.add_argument("--report")
    a = ap.parse_args()
    ts = [float(x) for x in a.ts.split()]

    if a.cmd == "init":
        if os.path.exists(a.chain):
            print(f"[chain] esiste gia': {a.chain}")
            return
        an = load(a.anchor)
        Tm, dH = float(an["Tm_K"]), float(an["dH_kJ"])
        table = {key(T): -(dH / R_KJ) * (1.0 / T - 1.0 / Tm) for T in grid(ts)}
        save(a.chain, {"anchor": os.path.abspath(a.anchor), "Tm0_K": Tm, "dH0_kJ": dH,
                       "L": {"0": table}, "measured": {"0": {}}})
        print(f"[chain] inizializzata da {a.anchor}: Tm(theta_0) {Tm:.2f} K, dH {dH:.1f} kJ/mol")
        return

    data = load(a.chain)
    if a.cmd == "get":
        t = data["L"].get(str(a.it))
        if t is None:
            sys.exit(f"[chain] manca l'iterazione {a.it} in {a.chain}")
        print(f"{interp(t, a.T):.6f}")
        return

    if a.cmd == "update":
        old = data["L"].get(str(a.prev))
        if old is None:
            sys.exit(f"[chain] manca l'iterazione {a.prev} in {a.chain}")
        meas = {}
        if a.report:
            r = load(a.report)
            for q, p in zip(r.get("pop", []), r.get("pop_final", [])):
                if q.get("L0_abs") is None:
                    sys.exit("[chain] il report non ha L0_abs: --pop senza sesto campo?")
                meas[float(p["T_K"])] = float(p["L"]) - float(p["L0"])
        Ts = sorted(set(float(k) for k in old) | set(grid(ts)))
        if len(meas) >= 2:
            xs = [1.0 / T for T in meas]
            ys = list(meas.values())
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
            dl = lambda T: my + b * (1.0 / T - mx)
        elif len(meas) == 1:
            v = next(iter(meas.values()))
            dl = lambda T: v
        else:
            dl = lambda T: 0.0
        new = {}
        for T in Ts:
            base = interp(old, T)
            new[key(T)] = base + (meas[T] if T in meas else dl(T))
        data["L"][str(a.it)] = new
        data.setdefault("measured", {})[str(a.it)] = {key(T): v for T, v in meas.items()}
        save(a.chain, data)
        s = "  ".join(f"{T:g}: {v:+.3f}" for T, v in sorted(meas.items()))
        print(f"[chain] it{a.it:02d}: dL misurato {s if s else '(nessuno: modello invariato)'}")
        return

    if a.cmd == "show":
        t = data["L"].get(str(a.it))
        if t is None:
            sys.exit(f"[chain] manca l'iterazione {a.it}")
        m = data.get("measured", {}).get(str(a.it), {})
        row = []
        for T in ts:
            L = interp(t, T)
            row.append(f"{T:g} K P_U {1 / (1 + math.exp(-L)):.3f} (esp {1 / (1 + math.exp(-exp_L(T))):.3f}){'' if key(T) in m or a.it == 0 else '*'}")
        print(f"[CHAIN] it{a.it:02d}  " + "  ".join(row) + ("   (* = dL interpolato)" if a.it > 0 else ""))


if __name__ == "__main__":
    main()
