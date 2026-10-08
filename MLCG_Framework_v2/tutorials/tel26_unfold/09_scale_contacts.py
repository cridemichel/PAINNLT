#!/usr/bin/env python3
"""Scala le profondita' dei contatti Morse pair-specific di un insieme di prior.

Due classi, riconosciute dalla geometria (system.json di tel26):
  tet  contatti fra due guanine della STESSA tetrade (B3-B3 e Hoogsteen N2-N7):
       30 per copia in lp2;
  oth  tutti gli altri contatti Morse (stacking fra tetradi, loop): 24 per copia.
D -> lam_tet * D e D -> lam_oth * D; a, r0, r_cut invariati (la coda smorzata a r_cut
resta la stessa forma).

Torsione di impilamento (diedri B3-B5-B5-B3, ruolo "twist", 8 per copia, K 157-310 kT):
  - passa alla forma smorzata CBT ("cbt": true).  Nel nativo gli angoli B3-B5-B5 sono
    70-98 gradi, g = 1 e nulla cambia; quando due guanine si separano un angolo puo' andare
    verso 0/180 gradi e il diedro a coseno da' forze ~K/sin(t): e' la causa dei crash della
    prima scansione (max_f > 10^4 ed E_kin a 10^4 in un intervallo di log, solo dove le
    tetradi si aprono);
  - K -> lam_twist * K, con lam_twist = lam_oth per default (e' un termine di impilamento).
Bond armonici, angoli, diedri di backbone e WCA non vengono toccati.

PRIOR DIPENDENTI DA T (energie libere di contatto)
  Al posto di un fattore costante, per ciascuna classe (tet, oth, twist) si possono dare
  h e T0:  D(T) = h D0 (1 - T/T0),  cioe'  D_H = h D0,  D_S = h D0 / T0
  (per le torsioni lo stesso su K).  h fissa la scala entalpica (sum D_H da confrontare con
  il dH di unfolding sperimentale, 352 kJ/mol), T0 la temperatura a cui il contatto si
  annulla.  Il file scrive D_H/D_S (k_H/k_S) e D = null: run_cg_md.py ed equilibrate.py
  calcolano D(T) alla temperatura del termostato (espresso_interactions.resolve_thermal_priors);
  un consumatore che non lo fa fallisce invece di usare una D sbagliata.

Uso:
  python3 09_scale_contacts.py --in cg_priors.lp2_1c.json --out cg_priors.lp2_1c.lt0.15_lo0.20.json \\
      --lam-tet 0.15 --lam-oth 0.20
  python3 09_scale_contacts.py --in cg_priors.lp2_1c.json --out cg_priors.lp2_1c.thermo.json \
      --h-tet 0.25 --t0-tet 340 --h-oth 0.20 --t0-oth 360      # twist segue oth
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from system_config import load_system  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    for c in ("tet", "oth", "twist"):
        ap.add_argument(f"--lam-{c}", type=float, default=None, help=f"fattore costante per {c}")
        ap.add_argument(f"--h-{c}", type=float, default=None, help=f"scala entalpica h per {c} (con --t0-{c})")
        ap.add_argument(f"--t0-{c}", type=float, default=None, help=f"temperatura (K) a cui {c} si annulla")
    ap.add_argument("--no-twist-cbt", action="store_true", help="lascia le torsioni a coseno puro")
    ap.add_argument("--report-T", default="300,330,360,400", help="temperature per il riepilogo di D(T)")
    args = ap.parse_args()
    spec = {}
    for c in ("tet", "oth", "twist"):
        lam, h, t0 = getattr(args, f"lam_{c}"), getattr(args, f"h_{c}"), getattr(args, f"t0_{c}")
        if lam is not None and (h is not None or t0 is not None):
            raise SystemExit(f"[ERROR] {c}: --lam-{c} oppure --h-{c}/--t0-{c}, non entrambi")
        if (h is None) != (t0 is None):
            raise SystemExit(f"[ERROR] {c}: --h-{c} e --t0-{c} vanno insieme")
        if lam is not None:
            if lam < 0:
                raise SystemExit(f"[ERROR] {c}: fattore negativo")
            spec[c] = ("lam", lam, None)
        elif h is not None:
            if h < 0 or t0 <= 0:
                raise SystemExit(f"[ERROR] {c}: h >= 0 e T0 > 0")
            spec[c] = ("thermo", h, t0)
    if "twist" not in spec and "oth" in spec:
        spec["twist"] = spec["oth"]
    for c in ("tet", "oth"):
        if c not in spec:
            raise SystemExit(f"[ERROR] manca la scala per {c}: --lam-{c} oppure --h-{c} --t0-{c}")
    report_T = [float(t) for t in args.report_T.split(",")]

    def apply(entry, key, mode):
        """Scala entry[key] (D o k) secondo la specifica; restituisce D0."""
        x0 = float(entry[key])
        kind, a, t0 = mode
        if kind == "lam":
            entry[key] = x0 * a
        else:
            entry[f"{key}_H"] = a * x0
            entry[f"{key}_S"] = a * x0 / t0
            entry[key] = None
        return x0

    def at_T(x0, mode, T):
        kind, a, t0 = mode
        return x0 * a if kind == "lam" else max(a * x0 * (1.0 - T / t0), 0.0)

    sysc = load_system(HERE.parent / "tel26")
    nuc = sysc.nuc
    tetrad_of = {r - 1: k for k, t in enumerate(sysc.tetrads) for r in t}
    pri = json.load(open(args.inp))
    stats = collections.defaultdict(lambda: [0, 0.0, [0.0] * len(report_T), 0.0])
    for e in pri["bonds"]:
        if e.get("type") != "morse":
            continue
        ri, rj = e["mol_i"] % nuc, e["mol_j"] % nuc
        if e["mol_i"] // nuc != e["mol_j"] // nuc:
            raise SystemExit(f"[ERROR] contatto Morse fra copie diverse: {e}")
        cls = "tet" if (ri in tetrad_of and rj in tetrad_of and tetrad_of[ri] == tetrad_of[rj]) else "oth"
        x0 = apply(e, "D", spec[cls])
        s = stats[cls]
        s[0] += 1; s[1] += x0
        s[2] = [v + at_T(x0, spec[cls], T) for v, T in zip(s[2], report_T)]
        s[3] += (e["D_H"] if "D_H" in e else 0.0)
    if not stats:
        raise SystemExit("[ERROR] nessun contatto Morse nei bond: insieme di prior inatteso")
    for d in pri.get("dihedrals", []):
        if d.get("role") != "twist":
            continue
        x0 = apply(d, "k", spec["twist"])
        s = stats["twist"]
        s[0] += 1; s[1] += x0
        s[2] = [v + at_T(x0, spec["twist"], T) for v, T in zip(s[2], report_T)]
        s[3] += (d["k_H"] if "k_H" in d else 0.0)
        if not args.no_twist_cbt:
            d["cbt"] = True
    if pri.get("morse_type_pairs"):
        print("[WARN] morse_type_pairs non vuoto: NON scalato (solo i contatti pair-specific)")
    ncopy = max(e["mol_i"] for e in pri["bonds"]) // nuc + 1
    meta = pri.setdefault("derived_prior_set", {})
    meta.setdefault("base", pathlib.Path(args.inp).name)
    meta["contact_scaling"] = {
        c: ({"lam": spec[c][1]} if spec[c][0] == "lam" else {"h": spec[c][1], "T0_K": spec[c][2]}) for c in spec}
    meta["contact_scaling"]["twist_cbt"] = not args.no_twist_cbt
    meta["contact_scaling"]["classes"] = ("tet = stessa tetrade (B3-B3, Hoogsteen); oth = stacking, loop; "
                                          "twist = torsioni B3-B5-B5-B3 (K)")
    meta["contact_scaling"]["thermal"] = any(spec[c][0] == "thermo" for c in spec)
    json.dump(pri, open(args.out, "w"), indent=1)
    hdr = "".join(f"{T:>9.0f} K" for T in report_T)
    print(f"  per copia (kJ/mol)         n   originale   sum D_H  {hdr}")
    tot = [0.0] * len(report_T); totH = 0.0
    for cls in ("tet", "oth", "twist"):
        if cls not in stats:
            continue
        n, x0, xs, xh = stats[cls]
        kind, a, t0 = spec[cls]
        lab = f"lam {a:g}" if kind == "lam" else f"h {a:g}, T0 {t0:g} K"
        print(f"  {cls:5s} {lab:16s} {n // ncopy:3d} {x0 / ncopy:10.1f} {xh / ncopy:9.1f}  "
              + "".join(f"{v / ncopy:11.1f}" for v in xs))
        if cls != "twist":
            tot = [t + v / ncopy for t, v in zip(tot, xs)]; totH += xh / ncopy
    print(f"  Morse totali                                {totH:9.1f}  " + "".join(f"{v:11.1f}" for v in tot)
          + "\n  (dH di unfolding sperimentale 352 kJ/mol, Tm 328 K)  -> " + args.out)


if __name__ == "__main__":
    main()
