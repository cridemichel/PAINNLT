#!/usr/bin/env python3
"""Coerenza fra il Debye-Hueckel del runtime ESPResSo e quello del dataset.

Il builder del dataset (preprocessing/build_cg_dataset.py) sottrae dalle forze
di riferimento un Debye-Hueckel scritto in NumPy (prior_kernels); il runtime lo
fa calcolare al solver DH di ESPResSo.  Se i due differiscono, il residuo ML
viene allenato contro un prior e simulato sopra un altro -- senza errori.

Controlla tre cose su pochi siti carichi in una scatola periodica:
  1. forze ESPResSo == kernel NumPy, coppia per coppia (anche oltre r_cut);
  2. energia ESPResSo == somma delle energie di coppia;
  3. le esclusioni di particella NON spengono l'elettrostatica: e' l'ipotesi
     su cui il builder somma TUTTE le coppie cariche, legate comprese.

USO (con l'ambiente del framework):
    pypresso simulation/diagnose_debye_huckel.py [--priors cg_priors.X.json]
Senza --priors usa lambda_D = 0.9 nm, eps_r = 78, r_cut = 3.5 nm.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "preprocessing"))
from prior_kernels import (  # noqa: E402
    debye_huckel_energy_array,
    debye_huckel_radial_force_array,
    normalize_debye_huckel,
)

import espressomd  # noqa: E402
import espressomd.electrostatics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priors", default=None)
    ap.add_argument("--tol", type=float, default=1e-6, help="tolleranza relativa")
    args = ap.parse_args()

    if args.priors:
        dh = normalize_debye_huckel(json.loads(Path(args.priors).read_text())["debye_huckel"])
    else:
        dh = normalize_debye_huckel({"charges_by_type": {"0": -1.0}, "prefactor": 138.935458 / 78.0,
                                     "kappa": 1.0 / 0.9, "r_cut": 3.5})
    pref, kappa, r_cut = dh["prefactor"], dh["kappa"], dh["r_cut"]
    print(f"[INFO] prefactor={pref:.6f} kJ/mol nm, kappa={kappa:.6f} 1/nm, r_cut={r_cut:.4f} nm")

    box = 12.0
    system = espressomd.System(box_l=[box] * 3)
    system.time_step = 0.001
    system.cell_system.skin = 0.4
    rng = np.random.default_rng(7)
    pos = rng.uniform(0.0, box, size=(12, 3))
    # due coppie vicine a distanze note, e una oltre il cutoff
    pos[1] = pos[0] + [0.6, 0.0, 0.0]
    pos[3] = pos[2] + [0.0, 1.1, 0.0]
    pos[5] = pos[4] + [0.0, 0.0, r_cut + 0.3]
    q = np.where(np.arange(12) % 3 == 0, 1.0, -1.0)
    for x, qi in zip(pos, q):
        system.part.add(pos=x, q=float(qi))
    system.electrostatics.solver = espressomd.electrostatics.DH(
        prefactor=pref, kappa=kappa, r_cut=r_cut)

    def reference():
        f = np.zeros_like(pos)
        e = 0.0
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = pos[i] - pos[j]
                d -= box * np.round(d / box)
                r = np.linalg.norm(d)
                fs = debye_huckel_radial_force_array([r], [q[i] * q[j]], pref, kappa, r_cut)[0]
                f[i] += fs * d / r
                f[j] -= fs * d / r
                e += debye_huckel_energy_array([r], [q[i] * q[j]], pref, kappa, r_cut)[0]
        return f, e

    f_ref, e_ref = reference()
    system.integrator.run(0, recalc_forces=True)
    f_esp = np.asarray([p.f for p in system.part])
    e_esp = float(system.analysis.energy()["coulomb"])
    scale = float(np.max(np.abs(f_ref)))
    ok = True

    err_f = float(np.max(np.abs(f_esp - f_ref))) / scale
    print(f"  1. forze: errore relativo massimo {err_f:.2e}")
    ok &= err_f < args.tol
    err_e = abs(e_esp - e_ref) / max(abs(e_ref), 1e-12)
    print(f"  2. energia: ESPResSo {e_esp:.8f}, NumPy {e_ref:.8f}, errore relativo {err_e:.2e}")
    ok &= err_e < args.tol

    system.part.by_id(0).add_exclusion(1)
    system.part.by_id(2).add_exclusion(3)
    system.integrator.run(0, recalc_forces=True)
    f_exc = np.asarray([p.f for p in system.part])
    err_x = float(np.max(np.abs(f_exc - f_ref))) / scale
    print(f"  3. con esclusioni 0-1, 2-3: scarto dalle forze senza esclusioni {err_x:.2e}")
    if err_x < args.tol:
        print("     le esclusioni NON agiscono sull'elettrostatica: il builder deve sommare tutte le coppie")
    else:
        print("     [ATTENZIONE] le esclusioni spengono l'elettrostatica in questa build: "
              "il builder, che somma tutte le coppie, NON e' coerente col runtime")
        ok = False

    print("[OK] Debye-Hueckel coerente fra runtime e dataset" if ok else "[FAIL] incoerenza")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
