#!/usr/bin/env python3
"""Coerenza del diedro a coseno fra runtime ESPResSo e dataset.

Il builder (build_cg_dataset.get_dihedral) definisce phi con b1 = p2-p1,
b2 = p3-p2, b3 = p4-p3 e atan2 in (-pi, pi]; run_cg_md.py mette il legame
Dihedral(bend=K, mult=n, phase=phi0) sulla SECONDA particella con partner
(p1, p3, p4), ed ESPResSo calcola phi in [0, 2 pi).  Le due definizioni
devono coincidere a meno di 2 pi, altrimenti la fase phi0 stimata dal
riferimento verrebbe applicata a un angolo diverso -- senza alcun errore.

Controlla energia e forze su configurazioni casuali, per piu' fasi e
molteplicita', contro l'energia del builder e il suo gradiente numerico.

USO (con l'ambiente del framework):
    pypresso simulation/diagnose_dihedral.py
"""
from __future__ import annotations

import sys

import numpy as np

import espressomd
import espressomd.interactions


def builder_phi(p1, p2, p3, p4):
    # copia di build_cg_dataset.get_dihedral (senza immagine minima: box grande)
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    m1, m2 = np.cross(b1, b2), np.cross(b2, b3)
    cos_phi = np.clip(np.dot(m1, m2) / np.sqrt(np.dot(m1, m1) * np.dot(m2, m2)), -1.0, 1.0)
    sin_phi = np.dot(b2, np.cross(m1, m2)) / (np.linalg.norm(b2) * np.sqrt(np.dot(m1, m1) * np.dot(m2, m2)))
    return np.arctan2(sin_phi, cos_phi)


def sin3(u, w):
    """Smorzamento angolare della CBT (come build_cg_dataset._cbt_factor, S0 = 0,3)."""
    c = np.clip(-np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)), -1.0, 1.0)
    x = (1.0 - c * c) / 0.09
    return 1.0 if x >= 1.0 else x * (2.0 - x)


def builder_energy(pos, K, n, phi0, cbt=False):
    e = K * (1.0 - np.cos(n * builder_phi(*pos) - phi0))
    if cbt:  # come build_cg_dataset.dihedral_energy(..., cbt=True)
        b1, b2, b3 = pos[1] - pos[0], pos[2] - pos[1], pos[3] - pos[2]
        e *= sin3(b1, b2) * sin3(b2, b3)
    return e


def builder_forces(pos, K, n, phi0, cbt=False, h=1e-6):
    f = np.zeros_like(pos)
    for a in range(4):
        for x in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[a, x] += h
            pm[a, x] -= h
            f[a, x] = -(builder_energy(pp, K, n, phi0, cbt) - builder_energy(pm, K, n, phi0, cbt)) / (2 * h)
    return f


def main():
    rng = np.random.default_rng(3)
    system = espressomd.System(box_l=[20.0] * 3)
    system.time_step = 0.001
    system.cell_system.skin = 0.3
    worst_e = worst_f = 0.0
    n_tests = 0
    worst_cbt_f = 0.0
    for n, cbt in ((1, False), (2, False), (1, True), (2, True)):
        for phi0 in (-2.5, -0.4, 0.0, 0.9, 2.8):
            K = 12.0
            # CBT (install_dihedral_cbt.py): mult < 0 in ESPResSo, cbt=True nel builder
            bond = espressomd.interactions.Dihedral(bend=K, mult=-n if cbt else n, phase=phi0)
            system.bonded_inter.add(bond)
            for trial in range(8 if cbt else 6):
                system.part.clear()
                pos = 10.0 + rng.normal(scale=0.4, size=(4, 3))
                if cbt and trial >= 4:
                    # quasi allineati: la forza del coseno semplice diverge, la CBT no
                    d = pos[2] - pos[1]
                    pos[0] = pos[1] - 0.8 * d + rng.normal(scale=10.0 ** -(1 + trial - 4), size=3)
                parts = [system.part.add(pos=x) for x in pos]
                # come run_cg_md.py: legame sulla SECONDA particella, partner (p1, p3, p4)
                parts[1].add_bond((bond, parts[0], parts[2], parts[3]))
                system.integrator.run(0, recalc_forces=True)
                e_esp = float(system.analysis.energy()["bonded"])
                f_esp = np.asarray([p.f for p in parts])
                e_ref = builder_energy(pos, K, n, phi0, cbt)
                f_ref = builder_forces(pos, K, n, phi0, cbt)
                if cbt:
                    worst_cbt_f = max(worst_cbt_f, float(np.max(np.abs(f_esp))))
                worst_e = max(worst_e, abs(e_esp - e_ref) / max(1.0, abs(e_ref)))
                worst_f = max(worst_f, float(np.max(np.abs(f_esp - f_ref))) / max(1.0, float(np.max(np.abs(f_ref)))))
                n_tests += 1
    print(f"  {n_tests} configurazioni, n = 1, 2, cinque fasi, coseno e CBT: errore relativo "
          f"massimo energia {worst_e:.1e}, forza {worst_f:.1e}")
    print(f"  CBT anche con tre siti quasi allineati: forza massima {worst_cbt_f:.1f} (K = 12)")
    ok = worst_e < 1e-6 and worst_f < 1e-5
    print("[OK] diedro a coseno coerente fra runtime e dataset" if ok
          else "[FAIL] convenzioni diverse: la fase stimata dal riferimento non sarebbe quella simulata")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
