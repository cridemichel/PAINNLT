#!/usr/bin/env python3
"""Coerenza dei contatti LJ pair-specific fra runtime ESPResSo e dataset.

Costruisce due corpi rigidi come fa run_cg_md.py (COM reale + siti virtuali
relativi), mette un contatto LJ fra un sito di ciascuno con lo stesso percorso
del runtime -- prepare_pair_specific_morse, marker virtuali, lennard_jones fra
tipi di marker -- e confronta forza e coppia sui COM con il kernel del builder
(forza 12-6 troncata a r_cut, energia spostata a zero al cutoff).  Ripete a
piu' distanze, dentro e fuori dal minimo e oltre il cutoff.

USO (con l'ambiente del framework):
    pypresso simulation/diagnose_pair_specific_lj.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import espressomd  # noqa: E402

from espresso_interactions import (  # noqa: E402
    configure_pair_specific_morse,
    create_pair_specific_morse_markers,
    prepare_pair_specific_morse,
)


def main():
    eps, sig = 17.0, 0.40
    r_cut = 3.5 * sig
    priors = {"bonds": [{"type": "lj", "mol_i": 0, "mol_j": 1, "site_i": 1, "site_j": 0,
                         "epsilon": eps, "sigma": sig, "r_cut": r_cut, "shift": "auto",
                         "exclude_wca": False}]}
    num_species = 3
    marker_types, contacts = prepare_pair_specific_morse(priors, num_species)
    assert contacts and contacts[0]["kind"] == "lj", contacts
    print(f"[INFO] contatto: {contacts[0]}")

    box = 10.0
    ok = True
    worst_f = worst_t = worst_e = 0.0
    offsets = [np.array([0.0, 0.0, 0.0]), np.array([0.25, 0.1, 0.0])]  # sito 0, sito 1 nel corpo
    # ESPResSo ammette un solo System per processo: lo si riusa, svuotandolo.
    system = espressomd.System(box_l=[box] * 3)
    system.time_step = 0.001
    system.cell_system.skin = 0.3
    for r_target in (0.40, 0.45, 0.52, 0.70, 1.00, 1.35, 1.50):
        system.part.clear()
        com_type = num_species + 1
        mol_com, mol_vs = {}, {}
        # corpo 0: sito 1 a (0.25, 0.1, 0) dal COM; corpo 1: sito 0 sul COM
        c0 = np.array([3.0, 3.0, 3.0])
        site0 = c0 + offsets[1]
        c1 = site0 + np.array([r_target, 0.0, 0.0])
        for mol, center in ((0, c0), (1, c1)):
            p = system.part.add(pos=center, type=com_type, mass=300.0,
                                rinertia=[10.0, 10.0, 10.0], rotation=[True] * 3, mol_id=mol)
            mol_com[mol] = p.id
            for s, off in enumerate(offsets):
                v = system.part.add(pos=center + off, type=s, mass=1e-5,
                                    rinertia=[1e-5] * 3, mol_id=mol)
                v.virtual = True
                v.vs_auto_relate_to(p.id)
                mol_vs[(mol, s)] = v.id
        create_pair_specific_morse_markers(system, marker_types, mol_com, mol_vs)
        configure_pair_specific_morse(system, contacts, marker_types)
        system.integrator.run(0, recalc_forces=True)

        pi = np.asarray(system.part.by_id(mol_vs[(0, 1)]).pos)
        pj = np.asarray(system.part.by_id(mol_vs[(1, 0)]).pos)
        d = pi - pj
        r = float(np.linalg.norm(d))
        if r < r_cut:
            sr6 = (sig / r) ** 6
            fs = 24.0 * eps * (2.0 * sr6 * sr6 - sr6) / r
            src6 = (sig / r_cut) ** 6
            e_ref = 4.0 * eps * (sr6 * sr6 - sr6 - (src6 * src6 - src6))
        else:
            fs, e_ref = 0.0, 0.0
        f_i = fs * d / r
        tau_i = np.cross(pi - c0, f_i)
        f_esp = np.asarray(system.part.by_id(mol_com[0]).f)
        f_esp_j = np.asarray(system.part.by_id(mol_com[1]).f)
        tau_esp = np.asarray(system.part.by_id(mol_com[0]).torque_lab)
        e_esp = float(system.analysis.energy()["non_bonded"])
        scale = max(1.0, abs(fs))
        ef = max(np.max(np.abs(f_esp - f_i)), np.max(np.abs(f_esp_j + f_i))) / scale
        et = float(np.max(np.abs(tau_esp - tau_i))) / scale
        ee = abs(e_esp - e_ref) / max(1.0, abs(e_ref))
        worst_f, worst_t, worst_e = max(worst_f, ef), max(worst_t, et), max(worst_e, ee)
        print(f"  r = {r:.3f} nm: F = {fs:10.3f} kJ/mol/nm, U = {e_ref:9.4f} kJ/mol | "
              f"errori rel. forza {ef:.1e}, coppia {et:.1e}, energia {ee:.1e}")

    ok = worst_f < 1e-6 and worst_t < 1e-6 and worst_e < 1e-6
    print("[OK] LJ pair-specific coerente fra runtime e dataset" if ok
          else f"[FAIL] forza {worst_f:.2e}, coppia {worst_t:.2e}, energia {worst_e:.2e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
