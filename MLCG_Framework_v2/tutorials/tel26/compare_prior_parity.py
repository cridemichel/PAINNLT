#!/usr/bin/env python3
"""Parita' dei prior fra dataset e runtime, sull'intero sistema assemblato.

PERCHE'
    Il residuo che la rete impara e' F_AA - F_prior(builder); in simulazione
    la rete si somma a F_prior(runtime).  Se i due F_prior differiscono di
    dF, la dinamica ML sente F_AA + dF: un errore sistematico che non dipende
    dalla quantita' di dati e cresce con la rigidita' dei prior.  I kernel sono
    stati verificati uno per uno (diedri, LJ, DH, Morse), ma mai il sistema
    completo: marker dei contatti, esclusioni WCA, coppie dei corpi rigidi.

CONFRONTO
    builder: build_cg_dataset.py --dump-prior-forces (DUMP_PRIOR_FORCES con
             PART=parity), somma dei prior sottratti per corpo, frame 0;
    runtime: run_cg_md.py --dump_initial_forces con --disable_ml, stessa
             configurazione (frame 0 del dataset), termostato spento.
    Per ogni corpo: forza sul centro di massa e coppia nel riferimento del
    laboratorio.

USO (in tutorials/tel26)
    python3 compare_prior_parity.py prior_forces_builder.npz prior_forces_runtime.npz
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("builder")
    ap.add_argument("runtime")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--seq", default="TTAGGGTTAGGGTTAGGGTTAGGGTT")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    b = np.load(args.builder)
    r = np.load(args.runtime)
    if bool(r["ml_active"]):
        raise SystemExit("[ERROR] il dump del runtime ha il ML attivo: rifallo con --disable_ml")
    cb, fb, tb = b["centers"][args.frame], b["prior_force"][args.frame], b["prior_torque"][args.frame]
    cr, fr, tr = r["com"], r["force"], r["torque_lab"]
    if cb.shape != cr.shape:
        raise SystemExit(f"[ERROR] corpi diversi: builder {cb.shape}, runtime {cr.shape}")
    dpos = np.max(np.abs(cb - cr))
    print(f"[INFO] {len(cb)} corpi; scarto massimo delle posizioni dei centri {dpos:.2e} nm "
          f"({'stessa configurazione' if dpos < 1e-4 else 'ATTENZIONE: configurazioni diverse'})")

    nuc, seq = len(args.seq), args.seq
    kind = np.array([seq[m % nuc] for m in range(len(cb))])

    def report(name, xb, xr):
        d = np.linalg.norm(xr - xb, axis=1)
        nb = np.linalg.norm(xb, axis=1)
        rel = d / np.maximum(nb, 1.0)
        slope = float(np.sum(xr * xb) / max(np.sum(xb * xb), 1e-30))
        print(f"\n  {name}: |builder| mediana {np.median(nb):9.2f}, |runtime - builder| mediana "
              f"{np.median(d):9.3e}, massimo {d.max():9.3e}; pendenza runtime/builder {slope:.6f}")
        for k in sorted(set(kind)):
            sel = kind == k
            print(f"    {k}: scarto relativo mediano {np.median(rel[sel]):.2e}, massimo {rel[sel].max():.2e}")
        worst = np.argsort(-d)[:args.top]
        print(f"    corpi peggiori (residuo, copia, |builder|, |runtime|, scarto):")
        for i in worst:
            print(f"      {i % nuc + 1:>2}{seq[i % nuc]} copia {i // nuc + 1:>2}  {nb[i]:10.2f} "
                  f"{np.linalg.norm(xr[i]):10.2f} {d[i]:10.3e}")
        return float(np.max(rel))

    wf = report("forza sul centro di massa", fb, fr)
    wt = report("coppia (laboratorio)", tb, tr)
    ok = wf < 1e-3 and wt < 1e-3
    print(f"\n[{'OK' if ok else 'FAIL'}] prior del runtime {'coincidenti' if ok else 'DIVERSI'} da quelli "
          f"sottratti nel dataset (soglia 1e-3 relativo): forza {wf:.1e}, coppia {wt:.1e}")


if __name__ == "__main__":
    main()
