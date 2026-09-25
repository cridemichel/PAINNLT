#!/usr/bin/env python3
"""Diedro indefinito (tre siti allineati): forza ed energia nulle invece dell'errore.

ESPResSo, quando |b1 x b2| o |b2 x b3| scende sotto 1e-4 nm^2, dichiara il
diedro indefinito.  Il suo commento dice "force zero", ma DihedralBond::forces
restituisce un optional vuoto e il chiamante lo tratta come legame rotto:
l'integrazione si ferma con "bond broken between particles ...".

Nel TEL26 con i diedri di backbone sui siti 0 di quattro nucleotidi
consecutivi succede nella coda al 3' (residui 23-26, copia 4, particelle
431, 438, 445, 447): una volta in circa 300 ps di corsa.  Coi seed fissi il
crash si ripete identico a ogni rilancio.  Nell'intorno dell'allineamento la
direzione del diedro non esiste; forza nulla nel passo in cui e' indefinito e'
la regolarizzazione che il commento di ESPResSo stesso dichiara.

Idempotente, e fallisce se non riconosce il sorgente.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SENTINEL = "MLCG: diedro indefinito"

FORCE_OLD = """  /* dihedral angle not defined - force zero */
  if (angle_is_undefined) {
    return {};
  }"""
FORCE_NEW = """  /* dihedral angle not defined - force zero */
  /* MLCG: diedro indefinito -> forza nulla, non legame rotto */
  if (angle_is_undefined) {
    return std::make_tuple(Utils::Vector3d{}, Utils::Vector3d{},
                           Utils::Vector3d{}, Utils::Vector3d{});
  }"""
ENERGY_OLD = """  /* dihedral angle not defined - energy zero */
  if (angle_is_undefined) {
    return {};
  }"""
ENERGY_NEW = """  /* dihedral angle not defined - energy zero */
  /* MLCG: diedro indefinito -> energia nulla, non legame rotto */
  if (angle_is_undefined) {
    return 0.;
  }"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--espresso-root", required=True, type=Path)
    args = ap.parse_args()
    path = args.espresso_root / "src/core/bonded_interactions/dihedral.hpp"
    if not path.is_file():
        raise SystemExit(f"[ERROR] non trovo {path}")
    text = path.read_text()
    if text.count(SENTINEL) == 2:
        print(f"[SKIP] diedro indefinito gia' a forza nulla: {path}")
        return
    if text.count(FORCE_OLD) != 1 or text.count(ENERGY_OLD) != 1:
        raise SystemExit(f"[ERROR] {path}: sorgente non riconosciuto, patch non applicata")
    text = text.replace(FORCE_OLD, FORCE_NEW).replace(ENERGY_OLD, ENERGY_NEW)
    path.write_text(text)
    print(f"[PASS] diedro indefinito a forza ed energia nulle: {path}")


if __name__ == "__main__":
    main()
