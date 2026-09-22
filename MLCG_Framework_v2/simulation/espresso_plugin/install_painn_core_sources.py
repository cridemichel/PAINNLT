#!/usr/bin/env python3
"""Aggiunge PaiNN_ML_Potential.cpp ai sorgenti del core di ESPResSo.

PERCHE' SERVE
    copy_plugin_files.sh copia PaiNN_ML_Potential.{hpp,cpp} dentro
    src/core/nonbonded_interactions/, ma ESPResSo elenca i propri sorgenti
    esplicitamente in target_sources(): un file copiato nella directory non
    viene compilato se non compare in quella lista.

    Senza questo passo il core non contiene global_painn_potential e il modulo
    Python muore all'import:

        ImportError: painn.so: undefined symbol: global_painn_potential

    Su macOS il sintomo non si vede in fase di build, perche' ESPResSo linka i
    moduli Cython con -undefined dynamic_lookup e la risoluzione slitta al
    runtime; su Linux il link e' stretto e l'errore arriva all'import.

E' idempotente: se il sorgente e' gia' elencato non tocca nulla.
"""
import argparse
import pathlib
import re
import sys

SOURCE = "PaiNN_ML_Potential.cpp"
ANCHOR = "wca.cpp"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--espresso-root", required=True)
    args = ap.parse_args()

    cmake = (pathlib.Path(args.espresso_root)
             / "src" / "core" / "nonbonded_interactions" / "CMakeLists.txt")
    if not cmake.is_file():
        sys.exit(f"[ERROR] non trovo {cmake}: l'albero di ESPResSo non ha il layout atteso")

    text = cmake.read_text()
    if SOURCE in text:
        print(f"[SKIP] {SOURCE} e' gia' fra i sorgenti del core")
        return

    # Si inserisce prima di wca.cpp, che chiude la lista: cosi' l'ultima riga
    # mantiene la parentesi di chiusura e il file resta valido.
    pattern = re.compile(r"^(\s*)(\$\{CMAKE_CURRENT_SOURCE_DIR\}/" + re.escape(ANCHOR) + r")",
                         re.MULTILINE)
    match = pattern.search(text)
    if not match:
        sys.exit(
            f"[ERROR] non trovo il riferimento a {ANCHOR} in {cmake}.\n"
            f"        Aggiungi a mano ${{CMAKE_CURRENT_SOURCE_DIR}}/{SOURCE} "
            f"alla lista target_sources() del core."
        )
    indent = match.group(1)
    insertion = f"{indent}${{CMAKE_CURRENT_SOURCE_DIR}}/{SOURCE}\n"
    text = text[:match.start()] + insertion + text[match.start():]
    cmake.write_text(text)
    print(f"[PASS] {SOURCE} aggiunto ai sorgenti del core in {cmake}")


if __name__ == "__main__":
    main()
