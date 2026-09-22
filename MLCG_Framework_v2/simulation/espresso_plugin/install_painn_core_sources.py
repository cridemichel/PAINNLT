#!/usr/bin/env python3
"""Innesta il potenziale PaiNN nel core di ESPResSo.

COSA FA, E PERCHE' NON BASTA COPIARE I FILE

copy_plugin_files.sh copia PaiNN_ML_Potential.{hpp,cpp} dentro
src/core/nonbonded_interactions/, ma un file copiato non e' un file
compilato, e un file compilato non e' un potenziale attivo.  Servono tre
innesti nell'albero di ESPResSo, che questo script applica in modo
idempotente:

  1. src/core/nonbonded_interactions/CMakeLists.txt
     ESPResSo elenca i sorgenti esplicitamente in target_sources(): senza
     questa riga PaiNN_ML_Potential.cpp non viene compilato e il modulo
     Python muore all'import con
         undefined symbol: global_painn_potential

  2. src/core/CMakeLists.txt
     find_package(Torch), il link a ${TORCH_LIBRARIES} e la define
     ESPRESSO_PAINN.  Senza il link, espresso_core.so resta con i simboli di
     LibTorch irrisolti:
         undefined symbol: _ZTIN3c105ErrorE   (typeinfo for c10::Error)

  3. src/core/forces.cpp
     La chiamata a global_painn_potential->calculate_forces() dentro
     System::calculate_forces().  E' il punto in cui il modello entra
     davvero nella dinamica: senza, ESPResSo compila, linka, importa e gira
     -- con i soli prior, senza dire nulla.  E' l'innesto che si nota di
     meno e conta di piu'.

Su macOS il primo e il secondo difetto non si manifestano in fase di build,
perche' i moduli Cython vengono linkati con -undefined dynamic_lookup e la
risoluzione slitta al runtime; il terzo non si manifesta affatto, se non nei
risultati.
"""
import argparse
import pathlib
import sys

SOURCE = "PaiNN_ML_Potential.cpp"


def patch(path, marker, anchor, insertion, what):
    """Inserisce `insertion` prima di `anchor`, se `marker` non c'e' gia'."""
    if not path.is_file():
        sys.exit(f"[ERROR] non trovo {path}: l'albero di ESPResSo non ha il layout atteso")
    text = path.read_text()
    if marker in text:
        print(f"[SKIP] {what}: gia' presente")
        return False
    if anchor not in text:
        sys.exit(
            f"[ERROR] {what}: non trovo il punto di innesto in {path}.\n"
            f"        Atteso:\n{anchor}\n"
            f"        L'albero di ESPResSo e' diverso da quello su cui il plugin e' validato "
            f"(commit 84cc1d924)."
        )
    path.write_text(text.replace(anchor, insertion + anchor, 1))
    print(f"[PASS] {what}: innestato in {path}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--espresso-root", required=True)
    args = ap.parse_args()
    root = pathlib.Path(args.espresso_root)
    core = root / "src" / "core"

    # ── 1. il sorgente entra nella compilazione del core ────────────────────
    patch(
        core / "nonbonded_interactions" / "CMakeLists.txt",
        marker=SOURCE,
        anchor="          ${CMAKE_CURRENT_SOURCE_DIR}/wca.cpp)",
        insertion=f"          ${{CMAKE_CURRENT_SOURCE_DIR}}/{SOURCE}\n",
        what="sorgente del potenziale",
    )

    # ── 2. il core linka LibTorch e definisce ESPRESSO_PAINN ────────────────
    cmake_core = core / "CMakeLists.txt"
    patch(
        cmake_core,
        marker="find_package(Torch",
        anchor="target_link_libraries(\n  espresso_core",
        insertion="find_package(Torch REQUIRED)\n\n",
        what="find_package(Torch)",
    )
    text = cmake_core.read_text()
    if '"${TORCH_LIBRARIES}"' not in text:
        anchor = "         $<$<BOOL:${ESPRESSO_BUILD_WITH_CUDA}>:OpenMP::OpenMP_CUDA>\n"
        if anchor not in text:
            sys.exit(f"[ERROR] non trovo dove aggiungere TORCH_LIBRARIES in {cmake_core}")
        text = text.replace(anchor, anchor + '         "${TORCH_LIBRARIES}"\n', 1)
        cmake_core.write_text(text)
        print(f"[PASS] link a TORCH_LIBRARIES: innestato in {cmake_core}")
    else:
        print("[SKIP] link a TORCH_LIBRARIES: gia' presente")
    patch(
        cmake_core,
        marker="ESPRESSO_PAINN",
        anchor="target_include_directories(espresso_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})",
        insertion="target_compile_definitions(espresso_core PUBLIC ESPRESSO_PAINN)\n\n",
        what="define ESPRESSO_PAINN",
    )

    # ── 3. la chiamata dentro System::calculate_forces() ────────────────────
    forces = core / "forces.cpp"
    patch(
        forces,
        marker="PaiNN_ML_Potential.hpp",
        anchor='#include <utils/Vector.hpp>',
        insertion=('#ifdef ESPRESSO_PAINN\n'
                   '#include "nonbonded_interactions/PaiNN_ML_Potential.hpp"\n'
                   '#endif\n\n'),
        what="include in forces.cpp",
    )
    patch(
        forces,
        marker="global_painn_potential->calculate_forces",
        anchor="  constraints->add_forces(particles, get_sim_time());",
        insertion=('#ifdef ESPRESSO_PAINN\n'
                   '  if (global_painn_potential) {\n'
                   '    global_painn_potential->calculate_forces(*cell_structure, verlet_criterion);\n'
                   '  }\n'
                   '#endif\n\n'),
        what="chiamata a calculate_forces",
    )

    # ── 4. visibilita' dei simboli di Kokkos ────────────────────────────────
    # Kokkos e' costruito come libreria condivisa e imposta sui PROPRI target
    # CXX_VISIBILITY_PRESET=hidden e VISIBILITY_INLINES_HIDDEN=ON: i metodi dei
    # template istanziati esplicitamente nella libreria (per esempio
    # SharedAllocationRecordCommon<HostSpace>::get_label) restano fuori dalla
    # tabella dinamica.  ESPResSo li usa dai suoi header, quindi espresso_core
    # resta con riferimenti irrisolti -- che non fanno fallire il link, perche'
    # una shared library non pretende di risolvere tutto, e si manifestano solo
    # all'import:
    #     undefined symbol: _ZNK6Kokkos4Impl28SharedAllocationRecordCommon...
    # Passare le variabili CMAKE_* al configure non basta: le property sul
    # target vincono, e vanno sovrascritte dopo FetchContent_MakeAvailable.
    # Su macOS non si presenta: Mach-O tratta la visibilita' diversamente.
    patch(
        root / "CMakeLists.txt",
        marker="MLCG: visibilita' dei simboli Kokkos",
        anchor="  # mark kokkos headers as system headers to disable compiler diagnostics",
        insertion=(
            "  # MLCG: visibilita' dei simboli Kokkos.  Le istanziazioni esplicite dei\n"
            "  # template restano fuori dalla tabella dinamica se le inline sono\n"
            "  # nascoste, e il core che le usa dagli header non le trova all'import.\n"
            "  foreach(kokkos_target IN ITEMS kokkoscore kokkoscontainers kokkossimd\n"
            "                                 kokkosalgorithms)\n"
            "    if(TARGET ${kokkos_target})\n"
            "      set_property(TARGET ${kokkos_target} PROPERTY CXX_VISIBILITY_PRESET default)\n"
            "      set_property(TARGET ${kokkos_target} PROPERTY VISIBILITY_INLINES_HIDDEN OFF)\n"
            "    endif()\n"
            "  endforeach()\n\n"
        ),
        what="visibilita' dei target Kokkos",
    )

    print("\n[DONE] innesto del core completato.")


if __name__ == "__main__":
    main()
