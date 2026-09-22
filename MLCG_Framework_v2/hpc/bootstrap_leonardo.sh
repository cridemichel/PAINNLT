#!/usr/bin/env bash
# Bootstrap del framework su Leonardo: innesta il plugin PaiNN nell'albero di
# ESPResSo, compila ESPResSo e il trainer.
#
#   bash hpc/submit_leonardo.sh configure     (dove c'e' la rete)
#   bash hpc/submit_leonardo.sh build          (dove ci sono i core)
#
# DUE STADI, E NON UNO
#   La configurazione di ESPResSo 5 scarica heFFTe, Kokkos e Cabana con
#   FetchContent -- Kokkos e Cabana incondizionatamente -- quindi vuole la
#   rete, che i nodi di calcolo non hanno.  Il configure va percio' su
#   lrd_all_serial (nodi di login), la compilazione su DCGP con 32 core.
#   STEP=configure|build|all sceglie cosa fare; all serve solo dove ci sono
#   entrambe le cose, cioe' su una macchina normale.
#
# PERCHE' ESPRESSO SENZA CUDA
#   La GPU qui serve a LibTorch, non a ESPResSo: l'inferenza del modello e la
#   sua backward stanno nel plugin.  Compilare ESPResSo senza CUDA toglie una
#   dipendenza dal toolkit senza togliere nulla alla simulazione.  Per
#   riattivarla:  ESPRESSO_CUDA=ON bash hpc/bootstrap_leonardo.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ESPRESSO_SRC="${ESPRESSO_SRC:-$FRAMEWORK_ROOT/espresso}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"
ESPRESSO_CUDA="${ESPRESSO_CUDA:-OFF}"
STEP="${STEP:-all}"
# waLBerla implementa il lattice Boltzmann, che questo framework non usa: e'
# il download e la compilazione piu' pesanti di tutto l'albero.
ESPRESSO_WALBERLA="${ESPRESSO_WALBERLA:-OFF}"

say() { printf '\n[bootstrap] %s\n' "$*"; }

# ── 0. ambiente ─────────────────────────────────────────────────────────────
if [[ -z "${LIBTORCH_ROOT:-}" && -f "${SCRIPT_DIR}/env_leonardo.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env_leonardo.sh"
fi

say "verifica dell'ambiente"
command -v cmake >/dev/null || { echo "[ERROR] cmake assente: carica i moduli (hpc/env_leonardo.sh)" >&2; exit 2; }

# LibTorch: da LIBTORCH_ROOT se c'e', altrimenti dal torch di Python (il caso
# del Mac, dove torch viene da pip).
if [[ -n "${LIBTORCH_ROOT:-}" && -f "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake" ]]; then
    TORCH_PREFIX="$LIBTORCH_ROOT"
    echo "  LibTorch   ${TORCH_PREFIX}"
elif python3 -c 'import torch' 2>/dev/null; then
    TORCH_PREFIX="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')"
    echo "  torch      $(python3 -c 'import torch;print(torch.__version__)') in ${TORCH_PREFIX}"
else
    echo "[ERROR] nessuna distribuzione LibTorch trovata." >&2
    echo "        Esegui prima:  bash hpc/submit_leonardo.sh setup" >&2
    exit 2
fi
echo "  python     $(command -v python3) ($(python3 --version 2>&1))"
echo "  compilatore $(command -v g++) ($(g++ --version 2>/dev/null | head -1))"
echo "  core       ${JOBS}"

[[ -d "${ESPRESSO_SRC}/.git" ]] || {
    echo "[ERROR] ESPResSo non clonato in ${ESPRESSO_SRC}." >&2
    echo "        Il clone ha bisogno di rete: bash hpc/submit_leonardo.sh setup" >&2
    exit 2
}
echo "  ESPResSo   $(git -C "$ESPRESSO_SRC" rev-parse --short HEAD)"

# ── 1. configurazione di ESPResSo, PRIMA del plugin ─────────────────────────
# install_switched_morse_nonbonded.py attiva la feature Morse scrivendo nel
# file di configurazione dentro espresso/build, quindi quella directory deve
# gia' esistere quando il plugin viene innestato.  Su un albero appena clonato
# non esiste: va configurato adesso.
configure_espresso() {
    # Una cache di CMake che punta a un compilatore diverso da quello corrente
    # non e' riutilizzabile: CMake rifiuta il cambio con "CXX compiler changed".
    # Capita dopo un tentativo fallito col gcc di sistema.
    local cache="$ESPRESSO_SRC/build/CMakeCache.txt"
    if [[ -f "$cache" && -n "${CXX:-}" ]]; then
        local cached
        cached="$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$cache" | head -1)"
        if [[ -n "$cached" && "$cached" != "$CXX" ]]; then
            say "la cache di CMake punta a ${cached}, ora si usa ${CXX}: la rigenero"
            rm -rf "$ESPRESSO_SRC/build"
        fi
    fi
    cmake -S "$ESPRESSO_SRC" -B "$ESPRESSO_SRC/build" \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
          -DESPRESSO_BUILD_WITH_CUDA="$ESPRESSO_CUDA" \
          -DESPRESSO_BUILD_WITH_WALBERLA="$ESPRESSO_WALBERLA" \
          -DESPRESSO_BUILD_TESTS=OFF \
          -DPython_EXECUTABLE="$(command -v python3)"
}

if [[ "$STEP" == "configure" || "$STEP" == "all" ]]; then
say "configuro ESPResSo (CUDA=${ESPRESSO_CUDA}, waLBerla=${ESPRESSO_WALBERLA})"
configure_espresso

# ── 2. innesto del plugin PaiNN ─────────────────────────────────────────────
say "innesto del plugin PaiNN nell'albero ESPResSo"
ESPRESSO_SRC="$ESPRESSO_SRC" PYTHON_BIN="$(command -v python3)" \
    bash "$FRAMEWORK_ROOT/simulation/espresso_plugin/copy_plugin_files.sh"

# ── 3. build di ESPResSo ────────────────────────────────────────────────────
# Riconfigurazione: l'innesto ha aggiunto sorgenti e toccato le liste di CMake
# e il file di configurazione delle feature.
say "riconfiguro dopo l'innesto"
configure_espresso
fi

if [[ "$STEP" == "configure" ]]; then
    say "configurazione completata.  Ora:  bash hpc/submit_leonardo.sh build"
    exit 0
fi

if [[ ! -f "$ESPRESSO_SRC/build/CMakeCache.txt" ]]; then
    echo "[ERROR] ESPResSo non e' configurato: la configurazione scarica heFFTe," >&2
    echo "        Kokkos e Cabana e vuole la rete.  Esegui prima:" >&2
    echo "          bash hpc/submit_leonardo.sh configure" >&2
    exit 2
fi

say "compilo ESPResSo (${JOBS} core)"
cmake --build "$ESPRESSO_SRC/build" -j "$JOBS"

# ── 4. build del trainer ────────────────────────────────────────────────────
# MLCG_TORCH_ROOT e' il meccanismo del CMakeLists della v2 per selezionare UNA
# distribuzione LibTorch: passarlo evita che CMake ne trovi due e linki header
# e librerie di versioni diverse.
# Come per il compilatore, una cache che punta a un'altra distribuzione
# LibTorch non e' riutilizzabile: TorchConfig scrive nei target percorsi
# assoluti (libkineto.a, libc10_cuda.so...), e dopo un cambio di LibTorch il
# link fallisce con "No rule to make target .../libkineto.a" anche se il
# find_package ha trovato correttamente quella nuova.
trainer_cache="$FRAMEWORK_ROOT/training/build/CMakeCache.txt"
if [[ -f "$trainer_cache" ]]; then
    cached_torch="$(sed -n 's/^MLCG_TORCH_ROOT:[^=]*=//p' "$trainer_cache" | head -1)"
    if [[ -n "$cached_torch" && "$cached_torch" != "$TORCH_PREFIX" ]]; then
        say "la cache del trainer punta a ${cached_torch}, ora si usa ${TORCH_PREFIX}: la rigenero"
        rm -rf "$FRAMEWORK_ROOT/training/build"
    fi
fi

say "compilo il trainer PaiNN"
cmake -S "$FRAMEWORK_ROOT/training" -B "$FRAMEWORK_ROOT/training/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DMLCG_TORCH_ROOT="$TORCH_PREFIX"
cmake --build "$FRAMEWORK_ROOT/training/build" -j "$JOBS"

# ── 5. verifica ─────────────────────────────────────────────────────────────
say "verifica"
ok=1
for f in "$ESPRESSO_SRC/build/pypresso" \
         "$ESPRESSO_SRC/build/src/python/espressomd/painn.so" \
         "$FRAMEWORK_ROOT/training/build/train_painn"; do
    if [[ -e "$f" ]]; then printf '  ok        %s\n' "$f"
    else printf '  ASSENTE   %s\n' "$f"; ok=0; fi
done

# Il plugin compilato deve conoscere i kwargs ordered-geometry, altrimenti
# equilibrate.py e run_cg_md.py li scarteranno con un WARN e un modello
# shared-geometry non sarebbe simulabile.
if [[ -e "$ESPRESSO_SRC/build/src/python/espressomd/painn.so" ]]; then
    if strings "$ESPRESSO_SRC/build/src/python/espressomd/painn.so" \
       | grep -q ordered_geometry_copies; then
        echo "  ok        painn.so include il supporto ordered-geometry"
    else
        echo "  [NOTA]    painn.so senza ordered-geometry: PaiNN puro funziona,"
        echo "            un modello shared-geometry no.  Riapplica il plugin e ricompila."
    fi
fi

(( ok )) || { echo "[ERROR] bootstrap incompleto" >&2; exit 1; }
say "bootstrap completato"
