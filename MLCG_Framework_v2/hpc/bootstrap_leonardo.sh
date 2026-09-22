#!/usr/bin/env bash
# Bootstrap del framework su Leonardo: innesta il plugin PaiNN nell'albero di
# ESPResSo, compila ESPResSo e il trainer.
#
#   bash hpc/submit_leonardo.sh bootstrap        (normalmente cosi')
#
# Non scarica nulla: LibTorch, il venv e il clone di ESPResSo li prepara
# setup_native.sh, che gira dove c'e' la rete.  Questo passo compila e basta,
# quindi puo' stare su DCGP con molti core.
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
echo "  core       ${JOBS}"

[[ -d "${ESPRESSO_SRC}/.git" ]] || {
    echo "[ERROR] ESPResSo non clonato in ${ESPRESSO_SRC}." >&2
    echo "        Il clone ha bisogno di rete: bash hpc/submit_leonardo.sh setup" >&2
    exit 2
}
echo "  ESPResSo   $(git -C "$ESPRESSO_SRC" rev-parse --short HEAD)"

# ── 1. innesto del plugin PaiNN ─────────────────────────────────────────────
say "innesto del plugin PaiNN nell'albero ESPResSo"
ESPRESSO_SRC="$ESPRESSO_SRC" PYTHON_BIN="$(command -v python3)" \
    bash "$FRAMEWORK_ROOT/simulation/espresso_plugin/copy_plugin_files.sh"

# ── 2. build di ESPResSo ────────────────────────────────────────────────────
say "compilo ESPResSo (${JOBS} core, CUDA=${ESPRESSO_CUDA})"
cmake -S "$ESPRESSO_SRC" -B "$ESPRESSO_SRC/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
      -DESPRESSO_BUILD_WITH_CUDA="$ESPRESSO_CUDA" \
      -DPython_EXECUTABLE="$(command -v python3)"
cmake --build "$ESPRESSO_SRC/build" -j "$JOBS"

# ── 3. build del trainer ────────────────────────────────────────────────────
# MLCG_TORCH_ROOT e' il meccanismo del CMakeLists della v2 per selezionare UNA
# distribuzione LibTorch: passarlo evita che CMake ne trovi due e linki header
# e librerie di versioni diverse.
say "compilo il trainer PaiNN"
cmake -S "$FRAMEWORK_ROOT/training" -B "$FRAMEWORK_ROOT/training/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DMLCG_TORCH_ROOT="$TORCH_PREFIX"
cmake --build "$FRAMEWORK_ROOT/training/build" -j "$JOBS"

# ── 4. verifica ─────────────────────────────────────────────────────────────
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
