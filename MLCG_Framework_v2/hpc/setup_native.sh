#!/usr/bin/env bash
# Prepara l'ambiente nativo su Leonardo: LibTorch, venv Python, clone di
# ESPResSo.  E' l'unico passo che ha bisogno della rete, quindi gira sul nodo
# di login o in un job su lrd_all_serial (che gira sui nodi di login); i nodi
# di calcolo non raggiungono internet.
#
# Non compila nulla: quello e' il bootstrap, che puo' andare su DCGP.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK="${FRAMEWORK:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$FRAMEWORK/.." && pwd)}"

LIBTORCH_ROOT="${LIBTORCH_ROOT:-${PROJECT_ROOT}/libtorch}"
MLCG_VENV="${MLCG_VENV:-${PROJECT_ROOT}/venv}"
ESPRESSO_SRC="${ESPRESSO_SRC:-${FRAMEWORK}/espresso}"
ESPRESSO_COMMIT="${ESPRESSO_COMMIT:-84cc1d924}"
ESPRESSO_REPO="${ESPRESSO_REPO:-https://github.com/espressomd/espresso.git}"

# LibTorch C++, variante cxx11-ABI.
#
# La scelta della build CUDA non e' libera: una libtorch cu124 vuole un driver
# NVIDIA >= 550, e su Leonardo il runtime di sistema e' CUDA 12.2.  cu121 e'
# la scelta prudente, e va bene per le A100 (compute capability 8.0).  Se il
# driver dei nodi boost risulta piu' recente, si puo' passare a cu124
# sovrascrivendo LIBTORCH_URL.
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.5.1}"
LIBTORCH_CUDA="${LIBTORCH_CUDA:-cu121}"
LIBTORCH_URL="${LIBTORCH_URL:-https://download.pytorch.org/libtorch/${LIBTORCH_CUDA}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA}.zip}"

say() { printf '\n[setup] %s\n' "$*"; }

say "ambiente"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_leonardo.sh"
echo "  python   $(command -v python3) ($(python3 --version 2>&1))"
echo "  cmake    $(command -v cmake) ($(cmake --version 2>/dev/null | head -1))"
echo "  progetto ${PROJECT_ROOT}"

# ── 1. LibTorch ─────────────────────────────────────────────────────────────
if [[ -f "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake" ]]; then
    say "LibTorch gia' presente in ${LIBTORCH_ROOT}"
else
    say "scarico LibTorch ${LIBTORCH_VERSION} ${LIBTORCH_CUDA} (~2.5 GB)"
    tmpzip="${PROJECT_ROOT}/.libtorch.zip"
    curl -L --fail --retry 3 -o "$tmpzip" "$LIBTORCH_URL"
    say "estraggo"
    rm -rf "${PROJECT_ROOT}/libtorch.new"
    mkdir -p "${PROJECT_ROOT}/libtorch.new"
    unzip -q "$tmpzip" -d "${PROJECT_ROOT}/libtorch.new"
    # lo zip contiene una directory libtorch/ di primo livello
    mv "${PROJECT_ROOT}/libtorch.new/libtorch" "${LIBTORCH_ROOT}"
    rmdir "${PROJECT_ROOT}/libtorch.new" 2>/dev/null || true
    rm -f "$tmpzip"
    [[ -f "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake" ]] \
        || { echo "[ERROR] TorchConfig.cmake assente dopo l'estrazione" >&2; exit 1; }
fi

# ── 2. virtualenv ───────────────────────────────────────────────────────────
# Il framework in Python usa MDAnalysis, numpy, scipy; ESPResSo vuole anche
# Cython per compilare i suoi moduli e numpy agli header.  Torch NON serve:
# l'inferenza passa dal plugin C++.
if [[ -f "${MLCG_VENV}/bin/activate" ]]; then
    say "venv gia' presente in ${MLCG_VENV}"
else
    say "creo il venv"
    python3 -m venv "${MLCG_VENV}"
fi
# shellcheck disable=SC1091
source "${MLCG_VENV}/bin/activate"
say "installo i pacchetti Python"
pip install --upgrade pip setuptools wheel >/dev/null
pip install numpy scipy matplotlib MDAnalysis h5py "cython<3.1" packaging
echo "  numpy       $(python3 -c 'import numpy;print(numpy.__version__)')"
echo "  MDAnalysis  $(python3 -c 'import MDAnalysis;print(MDAnalysis.__version__)')"

# ── 3. ESPResSo ─────────────────────────────────────────────────────────────
# Il plugin innesta file dentro src/core/nonbonded_interactions e
# src/python/espressomd e tocca le liste di CMake: la versione va fissata,
# altrimenti l'innesto fallisce in modi non ovvi.
if [[ -d "${ESPRESSO_SRC}/.git" ]]; then
    say "ESPResSo gia' presente ($(git -C "$ESPRESSO_SRC" rev-parse --short HEAD))"
else
    say "clono ESPResSo al commit ${ESPRESSO_COMMIT}"
    git clone "$ESPRESSO_REPO" "$ESPRESSO_SRC"
    git -C "$ESPRESSO_SRC" checkout "$ESPRESSO_COMMIT"
fi

say "setup completato.  Prossimo passo:  bash hpc/submit_leonardo.sh bootstrap"
