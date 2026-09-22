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

# DA DOVE VIENE LIBTORCH
#   pip  (default su Leonardo): il wheel di PyTorch, che e' manylinux_2_28 e
#        quindi compatibile con la glibc 2.28 di RHEL 8.  Porta la stessa
#        LibTorch C++ con i suoi share/cmake dentro site-packages/torch.
#   zip: la distribuzione LibTorch ufficiale.  E' costruita contro una glibc
#        piu' recente e su RHEL 8 il link fallisce con
#        "undefined reference to log2@GLIBC_2.29": quei simboli versionati non
#        esistono nella libm del sistema.
LIBTORCH_SOURCE="${LIBTORCH_SOURCE:-pip}"

say() { printf '\n[setup] %s\n' "$*"; }

say "ambiente"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_leonardo.sh"
echo "  python   $(command -v python3) ($(python3 --version 2>&1))"
echo "  cmake    $(command -v cmake) ($(cmake --version 2>/dev/null | head -1))"
echo "  progetto ${PROJECT_ROOT}"

# ── 1. virtualenv ───────────────────────────────────────────────────────────
# Il framework in Python usa MDAnalysis, numpy, scipy; ESPResSo vuole anche
# Cython per compilare i suoi moduli e numpy agli header.
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

# ── 2. LibTorch ─────────────────────────────────────────────────────────────
# L'ABI di libstdc++ conta: i wheel di PyTorch costruiti con
# _GLIBCXX_USE_CXX11_ABI=0 propagano quel flag, via TORCH_CXX_FLAGS, a tutto
# cio' che linka Torch.  Il core di ESPResSo verrebbe allora compilato con una
# std::string diversa da quella di Kokkos, Cabana e Boost, e l'import muore su
# simboli senza il tag [abi:cxx11]:
#     undefined symbol: ...SharedAllocationRecordCommon<HostSpace>::get_labelEv
# mentre la libreria definisce ...get_label[abi:cxx11]().  Serve un wheel con
# la ABI nuova.
if [[ "$LIBTORCH_SOURCE" == "pip" ]]; then
    have_torch="$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || true)"
    if [[ "$have_torch" == "${LIBTORCH_VERSION}"* ]]; then
        say "torch gia' installato nel venv (${have_torch})"
    else
        [[ -n "$have_torch" ]] && say "torch installato: ${have_torch}; ne serve ${LIBTORCH_VERSION}: reinstallo"
        say "installo torch ${LIBTORCH_VERSION}+${LIBTORCH_CUDA} dal wheel"
        pip install --force-reinstall "torch==${LIBTORCH_VERSION}" \
            --index-url "https://download.pytorch.org/whl/${LIBTORCH_CUDA}"
    fi
    TORCH_PREFIX="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')"
    echo "  torch       $(python3 -c 'import torch;print(torch.__version__)')"
    torch_abi="$(python3 -c 'import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)' 2>/dev/null || echo '?')"
    echo "  cxx11 ABI   ${torch_abi}"
    if [[ "$torch_abi" == "False" ]]; then
        echo "  [ATTENZIONE] wheel con ABI pre-C++11: il core di ESPResSo ereditera'"
        echo "               -D_GLIBCXX_USE_CXX11_ABI=0 e non linkera' con Kokkos/Boost."
        echo "               Prova una versione piu' recente:  LIBTORCH_VERSION=... "
    fi
    echo "  LibTorch    ${TORCH_PREFIX}"
    [[ -f "${TORCH_PREFIX}/share/cmake/Torch/TorchConfig.cmake" ]] \
        || { echo "[ERROR] TorchConfig.cmake assente in ${TORCH_PREFIX}" >&2; exit 1; }
elif [[ -f "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake" ]]; then
    say "LibTorch gia' presente in ${LIBTORCH_ROOT}"
else
    say "scarico LibTorch ${LIBTORCH_VERSION} ${LIBTORCH_CUDA} (~2.5 GB)"
    tmpzip="${PROJECT_ROOT}/.libtorch.zip"
    curl -L --fail --retry 3 -o "$tmpzip" "$LIBTORCH_URL"
    say "estraggo"
    rm -rf "${PROJECT_ROOT}/libtorch.new"
    mkdir -p "${PROJECT_ROOT}/libtorch.new"
    unzip -q "$tmpzip" -d "${PROJECT_ROOT}/libtorch.new"
    mv "${PROJECT_ROOT}/libtorch.new/libtorch" "${LIBTORCH_ROOT}"
    rmdir "${PROJECT_ROOT}/libtorch.new" 2>/dev/null || true
    rm -f "$tmpzip"
    [[ -f "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake" ]] \
        || { echo "[ERROR] TorchConfig.cmake assente dopo l'estrazione" >&2; exit 1; }
fi

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
