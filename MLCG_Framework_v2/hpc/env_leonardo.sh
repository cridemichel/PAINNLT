# Ambiente nativo del framework su Leonardo.  Da usare con "source".
#
# PERCHE' NIENTE CONTAINER
#   Leonardo non configura le mappe subuid/subgid, quindi
#   "singularity build --fakeroot" fallisce con "no valid mapping entry" e un
#   .def con %post non e' costruibile sul cluster.  Le dipendenze di ESPResSo
#   ci sono pero' tutte come moduli (boost, fftw, openmpi, cmake, gcc), e il
#   Python del framework non ha bisogno di torch: gli servono MDAnalysis,
#   numpy e scipy.  Torch serve solo come LibTorch C++, al trainer e al
#   plugin.  Quindi l'ambiente e': moduli + LibTorch scaricato + un venv.
#
# I nomi dei moduli sono quelli corti, che caricano il default del sistema.
# Per fissarne uno diverso:  MODULE_BOOST=boost/1.85.0--... source env_leonardo.sh

MLCG_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK="${FRAMEWORK:-$(cd -- "$MLCG_ENV_DIR/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$FRAMEWORK/.." && pwd)}"

MLCG_VENV="${MLCG_VENV:-${PROJECT_ROOT}/venv}"

# DOVE STA LIBTORCH
#   Su Leonardo viene dal wheel di PyTorch installato nel venv: RHEL 8 ha
#   glibc 2.28, mentre la distribuzione LibTorch ufficiale e' costruita contro
#   una glibc piu' recente e il link fallisce con "undefined reference to
#   log2@GLIBC_2.29" -- quei simboli versionati non esistono nella libm del
#   sistema.  I wheel sono manylinux_2_28, fatti apposta per questi sistemi.
#   La directory libtorch/ (zip) resta supportata se presente.

# Il gcc di sistema (RHEL8) e' il 8.5.0, mentre ESPResSo richiede
# >= 12.2.0.  Va caricato il modulo, che e' anche l'unica scelta coerente con
# boost, fftw e openmpi, compilati tutti con gcc 12.2.0.
# Il modulo CUDA deve essere ALMENO pari alla CUDA con cui e' costruito il
# wheel di torch: il link del trainer passa dalle librerie del toolkit, e
# torch 2.10+cu126 usa simboli introdotti in CUDA 12.5 --
#     undefined reference to cudaGetDriverEntryPointByVersion@libcudart.so.12
# che nel modulo di default (cuda/12.2) non esistono.  Su Leonardo sono
# disponibili 12.2, 12.3 e 12.6.
# CUDA va caricata PER ULTIMA: il modulo openmpi di Leonardo dipende da
# cuda/12.2 e la ricaricherebbe sopra la 12.6, con il risultato che CMake
# trova il toolkit 12.2 ("-- Found CUDA: ... found version 12.2") e il link
# del trainer fallisce sui simboli introdotti in 12.5.
for m in "${MODULE_GCC:-gcc}" \
         "${MODULE_CMAKE:-cmake}" \
         "${MODULE_OPENMPI:-openmpi}" \
         "${MODULE_BOOST:-boost}" \
         "${MODULE_FFTW:-fftw}" \
         "${MODULE_PYTHON:-python}"; do
    if ! module load "$m" 2>/dev/null; then
        echo "[env] ATTENZIONE: modulo non caricato: $m" >&2
    fi
done

# CUDA: serve uno SWAP, non un load.  Il modulo openmpi dipende da cuda/12.2 e
# la carica lui; un "module load cuda/12.6" successivo non la sostituisce --
# resta la 12.2, CMake la trova ("-- Found CUDA: ... version 12.2") e il link
# del trainer fallisce sui simboli introdotti in CUDA 12.5.
_cuda_wanted="${MODULE_CUDA:-cuda/12.6}"
module swap cuda "$_cuda_wanted" 2>/dev/null \
  || module switch cuda "$_cuda_wanted" 2>/dev/null \
  || module load "$_cuda_wanted" 2>/dev/null \
  || echo "[env] ATTENZIONE: non sono riuscito ad attivare ${_cuda_wanted}" >&2
if command -v nvcc >/dev/null 2>&1; then
    _cuda_active="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
    echo "[env] CUDA attiva: ${_cuda_active:-?}  (richiesta: ${_cuda_wanted})"
fi

if [[ -f "${MLCG_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${MLCG_VENV}/bin/activate"
fi

if [[ -z "${LIBTORCH_ROOT:-}" ]]; then
    # find_spec LOCALIZZA il pacchetto senza eseguirlo.  Prima qui c'era
    # "import torch", che per leggere un percorso caricava le librerie CUDA --
    # oltre un gigabyte di .so da Lustre -- e inizializzava il runtime: decine
    # di secondi a ogni "source", cioe' a ogni job e a ogni srun.
    if _torch_prefix="$(python3 -c 'import importlib.util as u,os;s=u.find_spec("torch");print(os.path.dirname(s.origin))' 2>/dev/null)" \
       && [[ -f "${_torch_prefix}/share/cmake/Torch/TorchConfig.cmake" ]]; then
        LIBTORCH_ROOT="$_torch_prefix"
    elif [[ -f "${PROJECT_ROOT}/libtorch/share/cmake/Torch/TorchConfig.cmake" ]]; then
        LIBTORCH_ROOT="${PROJECT_ROOT}/libtorch"
    else
        echo "[env] ATTENZIONE: nessuna LibTorch trovata; esegui hpc/submit_leonardo.sh setup" >&2
        LIBTORCH_ROOT="${PROJECT_ROOT}/libtorch"
    fi
fi

export PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

# CMake sceglie il compilatore da CC/CXX o dal PATH: senza questi, sui nodi di
# calcolo trova /usr/bin/gcc (8.5.0) e la configurazione di ESPResSo si ferma
# con "Unsupported compiler GNU 8.5.0 (required version >= 12.2.0)".
export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"
export FC="${FC:-$(command -v gfortran)}"

export LIBTORCH_ROOT
export CMAKE_PREFIX_PATH="${LIBTORCH_ROOT}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${LIBTORCH_ROOT}/lib:${LD_LIBRARY_PATH:-}"

# Gli header di LibTorch anche per i target che non li ereditano da CMake.
# Il CMakeLists del core di ESPResSo linka "${TORCH_LIBRARIES}", che e' una
# lista di percorsi di librerie e non un target: non porta con se' le include
# directory, e il modulo Cython painn fallisce con
#   fatal error: torch/torch.h: No such file or directory
# CPATH e' la stessa soluzione che usava il Dockerfile dell'immagine.
export CPATH="${LIBTORCH_ROOT}/include:${LIBTORCH_ROOT}/include/torch/csrc/api/include:${CPATH:-}"
export LIBRARY_PATH="${LIBTORCH_ROOT}/lib:${LIBRARY_PATH:-}"

# Le librerie CUDA che torch usa davvero sono quelle dei pacchetti nvidia-*
# installati accanto a lui nel venv, non quelle del modulo di sistema.  Vanno
# davanti nel percorso di ricerca, altrimenti il linker risolve
# libcudart.so.12 e libcupti.so.12 con le versioni del toolkit e i simboli
# nuovi mancano:
#     undefined reference to cudaGetDriverEntryPointByVersion@libcudart.so.12
# (quella API esiste da CUDA 12.5; il default di Leonardo e' 12.2).
_nvidia_root="$(dirname "${LIBTORCH_ROOT}")/nvidia"
if [[ -d "$_nvidia_root" ]]; then
    for _nv_lib in "$_nvidia_root"/*/lib; do
        [[ -d "$_nv_lib" ]] || continue
        LIBRARY_PATH="${_nv_lib}:${LIBRARY_PATH}"
        LD_LIBRARY_PATH="${_nv_lib}:${LD_LIBRARY_PATH}"
    done
    export LIBRARY_PATH LD_LIBRARY_PATH
fi

# Il toolkit CUDA serve anche dove non si compila codice CUDA: LibTorch e' una
# build CUDA, e TorchConfig.cmake include Caffe2Config, che pretende di
# risolvere le librerie del toolkit.  Senza, la configurazione del trainer si
# ferma con "Your installed Caffe2 version uses CUDA but I cannot find the
# CUDA libraries".  I nodi DCGP non hanno GPU, ma il toolkit c'e' lo stesso.
# Il toolkit CUDA per CMake si ricava SEMPRE da nvcc, non dalle variabili
# d'ambiente: dopo lo swap fra moduli, CUDA_HOME e CUDA_TOOLKIT_ROOT_DIR
# possono essere rimaste quelle del modulo sostituito, e CMake finirebbe per
# usare il toolkit vecchio ("-- Found CUDA: ... version 12.2") anche con la
# 12.6 attiva.  Serve comunque: TorchConfig include Caffe2Config, che per una
# LibTorch CUDA pretende di risolvere le librerie del toolkit -- anche sui
# nodi DCGP, che GPU non ne hanno.
if command -v nvcc >/dev/null 2>&1; then
    CUDA_TOOLKIT_ROOT_DIR="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
elif [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
    CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}"
fi
if [[ -n "${CUDA_TOOLKIT_ROOT_DIR:-}" ]]; then
    export CUDA_TOOLKIT_ROOT_DIR
    export CUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT_DIR"
    export CUDACXX="${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc"
    export CMAKE_PREFIX_PATH="${CUDA_TOOLKIT_ROOT_DIR}:${CMAKE_PREFIX_PATH}"
    echo "[env] toolkit CUDA: ${CUDA_TOOLKIT_ROOT_DIR}"
else
    echo "[env] ATTENZIONE: toolkit CUDA non trovato; il trainer non si configurera'." >&2
fi
