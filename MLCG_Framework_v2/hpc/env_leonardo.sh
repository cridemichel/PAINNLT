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

LIBTORCH_ROOT="${LIBTORCH_ROOT:-${PROJECT_ROOT}/libtorch}"
MLCG_VENV="${MLCG_VENV:-${PROJECT_ROOT}/venv}"

for m in "${MODULE_CMAKE:-cmake}" \
         "${MODULE_OPENMPI:-openmpi}" \
         "${MODULE_BOOST:-boost}" \
         "${MODULE_FFTW:-fftw}" \
         "${MODULE_PYTHON:-python}"; do
    if ! module load "$m" 2>/dev/null; then
        echo "[env] ATTENZIONE: modulo non caricato: $m" >&2
    fi
done

export LIBTORCH_ROOT
export CMAKE_PREFIX_PATH="${LIBTORCH_ROOT}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${LIBTORCH_ROOT}/lib:${LD_LIBRARY_PATH:-}"

if [[ -f "${MLCG_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${MLCG_VENV}/bin/activate"
fi

export PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
