#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
source "${HERE}/model_config.sh"
load_model_dependent_config step35 step36 step37
cd "${HERE}"
PYPRESSO="${PYPRESSO:-${ROOT}/espresso/build/pypresso}"
MODE="${1:-}"

if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--run" ]]; then
  echo "Usage: $0 --dry-run|--run" >&2
  exit 2
fi

STEP35="${IBI_DIHEDRAL_STEP35_OUT}"
STEP36="${IBI_DIHEDRAL_STEP36_OUT}"
ITER0="${STEP35}/ibi/iteration_000/cg_priors.json"
ITER1="${STEP35}/ibi/iteration_001/cg_priors.json"
CONS0="${STEP36}/candidates/${IBI_DIHEDRAL_CONSERVATIVITY_ZERO_UPDATE_CANDIDATE}/cg_priors.json"
CONS1="${STEP35}/conservative/cg_priors.json"
OUT="${IBI_DIHEDRAL_CONSERVATIVITY_OUT}"
REPORT="${OUT}/tabulated_dihedral_conservativity_report.json"
CONFIG_PROVENANCE="${OUT}/model_config_provenance.json"
PROBES="${IBI_DIHEDRAL_CONSERVATIVITY_PROBES}"
FD_EPS="${IBI_DIHEDRAL_CONSERVATIVITY_FD_EPS}"
SIN_MIN="${IBI_DIHEDRAL_CONSERVATIVITY_SIN_MIN}"
OVERWRITE="${OVERWRITE:-0}"

for path in "${ITER0}" "${ITER1}" "${CONS0}" "${CONS1}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing required artifact: ${path}" >&2; exit 1; }
done
[[ -x "${PYPRESSO}" ]] || { echo "[ERROR] pypresso not executable: ${PYPRESSO}" >&2; exit 1; }

cat <<EOF
[STEP 37 -- DIAGNOSE LEGACY TABULATED DIHEDRAL CONSERVATIVITY]
iteration 0 legacy      : ${ITER0}
iteration 0 conservative: ${CONS0}
iteration 1 legacy      : ${ITER1}
iteration 1 conservative: ${CONS1}
probes/group            : ${PROBES}
Cartesian FD epsilon    : ${FD_EPS}
minimum |sin(phi)|      : ${SIN_MIN}
output                   : ${REPORT}
[NOTE] Runtime-only diagnostic. No MD integration and no production modification.
[NOTE] Tests the actual ESPResSo force-factor convention dU/dphi=-factor*sin(phi).
EOF

if [[ "${MODE}" == "--dry-run" ]]; then
  echo "[PLAN] Compare 6 torsional groups before and after the first IBI update."
  echo "[PLAN] For each group: scalar table consistency, legacy runtime parity, Cartesian F=-grad(E), legacy-vs-conservative dF/dE."
  echo "[PLAN] No NVT/NVE trajectories are generated."
  exit 0
fi

if [[ -e "${OUT}" ]]; then
  if [[ "${OVERWRITE}" != "1" ]]; then
    echo "[ERROR] Output already exists: ${OUT}" >&2
    echo "[HINT] Remove it or rerun with OVERWRITE=1." >&2
    exit 1
  fi
  rm -rf "${OUT}"
fi
mkdir -p "${OUT}"
write_model_dependent_provenance "${CONFIG_PROVENANCE}"

"${PYPRESSO}" "${ROOT}/simulation/diagnose_tabulated_dihedral_conservativity.py" \
  --iteration0-priors "${ITER0}" \
  --iteration0-conservative "${CONS0}" \
  --iteration1-priors "${ITER1}" \
  --iteration1-conservative "${CONS1}" \
  --probes-per-group "${PROBES}" \
  --sin-min "${SIN_MIN}" \
  --fd-eps "${FD_EPS}" \
  --legacy-residual-min "${IBI_DIHEDRAL_CONSERVATIVITY_LEGACY_RESIDUAL_MIN}" \
  --ratio-min "${IBI_DIHEDRAL_CONSERVATIVITY_RATIO_MIN}" \
  --conservative-residual-max "${IBI_DIHEDRAL_CONSERVATIVITY_CONSERVATIVE_RESIDUAL_MAX}" \
  --model-config-provenance "${CONFIG_PROVENANCE}" --output "${REPORT}"

cat <<EOF
[DONE] report: ${REPORT}
[NOTE] Step 37 is diagnostic-only. Do not promote or regularize a torsional prior from this report alone.
EOF
