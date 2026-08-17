#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYRESSO="${PYRESSO:-${DEFAULT_PYPRESSO}}"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/model_config.sh"
load_model_dependent_config step29

MODEL="${IBI_MODEL}"
CONFIG="${TRAINING_CONFIG}"
DATASET="${IBI_DATASET}"
RB_INFO="${IBI_RB_INFO}"
IBI_PRIORS="${IBI_PRIORS}"
REFERENCE_PRIORS="${IBI_TIMESTEP_REFERENCE_PRIORS}"
SOURCE_CHECKPOINT="${IBI_TIMESTEP_SOURCE_CHECKPOINT}"
OUTPUT_DIR="${IBI_TIMESTEP_OUTPUT_DIR}"
DTS="${IBI_TIMESTEP_DTS}"
DURATION="${IBI_TIMESTEP_DURATION_PS}"
BRANCH_DT="${IBI_TIMESTEP_BRANCH_DT}"
BRANCH_DURATION="${IBI_TIMESTEP_BRANCH_DURATION_PS}"
BRANCH_KT="${IBI_TIMESTEP_BRANCH_KT}"
SEED_BASE="${IBI_TIMESTEP_SEED_BASE}"
DEVICE="${IBI_TIMESTEP_DEVICE}"
ML_PRECISION="${IBI_TIMESTEP_ML_PRECISION}"
NEIGHBOR="${IBI_TIMESTEP_NEIGHBOR_SEARCH}"


for path in "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" "${IBI_PRIORS}" "${REFERENCE_PRIORS}" "${SOURCE_CHECKPOINT}"; do
    if [[ ! -f "${path}" ]]; then echo "[ERROR] Missing required input: ${path}" >&2; exit 1; fi
done
read -r -a DT_ARGS <<< "${DTS}"

ARGS=(
  --pypresso "${PYRESSO}" --model "${MODEL}" --config "${CONFIG}"
  --old-priors "${REFERENCE_PRIORS}" --ibi-priors "${IBI_PRIORS}"
  --rb-info "${RB_INFO}" --dataset "${DATASET}" --source-checkpoint "${SOURCE_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}" --dts "${DT_ARGS[@]}" --duration-ps "${DURATION}"
  --branch-dt "${BRANCH_DT}" --branch-duration-ps "${BRANCH_DURATION}" --branch-kT "${BRANCH_KT}"
  --seed-base "${SEED_BASE}" --device "${DEVICE}" --ml-precision "${ML_PRECISION}" --neighbor-search "${NEIGHBOR}"
)
for arg in "$@"; do
    case "${arg}" in
        --dry-run|--overwrite|--resume) ARGS+=("${arg}") ;;
        *) echo "[ERROR] Unsupported argument: ${arg}" >&2; exit 2 ;;
    esac
done

cat <<EOF
[IBI COARSE-TIMESTEP/STIFFNESS DIAGNOSTIC]
reference    : ${REFERENCE_PRIORS}
IBI priors   : ${IBI_PRIORS}
source chk   : ${SOURCE_CHECKPOINT}
dt scan      : ${DTS} ps
NVE duration : ${DURATION} ps per dt
NVT branch   : ${BRANCH_DURATION} ps at dt=${BRANCH_DT} ps
variants     : reference / ibi_bonds_only / ibi_angles_only / full_ibi
output       : ${OUTPUT_DIR}
[NOTE] The configured reference restores the model-specific baseline bonded priors.
[NOTE] Diagnostic-only. No numerical kernel or certification artifact is modified.
EOF

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/diagnose_ibi_timestep_range.py" "${ARGS[@]}"

if [[ " $* " != *" --dry-run "* ]]; then
  write_model_dependent_provenance "${OUTPUT_DIR}/model_config_provenance.json"
fi
