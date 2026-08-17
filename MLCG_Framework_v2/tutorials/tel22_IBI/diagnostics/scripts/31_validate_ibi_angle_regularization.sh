#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/model_config.sh" || -f "${SCRIPT_DIR}/cg_priors.json" ]]; then
    TUTORIAL_DIR="${SCRIPT_DIR}"
else
    TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
source "${TUTORIAL_DIR}/model_config.sh"
load_model_dependent_config step31
cd "${TUTORIAL_DIR}"

PYPRESSO="${PYPRESSO:-${FRAMEWORK_ROOT}/espresso/build/pypresso}"
MODEL="${IBI_MODEL}"
CONFIG="${IBI_CONFIG_JSON}"
DATASET="${IBI_DATASET}"
RB_INFO="${IBI_RB_INFO}"
SOURCE_CHECKPOINT="${IBI_SOURCE_CHECKPOINT}"
IBI_SETTINGS="${IBI_SETTINGS}"
CURRENT_PRIORS="${IBI_PRIORS}"
CANDIDATE_01="${IBI_ANGLE_CANDIDATE_01_PRIORS}"
CANDIDATE_02="${IBI_ANGLE_CANDIDATE_02_PRIORS}"
OUTPUT_DIR="${IBI_ANGLE_VALIDATION_OUTPUT_DIR}"
DTS="${IBI_ANGLE_VALIDATION_DTS}"
NVE_DURATION="${IBI_ANGLE_VALIDATION_NVE_DURATION_PS}"
BRANCH_DT="${IBI_ANGLE_VALIDATION_BRANCH_DT}"
BRANCH_DURATION="${IBI_ANGLE_VALIDATION_BRANCH_DURATION_PS}"
KT="${IBI_ANGLE_VALIDATION_KT}"
SEED="${IBI_ANGLE_VALIDATION_THERMOSTAT_SEED}"
DEVICE="${IBI_ANGLE_VALIDATION_DEVICE}"
ML_PRECISION="${IBI_ANGLE_VALIDATION_ML_PRECISION}"
NEIGHBOR_SEARCH="${IBI_ANGLE_VALIDATION_NEIGHBOR_SEARCH}"
CONFIG_PROVENANCE="${OUTPUT_DIR}/model_config_provenance.json"

for path in "${PYPRESSO}" "${MODEL}" "${CONFIG}" "${DATASET}" "${RB_INFO}" "${SOURCE_CHECKPOINT}" "${IBI_SETTINGS}" "${CURRENT_PRIORS}" "${CANDIDATE_01}" "${CANDIDATE_02}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

read -r -a DT_ARRAY <<< "${DTS}"
ARGS=(
  "${FRAMEWORK_ROOT}/simulation/ibi_angle_candidate_validation.py"
  --pypresso "${PYPRESSO}"
  --model "${MODEL}"
  --config "${CONFIG}"
  --dataset "${DATASET}"
  --rb-info "${RB_INFO}"
  --source-checkpoint "${SOURCE_CHECKPOINT}"
  --ibi-config "${IBI_SETTINGS}"
  --variant "current=${CURRENT_PRIORS}"
  --variant "smooth_0p01=${CANDIDATE_01}"
  --variant "smooth_0p02=${CANDIDATE_02}"
  --dts "${DT_ARRAY[@]}"
  --duration-ps "${NVE_DURATION}"
  --branch-dt "${BRANCH_DT}"
  --branch-duration-ps "${BRANCH_DURATION}"
  --kT "${KT}"
  --thermostat-seed "${SEED}"
  --device "${DEVICE}"
  --ml-precision "${ML_PRECISION}"
  --neighbor-search "${NEIGHBOR_SEARCH}"
  --output-dir "${OUTPUT_DIR}"
)
for arg in "$@"; do
    case "${arg}" in
        --dry-run|--overwrite|--resume) ARGS+=("${arg}") ;;
        *) echo "[ERROR] Unsupported argument: ${arg}" >&2; exit 2 ;;
    esac
done

cat <<EOF
[IBI ANGLE REGULARIZATION MATCHED VALIDATION]
current priors : ${CURRENT_PRIORS}
candidate A    : ${CANDIDATE_01}
candidate B    : ${CANDIDATE_02}
source chk     : ${SOURCE_CHECKPOINT}
dt scan        : ${DTS} ps
NVE duration   : ${NVE_DURATION} ps per dt
NVT branch     : ${BRANCH_DURATION} ps at dt=${BRANCH_DT} ps kT=${KT}
thermostat seed: ${SEED} (same seed/protocol for all variants)
output         : ${OUTPUT_DIR}
[NOTE] Diagnostic-only. Candidates remain unvalidated and are never promoted automatically.
EOF

"${PYTHON_BIN}" "${ARGS[@]}"
if [[ " $* " != *" --dry-run "* ]]; then
  write_model_dependent_provenance "${CONFIG_PROVENANCE}"
fi
