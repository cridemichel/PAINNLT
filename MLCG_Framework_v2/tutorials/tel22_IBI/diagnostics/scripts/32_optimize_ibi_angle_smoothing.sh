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
load_model_dependent_config step32
cd "${TUTORIAL_DIR}"

PYPRESSO="${PYPRESSO:-${FRAMEWORK_ROOT}/espresso/build/pypresso}"
MODEL="${IBI_MODEL}"
CONFIG="${TRAINING_CONFIG}"
DATASET="${IBI_DATASET}"
RB_INFO="${IBI_RB_INFO}"
SOURCE_CHECKPOINT="${IBI_SOURCE_CHECKPOINT}"
IBI_SETTINGS="${IBI_SETTINGS}"
SOURCE_PRIORS="${IBI_PRIORS}"
STEP31_REPORT="${IBI_ANGLE_STEP31_REPORT}"
OUTPUT_DIR="${IBI_ANGLE_SWEEP_OUTPUT_DIR}"
NEW_SIGMAS="${IBI_ANGLE_SWEEP_SIGMAS}"
REUSE_SIGMA="${IBI_ANGLE_SWEEP_REUSE_SIGMA}"
REUSE_VARIANT="${IBI_ANGLE_SWEEP_REUSE_VARIANT}"
DTS="${IBI_ANGLE_SWEEP_DTS}"
NVE_DURATION="${IBI_ANGLE_SWEEP_NVE_DURATION_PS}"
BRANCH_DT="${IBI_ANGLE_SWEEP_BRANCH_DT_PS}"
BRANCH_DURATION="${IBI_ANGLE_SWEEP_BRANCH_DURATION_PS}"
KT="${IBI_ANGLE_SWEEP_KT}"
SEED="${IBI_ANGLE_SWEEP_SEED}"
DEVICE="${IBI_ANGLE_SWEEP_DEVICE}"
ML_PRECISION="${IBI_ANGLE_SWEEP_ML_PRECISION}"
NEIGHBOR_SEARCH="${IBI_ANGLE_SWEEP_NEIGHBOR_SEARCH}"
CONFIG_PROVENANCE="${OUTPUT_DIR}/model_config_provenance.json"

for path in "${PYPRESSO}" "${MODEL}" "${CONFIG}" "${DATASET}" "${RB_INFO}" "${SOURCE_CHECKPOINT}" "${IBI_SETTINGS}" "${SOURCE_PRIORS}" "${STEP31_REPORT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

read -r -a SIGMA_ARRAY <<< "${NEW_SIGMAS}"
read -r -a DT_ARRAY <<< "${DTS}"
ARGS=(
  "${FRAMEWORK_ROOT}/simulation/ibi_angle_smoothing_sweep.py"
  --python-bin "${PYTHON_BIN}"
  --pypresso "${PYPRESSO}"
  --model "${MODEL}"
  --config "${CONFIG}"
  --dataset "${DATASET}"
  --rb-info "${RB_INFO}"
  --source-checkpoint "${SOURCE_CHECKPOINT}"
  --ibi-config "${IBI_SETTINGS}"
  --source-priors "${SOURCE_PRIORS}"
  --step31-report "${STEP31_REPORT}"
  --new-sigmas "${SIGMA_ARRAY[@]}"
  --reuse-sigma "${REUSE_SIGMA}"
  --reuse-variant "${REUSE_VARIANT}"
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
[IBI ANGLE SMOOTHING LOCAL SWEEP]
source priors  : ${SOURCE_PRIORS}
step31 report  : ${STEP31_REPORT}
new sigmas     : ${NEW_SIGMAS} rad
reused sigma   : ${REUSE_SIGMA} rad (${REUSE_VARIANT}; no MD rerun)
common dt scan : ${DTS} ps
NVE duration   : ${NVE_DURATION} ps per dt
NVT branch     : ${BRANCH_DURATION} ps at dt=${BRANCH_DT} ps kT=${KT}
thermostat seed: ${SEED}
output         : ${OUTPUT_DIR}
[NOTE] Ranking prioritizes contiguous sigma_E/dt^2 behavior, not global p alone.
[NOTE] Diagnostic-only. No candidate is promoted automatically.
EOF

"${PYTHON_BIN}" "${ARGS[@]}"
if [[ " $* " != *" --dry-run "* ]]; then
  write_model_dependent_provenance "${CONFIG_PROVENANCE}"
fi
