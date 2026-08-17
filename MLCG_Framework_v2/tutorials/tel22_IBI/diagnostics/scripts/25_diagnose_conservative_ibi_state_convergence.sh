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

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYRESSO="${PYRESSO:-${DEFAULT_PYPRESSO}}"

cd "${TUTORIAL_DIR}"
source "${TUTORIAL_DIR}/model_config.sh"
load_model_dependent_config step25

MODEL="${IBI_MODEL}"
CONFIG="${TRAINING_CONFIG}"
DATASET="${IBI_DATASET}"
RB_INFO="${IBI_RB_INFO}"
PRIORS="${IBI_PRIORS}"
VALIDATION_REPORT="${IBI_VALIDATION_REPORT}"
RUNTIME_PARITY_REPORT="${IBI_RUNTIME_PARITY_REPORT}"
SOURCE_CHECKPOINT="${NVE_SOURCE_CHECKPOINT}"
REQUIRED_HAMILTONIAN_MODE="${NVE_REQUIRED_HAMILTONIAN_MODE}"


for path in \
    "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" \
    "${PRIORS}" "${VALIDATION_REPORT}" "${RUNTIME_PARITY_REPORT}" \
    "${SOURCE_CHECKPOINT}" "${NVE_EQ_CHECKPOINT}" "${NVE_EQ_REPORT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required state-convergence artifact: ${path}" >&2
        if [[ "${path}" == "${NVE_EQ_CHECKPOINT}" || "${path}" == "${NVE_EQ_REPORT}" ]]; then
            echo "[HINT] Run step 23 once to prepare the dedicated conservative IBI-only NVT checkpoint." >&2
        fi
        exit 1
    fi
done

cat <<EOF_PLAN
[CONSERVATIVE IBI-ONLY STATE-CONVERGENCE PLAN]
model anchor : ${MODEL} (PaiNN disabled)
priors       : ${PRIORS}
checkpoint   : ${NVE_EQ_CHECKPOINT}
dt ladder    : ${NVE_STATE_DTS} ps
reference dt : ${NVE_STATE_REFERENCE_DT} ps
duration     : ${NVE_STATE_DURATION_PS} ps
sample every : ${NVE_STATE_SAMPLE_INTERVAL_PS} ps
device       : ${NVE_STATE_DEVICE}
neighbor     : ${NVE_STATE_NEIGHBOR_SEARCH}
output       : ${NVE_STATE_OUTPUT_DIR}
[NOTE] This is diagnostic-only. It measures short-time state convergence and does not replace step-23 strict NVE certification.
EOF_PLAN

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/conservative_nve_preflight.py" \
    --priors "${PRIORS}" \
    --validation-report "${VALIDATION_REPORT}" \
    --runtime-parity-report "${RUNTIME_PARITY_REPORT}" \
    --output "${NVE_STATE_PREFLIGHT_REPORT}"

read -r -a DT_ARGS <<< "${NVE_STATE_DTS}"

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/nve_state_convergence.py" \
    --pypresso "${PYRESSO}" \
    --model "${MODEL}" \
    --config "${CONFIG}" \
    --priors "${PRIORS}" \
    --rb-info "${RB_INFO}" \
    --dataset "${DATASET}" \
    --checkpoint "${NVE_EQ_CHECKPOINT}" \
    --require-checkpoint-hamiltonian-mode "${REQUIRED_HAMILTONIAN_MODE}" \
    --require-checkpoint-source "${SOURCE_CHECKPOINT}" \
    --dts "${DT_ARGS[@]}" \
    --reference-dt "${NVE_STATE_REFERENCE_DT}" \
    --duration-ps "${NVE_STATE_DURATION_PS}" \
    --sample-interval-ps "${NVE_STATE_SAMPLE_INTERVAL_PS}" \
    --device "${NVE_STATE_DEVICE}" \
    --ml-precision "${NVE_STATE_ML_PRECISION}" \
    --neighbor-search "${NVE_STATE_NEIGHBOR_SEARCH}" \
    --output-dir "${NVE_STATE_OUTPUT_DIR}" \
    --order-min "${NVE_STATE_ORDER_MIN}" \
    --order-max "${NVE_STATE_ORDER_MAX}" \
    --min-r2 "${NVE_STATE_MIN_R2}" \
    "$@"

cat <<EOF_DONE
[CONSERVATIVE IBI-ONLY STATE-CONVERGENCE DIAGNOSTIC COMPLETE]
report : ${NVE_STATE_OUTPUT_DIR}/state_convergence_report.json
plan   : ${NVE_STATE_OUTPUT_DIR}/run_plan.json
[NOTE] Use Richardson position/velocity/orientation/omega orders to decide whether the underlying NVE trajectory is second-order convergent.
EOF_DONE

write_model_dependent_provenance "${NVE_STATE_OUTPUT_DIR}/model_config_provenance.json"
