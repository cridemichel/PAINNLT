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
load_model_dependent_config step24

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
        echo "[ERROR] Missing required NVE diagnostic artifact: ${path}" >&2
        if [[ "${path}" == "${NVE_EQ_CHECKPOINT}" || "${path}" == "${NVE_EQ_REPORT}" ]]; then
            echo "[HINT] Run step 23 once to prepare the dedicated conservative IBI-only NVT checkpoint." >&2
        fi
        exit 1
    fi
done

cat <<EOF_PLAN
[CONSERVATIVE IBI-ONLY NVE DIAGNOSTIC PLAN]
model anchor : ${MODEL} (PaiNN disabled)
priors       : ${PRIORS}
checkpoint   : ${NVE_EQ_CHECKPOINT}
dt scan      : ${NVE_DIAG_DTS} ps
duration     : ${NVE_DIAG_DURATION_PS} ps per dt
fine regime  : dt <= ${NVE_DIAG_FINE_MAX_DT} ps
coarse regime: dt >= ${NVE_DIAG_COARSE_MIN_DT} ps
local times  : ${NVE_DIAG_LOCAL_TIMES_PS} ps (fine regime only)
device       : ${NVE_DEVICE}
neighbor     : ${NVE_NEIGHBOR_SEARCH}
output       : ${NVE_DIAG_OUTPUT_DIR}
[NOTE] This is diagnostic-only. It does not relax or replace the strict step-23 NVE gate.
EOF_PLAN

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/conservative_nve_preflight.py" \
    --priors "${PRIORS}" \
    --validation-report "${VALIDATION_REPORT}" \
    --runtime-parity-report "${RUNTIME_PARITY_REPORT}" \
    --output "${NVE_DIAG_PREFLIGHT_REPORT}"

read -r -a DT_ARGS <<< "${NVE_DIAG_DTS}"
read -r -a LOCAL_TIME_ARGS <<< "${NVE_DIAG_LOCAL_TIMES_PS}"

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/certify_nve.py" \
    --pypresso "${PYRESSO}" \
    --model "${MODEL}" \
    --disable-ml \
    --config "${CONFIG}" \
    --priors "${PRIORS}" \
    --rb-info "${RB_INFO}" \
    --dataset "${DATASET}" \
    --checkpoint "${NVE_EQ_CHECKPOINT}" \
    --require-checkpoint-hamiltonian-mode "${REQUIRED_HAMILTONIAN_MODE}" \
    --require-checkpoint-source "${SOURCE_CHECKPOINT}" \
    --dts "${DT_ARGS[@]}" \
    --duration-ps "${NVE_DIAG_DURATION_PS}" \
    --device "${NVE_DEVICE}" \
    --ml-precision "${NVE_ML_PRECISION}" \
    --neighbor-search "${NVE_NEIGHBOR_SEARCH}" \
    --output-dir "${NVE_DIAG_OUTPUT_DIR}" \
    --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}" \
    --diagnostic-only \
    --diagnostic-fine-max-dt "${NVE_DIAG_FINE_MAX_DT}" \
    --diagnostic-coarse-min-dt "${NVE_DIAG_COARSE_MIN_DT}" \
    --local-times-ps "${LOCAL_TIME_ARGS[@]}" \
    --provenance-artifact "conservative_validation=${VALIDATION_REPORT}" \
    --provenance-artifact "conservative_runtime_parity=${RUNTIME_PARITY_REPORT}" \
    --provenance-artifact "ibi_only_nvt_equilibration=${NVE_EQ_REPORT}" \
    --provenance-artifact "diagnostic_preflight=${NVE_DIAG_PREFLIGHT_REPORT}" \
    --provenance-artifact "model_config=${NVE_DIAG_MODEL_CONFIG_PROVENANCE}" \
    "$@"

cat <<EOF_DONE
[CONSERVATIVE IBI-ONLY NVE DIAGNOSTIC COMPLETE]
report : ${NVE_DIAG_OUTPUT_DIR}/nve_diagnostic_report.json
table  : ${NVE_DIAG_OUTPUT_DIR}/nve_diagnostic_runs.csv
[NOTE] Interpret the fine-regime and local-energy fits before changing the conservative kernel.
EOF_DONE

write_model_dependent_provenance "${NVE_DIAG_OUTPUT_DIR}/model_config_provenance.json"
