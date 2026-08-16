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

MODEL="${IBI_MODEL:-tel22_model_ibi_conservative.pt}"
CONFIG="${TRAINING_CONFIG:-tel22_training_config.json}"
DATASET="${IBI_DATASET:-tel22_dataset_ibi_residual.bin}"
RB_INFO="${IBI_RB_INFO:-rigid_bodies_info_ibi.json}"
PRIORS="${IBI_PRIORS:-ibi_conservative/cg_priors.json}"
VALIDATION_REPORT="${IBI_VALIDATION_REPORT:-ibi_conservative/validation_report.json}"
RUNTIME_PARITY_REPORT="${IBI_RUNTIME_PARITY_REPORT:-ibi_conservative/runtime_parity_report.json}"
SOURCE_CHECKPOINT="${NVE_SOURCE_CHECKPOINT:-postibi_runtime_validation/equilibrated_postibi.npz}"
NVE_EQ_DIR="${NVE_EQ_DIR:-nve_equilibration_conservative_ibi_only}"
NVE_EQ_CHECKPOINT="${NVE_EQ_CHECKPOINT:-${NVE_EQ_DIR}/equilibrated_conservative_ibi_only.npz}"
NVE_EQ_REPORT="${NVE_EQ_REPORT:-${NVE_EQ_DIR}/equilibration_report.json}"
REQUIRED_HAMILTONIAN_MODE="conservative_classical_model_provenance_ml_disabled"

NVE_DEVICE="${NVE_DEVICE:-cpu}"
NVE_ML_PRECISION="${NVE_ML_PRECISION:-float32}"
NVE_NEIGHBOR_SEARCH="${NVE_NEIGHBOR_SEARCH:-link-cell}"
NVE_DIAG_DURATION_PS="${NVE_DIAG_DURATION_PS:-2.0}"
NVE_DIAG_DTS="${NVE_DIAG_DTS:-0.00025 0.0005 0.00075 0.001 0.0015 0.002 0.003 0.004 0.005}"
NVE_DIAG_FINE_MAX_DT="${NVE_DIAG_FINE_MAX_DT:-0.001}"
NVE_DIAG_COARSE_MIN_DT="${NVE_DIAG_COARSE_MIN_DT:-0.0015}"
# These times are exact multiples of every default fine-regime dt
# (0.00025, 0.0005, 0.00075, 0.001 ps), so no energy interpolation is needed.
NVE_DIAG_LOCAL_TIMES_PS="${NVE_DIAG_LOCAL_TIMES_PS:-0.012 0.024 0.048 0.096}"
NVE_DIAG_OUTPUT_DIR="${NVE_DIAG_OUTPUT_DIR:-nve_diagnostic_conservative_ibi_only}"
NVE_DIAG_PREFLIGHT_REPORT="${NVE_DIAG_PREFLIGHT_REPORT:-${NVE_DIAG_OUTPUT_DIR}_preflight.json}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

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
    "$@"

cat <<EOF_DONE
[CONSERVATIVE IBI-ONLY NVE DIAGNOSTIC COMPLETE]
report : ${NVE_DIAG_OUTPUT_DIR}/nve_diagnostic_report.json
table  : ${NVE_DIAG_OUTPUT_DIR}/nve_diagnostic_runs.csv
[NOTE] Interpret the fine-regime and local-energy fits before changing the conservative kernel.
EOF_DONE
