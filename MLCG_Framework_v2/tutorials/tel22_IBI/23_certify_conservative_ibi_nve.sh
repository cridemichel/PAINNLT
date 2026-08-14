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

DRY_RUN=0
for arg in "$@"; do
    if [[ "${arg}" == "--dry-run" ]]; then
        DRY_RUN=1
    fi
done

MODEL="${IBI_MODEL:-tel22_model_ibi_conservative.pt}"
CONFIG="${TRAINING_CONFIG:-tel22_training_config.json}"
DATASET="${IBI_DATASET:-tel22_dataset_ibi_residual.bin}"
RB_INFO="${IBI_RB_INFO:-rigid_bodies_info_ibi.json}"
PRIORS="${IBI_PRIORS:-ibi_conservative/cg_priors.json}"
VALIDATION_REPORT="${IBI_VALIDATION_REPORT:-ibi_conservative/validation_report.json}"
RUNTIME_PARITY_REPORT="${IBI_RUNTIME_PARITY_REPORT:-ibi_conservative/runtime_parity_report.json}"
CHECKPOINT="${NVE_CHECKPOINT:-postibi_runtime_validation/equilibrated_postibi.npz}"

NVE_DEVICE="${NVE_DEVICE:-cpu}"
NVE_ML_PRECISION="${NVE_ML_PRECISION:-float32}"
NVE_NEIGHBOR_SEARCH="${NVE_NEIGHBOR_SEARCH:-link-cell}"
NVE_DURATION_PS="${NVE_DURATION_PS:-5.0}"
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-nve_certification_conservative_ibi_only}"
NVE_DTS="${NVE_DTS:-0.001 0.0015 0.002 0.003 0.004 0.005}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.97}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"
NVE_PREFLIGHT_REPORT="${NVE_PREFLIGHT_REPORT:-${NVE_OUTPUT_DIR}_preflight.json}"

for path in \
    "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" \
    "${PRIORS}" "${VALIDATION_REPORT}" "${RUNTIME_PARITY_REPORT}" "${CHECKPOINT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required conservative-IBI NVE artifact: ${path}" >&2
        exit 1
    fi
done

cat <<EOF
[CONSERVATIVE IBI-ONLY NVE PLAN]
model anchor : ${MODEL} (PaiNN disabled during NVE)
config       : ${CONFIG}
dataset      : ${DATASET}
rb_info      : ${RB_INFO}
priors       : ${PRIORS}
checkpoint   : ${CHECKPOINT}
dt scan      : ${NVE_DTS} ps
duration     : ${NVE_DURATION_PS} ps per dt
device       : ${NVE_DEVICE}
neighbor     : ${NVE_NEIGHBOR_SEARCH}
output       : ${NVE_OUTPUT_DIR}
EOF

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/conservative_nve_preflight.py" \
    --priors "${PRIORS}" \
    --validation-report "${VALIDATION_REPORT}" \
    --runtime-parity-report "${RUNTIME_PARITY_REPORT}" \
    --output "${NVE_PREFLIGHT_REPORT}"

read -r -a DT_ARGS <<< "${NVE_DTS}"

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/certify_nve.py" \
    --pypresso "${PYRESSO}" \
    --model "${MODEL}" \
    --disable-ml \
    --config "${CONFIG}" \
    --priors "${PRIORS}" \
    --rb-info "${RB_INFO}" \
    --dataset "${DATASET}" \
    --checkpoint "${CHECKPOINT}" \
    --dts "${DT_ARGS[@]}" \
    --duration-ps "${NVE_DURATION_PS}" \
    --device "${NVE_DEVICE}" \
    --ml-precision "${NVE_ML_PRECISION}" \
    --neighbor-search "${NVE_NEIGHBOR_SEARCH}" \
    --output-dir "${NVE_OUTPUT_DIR}" \
    --slope-min "${NVE_SLOPE_MIN}" \
    --slope-max "${NVE_SLOPE_MAX}" \
    --min-r2 "${NVE_MIN_R2}" \
    --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}" \
    --provenance-artifact "conservative_phase2_preflight=${NVE_PREFLIGHT_REPORT}" \
    --provenance-artifact "conservative_validation=${VALIDATION_REPORT}" \
    --provenance-artifact "conservative_runtime_parity=${RUNTIME_PARITY_REPORT}" \
    "$@"

if [[ "${DRY_RUN}" == "1" ]]; then
    cat <<EOF
[CONSERVATIVE IBI-ONLY NVE DRY-RUN COMPLETE]
preflight: ${NVE_PREFLIGHT_REPORT}
run plan : ${NVE_OUTPUT_DIR}/run_plan.json
[NOTE] No NVE trajectory was launched.
EOF
else
    cat <<EOF
[CONSERVATIVE IBI-ONLY NVE COMPLETE]
preflight: ${NVE_PREFLIGHT_REPORT}
report   : ${NVE_OUTPUT_DIR}/nve_certification_report.json
table    : ${NVE_OUTPUT_DIR}/nve_certification_runs.csv
[NOTE] PaiNN was disabled in every trajectory; ${MODEL} was retained only to satisfy the provenance contract of the shared post-IBI checkpoint.
EOF
fi
