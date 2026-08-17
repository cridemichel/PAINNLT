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
load_model_dependent_config step27

MODEL="${IBI_MODEL}"
CONFIG="${TRAINING_CONFIG}"
DATASET="${IBI_DATASET}"
RB_INFO="${IBI_RB_INFO}"
PRIORS="${IBI_PRIORS}"
CHECKPOINT="${NVE_EQ_CHECKPOINT}"


for path in "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" "${PRIORS}" "${CHECKPOINT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required energy-localization artifact: ${path}" >&2
        if [[ "${path}" == "${CHECKPOINT}" ]]; then
            echo "[HINT] Run step 23 first to create the dedicated conservative IBI-only NVT checkpoint." >&2
        fi
        exit 1
    fi
done

read -r -a DT_ARGS <<< "${ENERGY_LOC_DTS}"

cat <<EOF
[CONSERVATIVE IBI ENERGY-SCALING LOCALIZATION]
model anchor : ${MODEL} (PaiNN disabled)
priors       : ${PRIORS}
checkpoint   : ${CHECKPOINT}
dt scan      : ${ENERGY_LOC_DTS} ps
full duration: ${ENERGY_LOC_DURATION_PS} ps per dt
micro duration: ${ENERGY_LOC_MICRO_DURATION_PS} ps per dt
trace dt     : ${ENERGY_LOC_TRACE_DT} ps
variants     : no_ibi / bonds_only / angles_only / full
extra probes : U'' knot jumps; knot-energy correlation; generalized force/torque FD;
               micro bond/angle crossing; rigid-angle torque; reversibility;
               link-cell vs nsquare
output       : ${ENERGY_LOC_OUTPUT_DIR}
[NOTE] Diagnostic-only. Steps 23-26 and their reports are not modified.
EOF

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/diagnose_conservative_ibi_localization.py" \
    --pypresso "${PYRESSO}" \
    --model "${MODEL}" \
    --config "${CONFIG}" \
    --priors "${PRIORS}" \
    --rb-info "${RB_INFO}" \
    --dataset "${DATASET}" \
    --checkpoint "${CHECKPOINT}" \
    --dts "${DT_ARGS[@]}" \
    --duration-ps "${ENERGY_LOC_DURATION_PS}" \
    --micro-duration-ps "${ENERGY_LOC_MICRO_DURATION_PS}" \
    --trace-dt "${ENERGY_LOC_TRACE_DT}" \
    --reversibility-dt "${ENERGY_LOC_REVERSIBILITY_DT}" \
    --reversibility-duration-ps "${ENERGY_LOC_REVERSIBILITY_DURATION_PS}" \
    --neighbor-duration-ps "${ENERGY_LOC_NEIGHBOR_DURATION_PS}" \
    --fd-max-bodies "${ENERGY_LOC_FD_MAX_BODIES}" \
    --fine-max-dt "${ENERGY_LOC_FINE_MAX_DT}" \
    --device "${ENERGY_LOC_DEVICE}" \
    --ml-precision "${ENERGY_LOC_ML_PRECISION}" \
    --output-dir "${ENERGY_LOC_OUTPUT_DIR}" \
    "$@"

write_model_dependent_provenance "${ENERGY_LOC_OUTPUT_DIR}/model_config_provenance.json"
