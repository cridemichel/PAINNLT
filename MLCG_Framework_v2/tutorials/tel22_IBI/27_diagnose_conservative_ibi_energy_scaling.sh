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
NVE_EQ_DIR="${NVE_EQ_DIR:-nve_equilibration_conservative_ibi_only}"
CHECKPOINT="${NVE_EQ_CHECKPOINT:-${NVE_EQ_DIR}/equilibrated_conservative_ibi_only.npz}"

ENERGY_LOC_DTS="${ENERGY_LOC_DTS:-0.001 0.00075 0.0005 0.000375 0.00025 0.0001875 0.000125}"
ENERGY_LOC_DURATION_PS="${ENERGY_LOC_DURATION_PS:-0.25}"
ENERGY_LOC_MICRO_DURATION_PS="${ENERGY_LOC_MICRO_DURATION_PS:-0.096}"
ENERGY_LOC_TRACE_DT="${ENERGY_LOC_TRACE_DT:-0.001}"
ENERGY_LOC_REVERSIBILITY_DT="${ENERGY_LOC_REVERSIBILITY_DT:-0.0005}"
ENERGY_LOC_REVERSIBILITY_DURATION_PS="${ENERGY_LOC_REVERSIBILITY_DURATION_PS:-0.024}"
ENERGY_LOC_NEIGHBOR_DURATION_PS="${ENERGY_LOC_NEIGHBOR_DURATION_PS:-0.024}"
ENERGY_LOC_FD_MAX_BODIES="${ENERGY_LOC_FD_MAX_BODIES:-8}"
ENERGY_LOC_DEVICE="${ENERGY_LOC_DEVICE:-cpu}"
ENERGY_LOC_ML_PRECISION="${ENERGY_LOC_ML_PRECISION:-float32}"
ENERGY_LOC_OUTPUT_DIR="${ENERGY_LOC_OUTPUT_DIR:-conservative_ibi_energy_localization}"

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

exec "${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/diagnose_conservative_ibi_localization.py" \
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
    --device "${ENERGY_LOC_DEVICE}" \
    --ml-precision "${ENERGY_LOC_ML_PRECISION}" \
    --output-dir "${ENERGY_LOC_OUTPUT_DIR}" \
    "$@"
