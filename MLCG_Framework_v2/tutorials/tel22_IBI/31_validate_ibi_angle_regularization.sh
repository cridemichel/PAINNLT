#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${SCRIPT_DIR}"

PYPRESSO="${PYPRESSO:-${FRAMEWORK_ROOT}/espresso/build/pypresso}"
MODEL="${IBI_MODEL:-tel22_model_ibi_conservative.pt}"
CONFIG="${IBI_CONFIG_JSON:-tel22_training_config.json}"
DATASET="${IBI_DATASET:-tel22_dataset_ibi_residual.bin}"
RB_INFO="${IBI_RB_INFO:-rigid_bodies_info_ibi.json}"
SOURCE_CHECKPOINT="${IBI_SOURCE_CHECKPOINT:-nve_equilibration_conservative_ibi_only/equilibrated_conservative_ibi_only.npz}"
IBI_SETTINGS="${IBI_SETTINGS:-ibi_settings.json}"
CURRENT_PRIORS="${IBI_PRIORS:-ibi_conservative/cg_priors.json}"
CANDIDATE_ROOT="${IBI_ANGLE_REG_OUTPUT_DIR:-ibi_angle_regularization_diagnostic}/candidates"
CANDIDATE_01="${IBI_ANGLE_CANDIDATE_01:-${CANDIDATE_ROOT}/smooth_0p01_wall_current/cg_priors.json}"
CANDIDATE_02="${IBI_ANGLE_CANDIDATE_02:-${CANDIDATE_ROOT}/smooth_0p02_wall_current/cg_priors.json}"
OUTPUT_DIR="${IBI_ANGLE_VALIDATION_OUTPUT_DIR:-ibi_angle_regularization_validation}"
DTS="${IBI_ANGLE_VALIDATION_DTS:-0.001 0.0015 0.002 0.003 0.004 0.005}"
NVE_DURATION="${IBI_ANGLE_VALIDATION_NVE_DURATION_PS:-1.0}"
BRANCH_DT="${IBI_ANGLE_VALIDATION_BRANCH_DT_PS:-0.0005}"
BRANCH_DURATION="${IBI_ANGLE_VALIDATION_BRANCH_DURATION_PS:-0.25}"
KT="${IBI_ANGLE_VALIDATION_KT:-2.49}"
SEED="${IBI_ANGLE_VALIDATION_SEED:-310000}"

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
  --device cpu
  --ml-precision float32
  --neighbor-search link-cell
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
candidate 0.01 : ${CANDIDATE_01}
candidate 0.02 : ${CANDIDATE_02}
source chk     : ${SOURCE_CHECKPOINT}
dt scan        : ${DTS} ps
NVE duration   : ${NVE_DURATION} ps per dt
NVT branch     : ${BRANCH_DURATION} ps at dt=${BRANCH_DT} ps kT=${KT}
thermostat seed: ${SEED} (same seed/protocol for all variants)
output         : ${OUTPUT_DIR}
[NOTE] Diagnostic-only. Candidates remain unvalidated and are never promoted automatically.
EOF

"${PYTHON_BIN}" "${ARGS[@]}"
