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
SOURCE_PRIORS="${IBI_PRIORS:-ibi_conservative/cg_priors.json}"
STEP31_REPORT="${IBI_ANGLE_STEP31_REPORT:-ibi_angle_regularization_validation/angle_candidate_validation_report.json}"
OUTPUT_DIR="${IBI_ANGLE_SWEEP_OUTPUT_DIR:-ibi_angle_smoothing_sweep}"
NEW_SIGMAS="${IBI_ANGLE_SWEEP_SIGMAS:-0.0075 0.0125 0.015}"
REUSE_SIGMA="${IBI_ANGLE_SWEEP_REUSE_SIGMA:-0.01}"
REUSE_VARIANT="${IBI_ANGLE_SWEEP_REUSE_VARIANT:-smooth_0p01}"
DTS="${IBI_ANGLE_SWEEP_DTS:-0.001 0.002 0.003 0.004 0.005}"
NVE_DURATION="${IBI_ANGLE_SWEEP_NVE_DURATION_PS:-1.0}"
BRANCH_DT="${IBI_ANGLE_SWEEP_BRANCH_DT_PS:-0.0005}"
BRANCH_DURATION="${IBI_ANGLE_SWEEP_BRANCH_DURATION_PS:-0.25}"
KT="${IBI_ANGLE_SWEEP_KT:-2.49}"
SEED="${IBI_ANGLE_SWEEP_SEED:-310000}"

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
