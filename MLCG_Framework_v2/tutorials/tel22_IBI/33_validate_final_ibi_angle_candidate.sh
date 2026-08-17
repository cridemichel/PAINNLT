#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYPRESSO="${PYPRESSO:-${ROOT}/espresso/build/pypresso}"
MODEL="${IBI_MODEL:-${HERE}/tel22_model_ibi_conservative.pt}"
CONFIG="${HERE}/tel22_training_config.json"
DATASET="${HERE}/tel22_dataset_ibi_residual.bin"
RB_INFO="${HERE}/rigid_bodies_info_ibi.json"
SOURCE_CHECKPOINT="${HERE}/nve_equilibration_conservative_ibi_only/equilibrated_conservative_ibi_only.npz"
IBI_CONFIG="${HERE}/ibi_settings.json"
CURRENT_PRIORS="${HERE}/ibi_conservative/cg_priors.json"
STEP32_REPORT="${HERE}/ibi_angle_smoothing_sweep/angle_smoothing_sweep_report.json"
CANDIDATE_PRIORS="${HERE}/ibi_angle_smoothing_sweep/candidates/smooth_0p0075_wall_current/cg_priors.json"
OUTPUT_DIR="${HERE}/ibi_angle_final_candidate_validation"
DTS=(0.001 0.002 0.003 0.004 0.005)
REPLICA_SEEDS=(330001 330002)

ARGS=(
  "${ROOT}/simulation/ibi_angle_final_validation.py"
  --python-bin "${PYTHON_BIN}"
  --pypresso "${PYPRESSO}"
  --model "${MODEL}"
  --config "${CONFIG}"
  --dataset "${DATASET}"
  --rb-info "${RB_INFO}"
  --source-checkpoint "${SOURCE_CHECKPOINT}"
  --ibi-config "${IBI_CONFIG}"
  --current-priors "${CURRENT_PRIORS}"
  --candidate-priors "${CANDIDATE_PRIORS}"
  --step32-report "${STEP32_REPORT}"
  --step32-candidate-name smooth_0p0075
  --expected-sigma-rad 0.0075
  --dts "${DTS[@]}"
  --nve-duration-ps 1.0
  --short-branch-dt 0.0005
  --short-branch-duration-ps 0.25
  --new-replica-seeds "${REPLICA_SEEDS[@]}"
  --long-branch-dt 0.0005
  --long-branch-duration-ps 2.0
  --long-thermostat-seed 330100
  --kT 2.49
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
[FINAL IBI ANGLE CANDIDATE VALIDATION]
candidate      : smooth_0p0075 (sigma=0.0075 rad)
current priors : ${CURRENT_PRIORS}
candidate      : ${CANDIDATE_PRIORS}
step32 report  : ${STEP32_REPORT}
replicas       : reuse step32 + ${REPLICA_SEEDS[*]}
dt scan        : ${DTS[*]} ps
NVE duration   : 1.0 ps per dt
long structure : 2.0 ps current + 2.0 ps candidate at dt=0.0005 ps
output         : ${OUTPUT_DIR}
[NOTE] Validation-only. No priors are promoted or overwritten.
EOF

"${PYTHON_BIN}" "${ARGS[@]}"
