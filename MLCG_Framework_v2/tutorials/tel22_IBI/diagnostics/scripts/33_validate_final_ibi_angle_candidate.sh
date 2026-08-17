#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${HERE}/model_config.sh" || -f "${HERE}/cg_priors.json" ]]; then
  TUTORIAL_DIR="${HERE}"
else
  TUTORIAL_DIR="$(cd "${HERE}/../.." && pwd)"
fi
ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
source "${TUTORIAL_DIR}/model_config.sh"
load_model_dependent_config step33
cd "${TUTORIAL_DIR}"
PYPRESSO="${PYPRESSO:-${ROOT}/espresso/build/pypresso}"

MODEL="${IBI_MODEL}"
CONFIG="${IBI_ANGLE_FINAL_CONFIG}"
DATASET="${IBI_ANGLE_FINAL_DATASET}"
RB_INFO="${IBI_ANGLE_FINAL_RB_INFO}"
SOURCE_CHECKPOINT="${IBI_ANGLE_FINAL_SOURCE_CHECKPOINT}"
IBI_CONFIG="${IBI_ANGLE_FINAL_IBI_CONFIG}"
CURRENT_PRIORS="${IBI_ANGLE_FINAL_CURRENT_PRIORS}"
STEP32_REPORT="${IBI_ANGLE_FINAL_STEP32_REPORT}"
CANDIDATE_PRIORS="${IBI_ANGLE_FINAL_CANDIDATE_PRIORS}"
OUTPUT_DIR="${IBI_ANGLE_FINAL_OUTPUT_DIR}"
CONFIG_PROVENANCE="${OUTPUT_DIR}/model_config_provenance.json"
read -r -a DTS <<< "${IBI_ANGLE_FINAL_DTS}"
read -r -a REPLICA_SEEDS <<< "${IBI_ANGLE_FINAL_NEW_REPLICA_SEEDS}"

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
  --step32-candidate-name "${IBI_ANGLE_FINAL_CANDIDATE_NAME}"
  --expected-sigma-rad "${IBI_ANGLE_FINAL_EXPECTED_SIGMA_RAD}"
  --dts "${DTS[@]}"
  --nve-duration-ps "${IBI_ANGLE_FINAL_NVE_DURATION_PS}"
  --short-branch-dt "${IBI_ANGLE_FINAL_SHORT_BRANCH_DT}"
  --short-branch-duration-ps "${IBI_ANGLE_FINAL_SHORT_BRANCH_DURATION_PS}"
  --new-replica-seeds "${REPLICA_SEEDS[@]}"
  --long-branch-dt "${IBI_ANGLE_FINAL_LONG_BRANCH_DT}"
  --long-branch-duration-ps "${IBI_ANGLE_FINAL_LONG_BRANCH_DURATION_PS}"
  --long-thermostat-seed "${IBI_ANGLE_FINAL_LONG_THERMOSTAT_SEED}"
  --kT "${IBI_ANGLE_FINAL_KT}"
  --device "${IBI_ANGLE_FINAL_DEVICE}"
  --ml-precision "${IBI_ANGLE_FINAL_ML_PRECISION}"
  --neighbor-search "${IBI_ANGLE_FINAL_NEIGHBOR_SEARCH}"
  --output-dir "${OUTPUT_DIR}"
  --min-clean-dt "${IBI_ANGLE_FINAL_MIN_CLEAN_DT}"
  --full-clean-dt "${IBI_ANGLE_FINAL_FULL_CLEAN_DT}"
  --common-p-min "${IBI_ANGLE_FINAL_COMMON_P_MIN}"
  --common-p-max "${IBI_ANGLE_FINAL_COMMON_P_MAX}"
  --common-r2-min "${IBI_ANGLE_FINAL_COMMON_R2_MIN}"
  --min-full-clean-replicas "${IBI_ANGLE_FINAL_MIN_FULL_CLEAN_REPLICAS}"
  --median-c2-spread-max "${IBI_ANGLE_FINAL_MEDIAN_C2_SPREAD_MAX}"
  --max-relative-block-drift "${IBI_ANGLE_FINAL_MAX_RELATIVE_BLOCK_DRIFT}"
  --weighted-angle-delta-max "${IBI_ANGLE_FINAL_WEIGHTED_ANGLE_DELTA_MAX}"
  --weighted-bond-delta-max "${IBI_ANGLE_FINAL_WEIGHTED_BOND_DELTA_MAX}"
  --max-group-angle-delta-max "${IBI_ANGLE_FINAL_MAX_GROUP_ANGLE_DELTA_MAX}"
  --kinetic-relative-delta-max "${IBI_ANGLE_FINAL_KINETIC_RELATIVE_DELTA_MAX}"
  --curvature-reduction-min "${IBI_ANGLE_FINAL_CURVATURE_REDUCTION_MIN}"
)
for arg in "$@"; do
  case "${arg}" in
    --dry-run|--overwrite|--resume) ARGS+=("${arg}") ;;
    *) echo "[ERROR] Unsupported argument: ${arg}" >&2; exit 2 ;;
  esac
done

cat <<EOF2
[FINAL IBI ANGLE CANDIDATE VALIDATION]
candidate      : ${IBI_ANGLE_FINAL_CANDIDATE_NAME} (sigma=${IBI_ANGLE_FINAL_EXPECTED_SIGMA_RAD} rad)
current priors : ${CURRENT_PRIORS}
candidate      : ${CANDIDATE_PRIORS}
step32 report  : ${STEP32_REPORT}
replicas       : reuse step32 + ${REPLICA_SEEDS[*]}
dt scan        : ${DTS[*]} ps
NVE duration   : ${IBI_ANGLE_FINAL_NVE_DURATION_PS} ps per dt
long structure : ${IBI_ANGLE_FINAL_LONG_BRANCH_DURATION_PS} ps current + candidate at dt=${IBI_ANGLE_FINAL_LONG_BRANCH_DT} ps
output         : ${OUTPUT_DIR}
[NOTE] Validation-only. No priors are promoted or overwritten.
EOF2

"${PYTHON_BIN}" "${ARGS[@]}"
if [[ " $* " != *" --dry-run "* ]]; then
  write_model_dependent_provenance "${CONFIG_PROVENANCE}"
fi
