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
PYPRESSO="${PYPRESSO:-${DEFAULT_PYPRESSO}}"

cd "${TUTORIAL_DIR}"
source "${TUTORIAL_DIR}/model_config.sh"
load_model_dependent_config step28

MODEL="${IBI_MODEL}"
CONFIG="${TRAINING_CONFIG}"
DATASET="${IBI_DATASET}"
RB_INFO="${IBI_RB_INFO}"
PRIORS="${IBI_PRIORS}"
BASE_CHECKPOINT="${SIGMA_REPLICA_BASE_CHECKPOINT}"
LOCALIZATION_REPORT="${SIGMA_REPLICA_LOCALIZATION_REPORT}"


for path in "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" "${PRIORS}" "${BASE_CHECKPOINT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required replica-scaling artifact: ${path}" >&2
        exit 1
    fi
done

read -r -a DT_ARGS <<< "${SIGMA_REPLICA_DTS}"
read -r -a DURATION_ARGS <<< "${SIGMA_REPLICA_DURATIONS_PS}"

EXTRA_ARGS=()
if [[ -f "${LOCALIZATION_REPORT}" ]]; then
    EXTRA_ARGS+=(--localization-report "${LOCALIZATION_REPORT}")
fi

cat <<EOF
[CONSERVATIVE IBI SIGMA(E) REPLICA/WINDOW DIAGNOSTIC]
model anchor : ${MODEL} (PaiNN disabled)
priors       : ${PRIORS}
base chk     : ${BASE_CHECKPOINT}
dt scan      : ${SIGMA_REPLICA_DTS} ps
durations    : ${SIGMA_REPLICA_DURATIONS_PS} ps (exact prefixes)
replicas     : ${SIGMA_REPLICA_COUNT}
NVT branch   : dt=${SIGMA_REPLICA_EQ_DT} ps duration=${SIGMA_REPLICA_EQ_DURATION_PS} ps kT=${SIGMA_REPLICA_KT}
seed base    : ${SIGMA_REPLICA_SEED_BASE}
neighbor     : ${SIGMA_REPLICA_NEIGHBOR_SEARCH}
output       : ${SIGMA_REPLICA_OUTPUT_DIR}
[NOTE] Each replica/dt is integrated once to the longest duration; shorter sigma(E) windows are prefixes.
[NOTE] sigma(E) is raw std(E_tot), ddof=0, with no detrending.
[NOTE] Diagnostic-only. No certification report is modified.
EOF

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/sigma_energy_replica_diagnostics.py" \
    --pypresso "${PYPRESSO}" \
    --model "${MODEL}" \
    --config "${CONFIG}" \
    --priors "${PRIORS}" \
    --rb-info "${RB_INFO}" \
    --dataset "${DATASET}" \
    --base-checkpoint "${BASE_CHECKPOINT}" \
    --dts "${DT_ARGS[@]}" \
    --durations "${DURATION_ARGS[@]}" \
    --replicas "${SIGMA_REPLICA_COUNT}" \
    --replica-equilibration-dt "${SIGMA_REPLICA_EQ_DT}" \
    --replica-equilibration-duration-ps "${SIGMA_REPLICA_EQ_DURATION_PS}" \
    --kT "${SIGMA_REPLICA_KT}" \
    --seed-base "${SIGMA_REPLICA_SEED_BASE}" \
    --bootstrap-samples "${SIGMA_REPLICA_BOOTSTRAP_SAMPLES}" \
    --bootstrap-seed "${SIGMA_REPLICA_BOOTSTRAP_SEED}" \
    --second-order-p-min "${SIGMA_REPLICA_P_MIN}" \
    --second-order-p-max "${SIGMA_REPLICA_P_MAX}" \
    --second-order-r2-min "${SIGMA_REPLICA_R2_MIN}" \
    --device "${SIGMA_REPLICA_DEVICE}" \
    --ml-precision "${SIGMA_REPLICA_ML_PRECISION}" \
    --neighbor-search "${SIGMA_REPLICA_NEIGHBOR_SEARCH}" \
    --output-dir "${SIGMA_REPLICA_OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

write_model_dependent_provenance "${SIGMA_REPLICA_OUTPUT_DIR}/model_config_provenance.json"
