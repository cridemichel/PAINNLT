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
PYPRESSO="${PYPRESSO:-${DEFAULT_PYPRESSO}}"

cd "${SCRIPT_DIR}"

MODEL="${IBI_MODEL:-tel22_model_ibi_conservative.pt}"
CONFIG="${TRAINING_CONFIG:-tel22_training_config.json}"
DATASET="${IBI_DATASET:-tel22_dataset_ibi_residual.bin}"
RB_INFO="${IBI_RB_INFO:-rigid_bodies_info_ibi.json}"
PRIORS="${IBI_PRIORS:-ibi_conservative/cg_priors.json}"
BASE_CHECKPOINT="${SIGMA_REPLICA_BASE_CHECKPOINT:-nve_equilibration_conservative_ibi_only/equilibrated_conservative_ibi_only.npz}"
LOCALIZATION_REPORT="${SIGMA_REPLICA_LOCALIZATION_REPORT:-conservative_ibi_energy_localization/localization_report.json}"

# The default grid is intentionally power-of-two in dt so every requested
# duration is an exact integer number of integration steps at every dt.
SIGMA_REPLICA_DTS="${SIGMA_REPLICA_DTS:-0.001 0.0005 0.00025 0.000125}"
SIGMA_REPLICA_DURATIONS_PS="${SIGMA_REPLICA_DURATIONS_PS:-0.125 0.25 0.5 1 2}"
SIGMA_REPLICA_COUNT="${SIGMA_REPLICA_COUNT:-4}"
SIGMA_REPLICA_EQ_DT="${SIGMA_REPLICA_EQ_DT:-0.0005}"
SIGMA_REPLICA_EQ_DURATION_PS="${SIGMA_REPLICA_EQ_DURATION_PS:-1.0}"
SIGMA_REPLICA_KT="${SIGMA_REPLICA_KT:-2.49}"
SIGMA_REPLICA_SEED_BASE="${SIGMA_REPLICA_SEED_BASE:-280000}"
SIGMA_REPLICA_BOOTSTRAP_SAMPLES="${SIGMA_REPLICA_BOOTSTRAP_SAMPLES:-1000}"
SIGMA_REPLICA_BOOTSTRAP_SEED="${SIGMA_REPLICA_BOOTSTRAP_SEED:-20260816}"
SIGMA_REPLICA_DEVICE="${SIGMA_REPLICA_DEVICE:-cpu}"
SIGMA_REPLICA_ML_PRECISION="${SIGMA_REPLICA_ML_PRECISION:-float32}"
SIGMA_REPLICA_NEIGHBOR_SEARCH="${SIGMA_REPLICA_NEIGHBOR_SEARCH:-link-cell}"
SIGMA_REPLICA_OUTPUT_DIR="${SIGMA_REPLICA_OUTPUT_DIR:-sigma_energy_replica_window_diagnostic}"

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

exec "${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/sigma_energy_replica_diagnostics.py" \
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
    --device "${SIGMA_REPLICA_DEVICE}" \
    --ml-precision "${SIGMA_REPLICA_ML_PRECISION}" \
    --neighbor-search "${SIGMA_REPLICA_NEIGHBOR_SEARCH}" \
    --output-dir "${SIGMA_REPLICA_OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
