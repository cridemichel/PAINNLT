#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
elif command -v pypresso >/dev/null 2>&1; then
    DEFAULT_PYPRESSO="$(command -v pypresso)"
else
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
fi

PYPRESSO="${PYPRESSO:-${DEFAULT_PYPRESSO}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE="${OVERWRITE:-0}"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/model_config.sh"
load_model_dependent_config step15
IBI_BEST_DIR="${IBI_BEST_DIR}"
IBI_VALIDATION_OUTDIR="${IBI_VALIDATION_OUTDIR}"
NEIGHBOR_SEARCH="${NEIGHBOR_SEARCH}"
VALIDATION_VELOCITY_SEED="${VALIDATION_VELOCITY_SEED}"
VALIDATION_THERMOSTAT_SEED="${VALIDATION_THERMOSTAT_SEED}"

BEST_PRIORS="${IBI_BEST_DIR}/best/cg_priors.json"
REFERENCE_SUMMARY="${IBI_BEST_DIR}/ibi_convergence_summary.json"
for path in "${IBI_TARGET_DATASET}" "${IBI_SETTINGS}" "${TRAINING_CONFIG}" "${VALIDATION_RB_INFO}" "${BEST_PRIORS}" "${REFERENCE_SUMMARY}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

args=(
    "${FRAMEWORK_ROOT}/ibi/validate_ibi_priors.py"
    --dataset "${IBI_TARGET_DATASET}"
    --priors "${BEST_PRIORS}"
    --reference-summary "${REFERENCE_SUMMARY}"
    --config "${TRAINING_CONFIG}"
    --rb_info "${VALIDATION_RB_INFO}"
    --pypresso "${PYPRESSO}"
    --outdir "${IBI_VALIDATION_OUTDIR}"
    --ibi-config "${IBI_SETTINGS}"
    --neighbor_search "${NEIGHBOR_SEARCH}"
    --velocity_seed "${VALIDATION_VELOCITY_SEED}"
    --thermostat_seed "${VALIDATION_THERMOSTAT_SEED}"
)
if [ "${OVERWRITE}" = "1" ]; then
    args+=(--overwrite)
fi

"${PYTHON_BIN}" "${args[@]}"

echo "[DONE] Read-only best-prior validation: ${IBI_VALIDATION_OUTDIR}/validation_report.json"
echo "[NEXT] If the independent L1 values are consistent with the selected best set, freeze the priors and rebuild the residual dataset with ./13_rebuild_residual_dataset.sh."

write_model_dependent_provenance "${IBI_VALIDATION_OUTDIR}/model_config_provenance.json"
