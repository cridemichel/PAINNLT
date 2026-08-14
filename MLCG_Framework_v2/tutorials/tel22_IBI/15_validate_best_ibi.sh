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
IBI_BEST_DIR="${IBI_BEST_DIR:-ibi_run_16ps_continue}"
IBI_VALIDATION_OUTDIR="${IBI_VALIDATION_OUTDIR:-ibi_validation_best}"
NEIGHBOR_SEARCH="${NEIGHBOR_SEARCH:-link-cell}"
VALIDATION_VELOCITY_SEED="${VALIDATION_VELOCITY_SEED:-271828}"
VALIDATION_THERMOSTAT_SEED="${VALIDATION_THERMOSTAT_SEED:-161803}"
OVERWRITE="${OVERWRITE:-0}"
cd "${SCRIPT_DIR}"

BEST_PRIORS="${IBI_BEST_DIR}/best/cg_priors.json"
REFERENCE_SUMMARY="${IBI_BEST_DIR}/ibi_convergence_summary.json"
for path in tel22_dataset.bin ibi_settings.json tel22_training_config.json rigid_bodies_info.json "${BEST_PRIORS}" "${REFERENCE_SUMMARY}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

args=(
    "${FRAMEWORK_ROOT}/ibi/validate_ibi_priors.py"
    --dataset tel22_dataset.bin
    --priors "${BEST_PRIORS}"
    --reference-summary "${REFERENCE_SUMMARY}"
    --config tel22_training_config.json
    --rb_info rigid_bodies_info.json
    --pypresso "${PYPRESSO}"
    --outdir "${IBI_VALIDATION_OUTDIR}"
    --ibi-config ibi_settings.json
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
