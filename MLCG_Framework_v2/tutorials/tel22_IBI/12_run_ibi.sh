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
IBI_ITERATIONS="${IBI_ITERATIONS:-5}"
IBI_OUTDIR="${IBI_OUTDIR:-ibi_run_16ps}"
NEIGHBOR_SEARCH="${NEIGHBOR_SEARCH:-link-cell}"
VELOCITY_SEED="${VELOCITY_SEED:-314159}"
THERMOSTAT_SEED="${THERMOSTAT_SEED:-42}"
OVERWRITE="${OVERWRITE:-0}"
cd "${SCRIPT_DIR}"

for path in tel22_dataset.bin cg_priors_ibi_seed.json ibi_settings.json tel22_training_config.json rigid_bodies_info.json; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

args=(
    "${FRAMEWORK_ROOT}/ibi/run_ibi_loop.py"
    --dataset tel22_dataset.bin
    --priors cg_priors_ibi_seed.json
    --config tel22_training_config.json
    --rb_info rigid_bodies_info.json
    --pypresso "${PYPRESSO}"
    --iterations "${IBI_ITERATIONS}"
    --outdir "${IBI_OUTDIR}"
    --ibi-config ibi_settings.json
    --neighbor_search "${NEIGHBOR_SEARCH}"
    --velocity_seed "${VELOCITY_SEED}"
    --thermostat_seed "${THERMOSTAT_SEED}"
)
if [ "${OVERWRITE}" = "1" ]; then
    args+=(--overwrite)
fi

"${PYTHON_BIN}" "${args[@]}"

summary_args=(
    "${SCRIPT_DIR}/summarize_ibi_convergence.py"
    --report "${IBI_OUTDIR}/ibi_report.json"
    --output "${IBI_OUTDIR}/ibi_convergence_summary.json"
    --best-dir "${IBI_OUTDIR}/best"
)
if [ "${OVERWRITE}" = "1" ]; then
    summary_args+=(--overwrite)
fi
"${PYTHON_BIN}" "${summary_args[@]}"

echo "[DONE] Final post-update IBI output: ${IBI_OUTDIR}/cg_priors_final.json"
echo "[DONE] Best evaluated priors: ${IBI_OUTDIR}/best/cg_priors.json"
echo "[NOTE] Defaults now run 5 iterations with 16 ps production per iteration."
