#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKER="${FRAMEWORK_ROOT}/training/residual_input_provenance.py"

cd "${SCRIPT_DIR}"

DATASET="${IBI_DATASET:-tel22_dataset_ibi_residual.bin}"
RB_INFO="${IBI_RB_INFO:-rigid_bodies_info_ibi.json}"
if [ -n "${IBI_PRIORS:-}" ]; then
    PRIORS="${IBI_PRIORS}"
elif [ -f "ibi_conservative/cg_priors.json" ]; then
    PRIORS="ibi_conservative/cg_priors.json"
elif [ -f "ibi_run_16ps_continue/best/cg_priors.json" ]; then
    PRIORS="ibi_run_16ps_continue/best/cg_priors.json"
else
    PRIORS="ibi_run_16ps/best/cg_priors.json"
fi
PROVENANCE="${IBI_RESIDUAL_PROVENANCE:-ibi_residual_build_manifest.json}"

for path in "${DATASET}" "${RB_INFO}" "${PRIORS}" "${PROVENANCE}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required IBI training artifact: ${path}" >&2
        echo "[HINT] Re-run ./13_rebuild_residual_dataset.sh after applying this patch." >&2
        exit 1
    fi
done

"${PYTHON_BIN}" "${CHECKER}" check \
    --manifest "${PROVENANCE}" \
    --dataset "${DATASET}" \
    --rb-info "${RB_INFO}" \
    --priors "${PRIORS}"
