#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILDER="${FRAMEWORK_ROOT}/preprocessing/build_cg_dataset.py"
PROVENANCE_TOOL="${FRAMEWORK_ROOT}/training/residual_input_provenance.py"
source "${SCRIPT_DIR}/model_config.sh"
load_model_dependent_config step13
if [[ -z "${IBI_PRIORS:-}" ]]; then
    for candidate in ${IBI_PRIOR_CANDIDATES}; do [[ -f "${SCRIPT_DIR}/${candidate}" ]] && { IBI_PRIORS="${candidate}"; break; }; done
fi
[[ -n "${IBI_PRIORS:-}" ]] || { echo "[ERROR] No configured IBI prior candidate exists." >&2; exit 1; }

cd "${SCRIPT_DIR}"

if [ -n "${IBI_VALIDATION_REPORT:-}" ]; then
    IBI_VALIDATION_REPORT="${IBI_VALIDATION_REPORT}"
elif [[ "${IBI_PRIORS}" == ibi_conservative/cg_priors.json || "${IBI_PRIORS}" == */ibi_conservative/cg_priors.json ]]; then
    IBI_VALIDATION_REPORT="$(dirname "${IBI_PRIORS}")/validation_report.json"
elif [ -f "$(dirname "${IBI_PRIORS}")/validation_report.json" ]; then
    IBI_VALIDATION_REPORT="$(dirname "${IBI_PRIORS}")/validation_report.json"
else
    IBI_VALIDATION_REPORT="${IBI_FALLBACK_VALIDATION_REPORT}"
fi
if [ -n "${IBI_RUNTIME_PARITY_REPORT:-}" ]; then
    IBI_RUNTIME_PARITY_REPORT="${IBI_RUNTIME_PARITY_REPORT}"
elif [[ "${IBI_PRIORS}" == ibi_conservative/cg_priors.json || "${IBI_PRIORS}" == */ibi_conservative/cg_priors.json ]]; then
    IBI_RUNTIME_PARITY_REPORT="$(dirname "${IBI_PRIORS}")/runtime_parity_report.json"
elif [ -f "$(dirname "${IBI_PRIORS}")/runtime_parity_report.json" ]; then
    IBI_RUNTIME_PARITY_REPORT="$(dirname "${IBI_PRIORS}")/runtime_parity_report.json"
else
    IBI_RUNTIME_PARITY_REPORT=""
fi



REQUIRED_INPUTS=(
    "${AA_TOPOLOGY}"
    "${AA_TRAJECTORY}"
    "${MAPPING_CONFIG}"
    "${IBI_PRIORS}"
    "${IBI_VALIDATION_REPORT}"
)
if [ -n "${IBI_RUNTIME_PARITY_REPORT}" ]; then
    REQUIRED_INPUTS+=("${IBI_RUNTIME_PARITY_REPORT}")
fi
for path in "${REQUIRED_INPUTS[@]}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" "${BUILDER}" \
    --topology "${AA_TOPOLOGY}" \
    --trajectory "${AA_TRAJECTORY}" \
    --config "${MAPPING_CONFIG}" \
    --priors "${IBI_PRIORS}" \
    --output "${OUTPUT_DATASET}" \
    --rb-info-output "${OUTPUT_RB_INFO}"

PROVENANCE_ARGS=(
    record
    --output "${OUTPUT_PROVENANCE}"
    --dataset "${OUTPUT_DATASET}"
    --rb-info "${OUTPUT_RB_INFO}"
    --priors "${IBI_PRIORS}"
    --aa-topology "${AA_TOPOLOGY}"
    --aa-trajectory "${AA_TRAJECTORY}"
    --mapping-config "${MAPPING_CONFIG}"
    --validation-report "${IBI_VALIDATION_REPORT}"
)
if [ -n "${IBI_RUNTIME_PARITY_REPORT}" ]; then
    PROVENANCE_ARGS+=(--runtime-parity-report "${IBI_RUNTIME_PARITY_REPORT}")
fi
"${PYTHON_BIN}" "${PROVENANCE_TOOL}" "${PROVENANCE_ARGS[@]}"

echo "[DONE] Residual dataset with selected IBI priors: ${OUTPUT_DATASET}"
echo "[DONE] Rigid-body metadata: ${OUTPUT_RB_INFO}"
echo "[DONE] Residual-build provenance: ${OUTPUT_PROVENANCE}"

write_model_dependent_provenance "${OUTPUT_PROVENANCE%.json}_model_config_provenance.json"
