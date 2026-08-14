#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILDER="${FRAMEWORK_ROOT}/preprocessing/build_cg_dataset.py"
PROVENANCE_TOOL="${FRAMEWORK_ROOT}/training/residual_input_provenance.py"
if [ -n "${IBI_PRIORS:-}" ]; then
    IBI_PRIORS="${IBI_PRIORS}"
elif [ -f "${SCRIPT_DIR}/ibi_conservative/cg_priors.json" ]; then
    IBI_PRIORS="ibi_conservative/cg_priors.json"
elif [ -f "${SCRIPT_DIR}/ibi_run_16ps_continue/best/cg_priors.json" ]; then
    IBI_PRIORS="ibi_run_16ps_continue/best/cg_priors.json"
else
    IBI_PRIORS="ibi_run_16ps/best/cg_priors.json"
fi
cd "${SCRIPT_DIR}"

OUTPUT_DATASET="${OUTPUT_DATASET:-tel22_dataset_ibi_residual.bin}"
OUTPUT_RB_INFO="${OUTPUT_RB_INFO:-rigid_bodies_info_ibi.json}"
OUTPUT_PROVENANCE="${OUTPUT_PROVENANCE:-ibi_residual_build_manifest.json}"
if [ -n "${IBI_VALIDATION_REPORT:-}" ]; then
    IBI_VALIDATION_REPORT="${IBI_VALIDATION_REPORT}"
elif [[ "${IBI_PRIORS}" == ibi_conservative/cg_priors.json || "${IBI_PRIORS}" == */ibi_conservative/cg_priors.json ]]; then
    IBI_VALIDATION_REPORT="$(dirname "${IBI_PRIORS}")/validation_report.json"
elif [ -f "$(dirname "${IBI_PRIORS}")/validation_report.json" ]; then
    IBI_VALIDATION_REPORT="$(dirname "${IBI_PRIORS}")/validation_report.json"
else
    IBI_VALIDATION_REPORT="ibi_validation_best/validation_report.json"
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

: "${AA_TOPOLOGY:=md.gro}"
: "${AA_TRAJECTORY:=md_whole.trr}"

REQUIRED_INPUTS=(
    "${AA_TOPOLOGY}"
    "${AA_TRAJECTORY}"
    tel22_topology.json
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
    --config tel22_topology.json \
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
    --mapping-config tel22_topology.json
    --validation-report "${IBI_VALIDATION_REPORT}"
)
if [ -n "${IBI_RUNTIME_PARITY_REPORT}" ]; then
    PROVENANCE_ARGS+=(--runtime-parity-report "${IBI_RUNTIME_PARITY_REPORT}")
fi
"${PYTHON_BIN}" "${PROVENANCE_TOOL}" "${PROVENANCE_ARGS[@]}"

echo "[DONE] Residual dataset with selected IBI priors: ${OUTPUT_DATASET}"
echo "[DONE] Rigid-body metadata: ${OUTPUT_RB_INFO}"
echo "[DONE] Residual-build provenance: ${OUTPUT_PROVENANCE}"
