#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${SCRIPT_DIR}"

if [[ -n "${IBI_PRIORS:-}" ]]; then
    SOURCE_PRIORS="${IBI_PRIORS}"
elif [[ -f "ibi_run_16ps_continue/best/cg_priors.json" ]]; then
    SOURCE_PRIORS="ibi_run_16ps_continue/best/cg_priors.json"
else
    SOURCE_PRIORS="ibi_run_16ps/best/cg_priors.json"
fi
OUTDIR="${CONSERVATIVE_IBI_OUTDIR:-ibi_conservative}"
OVERWRITE="${OVERWRITE:-0}"

args=(--priors "${SOURCE_PRIORS}" --output-dir "${OUTDIR}")
if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
fi
"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/ibi/convert_to_conservative_spline.py" "${args[@]}"

echo "[DONE] Conservative IBI priors: ${OUTDIR}/cg_priors.json"
echo "[DONE] Conversion provenance: ${OUTDIR}/conversion_report.json"
