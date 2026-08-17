#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/model_config.sh"
load_model_dependent_config step21

if [[ -n "${IBI_PRIORS:-}" ]]; then
  SOURCE_PRIORS="${IBI_PRIORS}"
else
  SOURCE_PRIORS=""
  for candidate in ${CONSERVATIVE_SOURCE_PRIOR_CANDIDATES}; do [[ -f "${candidate}" ]] && { SOURCE_PRIORS="${candidate}"; break; }; done
fi
[[ -n "${SOURCE_PRIORS}" ]] || { echo "[ERROR] No configured source prior candidate exists." >&2; exit 1; }
OUTDIR="${CONSERVATIVE_IBI_OUTDIR}"
OVERWRITE="${OVERWRITE:-0}"

args=(--priors "${SOURCE_PRIORS}" --output-dir "${OUTDIR}")
if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
fi
"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/ibi/convert_to_conservative_spline.py" "${args[@]}"

echo "[DONE] Conservative IBI priors: ${OUTDIR}/cg_priors.json"
echo "[DONE] Conversion provenance: ${OUTDIR}/conversion_report.json"

write_model_dependent_provenance "${OUTDIR}/model_config_provenance_step21.json"
