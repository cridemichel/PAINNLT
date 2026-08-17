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
PYRESSO="${PYRESSO:-${DEFAULT_PYPRESSO}}"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/model_config.sh"
load_model_dependent_config step22
OUTDIR="${CONSERVATIVE_IBI_OUTDIR}"

for path in "${OUTDIR}/cg_priors.json" "${OUTDIR}/conversion_report.json"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing ${path}; run ./21_convert_best_ibi_to_conservative.sh first." >&2; exit 1; }
done

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/ibi/validate_conservative_spline.py" \
    --conversion-report "${OUTDIR}/conversion_report.json"
"${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/diagnose_conservative_spline_parity.py" \
    --priors "${OUTDIR}/cg_priors.json" \
    --report "${OUTDIR}/runtime_parity_report.json"

cat <<EOF
[CONSERVATIVE IBI PHASE-2 GATE]
priors     : ${OUTDIR}/cg_priors.json
conversion : ${OUTDIR}/conversion_report.json
validation : ${OUTDIR}/validation_report.json
runtime    : ${OUTDIR}/runtime_parity_report.json
[PASS] Conservative spline U/derivative consistency and ESPResSo runtime/preprocessing parity passed.
[NEXT] Rebuild the residual dataset against these exact conservative priors, then retrain PaiNN before strict NVE certification.
EOF

write_model_dependent_provenance "${OUTDIR}/model_config_provenance_step22.json"
