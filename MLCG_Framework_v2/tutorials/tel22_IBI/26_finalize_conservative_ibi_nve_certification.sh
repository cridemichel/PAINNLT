#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/model_config.sh"
load_model_dependent_config step26

PRIORS="${IBI_PRIORS}"
VALIDATION_REPORT="${IBI_VALIDATION_REPORT}"
RUNTIME_PARITY_REPORT="${IBI_RUNTIME_PARITY_REPORT}"
NVE_PREFLIGHT_REPORT="${NVE_PREFLIGHT_REPORT}"
NVE_EQ_REPORT="${NVE_EQ_REPORT}"
STRICT_NVE_REPORT="${STRICT_NVE_REPORT}"
STATE_CONVERGENCE_REPORT="${STATE_CONVERGENCE_REPORT}"
FINAL_NVE_DIR="${FINAL_NVE_DIR}"
FINAL_NVE_REPORT="${FINAL_NVE_REPORT}"
NVE_STATE_ORDER_MIN="${NVE_STATE_ORDER_MIN}"
NVE_STATE_ORDER_MAX="${NVE_STATE_ORDER_MAX}"
NVE_STATE_MIN_R2="${NVE_STATE_MIN_R2}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT}"

for path in \
    "${PRIORS}" "${VALIDATION_REPORT}" "${RUNTIME_PARITY_REPORT}" \
    "${NVE_PREFLIGHT_REPORT}" "${NVE_EQ_REPORT}" "${STRICT_NVE_REPORT}" \
    "${STATE_CONVERGENCE_REPORT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing final-certification artifact: ${path}" >&2
        echo "[HINT] Complete steps 22, 23 and 25 before running step 26." >&2
        exit 1
    fi
done

mkdir -p "${FINAL_NVE_DIR}"
FINAL_NVE_MODEL_CONFIG_PROVENANCE="${FINAL_NVE_DIR}/model_config_provenance.json"
write_model_dependent_provenance "${FINAL_NVE_MODEL_CONFIG_PROVENANCE}"

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/finalize_conservative_nve_certification.py" \
    --priors "${PRIORS}" \
    --validation-report "${VALIDATION_REPORT}" \
    --runtime-parity-report "${RUNTIME_PARITY_REPORT}" \
    --nve-preflight-report "${NVE_PREFLIGHT_REPORT}" \
    --equilibration-report "${NVE_EQ_REPORT}" \
    --strict-nve-report "${STRICT_NVE_REPORT}" \
    --state-convergence-report "${STATE_CONVERGENCE_REPORT}" \
    --state-order-min "${NVE_STATE_ORDER_MIN}" \
    --state-order-max "${NVE_STATE_ORDER_MAX}" \
    --state-min-r2 "${NVE_STATE_MIN_R2}" \
    --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}" \
    --model-config-provenance "${FINAL_NVE_MODEL_CONFIG_PROVENANCE}" \
    --output "${FINAL_NVE_REPORT}"
