#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${SCRIPT_DIR}"

PRIORS="${IBI_PRIORS:-ibi_conservative/cg_priors.json}"
VALIDATION_REPORT="${IBI_VALIDATION_REPORT:-ibi_conservative/validation_report.json}"
RUNTIME_PARITY_REPORT="${IBI_RUNTIME_PARITY_REPORT:-ibi_conservative/runtime_parity_report.json}"
NVE_PREFLIGHT_REPORT="${NVE_PREFLIGHT_REPORT:-nve_certification_conservative_ibi_only_preflight.json}"
NVE_EQ_REPORT="${NVE_EQ_REPORT:-nve_equilibration_conservative_ibi_only/equilibration_report.json}"
STRICT_NVE_REPORT="${STRICT_NVE_REPORT:-nve_certification_conservative_ibi_only/nve_certification_report.json}"
STATE_CONVERGENCE_REPORT="${STATE_CONVERGENCE_REPORT:-nve_state_convergence_conservative_ibi_only/state_convergence_report.json}"
FINAL_NVE_DIR="${FINAL_NVE_DIR:-nve_final_certification_conservative_ibi_only}"
FINAL_NVE_REPORT="${FINAL_NVE_REPORT:-${FINAL_NVE_DIR}/conservative_ibi_nve_certification_report.json}"
NVE_STATE_ORDER_MIN="${NVE_STATE_ORDER_MIN:-1.7}"
NVE_STATE_ORDER_MAX="${NVE_STATE_ORDER_MAX:-2.3}"
NVE_STATE_MIN_R2="${NVE_STATE_MIN_R2:-0.95}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

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
    --output "${FINAL_NVE_REPORT}"
