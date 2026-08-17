#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/model_config.sh" || -f "${SCRIPT_DIR}/cg_priors.json" ]]; then
    TUTORIAL_DIR="${SCRIPT_DIR}"
else
    TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi

PYPRESSO="${PYPRESSO:-${PYRESSO:-${DEFAULT_PYPRESSO}}}"
MORSE_INDEX="${MORSE_INDEX:-0}"
MORSE_REVERSIBILITY_REPORT="${MORSE_REVERSIBILITY_REPORT:-diagnostics/morse/morse_reversibility_report.json}"

cd "${TUTORIAL_DIR}"

if [[ ! -f cg_priors.json ]]; then
    echo "[ERROR] Missing required input: cg_priors.json" >&2
    exit 1
fi

"${PYPRESSO}" "${FRAMEWORK_ROOT}/simulation/diagnose_morse_reversibility.py" \
    --priors cg_priors.json \
    --morse-index "${MORSE_INDEX}" \
    --output "${MORSE_REVERSIBILITY_REPORT}" \
    "$@"
