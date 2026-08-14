#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi

PYPRESSO="${PYPRESSO:-${PYRESSO:-${DEFAULT_PYPRESSO}}}"
MORSE_INDEX="${MORSE_INDEX:-0}"
MORSE_BREAKAGE_REPORT="${MORSE_BREAKAGE_REPORT:-morse_breakage_report.json}"

cd "${SCRIPT_DIR}"

if [[ ! -f cg_priors.json ]]; then
    echo "[ERROR] Missing required input: cg_priors.json" >&2
    exit 1
fi

"${PYPRESSO}" "${FRAMEWORK_ROOT}/simulation/diagnose_morse_breakage.py" \
    --priors cg_priors.json \
    --morse-index "${MORSE_INDEX}" \
    --output "${MORSE_BREAKAGE_REPORT}" \
    "$@"
