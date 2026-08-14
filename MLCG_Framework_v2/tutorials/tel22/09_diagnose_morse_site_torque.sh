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
MORSE_SITE_TORQUE_REPORT="${MORSE_SITE_TORQUE_REPORT:-morse_site_torque_report.json}"

cd "${SCRIPT_DIR}"

"${PYPRESSO}" "${FRAMEWORK_ROOT}/simulation/diagnose_pair_specific_morse_site_torque.py" \
    --output "${MORSE_SITE_TORQUE_REPORT}" \
    "$@"
