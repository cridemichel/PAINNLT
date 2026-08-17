#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ESPRESSO_ROOT="${ESPRESSO_ROOT:-${FRAMEWORK_ROOT}/espresso}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
JOBS="${JOBS:-4}"
INSTALLER="${FRAMEWORK_ROOT}/simulation/espresso_plugin/install_conservative_spline_bond.py"

if [[ ! -d "${ESPRESSO_ROOT}/src" || ! -d "${ESPRESSO_ROOT}/build" ]]; then
    echo "[ERROR] ESPResSo source/build tree not found: ${ESPRESSO_ROOT}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${INSTALLER}" --espresso-root "${ESPRESSO_ROOT}"
cmake --build "${ESPRESSO_ROOT}/build" -j "${JOBS}"
"${PYTHON_BIN}" "${INSTALLER}" --espresso-root "${ESPRESSO_ROOT}" --check

"${ESPRESSO_ROOT}/build/pypresso" - <<'PY'
import espressomd.interactions as ia
for name in ("ConservativeSplineDistance", "ConservativeSplineAngle", "ConservativeSplineDihedral"):
    if not hasattr(ia, name):
        raise SystemExit(f"missing espressomd.interactions.{name} after rebuild")
print("[PASS] ESPResSo exposes ConservativeSplineDistance, ConservativeSplineAngle, and ConservativeSplineDihedral.")
PY

"${ESPRESSO_ROOT}/build/pypresso" \
    "${FRAMEWORK_ROOT}/simulation/smoke_conservative_spline_runtime.py"
