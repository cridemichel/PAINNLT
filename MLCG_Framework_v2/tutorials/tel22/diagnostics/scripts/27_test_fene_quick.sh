#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/27_test_fene_quick.py"

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYPRESSO="${PYPRESSO:-${DEFAULT_PYPRESSO}}"

OUT_REL="${FENE_QUICK_OUT:-diagnostics/fene/fene_quick_closure}"
OUT="${TUTORIAL_DIR}/${OUT_REL}"
REPORT="${OUT}/fene_quick_summary.json"
FENE_K="${FENE_K:-30.0}"
FENE_R0="${FENE_R0:-1.2}"
FENE_RMAX="${FENE_RMAX:-0.5}"
FENE_NVE_DURATION_PS="${FENE_NVE_DURATION_PS:-4.0}"

usage() {
cat <<'USAGE'
Usage:
  27_test_fene_quick.sh [--dry-run | --overwrite]

Fast isolated FENE validation:
  analytic energy/force vs ESPResSo
  finite-difference force closure
  action/reaction and transverse-force checks
  six-dt Velocity-Verlet NVE scaling

No PaiNN, training, nonbonded interactions, thermostat, or production files.

Useful overrides:
  FENE_QUICK_OUT=diagnostics/fene/my_test
  FENE_K=30.0
  FENE_R0=1.2
  FENE_RMAX=0.5
  FENE_NVE_DURATION_PS=4.0
  PYPRESSO=/path/to/pypresso
USAGE
}

MODE="normal"
case "${1:-}" in
    "") ;;
    --dry-run) MODE="dry-run"; shift ;;
    --overwrite) MODE="overwrite"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: ${1}" >&2; usage >&2; exit 2 ;;
esac
[[ $# -eq 0 ]] || { echo "[ERROR] Unexpected arguments: $*" >&2; exit 2; }

[[ -f "${TEST_SCRIPT}" ]] || { echo "[ERROR] Missing test script: ${TEST_SCRIPT}" >&2; exit 1; }
case "${OUT}" in
    "${TUTORIAL_DIR}"/diagnostics/fene/*) ;;
    *) echo "[ERROR] Output must remain below diagnostics/fene: ${OUT}" >&2; exit 1 ;;
esac

echo "[FENE QUICK TEST]"
echo "parameters : k=${FENE_K}, r0=${FENE_R0}, d_r_max=${FENE_RMAX}"
echo "NVE        : six dt values, ${FENE_NVE_DURATION_PS} ps each"
echo "output     : ${OUT_REL}"
echo "scope      : isolated; production files untouched"

if [[ "${MODE}" == "dry-run" ]]; then
    command -v "${PYPRESSO}" >/dev/null 2>&1 || [[ -x "${PYPRESSO}" ]] || {
        echo "[ERROR] pypresso not found: ${PYPRESSO}" >&2
        exit 1
    }
    echo "[DRY-RUN] Would write: ${REPORT}"
    exit 0
fi

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf -- "${OUT}"
elif [[ -e "${OUT}" ]]; then
    echo "[ERROR] Output already exists: ${OUT}" >&2
    echo "        Use --overwrite or set FENE_QUICK_OUT." >&2
    exit 1
fi

command -v "${PYPRESSO}" >/dev/null 2>&1 || [[ -x "${PYPRESSO}" ]] || {
    echo "[ERROR] pypresso not found: ${PYPRESSO}" >&2
    exit 1
}
mkdir -p "${OUT}"

"${PYPRESSO}" "${TEST_SCRIPT}" \
    --output "${REPORT}" \
    --k "${FENE_K}" \
    --r0 "${FENE_R0}" \
    --rmax "${FENE_RMAX}" \
    --duration "${FENE_NVE_DURATION_PS}"
