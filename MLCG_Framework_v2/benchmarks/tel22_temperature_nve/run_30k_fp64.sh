#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${TEL22_TEMP_NVE_OUTPUT_DIR:-${SCRIPT_DIR}/results}"

if [[ ! -f "${RESULTS_DIR}/T30K_float32/nve_certification_report.json" ]]; then
  echo "[ERROR] Missing completed 30 K FP32 reference:" >&2
  echo "        ${RESULTS_DIR}/T30K_float32/nve_certification_report.json" >&2
  echo "Run the FP32 temperature sweep first." >&2
  exit 2
fi

export TEL22_TEMP_NVE_TEMPERATURES="30"
export TEL22_TEMP_NVE_PRECISIONS="float64"
export TEL22_TEMP_NVE_OUTPUT_DIR="${RESULTS_DIR}"

"${SCRIPT_DIR}/run.sh" "$@"

if [[ " $* " != *" --dry-run "* ]]; then
  python3 "${SCRIPT_DIR}/compare_30k_precision.py" \
    --results-dir "${RESULTS_DIR}" \
    --output "${RESULTS_DIR}/T30K_fp32_vs_fp64_closure.json"
fi
