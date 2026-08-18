#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TEL22_TEMP_NVE_SOURCE_T_K="${TEL22_TEMP_NVE_SOURCE_T_K:-300}"
TEL22_TEMP_NVE_TEMPERATURES="${TEL22_TEMP_NVE_TEMPERATURES:-300 100 30}"
TEL22_TEMP_NVE_PRECISIONS="${TEL22_TEMP_NVE_PRECISIONS:-float32}"
TEL22_TEMP_NVE_DTS="${TEL22_TEMP_NVE_DTS:-0.001 0.0015 0.002 0.003 0.004 0.005}"
TEL22_TEMP_NVE_DURATION_PS="${TEL22_TEMP_NVE_DURATION_PS:-2.0}"
TEL22_TEMP_NVE_OUTPUT_DIR="${TEL22_TEMP_NVE_OUTPUT_DIR:-${SCRIPT_DIR}/results}"

read -r -a TEMPERATURE_ARGS <<< "${TEL22_TEMP_NVE_TEMPERATURES}"
read -r -a PRECISION_ARGS <<< "${TEL22_TEMP_NVE_PRECISIONS}"
read -r -a DT_ARGS <<< "${TEL22_TEMP_NVE_DTS}"

CMD=(
  python3 "${SCRIPT_DIR}/run_temperature_sweep.py"
  --framework-root "${FRAMEWORK_ROOT}"
  --source-temperature-k "${TEL22_TEMP_NVE_SOURCE_T_K}"
  --temperatures "${TEMPERATURE_ARGS[@]}"
  --precisions "${PRECISION_ARGS[@]}"
  --dts "${DT_ARGS[@]}"
  --duration-ps "${TEL22_TEMP_NVE_DURATION_PS}"
  --output-dir "${TEL22_TEMP_NVE_OUTPUT_DIR}"
)
if [[ -n "${PYPRESSO:-}" ]]; then
  CMD+=(--pypresso "${PYPRESSO}")
fi
CMD+=("$@")

exec "${CMD[@]}"
