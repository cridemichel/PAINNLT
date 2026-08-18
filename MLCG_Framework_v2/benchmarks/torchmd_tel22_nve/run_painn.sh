#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

DEVICE="${TORCHMD_NVE_DEVICE:-cpu}"
PRECISION="${TORCHMD_NVE_PRECISION:-float64}"
DURATION_PS="${TORCHMD_PAINN_NVE_DURATION_PS:-0.60}"
OUTPUT_DIR="${TORCHMD_PAINN_NVE_OUTPUT_DIR:-results/painn_${DEVICE//:/_}_${PRECISION}}"

python3 selftest.py
python3 painn_selftest.py
python3 run_painn_certification.py \
  --device "${DEVICE}" \
  --precision "${PRECISION}" \
  --duration-ps "${DURATION_PS}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
