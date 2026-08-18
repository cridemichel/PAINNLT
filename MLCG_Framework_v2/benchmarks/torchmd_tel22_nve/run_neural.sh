#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

DEVICE="${TORCHMD_NVE_DEVICE:-cpu}"
PRECISION="${TORCHMD_NVE_PRECISION:-float64}"
DURATION_PS="${TORCHMD_NVE_DURATION_PS:-1.98}"
OUTPUT_DIR="${TORCHMD_NEURAL_NVE_OUTPUT_DIR:-results/neural_${DEVICE//:/_}_${PRECISION}}"

python3 selftest.py
python3 neural_selftest.py
python3 run_neural_certification.py \
  --device "${DEVICE}" \
  --precision "${PRECISION}" \
  --duration-ps "${DURATION_PS}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
