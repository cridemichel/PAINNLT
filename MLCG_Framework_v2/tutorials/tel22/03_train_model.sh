#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAINER="${TRAINER:-${FRAMEWORK_ROOT}/training/build/train_painn}"

cd "${SCRIPT_DIR}"

for path in tel22_dataset.bin tel22_training_config.json; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done
if [ ! -x "${TRAINER}" ]; then
    echo "[ERROR] train_painn not found/executable: ${TRAINER}" >&2
    echo "Build it first under training/build or set TRAINER=/path/to/train_painn." >&2
    exit 1
fi

"${TRAINER}" tel22_dataset.bin tel22_model.pt tel22_training_config.json

echo "[DONE] tel22_model.pt"
