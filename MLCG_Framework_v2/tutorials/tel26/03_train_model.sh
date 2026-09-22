#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAINER="${TRAINER:-${FRAMEWORK_ROOT}/training/build/train_painn}"
cd "${SCRIPT_DIR}"
for path in tel26_dataset.bin tel26_training_config.json; do
    [ -f "${path}" ] || { echo "[ERROR] manca: ${path}" >&2; exit 1; }
done
[ -x "${TRAINER}" ] || { echo "[ERROR] trainer non eseguibile: ${TRAINER}" >&2; exit 1; }
"${TRAINER}" tel26_dataset.bin tel26_model.pt tel26_training_config.json
echo "[DONE] tel26_model.pt"
