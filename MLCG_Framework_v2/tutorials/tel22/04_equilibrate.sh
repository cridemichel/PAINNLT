#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYRESSO="${PYRESSO:-pypresso}"
DEVICE="${DEVICE:-auto}"

cd "${SCRIPT_DIR}"

for path in tel22_model.pt tel22_training_config.json cg_priors.json rigid_bodies_info.json tel22_dataset.bin; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

"${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/equilibrate.py" \
    --model tel22_model.pt \
    --config tel22_training_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --out_checkpoint equilibrated.npz \
    --device "${DEVICE}" \
    --kT 2.49

echo "[DONE] equilibrated.npz"
