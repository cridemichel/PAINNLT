#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi

PYRESSO="${PYRESSO:-${DEFAULT_PYPRESSO}}"
DEVICE="${DEVICE:-auto}"
VELOCITY_SEED="${VELOCITY_SEED:-314159}"
NEIGHBOR_SEARCH="${NEIGHBOR_SEARCH:-link-cell}"

cd "${SCRIPT_DIR}"

for path in tel26_model.pt tel26_training_config.json cg_priors.json rigid_bodies_info.json tel26_dataset.bin; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

"${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/equilibrate.py" \
    --model tel26_model.pt \
    --config tel26_training_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel26_dataset.bin \
    --out_checkpoint equilibrated.npz \
    --device "${DEVICE}" \
    --neighbor_search "${NEIGHBOR_SEARCH}" \
    --kT 2.49 \
    --velocity_seed "${VELOCITY_SEED}"

echo "[DONE] equilibrated.npz"
