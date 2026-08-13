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
CG_STEPS="${CG_STEPS:-20000}"
CG_DT="${CG_DT:-0.001}"

cd "${SCRIPT_DIR}"

for path in tel22_model.pt tel22_training_config.json cg_priors.json rigid_bodies_info.json tel22_dataset.bin equilibrated.npz; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

"${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py" \
    --model tel22_model.pt \
    --config tel22_training_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --checkpoint equilibrated.npz \
    --steps "${CG_STEPS}" \
    --dt "${CG_DT}" \
    --kT 2.49 \
    --device "${DEVICE}"
