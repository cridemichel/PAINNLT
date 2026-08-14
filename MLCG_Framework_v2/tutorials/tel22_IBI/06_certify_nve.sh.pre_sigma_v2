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
NVE_DEVICE="${NVE_DEVICE:-cpu}"
NVE_DURATION_PS="${NVE_DURATION_PS:-5.0}"
NVE_LOG_INTERVAL_PS="${NVE_LOG_INTERVAL_PS:-0.01}"
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-nve_certification}"
NVE_DTS="${NVE_DTS:-0.002 0.001 0.0005 0.00025}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.97}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

cd "${SCRIPT_DIR}"

for path in tel22_model.pt tel22_training_config.json cg_priors.json rigid_bodies_info.json tel22_dataset.bin equilibrated.npz; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

read -r -a DT_ARGS <<< "${NVE_DTS}"

python3 "${FRAMEWORK_ROOT}/simulation/certify_nve.py" \
    --pypresso "${PYRESSO}" \
    --model tel22_model.pt \
    --config tel22_training_config.json \
    --priors cg_priors.json \
    --rb-info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --checkpoint equilibrated.npz \
    --dts "${DT_ARGS[@]}" \
    --duration-ps "${NVE_DURATION_PS}" \
    --log-interval-ps "${NVE_LOG_INTERVAL_PS}" \
    --device "${NVE_DEVICE}" \
    --output-dir "${NVE_OUTPUT_DIR}" \
    --slope-min "${NVE_SLOPE_MIN}" \
    --slope-max "${NVE_SLOPE_MAX}" \
    --min-r2 "${NVE_MIN_R2}" \
    --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}" \
    "$@"
