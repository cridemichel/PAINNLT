#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEL22_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TEL22_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINER="${TRAINER:-${FRAMEWORK_ROOT}/training/build/train_painn}"
SOURCE_RUN_DIR="${TEL22_SHARED_SOURCE_RUN_DIR:-${TEL22_DIR}/diagnostics/smoke/variant_a_long_1001f_15ep}"
RUN_DIR="${TEL22_SHARED_RUN_DIR:-${TEL22_DIR}/diagnostics/smoke/shared_head_1001f_15ep}"
CONFIG_SOURCE="${TEL22_SHARED_CONFIG:-${TEL22_DIR}/diagnostics/configs/tel22_training_config_shared_head_15ep.json}"

if [[ ! -x "${TRAINER}" ]]; then
    printf '[ERROR] Missing trainer: %s\n' "${TRAINER}" >&2
    exit 2
fi
for path in "${CONFIG_SOURCE}" "${SOURCE_RUN_DIR}/tel22_dataset.bin" \
            "${SOURCE_RUN_DIR}/cg_priors.json" "${SOURCE_RUN_DIR}/rigid_bodies_info.json"; do
    if [[ ! -s "${path}" ]]; then
        printf '[ERROR] Missing reusable Variant-A artifact: %s\n' "${path}" >&2
        exit 2
    fi
done
if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    printf '[ERROR] Test directory is not empty: %s\n' "${RUN_DIR}" >&2
    printf '        Select a fresh TEL22_SHARED_RUN_DIR; evidence is never overwritten.\n' >&2
    exit 2
fi
mkdir -p "${RUN_DIR}"

cp "${SOURCE_RUN_DIR}/tel22_dataset.bin" "${RUN_DIR}/tel22_dataset.bin"
cp "${SOURCE_RUN_DIR}/cg_priors.json" "${RUN_DIR}/cg_priors.json"
cp "${SOURCE_RUN_DIR}/rigid_bodies_info.json" "${RUN_DIR}/rigid_bodies_info.json"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_shared_head_cutoff.py" \
    --dataset "${RUN_DIR}/tel22_dataset.bin" \
    --config-in "${CONFIG_SOURCE}" \
    --config-out "${RUN_DIR}/tel22_training_config_shared_head_15ep.json" \
    --report "${RUN_DIR}/tel22_shared_cutoff_report.json"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_variant_a_topology.py" \
    --input "${RUN_DIR}/cg_priors.json"

cd "${RUN_DIR}"
"${TRAINER}" \
    tel22_dataset.bin \
    tel22_shared_head.pt \
    tel22_training_config_shared_head_15ep.json \
    2>&1 | tee training_stdout.log

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_shared_head_training.py" \
    --run-dir "${RUN_DIR}" \
    --report "${RUN_DIR}/tel22_shared_head_report.json"

printf '[PASS] TEL22 shared-head training screen completed in %s\n' "${RUN_DIR}"
printf '[INFO] Send tel22_shared_cutoff_report.json, tel22_shared_head_report.json, cg_training_log.csv, and training_stdout.log.\n'
