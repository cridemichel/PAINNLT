#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALA2_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${ALA2_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINER="${TRAINER:-${FRAMEWORK_ROOT}/training/build/train_painn}"
DATA_DIR="${ALA2_DATA_DIR:-${ALA2_DIR}/data}"
PRIOR_MODE="${ALA2_PRIOR_MODE:-harmonic}"
CONFIG_SOURCE="${ALA2_CONFIG_SOURCE:-${ALA2_DIR}/diagnostics/configs/ala2_training_config_50ep.json}"
RUN_DIR="${ALA2_RUN_DIR:-${ALA2_DIR}/diagnostics/smoke/cgnet_${PRIOR_MODE}_50ep}"

if [[ "${PRIOR_MODE}" != "harmonic" && "${PRIOR_MODE}" != "none" ]]; then
    printf '[ERROR] ALA2_PRIOR_MODE must be harmonic or none; got %s\n' "${PRIOR_MODE}" >&2
    exit 2
fi
if [[ ! -x "${TRAINER}" ]]; then
    printf '[ERROR] Missing executable trainer: %s\n' "${TRAINER}" >&2
    exit 2
fi
if [[ ! -f "${CONFIG_SOURCE}" ]]; then
    printf '[ERROR] Missing training config: %s\n' "${CONFIG_SOURCE}" >&2
    exit 2
fi
if ! "${PYTHON_BIN}" -c 'import numpy' >/dev/null 2>&1; then
    printf '[ERROR] %s cannot import numpy. Activate the framework Python environment first.\n' "${PYTHON_BIN}" >&2
    exit 2
fi
if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    printf '[ERROR] Benchmark directory is not empty: %s\n' "${RUN_DIR}" >&2
    printf '        Select a fresh ALA2_RUN_DIR; existing evidence is never overwritten.\n' >&2
    exit 2
fi

mkdir -p "${DATA_DIR}" "${RUN_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"

"${PYTHON_BIN}" "${SCRIPT_DIR}/download_cgnet_ala2.py" \
    --output-dir "${DATA_DIR}"

cp "${CONFIG_SOURCE}" "${RUN_DIR}/ala2_training_config_50ep.json"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_ala2_dataset.py" \
    --coordinates "${DATA_DIR}/ala2_coordinates.npy" \
    --forces "${DATA_DIR}/ala2_forces.npy" \
    --output "${RUN_DIR}/ala2_dataset.bin" \
    --priors-output "${RUN_DIR}/ala2_priors.json" \
    --rb-info-output "${RUN_DIR}/ala2_rigid_bodies_info.json" \
    --reference-output "${RUN_DIR}/ala2_reference.npz" \
    --report "${RUN_DIR}/ala2_conversion_report.json" \
    --prior-mode "${PRIOR_MODE}" \
    --validation-tail-frames 2000 \
    2>&1 | tee "${RUN_DIR}/conversion_stdout.log"

cd "${RUN_DIR}"

"${TRAINER}" \
    ala2_dataset.bin \
    ala2_model.pt \
    ala2_training_config_50ep.json \
    2>&1 | tee training_stdout.log

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/training/create_model_manifest.py" \
    --model ala2_model.pt \
    --config ala2_training_config_50ep.json \
    --dataset ala2_dataset.bin

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_ala2_benchmark.py" \
    --run-dir "${RUN_DIR}" \
    --report "${RUN_DIR}/ala2_benchmark_report.json"

printf '[PASS] Ala2 CGnet benchmark completed in %s\n' "${RUN_DIR}"
printf '[INFO] Send ala2_benchmark_report.json and cg_training_log.csv for comparison with TEL22.\n'
