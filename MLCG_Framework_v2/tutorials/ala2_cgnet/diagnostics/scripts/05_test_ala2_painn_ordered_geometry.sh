#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALA2_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${ALA2_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINER="${TRAINER:-${FRAMEWORK_ROOT}/training/build/train_painn}"
PYRESSO="${PYRESSO:-${FRAMEWORK_ROOT}/espresso/build/pypresso}"
SOURCE_RUN_DIR="${ALA2_ORDERED_SOURCE_RUN_DIR:-${ALA2_DIR}/diagnostics/smoke/cgnet_harmonic_50ep}"
RUN_DIR="${ALA2_ORDERED_RUN_DIR:-${ALA2_DIR}/diagnostics/smoke/painn_ordered_geometry_scale4p184_5ep}"
AB_RUN_DIR="${ALA2_ORDERED_AB_RUN_DIR:-${ALA2_DIR}/diagnostics/smoke/painn_ordered_geometry_scale4p184_fes_4x50k}"
CONFIG_SOURCE="${ALA2_ORDERED_CONFIG_SOURCE:-${ALA2_DIR}/diagnostics/configs/ala2_training_config_painn_ordered_geometry_5ep.json}"
MODE=training

case "${1:-}" in
    "") ;;
    --with-fes) MODE=both ;;
    --fes-only) MODE=fes ;;
    *) printf '[ERROR] Usage: %s [--with-fes|--fes-only]\n' "$0" >&2; exit 2 ;;
esac

if [[ "${MODE}" != fes && ! -x "${TRAINER}" ]]; then
    printf '[ERROR] Missing executable trainer: %s\n' "${TRAINER}" >&2
    printf '        Rebuild training/build after applying the ordered-geometry patch.\n' >&2
    exit 2
fi
if [[ "${MODE}" != fes && ! -f "${CONFIG_SOURCE}" ]]; then
    printf '[ERROR] Missing ordered-geometry config: %s\n' "${CONFIG_SOURCE}" >&2
    exit 2
fi

if [[ "${MODE}" != fes ]]; then
    for filename in \
        ala2_dataset.bin \
        ala2_conversion_report.json \
        ala2_priors.json \
        ala2_reference.npz; do
        if [[ ! -s "${SOURCE_RUN_DIR}/${filename}" ]]; then
            printf '[ERROR] Missing reusable baseline artifact: %s\n' "${SOURCE_RUN_DIR}/${filename}" >&2
            exit 2
        fi
    done
    if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
        printf '[ERROR] Ordered-geometry run directory is not empty: %s\n' "${RUN_DIR}" >&2
        printf '        Select a fresh ALA2_ORDERED_RUN_DIR; evidence is never overwritten.\n' >&2
        exit 2
    fi

    mkdir -p "${RUN_DIR}"
    RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
    for filename in \
        ala2_dataset.bin \
        ala2_conversion_report.json \
        ala2_priors.json \
        ala2_reference.npz; do
        cp "${SOURCE_RUN_DIR}/${filename}" "${RUN_DIR}/${filename}"
    done
    cp "${CONFIG_SOURCE}" "${RUN_DIR}/ala2_training_config_50ep.json"

    printf '[INFO] Reused identical Ala2 data/targets/priors from %s\n' "${SOURCE_RUN_DIR}"
    printf '[INFO] Architecture change: ordered 17-feature tanh head with independent 4.184 kJ/mol energy scale.\n'

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
        --report "${RUN_DIR}/ala2_benchmark_report.json" \
        --expected-cutoff 1.0 \
        --expected-spectral-strength 4.0 \
        --expected-architecture-variant painn_ordered_geometry_tanh_v2 \
        --expected-ordered-energy-scale-kj-mol 4.184 \
        --require-all-to-all \
        --require-cgnet-matched-controls \
        --require-ordered-geometry

    printf '[PASS] Ordered-geometry PaiNN training completed in %s\n' "${RUN_DIR}"
else
    for filename in \
        ala2_dataset.bin \
        ala2_priors.json \
        ala2_reference.npz \
        ala2_training_config_50ep.json \
        ala2_model.pt \
        ala2_model.pt.manifest.json \
        ala2_benchmark_report.json; do
        if [[ ! -s "${RUN_DIR}/${filename}" ]]; then
            printf '[ERROR] Missing completed ordered-geometry artifact: %s\n' "${RUN_DIR}/${filename}" >&2
            exit 2
        fi
    done
    RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
fi

if [[ "${MODE}" != training ]]; then
    if [[ ! -x "${PYRESSO}" ]]; then
        printf '[ERROR] Missing rebuilt ESPResSo executable: %s\n' "${PYRESSO}" >&2
        exit 2
    fi
    PYTHON_BIN="${PYTHON_BIN}" \
    PYRESSO="${PYRESSO}" \
    ALA2_TRAINING_RUN_DIR="${RUN_DIR}" \
    ALA2_AB_RUN_DIR="${AB_RUN_DIR}" \
    ALA2_AB_REPLICAS="${ALA2_AB_REPLICAS:-4}" \
    ALA2_AB_EQUIL_STEPS="${ALA2_AB_EQUIL_STEPS:-25000}" \
    ALA2_AB_PRODUCTION_STEPS="${ALA2_AB_PRODUCTION_STEPS:-50000}" \
    ALA2_AB_BURNIN_STEPS="${ALA2_AB_BURNIN_STEPS:-10000}" \
    ALA2_AB_SAMPLE_INTERVAL="${ALA2_AB_SAMPLE_INTERVAL:-50}" \
    DEVICE="${DEVICE:-auto}" \
    bash "${SCRIPT_DIR}/02_test_ala2_free_energy_ab.sh"
else
    printf '[INFO] Training-only mode. Inspect the report before using --fes-only.\n'
fi
