#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
ESPRESSO_ROOT="${ESPRESSO_SRC:-${ROOT}/espresso}"
ESPRESSO_BUILD="${ESPRESSO_BUILD:-${ESPRESSO_ROOT}/build}"
PROD_CPP="${ROOT}/simulation/espresso_plugin/PaiNN_ML_Potential.cpp"
ESP_CPP="${ESPRESSO_ROOT}/src/core/nonbonded_interactions/PaiNN_ML_Potential.cpp"
CASE_DIR="${MLCG_PAINN_PARITY_CASE_DIR:-${HERE}/../results/shared_painn_case}"
PRECISION="${MLCG_PAINN_PARITY_PRECISION:-float64}"
DEVICE="${MLCG_PAINN_PARITY_DEVICE:-cpu}"
DURATION="${MLCG_PAINN_PARITY_DURATION_PS:-0.60}"
JOBS="${MLCG_PAINN_PARITY_BUILD_JOBS:-4}"
OUTPUT_DIR="${MLCG_PAINN_PARITY_OUTPUT_DIR:-${HERE}/../results/mlcg_painn_${DEVICE//:/_}_${PRECISION}}"

DRY_RUN=0
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then DRY_RUN=1; fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    python3 "${HERE}/export_shared_case.py" --output-dir "${CASE_DIR}" --dry-run
    python3 "${HERE}/make_espresso_benchmark_override.py" \
        --source "${PROD_CPP}" --output "/tmp/mlcg_synthetic_painn_override.cpp"
    python3 "${HERE}/run_mlcg_certification.py" \
        --case-dir "${CASE_DIR}" --precision "${PRECISION}" --device "${DEVICE}" \
        --duration-ps "${DURATION}" --output-dir "${OUTPUT_DIR}" --dry-run
    echo "[DRY-RUN] ESPResSo source: ${ESPRESSO_ROOT}"
    echo "[DRY-RUN] ESPResSo build : ${ESPRESSO_BUILD}"
    echo "[DRY-RUN] benchmark temporarily replaces only PaiNN_ML_Potential.cpp, then restores and rebuilds production"
    exit 0
fi

if [[ ! -f "${ESP_CPP}" || ! -d "${ESPRESSO_BUILD}" ]]; then
    echo "[ERROR] ESPResSo source/build not found. Set ESPRESSO_SRC and ESPRESSO_BUILD if needed." >&2
    exit 2
fi

mkdir -p "${CASE_DIR}" "${OUTPUT_DIR}"
python3 "${HERE}/export_shared_case.py" --output-dir "${CASE_DIR}" --overwrite

# First synchronize the normal framework plugin.  This makes the backup below a
# known production source rather than an arbitrary stale ESPResSo file.
ESPRESSO_SRC="${ESPRESSO_ROOT}" bash "${ROOT}/simulation/espresso_plugin/copy_plugin_files.sh"
cmake --build "${ESPRESSO_BUILD}" -j"${JOBS}"

TMPDIR_BENCH="$(mktemp -d "${TMPDIR:-/tmp}/mlcg-painn-parity.XXXXXX")"
BACKUP_CPP="${TMPDIR_BENCH}/PaiNN_ML_Potential.production.cpp"
OVERRIDE_CPP="${TMPDIR_BENCH}/PaiNN_ML_Potential.benchmark.cpp"
cp -p "${ESP_CPP}" "${BACKUP_CPP}"
python3 "${HERE}/make_espresso_benchmark_override.py" --source "${PROD_CPP}" --output "${OVERRIDE_CPP}"

restored=0
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ -f "${BACKUP_CPP}" ]]; then
        cp -p "${BACKUP_CPP}" "${ESP_CPP}"
        if cmake --build "${ESPRESSO_BUILD}" -j"${JOBS}"; then
            restored=1
            echo "[PASS] restored and rebuilt production PaiNN plugin"
        else
            echo "[ERROR] production source restored but rebuild failed; rebuild ${ESPRESSO_BUILD} manually" >&2
            rc=2
        fi
    fi
    rm -rf "${TMPDIR_BENCH}"
    exit "$rc"
}
trap cleanup EXIT INT TERM

cp -f "${OVERRIDE_CPP}" "${ESP_CPP}"
cmake --build "${ESPRESSO_BUILD}" -j"${JOBS}"

set +e
PYTHONPATH="${ESPRESSO_BUILD}/src/python${PYTHONPATH:+:${PYTHONPATH}}" \
MLCG_SYNTHETIC_PAINN_CASE_DIR="${CASE_DIR}" \
python3 "${HERE}/run_mlcg_certification.py" \
    --case-dir "${CASE_DIR}" \
    --precision "${PRECISION}" \
    --device "${DEVICE}" \
    --duration-ps "${DURATION}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
status=$?
set -e
exit "$status"
