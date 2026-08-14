#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINER="${TRAINER:-${FRAMEWORK_ROOT}/training/build/train_painn}"
BENCHMARK_TOOL="${FRAMEWORK_ROOT}/training/multiseed_benchmark.py"
MANIFEST_TOOL="${FRAMEWORK_ROOT}/training/create_model_manifest.py"

cd "${SCRIPT_DIR}"

BASELINE_DATASET="${BASELINE_DATASET:-tel22_dataset.bin}"
IBI_DATASET="${IBI_DATASET:-tel22_dataset_ibi_residual.bin}"
CONFIG="${TRAINING_CONFIG:-tel22_training_config.json}"
OUTDIR="${MULTISEED_OUTDIR:-training_multiseed_benchmark}"
SEEDS_RAW="${MULTISEED_SEEDS:-11 42 73}"

# The post-IBI case is allowed into the benchmark only if its residual dataset,
# rigid-body metadata and selected validated priors still match the build
# provenance manifest.  This check is intentionally repeated here rather than
# relying on a previous interactive preflight.
bash ./16_check_ibi_training_inputs.sh

for path in "${BASELINE_DATASET}" "${IBI_DATASET}" "${CONFIG}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing benchmark input: ${path}" >&2
        exit 1
    fi
done
if [ ! -x "${TRAINER}" ]; then
    echo "[ERROR] train_painn not found/executable: ${TRAINER}" >&2
    exit 1
fi

read -r -a SEEDS <<< "${SEEDS_RAW}"
if [ "${#SEEDS[@]}" -lt 2 ]; then
    echo "[ERROR] MULTISEED_SEEDS must contain at least two distinct seeds." >&2
    exit 1
fi

MODE_ARG=()
if [ "${OVERWRITE:-0}" = "1" ] && [ "${RESUME:-0}" = "1" ]; then
    echo "[ERROR] OVERWRITE=1 and RESUME=1 are mutually exclusive." >&2
    exit 1
elif [ "${RESUME:-0}" = "1" ]; then
    MODE_ARG=(--resume)
elif [ "${OVERWRITE:-0}" = "1" ]; then
    MODE_ARG=(--overwrite)
fi

"${PYTHON_BIN}" "${BENCHMARK_TOOL}" \
    --trainer "${TRAINER}" \
    --manifest-tool "${MANIFEST_TOOL}" \
    --base-config "${CONFIG}" \
    --case baseline "${BASELINE_DATASET}" \
    --case ibi "${IBI_DATASET}" \
    --seeds "${SEEDS[@]}" \
    --output-dir "${OUTDIR}" \
    "${MODE_ARG[@]}"

printf '\n[DONE] Paired baseline/post-IBI multi-seed benchmark: %s\n' "${OUTDIR}/benchmark_summary.json"
printf '[DONE] Per-run table: %s\n' "${OUTDIR}/benchmark_runs.csv"
printf '[NOTE] Defaults use seeds: %s. Override with MULTISEED_SEEDS="11 23 42 73 101".\n' "${SEEDS_RAW}"
printf '[NOTE] Resume an interrupted benchmark with RESUME=1; do not combine it with OVERWRITE=1.\n'
