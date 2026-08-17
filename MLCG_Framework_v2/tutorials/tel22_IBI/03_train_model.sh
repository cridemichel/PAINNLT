#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAINER="${TRAINER:-${FRAMEWORK_ROOT}/training/build/train_painn}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST_TOOL="${FRAMEWORK_ROOT}/training/create_model_manifest.py"

cd "${SCRIPT_DIR}"

DATASET="${IBI_DATASET:-tel22_dataset_ibi_residual.bin}"
RB_INFO="${IBI_RB_INFO:-rigid_bodies_info_ibi.json}"
if [ -n "${IBI_PRIORS:-}" ]; then
    PRIORS="${IBI_PRIORS}"
elif [ -f "ibi_conservative/cg_priors.json" ]; then
    PRIORS="ibi_conservative/cg_priors.json"
elif [ -f "ibi_run_16ps_continue/best/cg_priors.json" ]; then
    PRIORS="ibi_run_16ps_continue/best/cg_priors.json"
else
    PRIORS="ibi_run_16ps/best/cg_priors.json"
fi
MODEL="${IBI_MODEL:-tel22_model_ibi.pt}"
CONFIG="${TRAINING_CONFIG:-tel22_training_config.json}"

# Fail closed before the C++ trainer sees the dataset.  This checks the exact
# residual dataset + rigid-body metadata + validated prior/tables against the
# provenance manifest written by 13_rebuild_residual_dataset.sh.
IBI_DATASET="${DATASET}" \
IBI_RB_INFO="${RB_INFO}" \
IBI_PRIORS="${PRIORS}" \
    bash ./16_check_ibi_training_inputs.sh

for path in "${DATASET}" "${CONFIG}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required training input: ${path}" >&2
        exit 1
    fi
done
if [ ! -x "${TRAINER}" ]; then
    echo "[ERROR] train_painn not found/executable: ${TRAINER}" >&2
    echo "Build it first under training/build or set TRAINER=/path/to/train_painn." >&2
    exit 1
fi

if [ "${MODEL}" = "tel22_model.pt" ]; then
    echo "[ERROR] Refusing to overwrite the pre-IBI baseline model tel22_model.pt." >&2
    echo "Use the default tel22_model_ibi.pt or set IBI_MODEL to a distinct filename." >&2
    exit 1
fi

"${TRAINER}" "${DATASET}" "${MODEL}" "${CONFIG}"

# Finalize the trainer sidecar with SHA256 hashes for model, dataset and config.
"${PYTHON_BIN}" "${MANIFEST_TOOL}" \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --config "${CONFIG}"

echo "[DONE] IBI residual model: ${MODEL}"
echo "[DONE] Model manifest: ${MODEL}.manifest.json"
echo "[INFO] Runtime Hamiltonian companions validated before training: ${RB_INFO}, ${PRIORS}"
