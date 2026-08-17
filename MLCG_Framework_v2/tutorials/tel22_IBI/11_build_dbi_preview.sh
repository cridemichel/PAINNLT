#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE="${OVERWRITE:-0}"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/model_config.sh"
load_model_dependent_config step11
OUTDIR="${DBI_OUTDIR}"

for path in "${IBI_TARGET_DATASET}" "${IBI_SEED_PRIORS}" "${IBI_SETTINGS}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

if [ -e "${OUTDIR}" ]; then
    if [ "${OVERWRITE}" != "1" ]; then
        echo "[ERROR] ${OUTDIR} already exists. Re-run with OVERWRITE=1 to replace it." >&2
        exit 1
    fi
    rm -rf "${OUTDIR}"
fi

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/ibi/build_dbi_priors.py" \
    --dataset "${IBI_TARGET_DATASET}" \
    --priors "${IBI_SEED_PRIORS}" \
    --outdir "${OUTDIR}" \
    --ibi-config "${IBI_SETTINGS}"

echo "[DONE] Initial DBI preview: ${OUTDIR}/cg_priors_dbi.json"

write_model_dependent_provenance "${OUTDIR}/model_config_provenance.json"
