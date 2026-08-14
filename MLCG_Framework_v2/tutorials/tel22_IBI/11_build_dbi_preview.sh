#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTDIR="${DBI_OUTDIR:-ibi_dbi_preview}"
OVERWRITE="${OVERWRITE:-0}"
cd "${SCRIPT_DIR}"

for path in tel22_dataset.bin cg_priors_ibi_seed.json ibi_settings.json; do
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
    --dataset tel22_dataset.bin \
    --priors cg_priors_ibi_seed.json \
    --outdir "${OUTDIR}" \
    --ibi-config ibi_settings.json

echo "[DONE] Initial DBI preview: ${OUTDIR}/cg_priors_dbi.json"
