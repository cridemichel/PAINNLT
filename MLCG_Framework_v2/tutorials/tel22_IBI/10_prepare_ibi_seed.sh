#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${SCRIPT_DIR}"

for path in cg_priors.json ibi_selection.json prepare_ibi_seed.py; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" prepare_ibi_seed.py \
    --priors cg_priors.json \
    --selection ibi_selection.json \
    --output cg_priors_ibi_seed.json \
    --report ibi_seed_report.json
