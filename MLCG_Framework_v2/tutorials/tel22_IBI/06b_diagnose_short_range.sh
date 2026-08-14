#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${SCRIPT_DIR}"

: "${NVE_DT_TAG:=0p005}"
: "${NVE_ENERGY_CSV:=nve_certification/dt_${NVE_DT_TAG}/energy.csv}"
: "${SHORT_RANGE_THRESHOLD_NM:=0.20}"
: "${SHORT_RANGE_REPORT:=nve_certification/short_range_diagnostic_${NVE_DT_TAG}.json}"

for path in tel22_dataset.bin cg_priors.json tel22_topology.json "${NVE_ENERGY_CSV}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/diagnose_short_range_pair.py" \
    --dataset tel22_dataset.bin \
    --priors cg_priors.json \
    --topology tel22_topology.json \
    --energy-csv "${NVE_ENERGY_CSV}" \
    --threshold-nm "${SHORT_RANGE_THRESHOLD_NM}" \
    --output "${SHORT_RANGE_REPORT}" \
    "$@"
