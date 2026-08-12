#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILDER="${FRAMEWORK_ROOT}/preprocessing/build_cg_dataset.py"

cd "${SCRIPT_DIR}"

: "${AA_TOPOLOGY:=md.gro}"
: "${AA_TRAJECTORY:=md_whole.trr}"

for path in "${AA_TOPOLOGY}" "${AA_TRAJECTORY}" tel22_topology.json; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" "${BUILDER}" \
    --topology "${AA_TOPOLOGY}" \
    --trajectory "${AA_TRAJECTORY}" \
    --config tel22_topology.json \
    --output tel22_dataset.bin \
    --priors-output cg_priors.json \
    --rb-info-output rigid_bodies_info.json

echo "[DONE] tel22_dataset.bin, cg_priors.json, rigid_bodies_info.json"
