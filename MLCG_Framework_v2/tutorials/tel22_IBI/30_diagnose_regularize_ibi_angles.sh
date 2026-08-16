#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${SCRIPT_DIR}"

IBI_PRIORS="${IBI_PRIORS:-ibi_conservative/cg_priors.json}"
OLD_PRIORS="${OLD_TEL22_PRIORS:-../tel22/cg_priors.json}"
TARGET_DATASET="${IBI_ANGLE_TARGET_DATASET:-tel22_dataset_ibi_residual.bin}"
IBI_CONFIG="${IBI_CONFIG:-ibi_settings.json}"
RUNTIME_SAMPLE="${IBI_ANGLE_RUNTIME_SAMPLE:-ibi_timestep_range_diagnostic/full_ibi/nvt_structured_sample.npz}"
OUTPUT_DIR="${IBI_ANGLE_REG_OUTPUT_DIR:-ibi_angle_regularization_diagnostic}"

for path in "${IBI_PRIORS}" "${OLD_PRIORS}" "${TARGET_DATASET}" "${IBI_CONFIG}" "${RUNTIME_SAMPLE}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

ARGS=(
  "${FRAMEWORK_ROOT}/ibi/angle_regularization_diagnostics.py"
  --dataset "${TARGET_DATASET}"
  --priors "${IBI_PRIORS}"
  --old-priors "${OLD_PRIORS}"
  --sample-npz "${RUNTIME_SAMPLE}"
  --ibi-config "${IBI_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
)
for arg in "$@"; do
    case "${arg}" in
        --dry-run|--overwrite) ARGS+=("${arg}") ;;
        *) echo "[ERROR] Unsupported argument: ${arg}" >&2; exit 2 ;;
    esac
done

cat <<EOF
[IBI ANGLE STIFFNESS/REGULARIZATION DIAGNOSTIC]
selected priors : ${IBI_PRIORS}
old priors      : ${OLD_PRIORS}
target dataset  : ${TARGET_DATASET}
runtime sample  : ${RUNTIME_SAMPLE}
IBI settings    : ${IBI_CONFIG}
output          : ${OUTPUT_DIR}
[NOTE] No MD is run. Selected priors are never modified.
[NOTE] Generated candidates are unvalidated and must not be promoted directly.
EOF

"${PYTHON_BIN}" "${ARGS[@]}"
