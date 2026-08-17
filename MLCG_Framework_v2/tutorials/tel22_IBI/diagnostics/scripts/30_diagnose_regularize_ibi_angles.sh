#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/model_config.sh" || -f "${SCRIPT_DIR}/cg_priors.json" ]]; then
    TUTORIAL_DIR="${SCRIPT_DIR}"
else
    TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${TUTORIAL_DIR}"
source "${TUTORIAL_DIR}/model_config.sh"
load_model_dependent_config step30

IBI_PRIORS="${IBI_PRIORS}"
REFERENCE_PRIORS="${IBI_ANGLE_REFERENCE_PRIORS}"
TARGET_DATASET="${IBI_ANGLE_TARGET_DATASET}"
IBI_CONFIG="${IBI_SETTINGS}"
RUNTIME_SAMPLE="${IBI_ANGLE_RUNTIME_SAMPLE}"
OUTPUT_DIR="${IBI_ANGLE_REG_OUTPUT_DIR}"


for path in "${IBI_PRIORS}" "${REFERENCE_PRIORS}" "${TARGET_DATASET}" "${IBI_CONFIG}" "${RUNTIME_SAMPLE}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

ARGS=(
  "${FRAMEWORK_ROOT}/ibi/angle_regularization_diagnostics.py"
  --dataset "${TARGET_DATASET}"
  --priors "${IBI_PRIORS}"
  --old-priors "${REFERENCE_PRIORS}"
  --sample-npz "${RUNTIME_SAMPLE}"
  --ibi-config "${IBI_CONFIG}"
  --candidate-specs-json "${IBI_ANGLE_REG_CANDIDATE_SPECS}"
  --hotspot-count "${IBI_ANGLE_REG_HOTSPOT_COUNT}"
  --hotspot-min-separation-rad "${IBI_ANGLE_REG_HOTSPOT_MIN_SEPARATION_RAD}"
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
reference priors: ${REFERENCE_PRIORS}
target dataset  : ${TARGET_DATASET}
runtime sample  : ${RUNTIME_SAMPLE}
IBI settings    : ${IBI_CONFIG}
output          : ${OUTPUT_DIR}
[NOTE] No MD is run. Selected priors are never modified.
[NOTE] Generated candidates are unvalidated and must not be promoted directly.
EOF

"${PYTHON_BIN}" "${ARGS[@]}"

if [[ " $* " != *" --dry-run "* ]]; then
  write_model_dependent_provenance "${OUTPUT_DIR}/model_config_provenance.json"
fi
