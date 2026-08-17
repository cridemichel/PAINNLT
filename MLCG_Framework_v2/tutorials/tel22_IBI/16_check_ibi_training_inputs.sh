#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKER="${FRAMEWORK_ROOT}/training/residual_input_provenance.py"

cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/model_config.sh"
load_model_dependent_config step16

DATASET="${IBI_DATASET}"
RB_INFO="${IBI_RB_INFO}"
if [[ -z "${IBI_PRIORS:-}" ]]; then
  for candidate in ${IBI_TRAINING_PRIOR_CANDIDATES}; do [[ -f "${candidate}" ]] && { IBI_PRIORS="${candidate}"; break; }; done
fi
[[ -n "${IBI_PRIORS:-}" ]] || { echo "[ERROR] No configured training prior candidate exists." >&2; exit 1; }
PRIORS="${IBI_PRIORS}"
PROVENANCE="${IBI_RESIDUAL_PROVENANCE}"

for path in "${DATASET}" "${RB_INFO}" "${PRIORS}" "${PROVENANCE}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required IBI training artifact: ${path}" >&2
        echo "[HINT] Re-run ./13_rebuild_residual_dataset.sh after applying this patch." >&2
        exit 1
    fi
done

"${PYTHON_BIN}" "${CHECKER}" check \
    --manifest "${PROVENANCE}" \
    --dataset "${DATASET}" \
    --rb-info "${RB_INFO}" \
    --priors "${PRIORS}"

write_model_dependent_provenance "${PROVENANCE%.json}_check_model_config_provenance.json"
