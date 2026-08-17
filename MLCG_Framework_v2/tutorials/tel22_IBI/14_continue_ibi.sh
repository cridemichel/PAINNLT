#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
elif command -v pypresso >/dev/null 2>&1; then
    DEFAULT_PYPRESSO="$(command -v pypresso)"
else
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
fi

PYPRESSO="${PYPRESSO:-${DEFAULT_PYPRESSO}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE="${OVERWRITE:-0}"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/model_config.sh"
load_model_dependent_config step14
IBI_PARENT_DIR="${IBI_PARENT_DIR}"
IBI_OUTDIR="${IBI_OUTDIR}"
IBI_ITERATIONS="${IBI_ITERATIONS}"
NEIGHBOR_SEARCH="${NEIGHBOR_SEARCH}"
VELOCITY_SEED="${VELOCITY_SEED}"
THERMOSTAT_SEED="${THERMOSTAT_SEED}"

PARENT_REPORT="${IBI_PARENT_DIR}/ibi_report.json"
RESUME_PRIORS="${IBI_PARENT_DIR}/best/cg_priors.json"
for path in "${IBI_TARGET_DATASET}" "${IBI_SETTINGS}" "${TRAINING_CONFIG}" rigid_bodies_info.json "${PARENT_REPORT}" "${RESUME_PRIORS}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

if [ -n "${IBI_ITERATION_OFFSET:-}" ]; then
    ITERATION_OFFSET="${IBI_ITERATION_OFFSET}"
else
    ITERATION_OFFSET="$(${PYTHON_BIN} - "${PARENT_REPORT}" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
iterations = [int(item["iteration"]) for item in report.get("metrics", [])]
if not iterations:
    raise SystemExit("parent report has no evaluated iterations")
print(max(iterations))
PY
)"
fi

args=(
    "${FRAMEWORK_ROOT}/ibi/run_ibi_loop.py"
    --dataset "${IBI_TARGET_DATASET}"
    --resume-priors "${RESUME_PRIORS}"
    --iteration-offset "${ITERATION_OFFSET}"
    --config "${TRAINING_CONFIG}"
    --rb_info rigid_bodies_info.json
    --pypresso "${PYPRESSO}"
    --iterations "${IBI_ITERATIONS}"
    --outdir "${IBI_OUTDIR}"
    --ibi-config "${IBI_SETTINGS}"
    --neighbor_search "${NEIGHBOR_SEARCH}"
    --velocity_seed "${VELOCITY_SEED}"
    --thermostat_seed "${THERMOSTAT_SEED}"
)
if [ "${OVERWRITE}" = "1" ]; then
    args+=(--overwrite)
fi

"${PYTHON_BIN}" "${args[@]}"

summary_args=(
    "${SCRIPT_DIR}/summarize_ibi_convergence.py"
    --previous-report "${PARENT_REPORT}"
    --report "${IBI_OUTDIR}/ibi_report.json"
    --output "${IBI_OUTDIR}/ibi_convergence_summary.json"
    --best-dir "${IBI_OUTDIR}/best"
)
if [ "${OVERWRITE}" = "1" ]; then
    summary_args+=(--overwrite)
fi
"${PYTHON_BIN}" "${summary_args[@]}"

echo "[DONE] Continued IBI output: ${IBI_OUTDIR}/cg_priors_final.json"
echo "[DONE] Best evaluated priors across parent + continuation: ${IBI_OUTDIR}/best/cg_priors.json"
echo "[NOTE] Resume source was the parent best evaluated set, not the unevaluated parent final update."

write_model_dependent_provenance "${IBI_OUTDIR}/model_config_provenance.json"
