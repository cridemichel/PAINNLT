#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYRESSO="${PYRESSO:-${DEFAULT_PYPRESSO}}"

STEPS="${SMOKE_STEPS:-10}"
DT="${SMOKE_DT:-0.0005}"
VELOCITY_SEED="${SMOKE_VELOCITY_SEED:-401001}"
THERMOSTAT_SEED="${SMOKE_THERMOSTAT_SEED:-401101}"
OUTDIR="${SMOKE_OUTDIR:-diagnostics/smoke/premerge_runtime}"
DRY_RUN=0
OVERWRITE=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        --overwrite) OVERWRITE=1 ;;
        *) echo "[ERROR] Unknown argument: ${arg}" >&2; exit 2 ;;
    esac
done

cd "${TUTORIAL_DIR}"
source "${TUTORIAL_DIR}/model_config.sh"
load_model_dependent_config step23

MODEL="${IBI_MODEL}"
CONFIG="${TRAINING_CONFIG}"
DATASET="${IBI_DATASET}"
RB_INFO="${IBI_RB_INFO}"
PRIORS="${IBI_PRIORS}"
ENERGY_FILE="${OUTDIR}/energy.csv"
RUN_LOG="${OUTDIR}/run.log"
REPORT="${OUTDIR}/report.json"

for path in "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" "${PRIORS}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing pre-merge smoke input: ${path}" >&2
        exit 1
    fi
done

cat <<EOF
[TEL22_IBI PRE-MERGE RUNTIME SMOKE]
model anchor : ${MODEL} (PaiNN disabled)
config       : ${CONFIG}
dataset      : ${DATASET}
rb_info      : ${RB_INFO}
priors       : ${PRIORS}
steps / dt   : ${STEPS} / ${DT} ps
neighbor     : ${NVE_NEIGHBOR_SEARCH}
output       : ${OUTDIR}
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[PLAN] Check both custom bonded installers in the current ESPResSo source tree."
    echo "[PLAN] Run ${STEPS} NVT steps with the promoted conservative IBI prior and PaiNN disabled."
    echo "[PLAN] Require exactly ${STEPS} completed steps, finite energies, and E_ml == 0 for every sample."
    exit 0
fi

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/espresso_plugin/install_analytic_morse_bond.py" \
    --espresso-root "${FRAMEWORK_ROOT}/espresso" --check
"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/espresso_plugin/install_conservative_spline_bond.py" \
    --espresso-root "${FRAMEWORK_ROOT}/espresso" --check

if [[ "${OVERWRITE}" == "1" ]]; then
    rm -rf "${OUTDIR}"
fi
mkdir -p "${OUTDIR}"
if [[ -e "${REPORT}" || -e "${ENERGY_FILE}" || -e "${RUN_LOG}" ]]; then
    echo "[ERROR] Smoke output already exists; use --overwrite: ${OUTDIR}" >&2
    exit 1
fi

"${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py" \
    --model "${MODEL}" \
    --disable_ml \
    --config "${CONFIG}" \
    --priors "${PRIORS}" \
    --rb_info "${RB_INFO}" \
    --dataset "${DATASET}" \
    --steps "${STEPS}" \
    --dt "${DT}" \
    --init_kT "${IBI_KT}" \
    --velocity_seed "${VELOCITY_SEED}" \
    --thermostat_seed "${THERMOSTAT_SEED}" \
    --device cpu \
    --ml_precision float32 \
    --neighbor_search "${NVE_NEIGHBOR_SEARCH}" \
    --log_interval 1 \
    --energy_file "${ENERGY_FILE}" \
    --no_vtf \
    2>&1 | tee "${RUN_LOG}"

"${PYTHON_BIN}" - "${ENERGY_FILE}" "${REPORT}" "${STEPS}" "${DT}" "${PRIORS}" <<'PY'
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

energy_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
expected_steps = int(sys.argv[3])
dt = float(sys.argv[4])
priors = Path(sys.argv[5])

with energy_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit("Smoke energy log is empty")

steps = [int(row["Step"]) for row in rows]
if steps[0] != 0 or steps[-1] != expected_steps:
    raise SystemExit(f"Smoke did not complete requested steps: observed {steps[0]}..{steps[-1]}, expected 0..{expected_steps}")
if steps != list(range(expected_steps + 1)):
    raise SystemExit("Smoke energy log does not contain every integration step")

finite_columns = ("E_tot", "E_kin", "E_class", "E_ml", "E_bonded", "E_non_bonded", "f_max", "torque_max")
for row in rows:
    for name in finite_columns:
        if not math.isfinite(float(row[name])):
            raise SystemExit(f"Non-finite {name} at step {row['Step']}")
    if abs(float(row["E_ml"])) > 1.0e-12:
        raise SystemExit(f"PaiNN-disabled smoke has nonzero E_ml at step {row['Step']}: {row['E_ml']}")

report = {
    "schema_version": 1,
    "kind": "tel22_ibi_premerge_runtime_smoke",
    "pass": True,
    "steps": expected_steps,
    "dt_ps": dt,
    "samples": len(rows),
    "ml_active": False,
    "max_abs_E_ml": max(abs(float(row["E_ml"])) for row in rows),
    "max_force": max(float(row["f_max"]) for row in rows),
    "max_torque": max(float(row["torque_max"]) for row in rows),
    "priors": str(priors),
    "priors_sha256": hashlib.sha256(priors.read_bytes()).hexdigest(),
}
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[PASS] TEL22_IBI pre-merge runtime smoke: {expected_steps} steps, E_ml=0, finite energies/forces")
print(f"[PASS] report: {report_path}")
PY
