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

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYRESSO="${PYRESSO:-${DEFAULT_PYPRESSO}}"

cd "${TUTORIAL_DIR}"
source "${TUTORIAL_DIR}/model_config.sh"
load_model_dependent_config step23

DRY_RUN=0
OVERWRITE=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        --overwrite) OVERWRITE=1 ;;
    esac
done

MODEL="${IBI_MODEL}"
CONFIG="${TRAINING_CONFIG}"
DATASET="${IBI_DATASET}"
RB_INFO="${IBI_RB_INFO}"
PRIORS="${IBI_PRIORS}"
VALIDATION_REPORT="${IBI_VALIDATION_REPORT}"
RUNTIME_PARITY_REPORT="${IBI_RUNTIME_PARITY_REPORT}"
SOURCE_CHECKPOINT="${NVE_SOURCE_CHECKPOINT}"
REQUIRED_HAMILTONIAN_MODE="${NVE_REQUIRED_HAMILTONIAN_MODE}"


for path in \
    "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" \
    "${PRIORS}" "${VALIDATION_REPORT}" "${RUNTIME_PARITY_REPORT}" "${SOURCE_CHECKPOINT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required conservative-IBI NVE artifact: ${path}" >&2
        exit 1
    fi
done

read -r EQ_STEPS EQ_ACTUAL_DURATION <<EOF_STEPS
$("${PYTHON_BIN}" - "${NVE_EQ_DT}" "${NVE_EQ_DURATION_PS}" <<'PY'
import sys
dt = float(sys.argv[1])
duration = float(sys.argv[2])
if dt <= 0.0 or duration <= 0.0:
    raise SystemExit("NVE_EQ_DT and NVE_EQ_DURATION_PS must be positive")
steps = int(round(duration / dt))
if steps < 1:
    raise SystemExit("IBI-only NVT equilibration requires at least one step")
print(steps, format(steps * dt, ".17g"))
PY
)
EOF_STEPS

cat <<EOF_PLAN
[CONSERVATIVE IBI-ONLY NVE PLAN]
model anchor : ${MODEL} (PaiNN disabled during equilibration and NVE)
config       : ${CONFIG}
dataset      : ${DATASET}
rb_info      : ${RB_INFO}
priors       : ${PRIORS}
source chk   : ${SOURCE_CHECKPOINT}
IBI-only chk : ${NVE_EQ_CHECKPOINT}
NVT prep     : dt=${NVE_EQ_DT} ps steps=${EQ_STEPS} duration=${EQ_ACTUAL_DURATION} ps kT=${NVE_EQ_KT}
dt scan      : ${NVE_DTS} ps
NVE duration : ${NVE_DURATION_PS} ps per dt
device       : ${NVE_DEVICE}
neighbor     : ${NVE_NEIGHBOR_SEARCH}
output       : ${NVE_OUTPUT_DIR}
EOF_PLAN

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/conservative_nve_preflight.py" \
    --priors "${PRIORS}" \
    --validation-report "${VALIDATION_REPORT}" \
    --runtime-parity-report "${RUNTIME_PARITY_REPORT}" \
    --output "${NVE_PREFLIGHT_REPORT}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[PLAN] IBI-only NVT: ${EQ_STEPS} steps at dt=${NVE_EQ_DT} ps -> ${NVE_EQ_CHECKPOINT}"
    read -r -a DT_ARGS <<< "${NVE_DTS}"
    "${PYTHON_BIN}" - "${NVE_DURATION_PS}" "${DT_ARGS[@]}" <<'PY'
import sys
duration = float(sys.argv[1])
for dt in sorted((float(x) for x in sys.argv[2:]), reverse=True):
    steps = int(round(duration / dt))
    print(f"[PLAN] NVE dt={dt:g} ps steps={steps} duration={steps*dt:g} ps log_every=1 steps")
PY
    cat <<EOF_DRY
[CONSERVATIVE IBI-ONLY NVE DRY-RUN COMPLETE]
preflight : ${NVE_PREFLIGHT_REPORT}
NVT source: ${SOURCE_CHECKPOINT}
NVT output: ${NVE_EQ_CHECKPOINT}
[NOTE] No NVT or NVE trajectory was launched.
EOF_DRY
    exit 0
fi

if [[ "${OVERWRITE}" == "1" ]]; then
    rm -rf "${NVE_EQ_DIR}"
fi
mkdir -p "${NVE_EQ_DIR}"
NVE_MODEL_CONFIG_PROVENANCE="${NVE_EQ_DIR}/model_config_provenance.json"
write_model_dependent_provenance "${NVE_MODEL_CONFIG_PROVENANCE}"

if [[ ! -s "${NVE_EQ_CHECKPOINT}" ]]; then
    echo "[INFO] Preparing dedicated conservative IBI-only NVT checkpoint..."
    "${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py" \
        --model "${MODEL}" \
        --disable_ml \
        --config "${CONFIG}" \
        --priors "${PRIORS}" \
        --rb_info "${RB_INFO}" \
        --dataset "${DATASET}" \
        --checkpoint "${SOURCE_CHECKPOINT}" \
        --steps "${EQ_STEPS}" \
        --dt "${NVE_EQ_DT}" \
        --kT "${NVE_EQ_KT}" \
        --thermostat_seed "${NVE_EQ_THERMOSTAT_SEED}" \
        --device "${NVE_DEVICE}" \
        --ml_precision "${NVE_ML_PRECISION}" \
        --neighbor_search "${NVE_NEIGHBOR_SEARCH}" \
        --log_interval "${NVE_EQ_LOG_INTERVAL}" \
        --energy_file "${NVE_EQ_ENERGY}" \
        --no_vtf \
        --out_checkpoint "${NVE_EQ_CHECKPOINT}" \
        2>&1 | tee "${NVE_EQ_LOG}"
else
    echo "[INFO] Reusing existing IBI-only checkpoint: ${NVE_EQ_CHECKPOINT}"
    echo "[INFO] Its Hamiltonian/source provenance will be revalidated before NVE."
fi

if [[ ! -s "${NVE_EQ_CHECKPOINT}" ]]; then
    echo "[ERROR] IBI-only NVT did not produce checkpoint: ${NVE_EQ_CHECKPOINT}" >&2
    exit 1
fi

"${PYTHON_BIN}" - \
    "${SOURCE_CHECKPOINT}" "${NVE_EQ_CHECKPOINT}" "${NVE_EQ_REPORT}" \
    "${NVE_EQ_DT}" "${EQ_STEPS}" "${EQ_ACTUAL_DURATION}" "${NVE_EQ_KT}" \
    "${NVE_EQ_THERMOSTAT_SEED}" "${NVE_NEIGHBOR_SEARCH}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

source, checkpoint, output = map(Path, sys.argv[1:4])
dt = float(sys.argv[4])
steps = int(sys.argv[5])
duration = float(sys.argv[6])
kT = float(sys.argv[7])
seed = int(sys.argv[8])
neighbor = sys.argv[9]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

with np.load(checkpoint, allow_pickle=False) as chk:
    if "metadata_json" not in chk.files:
        raise SystemExit("Dedicated IBI-only checkpoint lacks metadata_json")
    metadata = json.loads(str(np.asarray(chk["metadata_json"]).item()))

expected_mode = "conservative_classical_model_provenance_ml_disabled"
if metadata.get("hamiltonian_mode") != expected_mode:
    raise SystemExit(
        f"Dedicated checkpoint Hamiltonian mode mismatch: {metadata.get('hamiltonian_mode')!r}"
    )
if metadata.get("sampling_ensemble") != "NVT_Langevin":
    raise SystemExit(
        f"Dedicated checkpoint ensemble mismatch: {metadata.get('sampling_ensemble')!r}"
    )
if metadata.get("source_checkpoint_sha256") != sha(source):
    raise SystemExit("Dedicated checkpoint is not derived from the selected source checkpoint")
if abs(float(metadata.get("created_with_dt_ps", -1.0)) - dt) > 1.0e-15:
    raise SystemExit("Dedicated checkpoint dt does not match NVE_EQ_DT")
if abs(float(metadata.get("created_with_kT_kJ_mol", -1.0)) - kT) > 1.0e-12:
    raise SystemExit("Dedicated checkpoint kT does not match NVE_EQ_KT")
if int(metadata.get("completed_steps", -1)) != steps:
    raise SystemExit("Dedicated checkpoint step count does not match NVE_EQ_DURATION_PS/NVE_EQ_DT")
if metadata.get("neighbor_search") != neighbor:
    raise SystemExit("Dedicated checkpoint neighbor search does not match NVE_NEIGHBOR_SEARCH")
if int(metadata.get("thermostat_seed", -1)) != seed:
    raise SystemExit("Dedicated checkpoint thermostat seed does not match NVE_EQ_THERMOSTAT_SEED")

report = {
    "schema_version": 1,
    "framework": "MLCG_Framework_v2",
    "kind": "conservative_ibi_only_nvt_checkpoint",
    "source_checkpoint": str(source.resolve()),
    "source_checkpoint_sha256": sha(source),
    "checkpoint": str(checkpoint.resolve()),
    "checkpoint_sha256": sha(checkpoint),
    "hamiltonian_mode": expected_mode,
    "sampling_ensemble": "NVT_Langevin",
    "ml_active": False,
    "dt_ps": dt,
    "steps": steps,
    "duration_ps": duration,
    "kT_kJ_mol": kT,
    "thermostat_seed": seed,
    "neighbor_search": neighbor,
    "checkpoint_metadata": metadata,
    "pass": True,
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[PASS] IBI-only NVT checkpoint provenance written: {output}")
PY

read -r -a DT_ARGS <<< "${NVE_DTS}"

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/certify_nve.py" \
    --pypresso "${PYRESSO}" \
    --model "${MODEL}" \
    --disable-ml \
    --config "${CONFIG}" \
    --priors "${PRIORS}" \
    --rb-info "${RB_INFO}" \
    --dataset "${DATASET}" \
    --checkpoint "${NVE_EQ_CHECKPOINT}" \
    --require-checkpoint-hamiltonian-mode "${REQUIRED_HAMILTONIAN_MODE}" \
    --require-checkpoint-source "${SOURCE_CHECKPOINT}" \
    --dts "${DT_ARGS[@]}" \
    --duration-ps "${NVE_DURATION_PS}" \
    --device "${NVE_DEVICE}" \
    --ml-precision "${NVE_ML_PRECISION}" \
    --neighbor-search "${NVE_NEIGHBOR_SEARCH}" \
    --output-dir "${NVE_OUTPUT_DIR}" \
    --slope-min "${NVE_SLOPE_MIN}" \
    --slope-max "${NVE_SLOPE_MAX}" \
    --min-r2 "${NVE_MIN_R2}" \
    --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}" \
    --provenance-artifact "conservative_phase2_preflight=${NVE_PREFLIGHT_REPORT}" \
    --provenance-artifact "conservative_validation=${VALIDATION_REPORT}" \
    --provenance-artifact "conservative_runtime_parity=${RUNTIME_PARITY_REPORT}" \
    --provenance-artifact "ibi_only_nvt_equilibration=${NVE_EQ_REPORT}" \
    --provenance-artifact "model_config=${NVE_MODEL_CONFIG_PROVENANCE}" \
    "$@"

cat <<EOF_DONE
[CONSERVATIVE IBI-ONLY NVE COMPLETE]
preflight : ${NVE_PREFLIGHT_REPORT}
NVT source: ${SOURCE_CHECKPOINT}
NVT chk   : ${NVE_EQ_CHECKPOINT}
NVT report: ${NVE_EQ_REPORT}
report    : ${NVE_OUTPUT_DIR}/nve_certification_report.json
table     : ${NVE_OUTPUT_DIR}/nve_certification_runs.csv
[NOTE] PaiNN was disabled both while preparing the dedicated NVT checkpoint and in every NVE trajectory.
EOF_DONE
