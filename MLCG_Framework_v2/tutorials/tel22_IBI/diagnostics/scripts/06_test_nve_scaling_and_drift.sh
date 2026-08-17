#!/usr/bin/env bash
set -euo pipefail

# TEL22_IBI post-merge baseline recheck.
# Tests the CURRENT promoted conservative IBI Hamiltonian with PaiNN disabled.
# A fresh NVT checkpoint is built once, then reused by both this FP32-labelled
# run and 06b_test_nve_scaling_and_drift_fp64.sh for a matched A/B control.
#
# NOTE: with --disable_ml the active Hamiltonian is conservative-only, so
# --ml_precision does not change the force kernel. The FP32/FP64 labels are
# retained to verify that this inactive ML setting cannot perturb the result.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/model_config.sh" ]]; then
    TUTORIAL_DIR="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../../model_config.sh" ]]; then
    TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    echo "[ERROR] Cannot locate tutorials/tel22_IBI from ${SCRIPT_DIR}" >&2
    exit 1
fi
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
RUNNER="${FRAMEWORK_ROOT}/simulation/run_cg_md.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYPRESSO="${PYPRESSO:-${DEFAULT_PYPRESSO}}"

cd "${TUTORIAL_DIR}"
source ./model_config.sh
load_model_dependent_config step34

MODEL="${IBI_MODEL}"
CONFIG="${IBI_PROMOTION_CONFIG}"
DATASET="${IBI_PROMOTION_DATASET}"
RB_INFO="${IBI_PROMOTION_RB_INFO}"
PRIORS="${IBI_PROMOTION_CURRENT_DIR}/cg_priors.json"

NVE_DEVICE="${NVE_DEVICE:-cpu}"
NVE_ML_PRECISION="float32"
NVE_NEIGHBOR_SEARCH="${NVE_NEIGHBOR_SEARCH:-${IBI_PROMOTION_NEIGHBOR_SEARCH}}"
NVE_DTS="${NVE_DTS:-0.001 0.0015 0.002 0.003 0.004 0.005}"
NVE_DURATION_PS="${NVE_DURATION_PS:-2.0}"
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_scaling_drift_recheck_promoted_ibi_fp32}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.8}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.2}"
NVE_MIN_R2="${NVE_MIN_R2:-0.95}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-2e-5}"

# Shared preparation checkpoint for the matched 06/06b comparison.
PREP_DIR="${NVE_PREP_DIR:-diagnostics/nve/nve_scaling_drift_recheck_promoted_ibi_shared_nvt}"
PREP_CHECKPOINT="${PREP_DIR}/equilibrated_promoted_ibi_recheck.npz"
PREP_ENERGY="${PREP_DIR}/energy.csv"
PREP_LOG="${PREP_DIR}/run.log"
PREP_DT="${NVE_PREP_DT:-${IBI_PROMOTION_NVT_DT}}"
PREP_STEPS="${NVE_PREP_STEPS:-${IBI_PROMOTION_NVT_STEPS}}"
PREP_KT="${NVE_PREP_KT:-${IBI_PROMOTION_NVT_KT}}"
PREP_SEED="${NVE_PREP_SEED:-360601}"

usage() {
    cat <<'USAGE'
Usage:
  06_test_nve_scaling_and_drift.sh [--dry-run | --overwrite | --resume]

Post-merge TEL22_IBI conservative-only NVE baseline.
PaiNN is disabled but the promoted model remains bound for provenance.

Defaults:
  dt grid    : 0.001 0.0015 0.002 0.003 0.004 0.005 ps
  duration   : 2 ps per dt
  NVT seed   : 360601
  output     : diagnostics/nve/nve_scaling_drift_recheck_promoted_ibi_fp32

Useful overrides:
  NVE_DURATION_PS=5.0
  NVE_PREP_STEPS=2000
  NVE_DTS="0.001 0.0015 0.002 0.003 0.004 0.005"
USAGE
}

MODE="normal"
case "${1:-}" in
    "") ;;
    --dry-run) MODE="dry-run"; shift ;;
    --overwrite) MODE="overwrite"; shift ;;
    --resume) MODE="resume"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
esac
if (($# != 0)); then
    echo "[ERROR] Unexpected extra arguments: $*" >&2
    exit 2
fi

for path in "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" "${PRIORS}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing required IBI input: ${path}" >&2; exit 1; }
done
[[ -f "${CERTIFIER}" ]] || { echo "[ERROR] Missing certifier: ${CERTIFIER}" >&2; exit 1; }
[[ -f "${RUNNER}" ]] || { echo "[ERROR] Missing runner: ${RUNNER}" >&2; exit 1; }

SOURCE_CHECKPOINT=""
for candidate in \
    "${IBI_PROMOTION_SOURCE_CHECKPOINT}" \
    diagnostics/ml/postibi_runtime_validation/equilibrated_postibi.npz \
    equilibrated.npz; do
    if [[ -f "${candidate}" ]]; then
        SOURCE_CHECKPOINT="${candidate}"
        break
    fi
done
[[ -n "${SOURCE_CHECKPOINT}" ]] || {
    echo "[ERROR] No source checkpoint found. Run the relevant IBI equilibration first." >&2
    exit 1
}

read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} >= 3)) || { echo "[ERROR] NVE_DTS needs at least three values" >&2; exit 1; }

cat <<EOF_PLAN
[TEL22_IBI CONSERVATIVE NVE SCALING + DRIFT RECHECK -- FP32 LABEL]
priors            : ${PRIORS}
model provenance  : ${MODEL}
PaiNN dynamics    : DISABLED
source checkpoint : ${SOURCE_CHECKPOINT}
shared NVT        : ${PREP_STEPS} steps at dt=${PREP_DT} ps, kT=${PREP_KT}
shared checkpoint : ${PREP_CHECKPOINT}
dt grid [ps]      : ${NVE_DTS}
duration / dt     : ${NVE_DURATION_PS} ps
sampling          : every integration step
thermostat NVE    : OFF
device            : ${NVE_DEVICE}
ML precision flag : ${NVE_ML_PRECISION} (inactive because PaiNN is disabled)
neighbor search   : ${NVE_NEIGHBOR_SEARCH}
scaling gate      : ${NVE_SLOPE_MIN} <= p <= ${NVE_SLOPE_MAX}; R2 >= ${NVE_MIN_R2}
drift gate        : relative block-mean drift <= ${NVE_MAX_RELATIVE_DRIFT}
output            : ${NVE_OUTPUT_DIR}
EOF_PLAN

if [[ "${MODE}" == "dry-run" ]]; then
    echo "[PLAN] Would rebuild shared checkpoint ${PREP_CHECKPOINT} from ${SOURCE_CHECKPOINT}."
    for dt in "${DT_ARGS[@]}"; do
        echo "[PLAN] NVE dt=${dt} ps duration=${NVE_DURATION_PS} ps"
    done
    exit 0
fi

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${PREP_DIR}" "${NVE_OUTPUT_DIR}"
fi
mkdir -p "${PREP_DIR}"

if [[ "${MODE}" != "resume" || ! -s "${PREP_CHECKPOINT}" ]]; then
    rm -f "${PREP_CHECKPOINT}" "${PREP_ENERGY}" "${PREP_LOG}"
    echo "[RUN] Fresh promoted-IBI conservative NVT preparation"
    "${PYPRESSO}" "${RUNNER}" \
        --model "${MODEL}" \
        --disable_ml \
        --config "${CONFIG}" \
        --priors "${PRIORS}" \
        --rb_info "${RB_INFO}" \
        --dataset "${DATASET}" \
        --checkpoint "${SOURCE_CHECKPOINT}" \
        --allow_checkpoint_mismatch \
        --dt "${PREP_DT}" \
        --steps "${PREP_STEPS}" \
        --log_interval 1 \
        --device "${NVE_DEVICE}" \
        --ml_precision "${NVE_ML_PRECISION}" \
        --neighbor_search "${NVE_NEIGHBOR_SEARCH}" \
        --energy_file "${PREP_ENERGY}" \
        --no_vtf \
        --kT "${PREP_KT}" \
        --thermostat_seed "${PREP_SEED}" \
        --out_checkpoint "${PREP_CHECKPOINT}" \
        2>&1 | tee "${PREP_LOG}"
else
    echo "[REUSE] Shared NVT checkpoint: ${PREP_CHECKPOINT}"
fi
[[ -s "${PREP_CHECKPOINT}" ]] || { echo "[ERROR] NVT preparation did not produce ${PREP_CHECKPOINT}" >&2; exit 1; }

CERT_CMD=(
    "${PYTHON_BIN}" "${CERTIFIER}"
    --pypresso "${PYPRESSO}"
    --model "${MODEL}"
    --disable-ml
    --config "${CONFIG}"
    --priors "${PRIORS}"
    --rb-info "${RB_INFO}"
    --dataset "${DATASET}"
    --checkpoint "${PREP_CHECKPOINT}"
    --require-checkpoint-hamiltonian-mode conservative_classical_model_provenance_ml_disabled
    --require-checkpoint-source "${SOURCE_CHECKPOINT}"
    --dts "${DT_ARGS[@]}"
    --duration-ps "${NVE_DURATION_PS}"
    --device "${NVE_DEVICE}"
    --ml-precision "${NVE_ML_PRECISION}"
    --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
    --output-dir "${NVE_OUTPUT_DIR}"
    --slope-min "${NVE_SLOPE_MIN}"
    --slope-max "${NVE_SLOPE_MAX}"
    --min-r2 "${NVE_MIN_R2}"
    --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}"
)
if [[ "${MODE}" == "resume" ]]; then
    CERT_CMD+=(--reuse-existing)
elif [[ "${MODE}" == "overwrite" ]]; then
    CERT_CMD+=(--overwrite)
fi
"${CERT_CMD[@]}"

REPORT="${NVE_OUTPUT_DIR}/nve_certification_report.json"
[[ -f "${REPORT}" ]] || { echo "[ERROR] Missing report: ${REPORT}" >&2; exit 1; }

"${PYTHON_BIN}" - "${REPORT}" <<'PY'
import json, math, sys
p = sys.argv[1]
r = json.load(open(p, encoding="utf-8"))
c = r["certification"]
s = c["scaling"]
runs = sorted(r["runs"], key=lambda x: float(x["dt_ps"]))
print("\n[TEL22_IBI CONSERVATIVE NVE RECHECK SUMMARY -- FP32 LABEL]")
print(f"[SCALING] p={float(s['exponent_p']):.6f} R2={float(s['loglog_r2']):.6f} pass={c['scaling_pass']}")
worst = max(runs, key=lambda x: float(x["relative_block_mean_drift"]))
print(f"[DRIFT] max={float(worst['relative_block_mean_drift']):.3e} at dt={float(worst['dt_ps']):g} ps pass={c['drift_pass']}")
print("[TABLE] dt_ps       sigma_E       sigma_E/dt^2    rel_block_drift")
c2=[]
for x in runs:
    dt=float(x["dt_ps"]); sig=float(x["sigma_E"]); drift=float(x["relative_block_mean_drift"])
    val=sig/(dt*dt); c2.append(val)
    print(f"        {dt:<10g} {sig:<13.6g} {val:<15.6g} {drift:.3e}")
pos=[x for x in c2 if x>0 and math.isfinite(x)]
if pos:
    print(f"[C2] max/min spread={max(pos)/min(pos):.3f}")
print(f"[FINAL] pass={c['pass']}")
print(f"[REPORT] {p}")
PY
