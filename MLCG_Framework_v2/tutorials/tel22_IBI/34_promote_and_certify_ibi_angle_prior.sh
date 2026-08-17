#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
source "${HERE}/model_config.sh"
load_model_dependent_config step34
cd "${HERE}"
PYPRESSO="${PYPRESSO:-${ROOT}/espresso/build/pypresso}"

CURRENT_DIR="${IBI_PROMOTION_CURRENT_DIR}"
CURRENT_PRIORS="${CURRENT_DIR}/cg_priors.json"
BACKUP_DIR="${IBI_PROMOTION_BACKUP_DIR}"
CANDIDATE_PRIORS="${IBI_PROMOTION_CANDIDATE_PRIORS}"
STEP33_REPORT="${IBI_PROMOTION_STEP33_REPORT}"
EXPECTED_CANDIDATE_SHA256="${IBI_PROMOTION_EXPECTED_CANDIDATE_SHA256}"
EXPECTED_SIGMA="${IBI_PROMOTION_EXPECTED_SIGMA_RAD}"
MODEL="${IBI_MODEL}"
CONFIG="${IBI_PROMOTION_CONFIG}"
DATASET="${IBI_PROMOTION_DATASET}"
RB_INFO="${IBI_PROMOTION_RB_INFO}"
SOURCE_CHECKPOINT="${IBI_PROMOTION_SOURCE_CHECKPOINT}"
OUT="${IBI_PROMOTION_OUTPUT_DIR}"
NVT_DIR="${OUT}/nvt"
NVT_CHECKPOINT="${NVT_DIR}/equilibrated_promoted_ibi_only.npz"
NVT_ENERGY="${NVT_DIR}/energy.csv"
PREFLIGHT="${OUT}/promoted_nve_preflight.json"
NVE_DIR="${OUT}/nve_sigma_scaling"
NVE_REPORT="${NVE_DIR}/nve_certification_report.json"
STATE_DIR="${OUT}/state_convergence"
STATE_REPORT="${STATE_DIR}/state_convergence_report.json"
FINAL_REPORT="${OUT}/promoted_ibi_final_certification_report.json"
PROMOTION_REPORT="${CURRENT_DIR}/promotion_report.json"
VALIDATION_REPORT="${CURRENT_DIR}/validation_report.json"
PARITY_REPORT="${CURRENT_DIR}/runtime_parity_report.json"
RESIDUAL_STATUS="${CURRENT_DIR}/residual_ml_status.json"
CONFIG_PROVENANCE="${OUT}/model_config_provenance.json"
read -r -a DTS <<< "${IBI_PROMOTION_DTS}"
read -r -a RICHARDSON_DTS <<< "${IBI_PROMOTION_RICHARDSON_DTS}"
FULL_DT="${DTS[$((${#DTS[@]} - 1))]}"

MODE=""
for arg in "$@"; do
  case "${arg}" in
    --dry-run) MODE="dry-run" ;;
    --promote) MODE="promote" ;;
    --resume) MODE="resume" ;;
    *) echo "[ERROR] Unsupported argument: ${arg}" >&2; exit 2 ;;
  esac
done
if [[ -z "${MODE}" || $# -ne 1 ]]; then
  echo "[ERROR] Choose exactly one mode: --dry-run, --promote, or --resume" >&2
  exit 2
fi

for path in "${CANDIDATE_PRIORS}" "${STEP33_REPORT}" "${MODEL}" "${MODEL}.manifest.json" \
            "${CONFIG}" "${DATASET}" "${RB_INFO}" "${SOURCE_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing required artifact: ${path}" >&2; exit 1; }
done
[[ -x "${PYPRESSO}" ]] || { echo "[ERROR] pypresso not executable: ${PYPRESSO}" >&2; exit 1; }

cat <<EOF2
[PROMOTE + POST-PROMOTION CERTIFY CONSERVATIVE IBI]
candidate       : ${CANDIDATE_PRIORS} (expected sigma=${EXPECTED_SIGMA} rad)
candidate SHA   : ${EXPECTED_CANDIDATE_SHA256}
step33 report   : ${STEP33_REPORT}
production      : ${CURRENT_PRIORS}
backup          : ${BACKUP_DIR}
model           : ${MODEL} (PaiNN disabled in all certification dynamics)
residual dataset: ${DATASET} (stale for ML-active use after promotion)
fresh NVT       : ${IBI_PROMOTION_NVT_STEPS} steps at dt=${IBI_PROMOTION_NVT_DT} ps
fresh NVE scan  : ${DTS[*]} ps, ${IBI_PROMOTION_NVE_DURATION_PS} ps each
fresh Richardson: dt_ref=${IBI_PROMOTION_RICHARDSON_REFERENCE_DT} ps, duration=${IBI_PROMOTION_RICHARDSON_DURATION_PS} ps
config          : ${MODEL_DEPENDENT_CONFIG_PATH}
output          : ${OUT}
[NOTE] All TEL22/model-dependent numerical choices above come from the external workflow config.
[NOTE] Promotion never rebuilds or re-enables PaiNN; the residual/model are marked stale.
EOF2

PROMOTE_ARGS=(
  "${ROOT}/ibi/promote_validated_angle_candidate.py"
  --current-dir "${CURRENT_DIR}"
  --candidate-priors "${CANDIDATE_PRIORS}"
  --final-validation-report "${STEP33_REPORT}"
  --backup-dir "${BACKUP_DIR}"
  --expected-candidate-sha256 "${EXPECTED_CANDIDATE_SHA256}"
  --expected-sigma-rad "${EXPECTED_SIGMA}"
  --dataset "${DATASET}"
  --model "${MODEL}"
)

if [[ "${MODE}" == "dry-run" ]]; then
  "${PYTHON_BIN}" "${PROMOTE_ARGS[@]}" --dry-run
  "${PYTHON_BIN}" - "${IBI_PROMOTION_NVT_STEPS}" "${IBI_PROMOTION_NVE_DURATION_PS}" "${IBI_PROMOTION_RICHARDSON_DURATION_PS}" "${IBI_PROMOTION_RICHARDSON_REFERENCE_DT}" "${IBI_PROMOTION_DTS}" "${IBI_PROMOTION_RICHARDSON_DTS}" <<'PY'
import sys
nvt=int(sys.argv[1]); nve_dur=float(sys.argv[2]); rich_dur=float(sys.argv[3]); ref=float(sys.argv[4])
dts=[float(x) for x in sys.argv[5].split()]; rdts=[float(x) for x in sys.argv[6].split()]
steps=nvt+sum(round(nve_dur/x) for x in dts)+round(rich_dur/ref)+sum(round(rich_dur/x) for x in rdts)
print(f"[PLAN] Approximate new integration work after promotion: {steps} steps from configured grids.")
PY
  echo "[NOTE] No production priors or dynamics were modified/launched."
  exit 0
fi

if [[ "${MODE}" == "promote" ]]; then
  if [[ -e "${OUT}" ]]; then
    echo "[ERROR] Certification output already exists: ${OUT}" >&2
    echo "[HINT] Use --resume after an interrupted promoted run." >&2
    exit 1
  fi
  "${PYTHON_BIN}" "${PROMOTE_ARGS[@]}"
  mkdir -p "${OUT}"
else
  "${PYTHON_BIN}" "${PROMOTE_ARGS[@]}" --verify-only
  mkdir -p "${OUT}"
fi
write_model_dependent_provenance "${CONFIG_PROVENANCE}"

"${PYTHON_BIN}" "${ROOT}/ibi/validate_conservative_spline.py" --conversion-report "${CURRENT_DIR}/conversion_report.json"
"${PYPRESSO}" "${ROOT}/simulation/diagnose_conservative_spline_parity.py" --priors "${CURRENT_PRIORS}" --report "${PARITY_REPORT}"
"${PYTHON_BIN}" "${ROOT}/simulation/conservative_nve_preflight.py" \
  --priors "${CURRENT_PRIORS}" --validation-report "${VALIDATION_REPORT}" \
  --runtime-parity-report "${PARITY_REPORT}" --output "${PREFLIGHT}"

FINALIZE_ARGS=(
  "${ROOT}/simulation/finalize_promoted_ibi_certification.py"
  --priors "${CURRENT_PRIORS}" --promotion-report "${PROMOTION_REPORT}"
  --step33-report "${STEP33_REPORT}" --validation-report "${VALIDATION_REPORT}"
  --runtime-parity-report "${PARITY_REPORT}" --preflight-report "${PREFLIGHT}"
  --strict-nve-report "${NVE_REPORT}" --state-report "${STATE_REPORT}"
  --residual-ml-status "${RESIDUAL_STATUS}"
  --expected-candidate-sha256 "${EXPECTED_CANDIDATE_SHA256}"
  --sigma-p-min "${IBI_PROMOTION_SIGMA_P_MIN}" --sigma-p-max "${IBI_PROMOTION_SIGMA_P_MAX}"
  --sigma-r2-min "${IBI_PROMOTION_SIGMA_R2_MIN}" --sigma-c2-spread-max "${IBI_PROMOTION_SIGMA_C2_SPREAD_MAX}"
  --full-dt-ps "${FULL_DT}" --max-relative-drift "${IBI_PROMOTION_MAX_RELATIVE_DRIFT}"
  --state-p-min "${IBI_PROMOTION_STATE_P_MIN}" --state-p-max "${IBI_PROMOTION_STATE_P_MAX}"
  --state-r2-min "${IBI_PROMOTION_STATE_R2_MIN}"
  --model-config-provenance "${CONFIG_PROVENANCE}" --output "${FINAL_REPORT}"
)

if [[ "${MODE}" == "resume" && -f "${FINAL_REPORT}" ]]; then
  "${PYTHON_BIN}" "${FINALIZE_ARGS[@]}"
  exit 0
fi

REUSE_DYNAMICS=0
if [[ "${MODE}" == "resume" && -f "${NVE_REPORT}" && -f "${STATE_REPORT}" && -f "${NVT_CHECKPOINT}" ]]; then
  REUSE_DYNAMICS=1
fi
mkdir -p "${NVT_DIR}"
if [[ "${REUSE_DYNAMICS}" == "1" ]]; then
  echo "[REUSE] promoted NVT + sigma scaling + Richardson reports"
else
  rm -rf "${NVE_DIR}" "${STATE_DIR}"
  rm -f "${NVT_CHECKPOINT}" "${NVT_ENERGY}" "${NVT_DIR}/run.log"
  echo "[RUN] fresh promoted-prior NVT branch"
  "${PYPRESSO}" "${ROOT}/simulation/run_cg_md.py" \
    --model "${MODEL}" --disable_ml --config "${CONFIG}" --priors "${CURRENT_PRIORS}" \
    --rb_info "${RB_INFO}" --dataset "${DATASET}" --checkpoint "${SOURCE_CHECKPOINT}" --allow_checkpoint_mismatch \
    --dt "${IBI_PROMOTION_NVT_DT}" --steps "${IBI_PROMOTION_NVT_STEPS}" --log_interval "${IBI_PROMOTION_NVT_LOG_INTERVAL}" \
    --device "${IBI_PROMOTION_DEVICE}" --ml_precision "${IBI_PROMOTION_ML_PRECISION}" --neighbor_search "${IBI_PROMOTION_NEIGHBOR_SEARCH}" \
    --energy_file "${NVT_ENERGY}" --no_vtf --kT "${IBI_PROMOTION_NVT_KT}" --thermostat_seed "${IBI_PROMOTION_NVT_THERMOSTAT_SEED}" \
    --out_checkpoint "${NVT_CHECKPOINT}" > "${NVT_DIR}/run.log" 2>&1
fi

if [[ "${REUSE_DYNAMICS}" != "1" ]]; then
  echo "[RUN] fresh production-path sigma_E(dt) certification"
  "${PYTHON_BIN}" "${ROOT}/simulation/certify_nve.py" \
    --pypresso "${PYPRESSO}" --model "${MODEL}" --disable-ml --config "${CONFIG}" --priors "${CURRENT_PRIORS}" \
    --rb-info "${RB_INFO}" --dataset "${DATASET}" --checkpoint "${NVT_CHECKPOINT}" --dts "${DTS[@]}" \
    --duration-ps "${IBI_PROMOTION_NVE_DURATION_PS}" --device "${IBI_PROMOTION_DEVICE}" --ml-precision "${IBI_PROMOTION_ML_PRECISION}" \
    --neighbor-search "${IBI_PROMOTION_NEIGHBOR_SEARCH}" --output-dir "${NVE_DIR}" \
    --slope-min "${IBI_PROMOTION_SIGMA_P_MIN}" --slope-max "${IBI_PROMOTION_SIGMA_P_MAX}" --min-r2 "${IBI_PROMOTION_SIGMA_R2_MIN}" \
    --max-relative-drift "${IBI_PROMOTION_MAX_RELATIVE_DRIFT}" \
    --provenance-artifact "promotion=${PROMOTION_REPORT}" --provenance-artifact "step33_validation=${STEP33_REPORT}" \
    --provenance-artifact "post_promotion_preflight=${PREFLIGHT}" --provenance-artifact "model_config=${CONFIG_PROVENANCE}"

  echo "[RUN] fresh production-path Richardson state convergence"
  "${PYTHON_BIN}" "${ROOT}/simulation/nve_state_convergence.py" \
    --pypresso "${PYPRESSO}" --model "${MODEL}" --config "${CONFIG}" --priors "${CURRENT_PRIORS}" \
    --rb-info "${RB_INFO}" --dataset "${DATASET}" --checkpoint "${NVT_CHECKPOINT}" \
    --require-checkpoint-hamiltonian-mode conservative_classical_model_provenance_ml_disabled \
    --require-checkpoint-source "${SOURCE_CHECKPOINT}" --dts "${RICHARDSON_DTS[@]}" \
    --reference-dt "${IBI_PROMOTION_RICHARDSON_REFERENCE_DT}" --duration-ps "${IBI_PROMOTION_RICHARDSON_DURATION_PS}" \
    --sample-interval-ps "${IBI_PROMOTION_RICHARDSON_SAMPLE_INTERVAL_PS}" --device "${IBI_PROMOTION_DEVICE}" \
    --ml-precision "${IBI_PROMOTION_ML_PRECISION}" --neighbor-search "${IBI_PROMOTION_NEIGHBOR_SEARCH}" --output-dir "${STATE_DIR}" \
    --order-min "${IBI_PROMOTION_STATE_P_MIN}" --order-max "${IBI_PROMOTION_STATE_P_MAX}" --min-r2 "${IBI_PROMOTION_STATE_R2_MIN}"
fi

"${PYTHON_BIN}" "${FINALIZE_ARGS[@]}"

cat <<EOF2
[DONE] promoted priors : ${CURRENT_PRIORS}
[DONE] backup          : ${BACKUP_DIR}
[DONE] certification   : ${FINAL_REPORT}
[NOTE] PaiNN remains stale/disabled. Rebuild residual labels and retrain before any ML-active use of the promoted priors.
EOF2
