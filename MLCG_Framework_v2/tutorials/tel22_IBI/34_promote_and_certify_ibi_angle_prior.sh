#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYPRESSO="${PYPRESSO:-${ROOT}/espresso/build/pypresso}"

CURRENT_DIR="${HERE}/ibi_conservative"
CURRENT_PRIORS="${CURRENT_DIR}/cg_priors.json"
BACKUP_DIR="${HERE}/ibi_conservative_pre_smooth_0p0075"
CANDIDATE_PRIORS="${HERE}/ibi_angle_smoothing_sweep/candidates/smooth_0p0075_wall_current/cg_priors.json"
STEP33_REPORT="${HERE}/ibi_angle_final_candidate_validation/angle_final_candidate_validation_report.json"
EXPECTED_CANDIDATE_SHA256="c31f6d0d53f053071ab694f91d8271c83fc90a90ada291ba60c206adf82a3799"
EXPECTED_SIGMA="0.0075"

MODEL="${IBI_MODEL:-${HERE}/tel22_model_ibi_conservative.pt}"
CONFIG="${HERE}/tel22_training_config.json"
DATASET="${HERE}/tel22_dataset_ibi_residual.bin"
RB_INFO="${HERE}/rigid_bodies_info_ibi.json"
SOURCE_CHECKPOINT="${HERE}/nve_equilibration_conservative_ibi_only/equilibrated_conservative_ibi_only.npz"

OUT="${HERE}/ibi_promoted_final_certification"
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

DTS=(0.001 0.002 0.003 0.004 0.005)
NVT_DT="0.0005"
NVT_STEPS="1000"
NVE_DURATION="1.0"

MODE=""
for arg in "$@"; do
  case "${arg}" in
    --dry-run) MODE="dry-run" ;;
    --promote) MODE="promote" ;;
    --resume) MODE="resume" ;;
    *) echo "[ERROR] Unsupported argument: ${arg}" >&2; exit 2 ;;
  esac
done
if [[ -z "${MODE}" ]]; then
  echo "[ERROR] Choose exactly one of --dry-run, --promote, or --resume" >&2
  exit 2
fi
if [[ $# -ne 1 ]]; then
  echo "[ERROR] Choose exactly one mode: --dry-run, --promote, or --resume" >&2
  exit 2
fi

for path in "${CANDIDATE_PRIORS}" "${STEP33_REPORT}" "${MODEL}" "${MODEL}.manifest.json" \
            "${CONFIG}" "${DATASET}" "${RB_INFO}" "${SOURCE_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing required artifact: ${path}" >&2; exit 1; }
done
[[ -x "${PYPRESSO}" ]] || { echo "[ERROR] pypresso not executable: ${PYPRESSO}" >&2; exit 1; }

cat <<EOF
[PROMOTE + POST-PROMOTION CERTIFY CONSERVATIVE IBI]
candidate       : smooth_0p0075 (${EXPECTED_SIGMA} rad)
candidate SHA   : ${EXPECTED_CANDIDATE_SHA256}
step33 report   : ${STEP33_REPORT}
production      : ${CURRENT_PRIORS}
backup          : ${BACKUP_DIR}
model           : ${MODEL} (PaiNN disabled in all certification dynamics)
residual dataset: ${DATASET} (stale for ML-active use after promotion)
fresh NVT       : ${NVT_STEPS} steps at dt=${NVT_DT} ps
fresh NVE scan  : ${DTS[*]} ps, ${NVE_DURATION} ps each
fresh Richardson: dt_ref=0.0000625 ps, duration=0.096 ps
output          : ${OUT}
[NOTE] sigma_E scaling is gating again in this post-promotion certification.
[NOTE] Promotion never rebuilds or re-enables PaiNN; the residual/model are marked stale.
EOF

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
  echo "[PLAN] New integration after promotion: about 6009 steps (1000 NVT + 2033 sigma scan + 2976 Richardson)."
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

# Rebuild validation/parity against the exact production path.  Old reports are
# deliberately not copied through the promotion transaction.
"${PYTHON_BIN}" "${ROOT}/ibi/validate_conservative_spline.py" \
  --conversion-report "${CURRENT_DIR}/conversion_report.json"
"${PYPRESSO}" "${ROOT}/simulation/diagnose_conservative_spline_parity.py" \
  --priors "${CURRENT_PRIORS}" \
  --report "${PARITY_REPORT}"
"${PYTHON_BIN}" "${ROOT}/simulation/conservative_nve_preflight.py" \
  --priors "${CURRENT_PRIORS}" \
  --validation-report "${VALIDATION_REPORT}" \
  --runtime-parity-report "${PARITY_REPORT}" \
  --output "${PREFLIGHT}"

# A completed final report is the strongest possible resume marker; re-run the
# finalizer to verify all hashes before returning.
if [[ "${MODE}" == "resume" && -f "${FINAL_REPORT}" ]]; then
  "${PYTHON_BIN}" "${ROOT}/simulation/finalize_promoted_ibi_certification.py" \
    --priors "${CURRENT_PRIORS}" --promotion-report "${PROMOTION_REPORT}" \
    --step33-report "${STEP33_REPORT}" --validation-report "${VALIDATION_REPORT}" \
    --runtime-parity-report "${PARITY_REPORT}" --preflight-report "${PREFLIGHT}" \
    --strict-nve-report "${NVE_REPORT}" --state-report "${STATE_REPORT}" \
    --residual-ml-status "${RESIDUAL_STATUS}" \
    --expected-candidate-sha256 "${EXPECTED_CANDIDATE_SHA256}" --output "${FINAL_REPORT}"
  exit 0
fi

# Resume is fail-safe rather than maximally granular.  If both expensive reports
# are complete they are reused together.  Otherwise regenerate the short NVT
# checkpoint and both diagnostics so no partial report can be bound to a
# different checkpoint.
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
    --model "${MODEL}" --disable_ml \
    --config "${CONFIG}" --priors "${CURRENT_PRIORS}" --rb_info "${RB_INFO}" --dataset "${DATASET}" \
    --checkpoint "${SOURCE_CHECKPOINT}" --allow_checkpoint_mismatch \
    --dt "${NVT_DT}" --steps "${NVT_STEPS}" --log_interval 1 \
    --device cpu --ml_precision float32 --neighbor_search link-cell \
    --energy_file "${NVT_ENERGY}" --no_vtf --kT 2.49 --thermostat_seed 340001 \
    --out_checkpoint "${NVT_CHECKPOINT}" > "${NVT_DIR}/run.log" 2>&1
fi

if [[ "${REUSE_DYNAMICS}" == "1" ]]; then
  echo "[REUSE] fresh production-path sigma scaling report: ${NVE_REPORT}"
else
  echo "[RUN] fresh production-path sigma_E(dt) certification"
  "${PYTHON_BIN}" "${ROOT}/simulation/certify_nve.py" \
    --pypresso "${PYPRESSO}" --model "${MODEL}" --disable-ml \
    --config "${CONFIG}" --priors "${CURRENT_PRIORS}" --rb-info "${RB_INFO}" --dataset "${DATASET}" \
    --checkpoint "${NVT_CHECKPOINT}" --dts "${DTS[@]}" --duration-ps "${NVE_DURATION}" \
    --device cpu --ml-precision float32 --neighbor-search link-cell \
    --output-dir "${NVE_DIR}" --slope-min 1.8 --slope-max 2.2 --min-r2 0.95 \
    --max-relative-drift 2e-5 \
    --provenance-artifact "promotion=${PROMOTION_REPORT}" \
    --provenance-artifact "step33_validation=${STEP33_REPORT}" \
    --provenance-artifact "post_promotion_preflight=${PREFLIGHT}"
fi

if [[ "${REUSE_DYNAMICS}" == "1" ]]; then
  echo "[REUSE] fresh production-path Richardson report: ${STATE_REPORT}"
else
  echo "[RUN] fresh production-path Richardson state convergence"
  "${PYTHON_BIN}" "${ROOT}/simulation/nve_state_convergence.py" \
    --pypresso "${PYPRESSO}" --model "${MODEL}" --config "${CONFIG}" \
    --priors "${CURRENT_PRIORS}" --rb-info "${RB_INFO}" --dataset "${DATASET}" \
    --checkpoint "${NVT_CHECKPOINT}" \
    --require-checkpoint-hamiltonian-mode conservative_classical_model_provenance_ml_disabled \
    --require-checkpoint-source "${SOURCE_CHECKPOINT}" \
    --dts 0.001 0.0005 0.00025 0.000125 --reference-dt 0.0000625 \
    --duration-ps 0.096 --sample-interval-ps 0.012 \
    --device cpu --ml-precision float32 --neighbor-search link-cell \
    --output-dir "${STATE_DIR}" --order-min 1.7 --order-max 2.3 --min-r2 0.95
fi

"${PYTHON_BIN}" "${ROOT}/simulation/finalize_promoted_ibi_certification.py" \
  --priors "${CURRENT_PRIORS}" --promotion-report "${PROMOTION_REPORT}" \
  --step33-report "${STEP33_REPORT}" --validation-report "${VALIDATION_REPORT}" \
  --runtime-parity-report "${PARITY_REPORT}" --preflight-report "${PREFLIGHT}" \
  --strict-nve-report "${NVE_REPORT}" --state-report "${STATE_REPORT}" \
  --residual-ml-status "${RESIDUAL_STATUS}" \
  --expected-candidate-sha256 "${EXPECTED_CANDIDATE_SHA256}" --output "${FINAL_REPORT}"

cat <<EOF
[DONE] promoted priors : ${CURRENT_PRIORS}
[DONE] backup          : ${BACKUP_DIR}
[DONE] certification   : ${FINAL_REPORT}
[NOTE] PaiNN remains stale/disabled. Rebuild residual labels and retrain before any ML-active use of the promoted priors.
EOF
