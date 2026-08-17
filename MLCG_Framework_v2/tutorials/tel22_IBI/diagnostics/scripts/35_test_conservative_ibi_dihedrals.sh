#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${HERE}/model_config.sh" || -f "${HERE}/cg_priors.json" ]]; then
  TUTORIAL_DIR="${HERE}"
else
  TUTORIAL_DIR="$(cd "${HERE}/../.." && pwd)"
fi
ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
source "${TUTORIAL_DIR}/model_config.sh"
load_model_dependent_config step35
cd "${TUTORIAL_DIR}"
PYPRESSO="${PYPRESSO:-${ROOT}/espresso/build/pypresso}"

MODE="${1:-}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--run" && "${MODE}" != "--resume-nve" ]]; then
  echo "Usage: $0 --dry-run|--run|--resume-nve" >&2
  exit 2
fi

BASE_PRIORS="${IBI_DIHEDRAL_BASE_PRIORS}"
TARGET_DATASET="${IBI_DIHEDRAL_TARGET_DATASET}"
RUNTIME_DATASET="${IBI_DIHEDRAL_RUNTIME_DATASET}"
CONFIG="${IBI_DIHEDRAL_CONFIG}"
RB_INFO="${IBI_DIHEDRAL_RB_INFO}"
MODEL="${IBI_DIHEDRAL_MODEL}"
SETTINGS="${IBI_DIHEDRAL_SETTINGS}"
SOURCE_CHECKPOINT="${IBI_DIHEDRAL_SOURCE_CHECKPOINT}"
BASELINE_CERT="${IBI_DIHEDRAL_BASELINE_CERT}"
OUT="${IBI_DIHEDRAL_TEST_OUT}"
IBI_ITERATIONS="${IBI_DIHEDRAL_ITERATIONS}"
OVERWRITE="${OVERWRITE:-0}"
if ! [[ "${IBI_ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] IBI_DIHEDRAL_ITERATIONS must be a positive integer" >&2
  exit 2
fi

SEED_DIR="${OUT}/seed"
SEED_PRIORS="${SEED_DIR}/cg_priors_dihedral_ibi_seed.json"
SEED_REPORT="${SEED_DIR}/seed_report.json"
IBI_OUT="${OUT}/ibi"
IBI_FINAL="${IBI_OUT}/cg_priors_final.json"
IBI_REPORT="${IBI_OUT}/ibi_report.json"
CONSERVATIVE_DIR="${OUT}/conservative"
CONSERVATIVE_PRIORS="${CONSERVATIVE_DIR}/cg_priors.json"
CONVERSION_REPORT="${CONSERVATIVE_DIR}/conversion_report.json"
VALIDATION_REPORT="${CONSERVATIVE_DIR}/validation_report.json"
PARITY_REPORT="${CONSERVATIVE_DIR}/runtime_parity_report.json"
NVT_DIR="${OUT}/nvt"
NVT_CHECKPOINT="${NVT_DIR}/equilibrated_conservative_dihedral_test.npz"
NVT_SAMPLE="${NVT_DIR}/trajectory.npz"
NVT_ENERGY="${NVT_DIR}/energy.csv"
STRUCTURE_REPORT="${OUT}/runtime_structure_report.json"
NVE_DIR="${OUT}/nve"
NVE_REPORT="${NVE_DIR}/nve_diagnostic_report.json"
FINAL_REPORT="${OUT}/dihedral_ibi_test_report.json"
CONFIG_PROVENANCE="${OUT}/model_config_provenance.json"

for path in "${BASE_PRIORS}" "${TARGET_DATASET}" "${RUNTIME_DATASET}" "${CONFIG}" "${RB_INFO}" \
            "${MODEL}" "${MODEL}.manifest.json" "${SETTINGS}" "${SOURCE_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing required artifact: ${path}" >&2; exit 1; }
done
[[ -x "${PYPRESSO}" ]] || { echo "[ERROR] pypresso not executable: ${PYPRESSO}" >&2; exit 1; }

"${PYPRESSO}" - <<'PY'
import espressomd
for name in ("ConservativeSplineDistance", "ConservativeSplineAngle", "ConservativeSplineDihedral"):
    if not hasattr(espressomd.interactions, name):
        raise RuntimeError(f"Missing espressomd.interactions.{name}")
print("[PASS] ESPResSo conservative distance/angle/dihedral bindings are available.")
PY

cat <<EOF
[STEP 35 -- TEST PERIODIC CONSERVATIVE IBI DIHEDRALS]
base production priors : ${BASE_PRIORS}
target geometry dataset: ${TARGET_DATASET}
IBI iterations          : ${IBI_ITERATIONS}
IBI test settings       : ${SETTINGS}
conservative candidate  : ${CONSERVATIVE_PRIORS}
short candidate NVT     : ${IBI_DIHEDRAL_NVT_STEPS} steps at dt=${IBI_DIHEDRAL_NVT_DT} ps
NVE diagnostic          : dt=${IBI_DIHEDRAL_NVE_DTS} ps, ${IBI_DIHEDRAL_NVE_DURATION_PS} ps each
output                  : ${OUT}
[NOTE] Test only: production priors are never modified or promoted.
[NOTE] The final post-update IBI torsional prior is sampled again after conservative conversion.
EOF

read -r -a NVE_DTS <<< "${IBI_DIHEDRAL_NVE_DTS}"

run_nve_diagnostic() {
  local reuse_flag="${1:-}"
  local extra=()
  if [[ "${reuse_flag}" == "--reuse-existing" ]]; then
    extra+=(--reuse-existing)
  fi
  "${PYTHON_BIN}" "${ROOT}/simulation/certify_nve.py" \
    --pypresso "${PYPRESSO}" --model "${MODEL}" --disable-ml \
    --config "${CONFIG}" --priors "${CONSERVATIVE_PRIORS}" --rb-info "${RB_INFO}" --dataset "${RUNTIME_DATASET}" \
    --checkpoint "${NVT_CHECKPOINT}" --dts "${NVE_DTS[@]}" --duration-ps "${IBI_DIHEDRAL_NVE_DURATION_PS}" \
    --device "${IBI_DIHEDRAL_DEVICE}" --ml-precision "${IBI_DIHEDRAL_ML_PRECISION}" --neighbor-search "${IBI_DIHEDRAL_NEIGHBOR_SEARCH}" \
    --output-dir "${NVE_DIR}" --slope-min "${IBI_DIHEDRAL_NVE_P_MIN}" --slope-max "${IBI_DIHEDRAL_NVE_P_MAX}" --min-r2 "${IBI_DIHEDRAL_NVE_R2_MIN}" --max-relative-drift "${IBI_DIHEDRAL_NVE_MAX_RELATIVE_DRIFT}" \
    --diagnostic-only --diagnostic-fine-max-dt "${IBI_DIHEDRAL_NVE_FINE_MAX_DT}" --diagnostic-coarse-min-dt "${IBI_DIHEDRAL_NVE_COARSE_MIN_DT}" \
    --provenance-artifact "dihedral_seed=${SEED_REPORT}" \
    --provenance-artifact "dihedral_ibi=${IBI_REPORT}" \
    --provenance-artifact "dihedral_conversion=${CONVERSION_REPORT}" \
    --provenance-artifact "dihedral_runtime_parity=${PARITY_REPORT}" \
    --provenance-artifact "model_config=${CONFIG_PROVENANCE}" \
    "${extra[@]}"
}

finalize_test() {
  local final_args=(
    "${ROOT}/simulation/finalize_dihedral_ibi_test.py"
    --seed-report "${SEED_REPORT}"
    --ibi-report "${IBI_REPORT}"
    --conversion-report "${CONVERSION_REPORT}"
    --validation-report "${VALIDATION_REPORT}"
    --parity-report "${PARITY_REPORT}"
    --structure-report "${STRUCTURE_REPORT}"
    --nve-report "${NVE_REPORT}"
    --candidate-priors "${CONSERVATIVE_PRIORS}"
    --nve-p-min "${IBI_DIHEDRAL_NVE_P_MIN}" --nve-p-max "${IBI_DIHEDRAL_NVE_P_MAX}"
    --nve-r2-min "${IBI_DIHEDRAL_NVE_R2_MIN}" --nve-c2-spread-max "${IBI_DIHEDRAL_NVE_C2_SPREAD_MAX}"
    --nve-max-relative-drift "${IBI_DIHEDRAL_NVE_MAX_RELATIVE_DRIFT}" --nve-required-max-dt "${IBI_DIHEDRAL_NVE_REQUIRED_MAX_DT}"
    --model-config-provenance "${CONFIG_PROVENANCE}" --output "${FINAL_REPORT}"
  )
  if [[ -f "${BASELINE_CERT}" ]]; then
    final_args+=(--baseline-certification "${BASELINE_CERT}")
  fi
  "${PYTHON_BIN}" "${final_args[@]}"
}

if [[ "${MODE}" == "--dry-run" ]]; then
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' EXIT
  "${PYTHON_BIN}" "${ROOT}/ibi/prepare_dihedral_ibi_test_seed.py" \
    --base-priors "${BASE_PRIORS}" --output "${tmpdir}/seed.json" --report "${tmpdir}/seed_report.json" --grouping-strategy "${IBI_DIHEDRAL_GROUPING_STRATEGY}"
  "${PYTHON_BIN}" - <<PY
import json
r=json.load(open("${tmpdir}/seed_report.json"))
print(f"[PLAN] derived periodic dihedrals: {r['dihedral_occurrences']} occurrences in {r['unique_groups']} pooled groups")
PY
  echo "[PLAN] New MD work follows the configured IBI/NVT/NVE protocols; no model-dependent step count is embedded in this wrapper."
  echo "[PLAN] No production artifact will be modified."
  exit 0
fi

if [[ "${MODE}" == "--resume-nve" ]]; then
  for path in "${SEED_REPORT}" "${IBI_REPORT}" "${CONVERSION_REPORT}" "${VALIDATION_REPORT}" \
              "${PARITY_REPORT}" "${STRUCTURE_REPORT}" "${CONSERVATIVE_PRIORS}" "${NVT_CHECKPOINT}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Cannot resume; missing artifact: ${path}" >&2; exit 1; }
  done
  [[ -f "${CONFIG_PROVENANCE}" ]] || write_model_dependent_provenance "${CONFIG_PROVENANCE}"
  echo "[RESUME] Reusing completed NVE branches and running only missing dt values."
  echo "[RESUME] Missing configured NVE dt values will be run; completed branches are reused."
  run_nve_diagnostic --reuse-existing
  finalize_test
  echo "[DONE] resumed test report: ${FINAL_REPORT}"
  exit 0
fi

if [[ -e "${OUT}" ]]; then
  if [[ "${OVERWRITE}" != "1" ]]; then
    echo "[ERROR] Output already exists: ${OUT}" >&2
    echo "[HINT] Remove it or rerun with OVERWRITE=1." >&2
    exit 1
  fi
  rm -rf "${OUT}"
fi
mkdir -p "${SEED_DIR}" "${NVT_DIR}"
write_model_dependent_provenance "${CONFIG_PROVENANCE}"

"${PYTHON_BIN}" "${ROOT}/ibi/prepare_dihedral_ibi_test_seed.py" \
  --base-priors "${BASE_PRIORS}" --output "${SEED_PRIORS}" --report "${SEED_REPORT}" --grouping-strategy "${IBI_DIHEDRAL_GROUPING_STRATEGY}"

"${PYTHON_BIN}" "${ROOT}/ibi/run_ibi_loop.py" \
  --dataset "${TARGET_DATASET}" --priors "${SEED_PRIORS}" \
  --config "${CONFIG}" --rb_info "${RB_INFO}" --iterations "${IBI_ITERATIONS}" \
  --outdir "${IBI_OUT}" --ibi-config "${SETTINGS}" --pypresso "${PYPRESSO}" \
  --neighbor_search "${IBI_DIHEDRAL_NEIGHBOR_SEARCH}" --velocity_seed "${IBI_DIHEDRAL_IBI_VELOCITY_SEED}" --thermostat_seed "${IBI_DIHEDRAL_IBI_THERMOSTAT_SEED}"

"${PYTHON_BIN}" "${ROOT}/ibi/convert_to_conservative_spline.py" \
  --priors "${IBI_FINAL}" --output-dir "${CONSERVATIVE_DIR}"
"${PYTHON_BIN}" "${ROOT}/ibi/validate_conservative_spline.py" \
  --conversion-report "${CONVERSION_REPORT}"
"${PYPRESSO}" "${ROOT}/simulation/diagnose_conservative_spline_parity.py" \
  --priors "${CONSERVATIVE_PRIORS}" --report "${PARITY_REPORT}"

# Re-equilibrate the exact conservative candidate.  The source checkpoint is
# production-smoothed IBI without torsions, so its prior hash mismatch is
# expected and is explicitly limited to this test branch.
"${PYPRESSO}" "${ROOT}/simulation/run_cg_md.py" \
  --model "${MODEL}" --disable_ml \
  --config "${CONFIG}" --priors "${CONSERVATIVE_PRIORS}" --rb_info "${RB_INFO}" --dataset "${RUNTIME_DATASET}" \
  --checkpoint "${SOURCE_CHECKPOINT}" --allow_checkpoint_mismatch \
  --dt "${IBI_DIHEDRAL_NVT_DT}" --steps "${IBI_DIHEDRAL_NVT_STEPS}" --log_interval "${IBI_DIHEDRAL_NVT_LOG_INTERVAL}" --sample_start_step "${IBI_DIHEDRAL_NVT_SAMPLE_START}" \
  --sample_npz "${NVT_SAMPLE}" --energy_file "${NVT_ENERGY}" --no_vtf \
  --kT "${IBI_DIHEDRAL_NVT_KT}" --thermostat_seed "${IBI_DIHEDRAL_NVT_THERMOSTAT_SEED}" --neighbor_search "${IBI_DIHEDRAL_NEIGHBOR_SEARCH}" \
  --device "${IBI_DIHEDRAL_DEVICE}" --ml_precision "${IBI_DIHEDRAL_ML_PRECISION}" --out_checkpoint "${NVT_CHECKPOINT}" > "${NVT_DIR}/run.log" 2>&1

echo "[PASS] Conservative torsional candidate completed the short NVT branch."

"${PYTHON_BIN}" "${ROOT}/ibi/validate_runtime_structure.py" \
  --dataset "${TARGET_DATASET}" --priors "${CONSERVATIVE_PRIORS}" \
  --sample-npz "${NVT_SAMPLE}" --ibi-config "${SETTINGS}" --output "${STRUCTURE_REPORT}"

# Diagnostic-only is intentional: if a raw/short-IBI torsional prior is too
# stiff at coarse dt, that is a scientific result of this test, not a reason to
# lose the completed trajectories/report.
run_nve_diagnostic
finalize_test

cat <<EOF
[DONE] test report          : ${FINAL_REPORT}
[DONE] conservative candidate: ${CONSERVATIVE_PRIORS}
[NOTE] Nothing was promoted. Do not use this candidate as production priors based on step 35 alone.
EOF
