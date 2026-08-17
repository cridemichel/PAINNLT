#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYPRESSO="${PYPRESSO:-${ROOT}/espresso/build/pypresso}"

MODE="${1:-}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--run" && "${MODE}" != "--resume-nve" ]]; then
  echo "Usage: $0 --dry-run|--run|--resume-nve" >&2
  exit 2
fi

BASE_PRIORS="${IBI_DIHEDRAL_BASE_PRIORS:-${HERE}/ibi_conservative/cg_priors.json}"
TARGET_DATASET="${IBI_DIHEDRAL_TARGET_DATASET:-${HERE}/tel22_dataset.bin}"
RUNTIME_DATASET="${IBI_DIHEDRAL_RUNTIME_DATASET:-${HERE}/tel22_dataset_ibi_residual.bin}"
CONFIG="${IBI_DIHEDRAL_CONFIG:-${HERE}/tel22_training_config.json}"
RB_INFO="${IBI_DIHEDRAL_RB_INFO:-${HERE}/rigid_bodies_info_ibi.json}"
MODEL="${IBI_DIHEDRAL_MODEL:-${HERE}/tel22_model_ibi_conservative.pt}"
SETTINGS="${IBI_DIHEDRAL_SETTINGS:-${HERE}/ibi_dihedral_test_settings.json}"
SOURCE_CHECKPOINT="${IBI_DIHEDRAL_SOURCE_CHECKPOINT:-${HERE}/ibi_promoted_final_certification/nvt/equilibrated_promoted_ibi_only.npz}"
BASELINE_CERT="${IBI_DIHEDRAL_BASELINE_CERT:-${HERE}/ibi_promoted_final_certification/promoted_ibi_final_certification_report.json}"
OUT="${IBI_DIHEDRAL_TEST_OUT:-${HERE}/ibi_dihedral_candidate_test}"
IBI_ITERATIONS="${IBI_DIHEDRAL_ITERATIONS:-1}"
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
short candidate NVT     : 1000 steps at dt=0.0005 ps
NVE diagnostic          : dt=0.001 0.0015 0.002 0.003 0.004 0.005 ps, 0.5 ps each
output                  : ${OUT}
[NOTE] Test only: production priors are never modified or promoted.
[NOTE] The final post-update IBI torsional prior is sampled again after conservative conversion.
EOF

run_nve_diagnostic() {
  local reuse_flag="${1:-}"
  local extra=()
  if [[ "${reuse_flag}" == "--reuse-existing" ]]; then
    extra+=(--reuse-existing)
  fi
  "${PYTHON_BIN}" "${ROOT}/simulation/certify_nve.py" \
    --pypresso "${PYPRESSO}" --model "${MODEL}" --disable-ml \
    --config "${CONFIG}" --priors "${CONSERVATIVE_PRIORS}" --rb-info "${RB_INFO}" --dataset "${RUNTIME_DATASET}" \
    --checkpoint "${NVT_CHECKPOINT}" --dts 0.001 0.0015 0.002 0.003 0.004 0.005 --duration-ps 0.5 \
    --device cpu --ml-precision float32 --neighbor-search link-cell \
    --output-dir "${NVE_DIR}" --slope-min 1.8 --slope-max 2.2 --min-r2 0.95 --max-relative-drift 2e-5 \
    --diagnostic-only --diagnostic-fine-max-dt 0.002 --diagnostic-coarse-min-dt 0.003 \
    --provenance-artifact "dihedral_seed=${SEED_REPORT}" \
    --provenance-artifact "dihedral_ibi=${IBI_REPORT}" \
    --provenance-artifact "dihedral_conversion=${CONVERSION_REPORT}" \
    --provenance-artifact "dihedral_runtime_parity=${PARITY_REPORT}" \
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
    --output "${FINAL_REPORT}"
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
    --base-priors "${BASE_PRIORS}" --output "${tmpdir}/seed.json" --report "${tmpdir}/seed_report.json"
  "${PYTHON_BIN}" - <<PY
import json
r=json.load(open("${tmpdir}/seed_report.json"))
print(f"[PLAN] derived periodic dihedrals: {r['dihedral_occurrences']} occurrences in {r['unique_groups']} pooled groups")
PY
  echo "[PLAN] New MD work: $((IBI_ITERATIONS * 2500 + 1000 + 1475)) steps approximately."
  echo "[PLAN] No production artifact will be modified."
  exit 0
fi

if [[ "${MODE}" == "--resume-nve" ]]; then
  for path in "${SEED_REPORT}" "${IBI_REPORT}" "${CONVERSION_REPORT}" "${VALIDATION_REPORT}" \
              "${PARITY_REPORT}" "${STRUCTURE_REPORT}" "${CONSERVATIVE_PRIORS}" "${NVT_CHECKPOINT}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Cannot resume; missing artifact: ${path}" >&2; exit 1; }
  done
  echo "[RESUME] Reusing completed NVE branches and running only missing dt values."
  echo "[RESUME] Added fine-regime point: dt=0.0015 ps (about 333 integration steps)."
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

"${PYTHON_BIN}" "${ROOT}/ibi/prepare_dihedral_ibi_test_seed.py" \
  --base-priors "${BASE_PRIORS}" --output "${SEED_PRIORS}" --report "${SEED_REPORT}"

"${PYTHON_BIN}" "${ROOT}/ibi/run_ibi_loop.py" \
  --dataset "${TARGET_DATASET}" --priors "${SEED_PRIORS}" \
  --config "${CONFIG}" --rb_info "${RB_INFO}" --iterations "${IBI_ITERATIONS}" \
  --outdir "${IBI_OUT}" --ibi-config "${SETTINGS}" --pypresso "${PYPRESSO}" \
  --neighbor_search link-cell --velocity_seed 350001 --thermostat_seed 350101

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
  --dt 0.0005 --steps 1000 --log_interval 10 --sample_start_step 200 \
  --sample_npz "${NVT_SAMPLE}" --energy_file "${NVT_ENERGY}" --no_vtf \
  --kT 2.49 --thermostat_seed 350201 --neighbor_search link-cell \
  --device cpu --ml_precision float32 --out_checkpoint "${NVT_CHECKPOINT}" > "${NVT_DIR}/run.log" 2>&1

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
