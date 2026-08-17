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
load_model_dependent_config step35 step36
cd "${TUTORIAL_DIR}"
PYPRESSO="${PYPRESSO:-${ROOT}/espresso/build/pypresso}"

MODE="${1:-}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--run" && "${MODE}" != "--resume" ]]; then
  echo "Usage: $0 --dry-run|--run|--resume" >&2
  exit 2
fi

STEP35="${IBI_DIHEDRAL_STEP35_OUT}"
ITER0="${STEP35}/ibi/iteration_000/cg_priors.json"
ITER1="${STEP35}/ibi/iteration_001/cg_priors.json"
STEP35_CONSERVATIVE="${STEP35}/conservative/cg_priors.json"
STEP35_REPORT="${STEP35}/dihedral_ibi_test_report.json"
STEP35_IBI_REPORT="${STEP35}/ibi/ibi_report.json"
TARGET_DATASET="${IBI_DIHEDRAL_TARGET_DATASET}"
RUNTIME_DATASET="${IBI_DIHEDRAL_RUNTIME_DATASET}"
CONFIG="${IBI_DIHEDRAL_CONFIG}"
RB_INFO="${IBI_DIHEDRAL_RB_INFO}"
MODEL="${IBI_DIHEDRAL_MODEL}"
SETTINGS="${IBI_DIHEDRAL_SETTINGS}"
SOURCE_CHECKPOINT="${IBI_DIHEDRAL_SOURCE_CHECKPOINT}"
OUT="${IBI_DIHEDRAL_UPDATE_LOCALIZATION_OUT}"
REGISTRY="${OUT}/candidate_registry.json"
STRUCTURE_ROOT="${OUT}/short_nvt"
FINAL_REPORT="${OUT}/dihedral_update_localization_report.json"
CONFIG_PROVENANCE="${OUT}/model_config_provenance.json"
NVT_STEPS="${IBI_DIHEDRAL_LOCALIZATION_NVT_STEPS}"
SAMPLE_START="${IBI_DIHEDRAL_LOCALIZATION_SAMPLE_START}"
OVERWRITE="${OVERWRITE:-0}"

if ! [[ "${NVT_STEPS}" =~ ^[1-9][0-9]*$ && "${SAMPLE_START}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] NVT/SAMPLE_START must be non-negative integers" >&2
  exit 2
fi
if (( SAMPLE_START >= NVT_STEPS )); then
  echo "[ERROR] SAMPLE_START must be smaller than NVT_STEPS" >&2
  exit 2
fi

for path in "${ITER0}" "${ITER1}" "${STEP35_CONSERVATIVE}" "${STEP35_REPORT}" "${STEP35_IBI_REPORT}" "${TARGET_DATASET}" \
            "${RUNTIME_DATASET}" "${CONFIG}" "${RB_INFO}" "${MODEL}" "${MODEL}.manifest.json" "${SOURCE_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing required artifact: ${path}" >&2; exit 1; }
done
[[ -x "${PYPRESSO}" ]] || { echo "[ERROR] pypresso not executable: ${PYPRESSO}" >&2; exit 1; }

cat <<EOF
[STEP 36 -- LOCALIZE PERIODIC DIHEDRAL IBI UPDATE]
step35 iteration 0 : ${ITER0}
step35 iteration 1 : ${ITER1}
step35 conservative: ${STEP35_CONSERVATIVE}
short NVT           : ${NVT_STEPS} steps, sample from step ${SAMPLE_START}, dt=${IBI_DIHEDRAL_LOCALIZATION_DT} ps
output              : ${OUT}
[NOTE] Test only. No production prior is modified or promoted.
[NOTE] Sweep scales the observed step-35 update downward; it does not extrapolate to larger alpha.
EOF

if [[ "${MODE}" == "--dry-run" ]]; then
  echo "[PLAN] Candidate sweep is read from the external model-dependent config."
  N_CANDIDATES="$("${PYTHON_BIN}" -c 'import json,sys; print(len(json.loads(sys.argv[1])))' "${IBI_DIHEDRAL_LOCALIZATION_CANDIDATES}")"
  echo "[PLAN] Approximate new MD work: $((N_CANDIDATES * NVT_STEPS)) steps."
  echo "[PLAN] No NVE is run in step 36; candidate NVE screening is a later decision."
  exit 0
fi

if [[ "${MODE}" == "--run" ]]; then
  if [[ -e "${OUT}" ]]; then
    if [[ "${OVERWRITE}" != "1" ]]; then
      echo "[ERROR] Output already exists: ${OUT}" >&2
      echo "[HINT] Remove it or rerun with OVERWRITE=1." >&2
      exit 1
    fi
    rm -rf "${OUT}"
  fi
  mkdir -p "${OUT}"
  write_model_dependent_provenance "${CONFIG_PROVENANCE}"
  "${PYTHON_BIN}" "${ROOT}/ibi/generate_dihedral_update_localization_candidates.py" \
    --iteration0-priors "${ITER0}" --iteration1-priors "${ITER1}" \
    --conservative-priors "${STEP35_CONSERVATIVE}" --target-dataset "${TARGET_DATASET}" \
    --ibi-report "${STEP35_IBI_REPORT}" --output-dir "${OUT}" --kT "${IBI_DIHEDRAL_LOCALIZATION_KT}" \
    --candidate-specs-json "${IBI_DIHEDRAL_LOCALIZATION_CANDIDATES}"
else
  [[ -f "${REGISTRY}" ]] || { echo "[ERROR] Cannot resume; missing registry: ${REGISTRY}" >&2; exit 1; }
  [[ -f "${CONFIG_PROVENANCE}" ]] || write_model_dependent_provenance "${CONFIG_PROVENANCE}"
  echo "[RESUME] Reusing generated candidates and completed short-NVT branches."
fi

CANDIDATES="$("${PYTHON_BIN}" - "${REGISTRY}" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
print(" ".join(c["name"] for c in r["candidates"]))
PY
)"

mkdir -p "${STRUCTURE_ROOT}"
for name in ${CANDIDATES}; do
  cdir="${OUT}/candidates/${name}"
  priors="${cdir}/cg_priors.json"
  run_dir="${STRUCTURE_ROOT}/${name}"
  sample="${run_dir}/trajectory.npz"
  checkpoint="${run_dir}/checkpoint.npz"
  structure="${run_dir}/runtime_structure_report.json"
  mkdir -p "${run_dir}"
  if [[ "${MODE}" == "--resume" && -f "${structure}" ]]; then
    echo "[REUSE] ${name}"
    continue
  fi
  if [[ "${MODE}" == "--resume" && -f "${sample}" ]]; then
    echo "[REUSE SAMPLE] ${name}; rebuilding only structure report"
  else
    echo "[RUN] ${name}"
    "${PYPRESSO}" "${ROOT}/simulation/run_cg_md.py" \
      --model "${MODEL}" --disable_ml \
      --config "${CONFIG}" --priors "${priors}" --rb_info "${RB_INFO}" --dataset "${RUNTIME_DATASET}" \
      --checkpoint "${SOURCE_CHECKPOINT}" --allow_checkpoint_mismatch \
      --dt "${IBI_DIHEDRAL_LOCALIZATION_DT}" --steps "${NVT_STEPS}" --log_interval 10 --sample_start_step "${SAMPLE_START}" \
      --sample_npz "${sample}" --no_vtf --no_log \
      --kT "${IBI_DIHEDRAL_LOCALIZATION_KT}" --thermostat_seed "${IBI_DIHEDRAL_LOCALIZATION_THERMOSTAT_SEED}" --neighbor_search "${IBI_DIHEDRAL_LOCALIZATION_NEIGHBOR_SEARCH}" \
      --device "${IBI_DEVICE}" --ml_precision "${IBI_ML_PRECISION}" --out_checkpoint "${checkpoint}" > "${run_dir}/run.log" 2>&1
  fi
  "${PYTHON_BIN}" "${ROOT}/ibi/validate_runtime_structure.py" \
    --dataset "${TARGET_DATASET}" --priors "${priors}" --sample-npz "${sample}" \
    --ibi-config "${SETTINGS}" --output "${structure}" > "${run_dir}/structure.log" 2>&1
  "${PYTHON_BIN}" - "${name}" "${structure}" <<'PY'
import json, sys
name=sys.argv[1]; r=json.load(open(sys.argv[2]))
print(f"[RESULT] {name}: dihedral mean L1={r['mean_l1_by_kind']['dihedral']:.6f} max={r['max_l1']:.6f}")
PY
done

"${PYTHON_BIN}" "${ROOT}/simulation/finalize_dihedral_update_localization.py" \
  --candidate-registry "${REGISTRY}" --step35-report "${STEP35_REPORT}" \
  --structure-root "${STRUCTURE_ROOT}" --output "${FINAL_REPORT}" \
  --representation-dominance-factor "${IBI_DIHEDRAL_REPRESENTATION_DOMINANCE_FACTOR}" \
  --gain-dominance-factor "${IBI_DIHEDRAL_GAIN_DOMINANCE_FACTOR}" --gain-floor "${IBI_DIHEDRAL_GAIN_FLOOR}" \
  --model-config-provenance "${CONFIG_PROVENANCE}"

cat <<EOF
[DONE] report: ${FINAL_REPORT}
[NOTE] Step 36 is localization-only. Do not promote any candidate from this report.
EOF
