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
load_model_dependent_config step18

MODEL="${POSTIBI_MODEL}"
CONFIG="${TRAINING_CONFIG}"
DATASET="${IBI_DATASET}"
RB_INFO="${IBI_RB_INFO}"
if [[ -z "${IBI_PRIORS:-}" ]]; then
  for candidate in ${POSTIBI_PRIOR_CANDIDATES}; do [[ -f "${candidate}" ]] && { IBI_PRIORS="${candidate}"; break; }; done
fi
[[ -n "${IBI_PRIORS:-}" ]] || { echo "[ERROR] No configured post-IBI prior candidate exists." >&2; exit 1; }
PRIORS="${IBI_PRIORS}"
RESIDUAL_MANIFEST="${IBI_RESIDUAL_PROVENANCE}"
IBI_CONFIG="${IBI_SETTINGS}"
OUTDIR="${POSTIBI_RUNTIME_OUTDIR}"
OVERWRITE="${OVERWRITE:-0}"
DT="${POSTIBI_DT}"
EQ_SD_STEPS="${POSTIBI_EQ_SD_STEPS}"
EQ_CLASSICAL_STEPS="${POSTIBI_EQ_CLASSICAL_STEPS}"
EQ_ML_CAPPED_STEPS="${POSTIBI_EQ_ML_CAPPED_STEPS}"
EQ_ML_UNCAPPED_STEPS="${POSTIBI_EQ_ML_UNCAPPED_STEPS}"
EQ_CHUNK="${POSTIBI_EQ_CHUNK}"
NVT_STEPS="${POSTIBI_NVT_STEPS}"
NVT_LOG_INTERVAL="${POSTIBI_NVT_LOG_INTERVAL}"
NVT_SAMPLE_START="${POSTIBI_NVT_SAMPLE_START}"
VELOCITY_SEED="${POSTIBI_VELOCITY_SEED}"
THERMOSTAT_SEED="${POSTIBI_THERMOSTAT_SEED}"
KT="${POSTIBI_KT}"


for path in "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" "${PRIORS}" "${RESIDUAL_MANIFEST}" "${IBI_CONFIG}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required post-IBI runtime artifact: ${path}" >&2
        exit 1
    fi
done

if [[ -e "${OUTDIR}" ]]; then
    if [[ "${OVERWRITE}" != "1" ]]; then
        echo "[ERROR] Output directory already exists: ${OUTDIR}" >&2
        echo "Set OVERWRITE=1 to replace this validation run." >&2
        exit 1
    fi
    rm -rf "${OUTDIR}"
fi
mkdir -p "${OUTDIR}"

PREFLIGHT_JSON="${OUTDIR}/runtime_preflight.json"
CHECKPOINT="${OUTDIR}/equilibrated_postibi.npz"
EQ_LOG="${OUTDIR}/equilibrate.log"
ENERGY_CSV="${OUTDIR}/nvt_energy.csv"
NVT_LOG="${OUTDIR}/nvt_run.log"
SAMPLE_NPZ="${OUTDIR}/nvt_sample.npz"
NVT_REPORT="${OUTDIR}/nvt_smoke_report.json"
STRUCTURE_REPORT="${OUTDIR}/runtime_structure_report.json"

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/runtime_preflight.py" \
    --model "${MODEL}" \
    --config "${CONFIG}" \
    --dataset "${DATASET}" \
    --priors "${PRIORS}" \
    --rb-info "${RB_INFO}" \
    --residual-manifest "${RESIDUAL_MANIFEST}" \
    --output "${PREFLIGHT_JSON}"

# Generate a checkpoint with the exact post-IBI Hamiltonian.  This also stores
# hashes for model/config/dataset/priors/rb_info, which run_cg_md validates on load.
echo "[INFO] Post-IBI equilibration -> ${CHECKPOINT}"
"${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/equilibrate.py" \
    --model "${MODEL}" \
    --config "${CONFIG}" \
    --priors "${PRIORS}" \
    --rb_info "${RB_INFO}" \
    --dataset "${DATASET}" \
    --out_checkpoint "${CHECKPOINT}" \
    --device "${DEVICE}" \
    --neighbor_search "${NEIGHBOR_SEARCH}" \
    --dt "${DT}" \
    --kT "${KT}" \
    --velocity_seed "${VELOCITY_SEED}" \
    --steps_sd "${EQ_SD_STEPS}" \
    --steps_md "${EQ_CLASSICAL_STEPS}" \
    --steps_ml_capped "${EQ_ML_CAPPED_STEPS}" \
    --steps_ml_uncapped "${EQ_ML_UNCAPPED_STEPS}" \
    --warmup_chunk "${EQ_CHUNK}" \
    2>&1 | tee "${EQ_LOG}"

if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "[ERROR] Equilibration did not produce checkpoint: ${CHECKPOINT}" >&2
    tail -80 "${EQ_LOG}" >&2 || true
    exit 1
fi

echo "[INFO] Post-IBI NVT smoke: steps=${NVT_STEPS}, dt=${DT} ps, sample_start=${NVT_SAMPLE_START}"
"${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py" \
    --model "${MODEL}" \
    --config "${CONFIG}" \
    --priors "${PRIORS}" \
    --rb_info "${RB_INFO}" \
    --dataset "${DATASET}" \
    --checkpoint "${CHECKPOINT}" \
    --steps "${NVT_STEPS}" \
    --dt "${DT}" \
    --kT "${KT}" \
    --thermostat_seed "${THERMOSTAT_SEED}" \
    --device "${DEVICE}" \
    --neighbor_search "${NEIGHBOR_SEARCH}" \
    --energy_file "${ENERGY_CSV}" \
    --no_vtf \
    --sample_npz "${SAMPLE_NPZ}" \
    --sample_start_step "${NVT_SAMPLE_START}" \
    --log_interval "${NVT_LOG_INTERVAL}" \
    2>&1 | tee "${NVT_LOG}"

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/analyze_nvt_smoke.py" \
    --energy-csv "${ENERGY_CSV}" \
    --expected-steps "${NVT_STEPS}" \
    --output "${NVT_REPORT}"

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/ibi/validate_runtime_structure.py" \
    --dataset "${DATASET}" \
    --priors "${PRIORS}" \
    --sample-npz "${SAMPLE_NPZ}" \
    --ibi-config "${IBI_CONFIG}" \
    --output "${STRUCTURE_REPORT}"

cat <<EOF
[POST-IBI RUNTIME VALIDATION COMPLETE]
preflight : ${PREFLIGHT_JSON}
checkpoint: ${CHECKPOINT}
nvt report: ${NVT_REPORT}
structure : ${STRUCTURE_REPORT}
energy csv: ${ENERGY_CSV}
logs      : ${EQ_LOG}, ${NVT_LOG}
[PASS] Provenance-consistent IBI+PaiNN Hamiltonian completed the NVT smoke validation.
[NOTE] This NVT smoke is not an NVE conservation certification; use the dedicated NVE gate afterwards.
EOF

write_model_dependent_provenance "${OUTDIR}/model_config_provenance.json"
