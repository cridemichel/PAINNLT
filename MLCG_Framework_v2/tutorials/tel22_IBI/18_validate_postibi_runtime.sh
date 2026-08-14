#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYRESSO="${PYRESSO:-${DEFAULT_PYPRESSO}}"

cd "${SCRIPT_DIR}"

MODEL="${IBI_MODEL:-tel22_model_ibi.pt}"
CONFIG="${TRAINING_CONFIG:-tel22_training_config.json}"
DATASET="${IBI_DATASET:-tel22_dataset_ibi_residual.bin}"
RB_INFO="${IBI_RB_INFO:-rigid_bodies_info_ibi.json}"
if [[ -n "${IBI_PRIORS:-}" ]]; then
    PRIORS="${IBI_PRIORS}"
elif [[ -f "ibi_conservative/cg_priors.json" ]]; then
    PRIORS="ibi_conservative/cg_priors.json"
elif [[ -f "ibi_run_16ps_continue/best/cg_priors.json" ]]; then
    PRIORS="ibi_run_16ps_continue/best/cg_priors.json"
else
    PRIORS="ibi_run_16ps/best/cg_priors.json"
fi
RESIDUAL_MANIFEST="${IBI_RESIDUAL_PROVENANCE:-ibi_residual_build_manifest.json}"
IBI_CONFIG="${IBI_CONFIG:-ibi_settings.json}"
OUTDIR="${POSTIBI_RUNTIME_OUTDIR:-postibi_runtime_validation}"
OVERWRITE="${OVERWRITE:-0}"
DEVICE="${DEVICE:-auto}"
NEIGHBOR_SEARCH="${NEIGHBOR_SEARCH:-link-cell}"

DT="${POSTIBI_DT:-0.0005}"
EQ_SD_STEPS="${POSTIBI_EQ_SD_STEPS:-1000}"
EQ_CLASSICAL_STEPS="${POSTIBI_EQ_CLASSICAL_STEPS:-1000}"
EQ_ML_CAPPED_STEPS="${POSTIBI_EQ_ML_CAPPED_STEPS:-1000}"
EQ_ML_UNCAPPED_STEPS="${POSTIBI_EQ_ML_UNCAPPED_STEPS:-2000}"
EQ_CHUNK="${POSTIBI_EQ_CHUNK:-100}"
NVT_STEPS="${POSTIBI_NVT_STEPS:-4000}"
NVT_LOG_INTERVAL="${POSTIBI_NVT_LOG_INTERVAL:-100}"
NVT_SAMPLE_START="${POSTIBI_NVT_SAMPLE_START:-1000}"
VELOCITY_SEED="${POSTIBI_VELOCITY_SEED:-424242}"
THERMOSTAT_SEED="${POSTIBI_THERMOSTAT_SEED:-171717}"
KT="${POSTIBI_KT:-2.49}"

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
