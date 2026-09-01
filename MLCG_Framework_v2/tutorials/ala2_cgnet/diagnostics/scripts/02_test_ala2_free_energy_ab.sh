#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALA2_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${ALA2_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYRESSO="${PYRESSO:-${FRAMEWORK_ROOT}/espresso/build/pypresso}"
TRAINING_RUN_DIR="${ALA2_TRAINING_RUN_DIR:-${ALA2_DIR}/diagnostics/smoke/cgnet_harmonic_50ep}"
RUN_DIR="${ALA2_AB_RUN_DIR:-${ALA2_DIR}/diagnostics/smoke/fes_ab_4x250k}"
DEVICE="${DEVICE:-auto}"

REPLICAS="${ALA2_AB_REPLICAS:-4}"
EQUIL_STEPS="${ALA2_AB_EQUIL_STEPS:-25000}"
PRODUCTION_STEPS="${ALA2_AB_PRODUCTION_STEPS:-250000}"
BURNIN_STEPS="${ALA2_AB_BURNIN_STEPS:-50000}"
SAMPLE_INTERVAL="${ALA2_AB_SAMPLE_INTERVAL:-100}"
DT_PS="${ALA2_AB_DT_PS:-0.002}"
KT_KJ_MOL="${ALA2_AB_KT_KJ_MOL:-2.4943387854}"
CGNET_SAMPLES="${ALA2_CGNET_SAMPLES:-}"
CGNET_UNITS="${ALA2_CGNET_UNITS:-angstrom}"

MODEL="${TRAINING_RUN_DIR}/ala2_model.pt"
CONFIG="${TRAINING_RUN_DIR}/ala2_training_config_50ep.json"
DATASET="${TRAINING_RUN_DIR}/ala2_dataset.bin"
PRIORS="${TRAINING_RUN_DIR}/ala2_priors.json"
REFERENCE="${TRAINING_RUN_DIR}/ala2_reference.npz"
TRAINING_REPORT="${TRAINING_RUN_DIR}/ala2_benchmark_report.json"

for path in "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${PRIORS}" "${REFERENCE}"; do
    if [[ ! -f "${path}" ]]; then
        printf '[ERROR] Missing completed Ala2 training artifact: %s\n' "${path}" >&2
        exit 2
    fi
done
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    printf '[ERROR] Missing Python executable: %s\n' "${PYTHON_BIN}" >&2
    exit 2
fi
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
if [[ ! -x "${PYRESSO}" ]]; then
    printf '[ERROR] Missing executable: %s\n' "${PYRESSO}" >&2
    exit 2
fi
if [[ -n "${CGNET_SAMPLES}" && ! -f "${CGNET_SAMPLES}" ]]; then
    printf '[ERROR] Missing optional CGnet trajectory: %s\n' "${CGNET_SAMPLES}" >&2
    exit 2
fi
if (( REPLICAS < 2 || EQUIL_STEPS < 0 || PRODUCTION_STEPS <= 0 || BURNIN_STEPS < 0 || BURNIN_STEPS >= PRODUCTION_STEPS || SAMPLE_INTERVAL <= 0 )); then
    printf '[ERROR] Invalid replica/step configuration.\n' >&2
    exit 2
fi
if (( BURNIN_STEPS % SAMPLE_INTERVAL != 0 )); then
    printf '[ERROR] ALA2_AB_BURNIN_STEPS must be divisible by ALA2_AB_SAMPLE_INTERVAL.\n' >&2
    exit 2
fi
if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    printf '[ERROR] A/B directory is not empty: %s\n' "${RUN_DIR}" >&2
    printf '        Select a fresh ALA2_AB_RUN_DIR; existing evidence is never overwritten.\n' >&2
    exit 2
fi

mkdir -p "${RUN_DIR}/runtime" "${RUN_DIR}/replicas"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_ala2_runtime.py" \
    --dataset "${DATASET}" \
    --priors "${PRIORS}" \
    --reference "${REFERENCE}" \
    --output-dir "${RUN_DIR}/runtime" \
    --replicas "${REPLICAS}" \
    --report "${RUN_DIR}/ala2_runtime_preparation_report.json" \
    2>&1 | tee "${RUN_DIR}/runtime_preparation_stdout.log"

RUNTIME_PRIORS="${RUN_DIR}/runtime/ala2_runtime_priors.json"
RB_INFO="${RUN_DIR}/runtime/ala2_rigid_bodies_info.json"
prior_samples=()
ml_samples=()

for ((replica=0; replica<REPLICAS; replica++)); do
    label="$(printf '%02d' "${replica}")"
    replica_dir="${RUN_DIR}/replicas/replica_${label}"
    replica_dataset="${RUN_DIR}/runtime/replica_${label}_dataset.bin"
    common_checkpoint="${replica_dir}/common_equilibrated.npz"
    mkdir -p "${replica_dir}"

    printf '[INFO] Replica %s/%s: common prior-only equilibration (%s steps).\n' "$((replica + 1))" "${REPLICAS}" "${EQUIL_STEPS}"
    (
        cd "${replica_dir}"
        "${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py" \
            --model "${MODEL}" \
            --disable_ml \
            --config "${CONFIG}" \
            --priors "${RUNTIME_PRIORS}" \
            --rb_info "${RB_INFO}" \
            --dataset "${replica_dataset}" \
            --init_kT "${KT_KJ_MOL}" \
            --kT "${KT_KJ_MOL}" \
            --velocity_seed "$((314159 + replica))" \
            --thermostat_seed "$((4100 + replica))" \
            --steps "${EQUIL_STEPS}" \
            --dt "${DT_PS}" \
            --device "${DEVICE}" \
            --neighbor_search nsquare \
            --log_interval 1000 \
            --no_log \
            --out_checkpoint "${common_checkpoint}" \
            2>&1 | tee equilibration_stdout.log
    )

    prior_sample="${replica_dir}/prior_only_samples.npz"
    ml_sample="${replica_dir}/prior_plus_painn_samples.npz"
    prior_samples+=("${prior_sample}")
    ml_samples+=("${ml_sample}")

    printf '[INFO] Replica %s/%s: prior-only production (%s steps).\n' "$((replica + 1))" "${REPLICAS}" "${PRODUCTION_STEPS}"
    (
        cd "${replica_dir}"
        "${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py" \
            --model "${MODEL}" \
            --disable_ml \
            --config "${CONFIG}" \
            --priors "${RUNTIME_PRIORS}" \
            --rb_info "${RB_INFO}" \
            --dataset "${replica_dataset}" \
            --checkpoint "${common_checkpoint}" \
            --kT "${KT_KJ_MOL}" \
            --thermostat_seed "$((4200 + replica))" \
            --steps "${PRODUCTION_STEPS}" \
            --sample_start_step "${BURNIN_STEPS}" \
            --log_interval "${SAMPLE_INTERVAL}" \
            --sample_npz "${prior_sample}" \
            --dt "${DT_PS}" \
            --device "${DEVICE}" \
            --neighbor_search nsquare \
            --no_log \
            2>&1 | tee prior_only_stdout.log
    )

    printf '[INFO] Replica %s/%s: prior+PaiNN production (%s steps).\n' "$((replica + 1))" "${REPLICAS}" "${PRODUCTION_STEPS}"
    (
        cd "${replica_dir}"
        "${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py" \
            --model "${MODEL}" \
            --config "${CONFIG}" \
            --priors "${RUNTIME_PRIORS}" \
            --rb_info "${RB_INFO}" \
            --dataset "${replica_dataset}" \
            --checkpoint "${common_checkpoint}" \
            --kT "${KT_KJ_MOL}" \
            --thermostat_seed "$((4200 + replica))" \
            --steps "${PRODUCTION_STEPS}" \
            --sample_start_step "${BURNIN_STEPS}" \
            --log_interval "${SAMPLE_INTERVAL}" \
            --sample_npz "${ml_sample}" \
            --dt "${DT_PS}" \
            --device "${DEVICE}" \
            --neighbor_search nsquare \
            --no_log \
            2>&1 | tee prior_plus_painn_stdout.log
    )
done

analysis_args=(
    --reference "${REFERENCE}"
    --prior-samples "${prior_samples[@]}"
    --ml-samples "${ml_samples[@]}"
    --report "${RUN_DIR}/ala2_fes_ab_report.json"
    --plot "${RUN_DIR}/ala2_fes_ab.png"
)
if [[ -f "${TRAINING_REPORT}" ]]; then
    analysis_args+=(--training-report "${TRAINING_REPORT}")
fi
if [[ -n "${CGNET_SAMPLES}" ]]; then
    analysis_args+=(--cgnet-samples "${CGNET_SAMPLES}" --cgnet-units "${CGNET_UNITS}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_ala2_fes_ab.py" "${analysis_args[@]}" \
    2>&1 | tee "${RUN_DIR}/analysis_stdout.log"

printf '[PASS] Matched Ala2 FES A/B diagnostic completed in %s\n' "${RUN_DIR}"
printf '[INFO] Send ala2_fes_ab_report.json and ala2_fes_ab.png for interpretation.\n'
