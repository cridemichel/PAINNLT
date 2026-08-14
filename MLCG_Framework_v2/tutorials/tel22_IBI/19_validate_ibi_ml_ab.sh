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
elif [[ -f "ibi_run_16ps_continue/best/cg_priors.json" ]]; then
    PRIORS="ibi_run_16ps_continue/best/cg_priors.json"
else
    PRIORS="ibi_run_16ps/best/cg_priors.json"
fi
RESIDUAL_MANIFEST="${IBI_RESIDUAL_PROVENANCE:-ibi_residual_build_manifest.json}"
IBI_CONFIG="${IBI_CONFIG:-ibi_settings.json}"
CHECKPOINT="${AB_CHECKPOINT:-postibi_runtime_validation/equilibrated_postibi.npz}"
OUTDIR="${AB_OUTDIR:-ibi_ml_ab_validation}"
OVERWRITE="${OVERWRITE:-0}"
AB_RESUME="${AB_RESUME:-0}"
DEVICE="${DEVICE:-auto}"
NEIGHBOR_SEARCH="${NEIGHBOR_SEARCH:-link-cell}"

DT="${AB_DT:-0.0005}"
BURNIN_PS="${AB_BURNIN_PS:-1.0}"
PRODUCTION_PS="${AB_PRODUCTION_PS:-8.0}"
LOG_INTERVAL="${AB_LOG_INTERVAL:-100}"
THERMOSTAT_SEED="${AB_THERMOSTAT_SEED:-272727}"
KT="${AB_KT:-2.49}"

steps_for_ps() {
    "${PYTHON_BIN}" - "$1" "${DT}" <<'PY'
import math
import sys
ps = float(sys.argv[1])
dt = float(sys.argv[2])
if ps < 0.0 or dt <= 0.0:
    raise SystemExit("time must be >=0 and dt must be >0")
raw = ps / dt
steps = int(round(raw))
if not math.isclose(raw, steps, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit(f"Requested time {ps} ps is not an integer number of steps at dt={dt} ps")
print(steps)
PY
}

BURNIN_STEPS="$(steps_for_ps "${BURNIN_PS}")"
PRODUCTION_STEPS="$(steps_for_ps "${PRODUCTION_PS}")"
TOTAL_STEPS="$((BURNIN_STEPS + PRODUCTION_STEPS))"
if (( TOTAL_STEPS <= 0 )); then
    echo "[ERROR] A/B test requires at least one integration step." >&2
    exit 1
fi
if (( BURNIN_STEPS % LOG_INTERVAL != 0 )); then
    echo "[ERROR] Burn-in steps (${BURNIN_STEPS}) must be a multiple of AB_LOG_INTERVAL (${LOG_INTERVAL})." >&2
    exit 1
fi

for path in "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" "${PRIORS}" "${RESIDUAL_MANIFEST}" "${IBI_CONFIG}" "${CHECKPOINT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required A/B runtime artifact: ${path}" >&2
        if [[ "${path}" == "${CHECKPOINT}" ]]; then
            echo "[ERROR] Run ./18_validate_postibi_runtime.sh first or set AB_CHECKPOINT explicitly." >&2
        fi
        exit 1
    fi
done

if [[ "${OVERWRITE}" == "1" && "${AB_RESUME}" == "1" ]]; then
    echo "[ERROR] OVERWRITE=1 and AB_RESUME=1 are mutually exclusive." >&2
    exit 1
fi
if [[ -e "${OUTDIR}" ]]; then
    if [[ "${OVERWRITE}" == "1" ]]; then
        rm -rf "${OUTDIR}"
    elif [[ "${AB_RESUME}" != "1" ]]; then
        echo "[ERROR] Output directory already exists: ${OUTDIR}" >&2
        echo "Set AB_RESUME=1 to reuse verified completed branches, or OVERWRITE=1 to replace the run." >&2
        exit 1
    fi
fi
mkdir -p "${OUTDIR}/ibi_only" "${OUTDIR}/ibi_plus_ml"

PREFLIGHT_JSON="${OUTDIR}/runtime_preflight.json"
"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/runtime_preflight.py" \
    --model "${MODEL}" \
    --config "${CONFIG}" \
    --dataset "${DATASET}" \
    --priors "${PRIORS}" \
    --rb-info "${RB_INFO}" \
    --residual-manifest "${RESIDUAL_MANIFEST}" \
    --output "${PREFLIGHT_JSON}"

cat <<EOF
[IBI/ML MATCHED A/B PLAN]
checkpoint       : ${CHECKPOINT}
dt               : ${DT} ps
branch burn-in   : ${BURNIN_PS} ps (${BURNIN_STEPS} steps)
branch production: ${PRODUCTION_PS} ps (${PRODUCTION_STEPS} steps)
total per branch : ${TOTAL_STEPS} steps
thermostat seed  : ${THERMOSTAT_SEED}
log/sample every : ${LOG_INTERVAL} steps
A                : IBI + WCA + Morse, PaiNN disabled but model provenance retained
B                : IBI + WCA + Morse + PaiNN
EOF

branch_complete() {
    local branch_dir="$1"
    local disable_ml="$2"
    local energy_csv="${branch_dir}/energy.csv"
    local sample_npz="${branch_dir}/sample.npz"
    local run_log="${branch_dir}/run.log"
    local nvt_report="${branch_dir}/nvt_smoke_report.json"
    local structure_report="${branch_dir}/runtime_structure_report.json"

    for path in "${energy_csv}" "${sample_npz}" "${run_log}" "${nvt_report}" "${structure_report}"; do
        [[ -s "${path}" ]] || return 1
    done
    grep -q "Checkpoint provenance and particle identity validated." "${run_log}" || return 1
    grep -q "Simulation finished successfully." "${run_log}" || return 1
    if [[ "${disable_ml}" == "1" ]]; then
        grep -q "PaiNN disabled by --disable_ml" "${run_log}" || return 1
    else
        grep -q "PaiNN ML Potential attivato:" "${run_log}" || return 1
    fi

    "${PYTHON_BIN}" - "${nvt_report}" "${structure_report}" "${TOTAL_STEPS}" <<'PY'
import json
import sys
from pathlib import Path
nvt = json.loads(Path(sys.argv[1]).read_text())
structure = json.loads(Path(sys.argv[2]).read_text())
expected = int(sys.argv[3])
if nvt.get("pass") is not True or int(nvt.get("final_step", -1)) != expected:
    raise SystemExit(1)
if structure.get("pass") is not True or not structure.get("groups"):
    raise SystemExit(1)
PY
}

run_branch() {
    local label="$1"
    local branch_dir="$2"
    local disable_ml="$3"
    local energy_csv="${branch_dir}/energy.csv"
    local sample_npz="${branch_dir}/sample.npz"
    local run_log="${branch_dir}/run.log"
    local nvt_report="${branch_dir}/nvt_smoke_report.json"
    local structure_report="${branch_dir}/runtime_structure_report.json"
    local -a cmd

    if [[ "${AB_RESUME}" == "1" ]] && branch_complete "${branch_dir}" "${disable_ml}"; then
        echo "[RESUME] Reusing verified completed branch ${label}: ${branch_dir}"
        return 0
    fi
    if [[ "${AB_RESUME}" == "1" ]]; then
        echo "[RESUME] Branch ${label} is incomplete or unverifiable; rebuilding it."
        rm -rf "${branch_dir}"
        mkdir -p "${branch_dir}"
    fi

    # Keep the command array non-empty on all code paths. macOS ships Bash 3.2,
    # where expanding an empty local array under `set -u` can raise
    # "unbound variable". This also avoids shell-version-dependent behavior.
    cmd=(
        "${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py"
        --model "${MODEL}"
    )
    if [[ "${disable_ml}" == "1" ]]; then
        cmd+=(--disable_ml)
    fi
    cmd+=(
        --config "${CONFIG}"
        --priors "${PRIORS}"
        --rb_info "${RB_INFO}"
        --dataset "${DATASET}"
        --checkpoint "${CHECKPOINT}"
        --steps "${TOTAL_STEPS}"
        --dt "${DT}"
        --kT "${KT}"
        --thermostat_seed "${THERMOSTAT_SEED}"
        --device "${DEVICE}"
        --neighbor_search "${NEIGHBOR_SEARCH}"
        --energy_file "${energy_csv}"
        --no_vtf
        --sample_npz "${sample_npz}"
        --sample_start_step "${BURNIN_STEPS}"
        --log_interval "${LOG_INTERVAL}"
    )

    echo "[RUN] ${label}: ${TOTAL_STEPS} steps from shared checkpoint"
    PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "${run_log}"

    "${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/analyze_nvt_smoke.py" \
        --energy-csv "${energy_csv}" \
        --expected-steps "${TOTAL_STEPS}" \
        --output "${nvt_report}"

    "${PYTHON_BIN}" "${FRAMEWORK_ROOT}/ibi/validate_runtime_structure.py" \
        --dataset "${DATASET}" \
        --priors "${PRIORS}" \
        --sample-npz "${sample_npz}" \
        --ibi-config "${IBI_CONFIG}" \
        --output "${structure_report}"
}

run_branch "A/IBI-only" "${OUTDIR}/ibi_only" 1
run_branch "B/IBI+PaiNN" "${OUTDIR}/ibi_plus_ml" 0

COMPARISON="${OUTDIR}/ab_structure_comparison.json"
"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/ibi/compare_runtime_structure.py" \
    --report-a "${OUTDIR}/ibi_only/runtime_structure_report.json" \
    --report-b "${OUTDIR}/ibi_plus_ml/runtime_structure_report.json" \
    --label-a "IBI-only" \
    --label-b "IBI+PaiNN" \
    --output "${COMPARISON}"

cat <<EOF
[IBI/ML MATCHED A/B VALIDATION COMPLETE]
preflight : ${PREFLIGHT_JSON}
A report  : ${OUTDIR}/ibi_only/runtime_structure_report.json
B report  : ${OUTDIR}/ibi_plus_ml/runtime_structure_report.json
comparison: ${COMPARISON}
[PASS] Both branches completed from the same provenance-validated checkpoint with identical NVT seed and sampling schedule.
[NOTE] Structural L1 is diagnostic. This A/B gate isolates the effect of PaiNN before conservative-IBI conversion.
EOF
