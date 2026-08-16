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

MODEL="${IBI_MODEL:-tel22_model_ibi_conservative.pt}"
CONFIG="${TRAINING_CONFIG:-tel22_training_config.json}"
DATASET="${IBI_DATASET:-tel22_dataset_ibi_residual.bin}"
RB_INFO="${IBI_RB_INFO:-rigid_bodies_info_ibi.json}"
IBI_PRIORS="${IBI_PRIORS:-ibi_conservative/cg_priors.json}"
OLD_PRIORS="${OLD_TEL22_PRIORS:-../tel22/cg_priors.json}"
SOURCE_CHECKPOINT="${IBI_TIMESTEP_SOURCE_CHECKPOINT:-nve_equilibration_conservative_ibi_only/equilibrated_conservative_ibi_only.npz}"
OUTPUT_DIR="${IBI_TIMESTEP_OUTPUT_DIR:-ibi_timestep_range_diagnostic}"
DTS="${IBI_TIMESTEP_DTS:-0.001 0.0015 0.002 0.003 0.004 0.005}"
DURATION="${IBI_TIMESTEP_DURATION_PS:-1.0}"
BRANCH_DT="${IBI_TIMESTEP_BRANCH_DT:-0.0005}"
BRANCH_DURATION="${IBI_TIMESTEP_BRANCH_DURATION_PS:-0.25}"
BRANCH_KT="${IBI_TIMESTEP_BRANCH_KT:-2.49}"
SEED_BASE="${IBI_TIMESTEP_SEED_BASE:-290000}"
DEVICE="${IBI_TIMESTEP_DEVICE:-cpu}"
ML_PRECISION="${IBI_TIMESTEP_ML_PRECISION:-float32}"
NEIGHBOR="${IBI_TIMESTEP_NEIGHBOR_SEARCH:-link-cell}"

for path in "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" "${IBI_PRIORS}" "${OLD_PRIORS}" "${SOURCE_CHECKPOINT}"; do
    if [[ ! -f "${path}" ]]; then echo "[ERROR] Missing required input: ${path}" >&2; exit 1; fi
done
read -r -a DT_ARGS <<< "${DTS}"

ARGS=(
  --pypresso "${PYRESSO}" --model "${MODEL}" --config "${CONFIG}"
  --old-priors "${OLD_PRIORS}" --ibi-priors "${IBI_PRIORS}"
  --rb-info "${RB_INFO}" --dataset "${DATASET}" --source-checkpoint "${SOURCE_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}" --dts "${DT_ARGS[@]}" --duration-ps "${DURATION}"
  --branch-dt "${BRANCH_DT}" --branch-duration-ps "${BRANCH_DURATION}" --branch-kT "${BRANCH_KT}"
  --seed-base "${SEED_BASE}" --device "${DEVICE}" --ml-precision "${ML_PRECISION}" --neighbor-search "${NEIGHBOR}"
)
for arg in "$@"; do
    case "${arg}" in
        --dry-run|--overwrite|--resume) ARGS+=("${arg}") ;;
        *) echo "[ERROR] Unsupported argument: ${arg}" >&2; exit 2 ;;
    esac
done

cat <<EOF
[IBI COARSE-TIMESTEP/STIFFNESS DIAGNOSTIC]
old priors   : ${OLD_PRIORS}
IBI priors   : ${IBI_PRIORS}
source chk   : ${SOURCE_CHECKPOINT}
dt scan      : ${DTS} ps
NVE duration : ${DURATION} ps per dt
NVT branch   : ${BRANCH_DURATION} ps at dt=${BRANCH_DT} ps
variants     : old_tel22 / ibi_bonds_only / ibi_angles_only / full_ibi
output       : ${OUTPUT_DIR}
[NOTE] Unlike step 27 no_ibi, old_tel22 restores the original harmonic bonded priors.
[NOTE] Diagnostic-only. No numerical kernel or certification artifact is modified.
EOF

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/simulation/diagnose_ibi_timestep_range.py" "${ARGS[@]}"
