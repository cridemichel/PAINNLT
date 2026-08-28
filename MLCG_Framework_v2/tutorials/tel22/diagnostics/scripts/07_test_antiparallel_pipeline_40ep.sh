#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEL22_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TEL22_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINER="${TRAINER:-${FRAMEWORK_ROOT}/training/build/train_painn}"
PYRESSO="${PYRESSO:-${FRAMEWORK_ROOT}/espresso/build/pypresso}"
AA_TOPOLOGY="${AA_TOPOLOGY:-${TEL22_DIR}/md.gro}"
AA_TRAJECTORY="${AA_TRAJECTORY:-${TEL22_DIR}/md_whole.trr}"
TOPOLOGY_CONFIG="${TOPOLOGY_CONFIG:-${TEL22_DIR}/tel22_topology.json}"
PDB_REFERENCE="${PDB_REFERENCE:-${TEL22_DIR}/143D.pdb}"
TRAINING_CONFIG_SOURCE="${TRAINING_CONFIG_SOURCE:-${TEL22_DIR}/diagnostics/configs/tel22_training_config_pipeline40.json}"
RUN_DIR="${PIPELINE_TEST_RUN_DIR:-${TEL22_DIR}/diagnostics/smoke/antiparallel_pipeline_40ep}"
DEVICE="${DEVICE:-auto}"

STEPS_SD="${PIPELINE_TEST_STEPS_SD:-200}"
STEPS_MD="${PIPELINE_TEST_STEPS_MD:-200}"
STEPS_ML_CAPPED="${PIPELINE_TEST_STEPS_ML_CAPPED:-200}"
STEPS_ML_UNCAPPED="${PIPELINE_TEST_STEPS_ML_UNCAPPED:-200}"
PRODUCTION_STEPS="${PIPELINE_TEST_PRODUCTION_STEPS:-500}"

required_files=(
    "${AA_TOPOLOGY}"
    "${AA_TRAJECTORY}"
    "${TOPOLOGY_CONFIG}"
    "${PDB_REFERENCE}"
    "${TRAINING_CONFIG_SOURCE}"
)
for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        printf '[ERROR] Missing required input: %s\n' "${path}" >&2
        exit 2
    fi
done
for executable in "${TRAINER}" "${PYRESSO}"; do
    if [[ ! -x "${executable}" ]]; then
        printf '[ERROR] Missing executable: %s\n' "${executable}" >&2
        exit 2
    fi
done
if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    printf '[ERROR] Test directory is not empty: %s\n' "${RUN_DIR}" >&2
    printf '        Select a fresh PIPELINE_TEST_RUN_DIR; existing evidence is never overwritten.\n' >&2
    exit 2
fi
mkdir -p "${RUN_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_antiparallel_topology.py" \
    --topology "${TOPOLOGY_CONFIG}" \
    --pdb "${PDB_REFERENCE}" \
    --r0-mode auto \
    --require-reference-metadata

cp "${TOPOLOGY_CONFIG}" "${RUN_DIR}/tel22_topology.json"
cp "${TRAINING_CONFIG_SOURCE}" "${RUN_DIR}/tel22_training_config_pipeline40.json"

cd "${RUN_DIR}"

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/preprocessing/build_cg_dataset.py" \
    --topology "${AA_TOPOLOGY}" \
    --trajectory "${AA_TRAJECTORY}" \
    --config tel22_topology.json \
    --output tel22_dataset.bin \
    --priors-output cg_priors.json \
    --rb-info-output rigid_bodies_info.json \
    2>&1 | tee preprocessing_stdout.log

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_antiparallel_topology.py" \
    --topology cg_priors.json \
    --r0-mode numeric

"${TRAINER}" \
    tel22_dataset.bin \
    tel22_model.pt \
    tel22_training_config_pipeline40.json \
    2>&1 | tee training_stdout.log

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/training/create_model_manifest.py" \
    --model tel22_model.pt \
    --config tel22_training_config_pipeline40.json \
    --dataset tel22_dataset.bin

"${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/equilibrate.py" \
    --model tel22_model.pt \
    --config tel22_training_config_pipeline40.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --out_checkpoint equilibrated.npz \
    --device "${DEVICE}" \
    --neighbor_search link-cell \
    --dt 0.001 \
    --steps_sd "${STEPS_SD}" \
    --steps_md "${STEPS_MD}" \
    --steps_ml_capped "${STEPS_ML_CAPPED}" \
    --steps_ml_uncapped "${STEPS_ML_UNCAPPED}" \
    2>&1 | tee equilibration_stdout.log

"${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py" \
    --model tel22_model.pt \
    --config tel22_training_config_pipeline40.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --checkpoint equilibrated.npz \
    --steps "${PRODUCTION_STEPS}" \
    --dt 0.001 \
    --device "${DEVICE}" \
    --neighbor_search link-cell \
    --energy_file energy.csv \
    --trajectory_file cg_trajectory.vtf \
    --out_checkpoint production_final.npz \
    2>&1 | tee production_stdout.log

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_pipeline_test.py" \
    --run-dir "${RUN_DIR}" \
    --expected-epochs 40 \
    --report "${RUN_DIR}/pipeline_test_report.json"

printf '[PASS] Functional pipeline test completed in %s\n' "${RUN_DIR}"
printf '[NOTE] This smoke test does not replace production-length structural validation or NVE certification.\n'
