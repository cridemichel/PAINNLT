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
SOURCE_TOPOLOGY="${TOPOLOGY_CONFIG:-${TEL22_DIR}/tel22_topology.json}"
PDB_REFERENCE="${PDB_REFERENCE:-${TEL22_DIR}/143D.pdb}"
TRAINING_CONFIG_SOURCE="${TRAINING_CONFIG_SOURCE:-${TEL22_DIR}/diagnostics/configs/tel22_training_config_variant_a_15ep.json}"
RUN_DIR="${VARIANT_A_RUN_DIR:-${PIPELINE_TEST_RUN_DIR:-${TEL22_DIR}/diagnostics/smoke/variant_a_15ep}}"
DEVICE="${DEVICE:-auto}"
ALLOW_EARLY_STOP="${PIPELINE_ALLOW_EARLY_STOP:-0}"
REUSE_DATASET_DIR="${VARIANT_A_REUSE_DATASET_DIR:-}"

STEPS_SD="${PIPELINE_TEST_STEPS_SD:-200}"
STEPS_MD="${PIPELINE_TEST_STEPS_MD:-200}"
STEPS_ML_CAPPED="${PIPELINE_TEST_STEPS_ML_CAPPED:-200}"
STEPS_ML_UNCAPPED="${PIPELINE_TEST_STEPS_ML_UNCAPPED:-200}"
PRODUCTION_STEPS="${PIPELINE_TEST_PRODUCTION_STEPS:-500}"

required_files=(
    "${AA_TOPOLOGY}"
    "${AA_TRAJECTORY}"
    "${SOURCE_TOPOLOGY}"
    "${PDB_REFERENCE}"
    "${TRAINING_CONFIG_SOURCE}"
)
for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        printf '[ERROR] Missing required input: %s\n' "${path}" >&2
        exit 2
    fi
done

# The runner later changes into RUN_DIR. Resolve every input first so callers
# may safely provide either absolute or repository-relative paths.
absolute_file() {
    local path="$1"
    printf '%s/%s\n' "$(cd "$(dirname "${path}")" && pwd)" "$(basename "${path}")"
}
AA_TOPOLOGY="$(absolute_file "${AA_TOPOLOGY}")"
AA_TRAJECTORY="$(absolute_file "${AA_TRAJECTORY}")"
SOURCE_TOPOLOGY="$(absolute_file "${SOURCE_TOPOLOGY}")"
PDB_REFERENCE="$(absolute_file "${PDB_REFERENCE}")"
TRAINING_CONFIG_SOURCE="$(absolute_file "${TRAINING_CONFIG_SOURCE}")"

if [[ -n "${REUSE_DATASET_DIR}" ]]; then
    if [[ ! -d "${REUSE_DATASET_DIR}" ]]; then
        printf '[ERROR] Reuse directory does not exist: %s\n' "${REUSE_DATASET_DIR}" >&2
        exit 2
    fi
    REUSE_DATASET_DIR="$(cd "${REUSE_DATASET_DIR}" && pwd)"
    for name in tel22_dataset.bin cg_priors.json rigid_bodies_info.json; do
        if [[ ! -s "${REUSE_DATASET_DIR}/${name}" ]]; then
            printf '[ERROR] Missing reusable Variant-A artifact: %s\n' "${REUSE_DATASET_DIR}/${name}" >&2
            exit 2
        fi
    done
fi

for executable in "${TRAINER}" "${PYRESSO}"; do
    if [[ ! -x "${executable}" ]]; then
        printf '[ERROR] Missing executable: %s\n' "${executable}" >&2
        exit 2
    fi
done
if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    printf '[ERROR] Test directory is not empty: %s\n' "${RUN_DIR}" >&2
    printf '        Select a fresh VARIANT_A_RUN_DIR; existing evidence is never overwritten.\n' >&2
    exit 2
fi
mkdir -p "${RUN_DIR}"

TRAINING_CONFIG_NAME="$(basename "${TRAINING_CONFIG_SOURCE}")"
EXPECTED_EPOCHS="$(
    "${PYTHON_BIN}" -c \
        'import json, sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["epochs"]))' \
        "${TRAINING_CONFIG_SOURCE}"
)"

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_antiparallel_topology.py" \
    --topology "${SOURCE_TOPOLOGY}" \
    --pdb "${PDB_REFERENCE}" \
    --r0-mode auto \
    --require-reference-metadata

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_variant_a_topology.py" \
    --input "${SOURCE_TOPOLOGY}" \
    --output "${RUN_DIR}/tel22_topology_variant_a.json"
cp "${TRAINING_CONFIG_SOURCE}" "${RUN_DIR}/${TRAINING_CONFIG_NAME}"

cd "${RUN_DIR}"

if [[ -n "${REUSE_DATASET_DIR}" ]]; then
    cp "${REUSE_DATASET_DIR}/tel22_dataset.bin" tel22_dataset.bin
    cp "${REUSE_DATASET_DIR}/cg_priors.json" cg_priors.json
    cp "${REUSE_DATASET_DIR}/rigid_bodies_info.json" rigid_bodies_info.json
    printf '[INFO] Reused Variant-A dataset and priors from %s\n' "${REUSE_DATASET_DIR}"
else
    "${PYTHON_BIN}" "${FRAMEWORK_ROOT}/preprocessing/build_cg_dataset.py" \
        --topology "${AA_TOPOLOGY}" \
        --trajectory "${AA_TRAJECTORY}" \
        --config tel22_topology_variant_a.json \
        --output tel22_dataset.bin \
        --priors-output cg_priors.json \
        --rb-info-output rigid_bodies_info.json \
        2>&1 | tee preprocessing_stdout.log
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_variant_a_topology.py" \
    --input cg_priors.json

"${TRAINER}" \
    tel22_dataset.bin \
    tel22_model.pt \
    "${TRAINING_CONFIG_NAME}" \
    2>&1 | tee training_stdout.log

"${PYTHON_BIN}" "${FRAMEWORK_ROOT}/training/create_model_manifest.py" \
    --model tel22_model.pt \
    --config "${TRAINING_CONFIG_NAME}" \
    --dataset tel22_dataset.bin

"${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/equilibrate.py" \
    --model tel22_model.pt \
    --config "${TRAINING_CONFIG_NAME}" \
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
    --config "${TRAINING_CONFIG_NAME}" \
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

validator_args=(
    --run-dir "${RUN_DIR}" \
    --expected-epochs "${EXPECTED_EPOCHS}" \
    --config-name "${TRAINING_CONFIG_NAME}" \
    --topology-mode variant-a \
    --report "${RUN_DIR}/pipeline_test_report.json"
)
if [[ "${ALLOW_EARLY_STOP}" == "1" ]]; then
    validator_args+=(--allow-early-stop)
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_pipeline_test.py" "${validator_args[@]}"

printf '[PASS] TEL22 Variant-A pipeline test completed in %s\n' "${RUN_DIR}"
printf '[NOTE] Compare best validation MAE/loss with the Morse run; this smoke test is not a thermodynamic certification.\n'
