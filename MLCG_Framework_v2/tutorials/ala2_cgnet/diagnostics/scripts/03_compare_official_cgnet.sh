#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALA2_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINING_RUN_DIR="${ALA2_TRAINING_RUN_DIR:-${ALA2_DIR}/diagnostics/smoke/cgnet_harmonic_50ep}"
AB_RUN_DIR="${ALA2_AB_RUN_DIR:-${ALA2_DIR}/diagnostics/smoke/fes_ab_quick_4x50k}"
RUN_DIR="${ALA2_CGNET_COMPARATOR_RUN_DIR:-${ALA2_DIR}/diagnostics/smoke/official_cgnet_comparator}"
DATA_DIR="${ALA2_DATA_DIR:-${ALA2_DIR}/data}"

CGNET_EPOCHS="${ALA2_CGNET_EPOCHS:-5}"
CGNET_DEVICE="${ALA2_CGNET_DEVICE:-cpu}"
CGNET_EQUIL_STEPS="${ALA2_CGNET_EQUIL_STEPS:-25000}"
CGNET_BURNIN_STEPS="${ALA2_CGNET_BURNIN_STEPS:-10000}"
CGNET_SAMPLE_INTERVAL="${ALA2_CGNET_SAMPLE_INTERVAL:-50}"
CGNET_DT="${ALA2_CGNET_DT:-0.0005}"
FES_BINS="${ALA2_FES_BINS:-24}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    printf '[ERROR] Missing Python executable: %s\n' "${PYTHON_BIN}" >&2
    exit 2
fi
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
if ! "${PYTHON_BIN}" -c 'import numpy, scipy, torch, matplotlib' >/dev/null 2>&1; then
    printf '[ERROR] The active Python needs numpy, scipy, torch and matplotlib.\n' >&2
    exit 2
fi
for path in \
    "${TRAINING_RUN_DIR}/ala2_reference.npz" \
    "${AB_RUN_DIR}/ala2_fes_ab_report.json"; do
    if [[ ! -f "${path}" ]]; then
        printf '[ERROR] Missing prerequisite artifact: %s\n' "${path}" >&2
        exit 2
    fi
done
if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    printf '[ERROR] Comparator directory is not empty: %s\n' "${RUN_DIR}" >&2
    printf '        Select a fresh ALA2_CGNET_COMPARATOR_RUN_DIR.\n' >&2
    exit 2
fi

prior_samples=("${AB_RUN_DIR}"/replicas/replica_*/prior_only_samples.npz)
ml_samples=("${AB_RUN_DIR}"/replicas/replica_*/prior_plus_painn_samples.npz)
if [[ ${#prior_samples[@]} -lt 2 || ${#prior_samples[@]} -ne ${#ml_samples[@]} ]]; then
    printf '[ERROR] Expected at least two matched prior/PaiNN replica pairs in %s\n' "${AB_RUN_DIR}" >&2
    exit 2
fi
for path in "${prior_samples[@]}" "${ml_samples[@]}"; do
    if [[ ! -f "${path}" ]]; then
        printf '[ERROR] Missing matched A/B trajectory: %s\n' "${path}" >&2
        exit 2
    fi
done

mkdir -p "${RUN_DIR}" "${DATA_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"

"${PYTHON_BIN}" "${SCRIPT_DIR}/download_cgnet_ala2.py" \
    --output-dir "${DATA_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/download_official_cgnet.py" \
    --output-dir "${RUN_DIR}/official_cgnet_source" \
    --report "${RUN_DIR}/official_cgnet_source_report.json" \
    2>&1 | tee "${RUN_DIR}/source_download_stdout.log"

CGNET_SOURCE_ROOT="$("${PYTHON_BIN}" - "${RUN_DIR}/official_cgnet_source_report.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    print(json.load(handle)["source_root"])
PY
)"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_official_cgnet_comparator.py" \
    --cgnet-source "${CGNET_SOURCE_ROOT}" \
    --coordinates "${DATA_DIR}/ala2_coordinates.npy" \
    --forces "${DATA_DIR}/ala2_forces.npy" \
    --matched-runtime-samples "${prior_samples[@]}" \
    --output-dir "${RUN_DIR}" \
    --report "${RUN_DIR}/official_cgnet_training_report.json" \
    --epochs "${CGNET_EPOCHS}" \
    --device "${CGNET_DEVICE}" \
    --equil-steps "${CGNET_EQUIL_STEPS}" \
    --burnin-steps "${CGNET_BURNIN_STEPS}" \
    --sample-interval "${CGNET_SAMPLE_INTERVAL}" \
    --dt "${CGNET_DT}" \
    2>&1 | tee "${RUN_DIR}/official_cgnet_stdout.log"

cgnet_samples=("${RUN_DIR}"/official_cgnet_replica_*.npy)
if [[ ${#cgnet_samples[@]} -ne ${#prior_samples[@]} ]]; then
    printf '[ERROR] Official CGnet did not produce one trajectory per A/B replica.\n' >&2
    exit 2
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_ala2_fes_ab.py" \
    --reference "${TRAINING_RUN_DIR}/ala2_reference.npz" \
    --prior-samples "${prior_samples[@]}" \
    --ml-samples "${ml_samples[@]}" \
    --cgnet-samples "${cgnet_samples[@]}" \
    --cgnet-units angstrom \
    --training-report "${TRAINING_RUN_DIR}/ala2_benchmark_report.json" \
    --bins "${FES_BINS}" \
    --report "${RUN_DIR}/ala2_painn_vs_official_cgnet_report.json" \
    --plot "${RUN_DIR}/ala2_painn_vs_official_cgnet.png" \
    2>&1 | tee "${RUN_DIR}/comparison_analysis_stdout.log"

printf '[PASS] Official CGnet comparison completed in %s\n' "${RUN_DIR}"
printf '[INFO] Send official_cgnet_training_report.json, ala2_painn_vs_official_cgnet_report.json and ala2_painn_vs_official_cgnet.png.\n'
