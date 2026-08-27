#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
VALIDATE_FULL="${SCRIPT_DIR}/13_validate_full_baseline.py"
COMPARE="${SCRIPT_DIR}/13_compare_full_vs_priors_only.py"

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYPRESSO="${PYPRESSO:-${DEFAULT_PYPRESSO}}"
NVE_DEVICE="${NVE_DEVICE:-cpu}"
NVE_ML_PRECISION="${NVE_ML_PRECISION:-float32}"
NVE_NEIGHBOR_SEARCH="${NVE_NEIGHBOR_SEARCH:-link-cell}"
NVE_DTS="${NVE_DTS:-0.001 0.0015 0.002 0.003 0.004 0.005}"
NVE_DURATION_PS="${NVE_DURATION_PS:-2.0}"
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_full_vs_priors_only}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.97}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"
FULL_BASELINE_REPORT="${FULL_BASELINE_REPORT:-diagnostics/nve/nve_prior_ablation_morse_dihedral/baseline/nve_certification_report.json}"
FULL_BASELINE_REUSE="${FULL_BASELINE_REUSE:-1}"

usage() {
    cat <<'EOF'
Usage:
  13_test_nve_priors_only.sh [--dry-run | --overwrite | --resume]

Final TEL22 NVE attribution test:
  full        = production priors + trained PaiNN
  priors_only = identical priors/topology/checkpoint/model provenance, but PaiNN disabled
                with certify_nve.py --disable-ml.

For the default FP32 protocol the script reuses the previously completed full
baseline only after validating input hashes, dt grid, duration, device, neighbor
search and the exact --ml_precision float32 command stored in run_plan.json.
If validation fails it safely computes a fresh full baseline.
EOF
}

MODE="normal"
case "${1:-}" in
    "") ;;
    --dry-run) MODE="dry-run"; shift ;;
    --overwrite) MODE="overwrite"; shift ;;
    --resume) MODE="resume"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
esac
if (($# != 0)); then
    echo "[ERROR] Unexpected extra arguments: $*" >&2
    exit 2
fi

cd "${TUTORIAL_DIR}"
for path in tel22_model.pt tel22_model.pt.manifest.json tel22_training_config.json cg_priors.json rigid_bodies_info.json tel22_dataset.bin equilibrated.npz; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing required input: ${path}" >&2; exit 1; }
done
for path in "${CERTIFIER}" "${VALIDATE_FULL}" "${COMPARE}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing diagnostic component: ${path}" >&2; exit 1; }
done
read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} >= 3)) || { echo "[ERROR] NVE_DTS needs at least three values" >&2; exit 1; }

OUT_ABS="${TUTORIAL_DIR}/${NVE_OUTPUT_DIR}"
FULL_LOCAL_OUT="${OUT_ABS}/full"
PRIORS_ONLY_OUT="${OUT_ABS}/priors_only"
SUMMARY="${OUT_ABS}/full_vs_priors_only_summary.json"
FULL_DEFAULT_ABS="${TUTORIAL_DIR}/${FULL_BASELINE_REPORT}"

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT_ABS}"
elif [[ "${MODE}" == "normal" && -e "${OUT_ABS}" ]]; then
    echo "[ERROR] Output exists: ${OUT_ABS}. Use --overwrite, --resume, or set NVE_OUTPUT_DIR." >&2
    exit 1
fi
mkdir -p "${OUT_ABS}"

cat <<EOF_PLAN

[TEL22 NVE FULL vs PRIORS-ONLY]
full Hamiltonian    : cg_priors.json + tel22_model.pt
priors-only branch  : identical cg_priors.json; tel22_model.pt retained for provenance, PaiNN disabled
checkpoint state    : exact same equilibrated.npz
precision full      : ${NVE_ML_PRECISION}
device              : ${NVE_DEVICE}
neighbor search     : ${NVE_NEIGHBOR_SEARCH}
dt grid [ps]        : ${NVE_DTS}
duration / dt       : ${NVE_DURATION_PS} ps
NOTE                : this isolates activation of the trained TEL22 residual; no prior is removed.
EOF_PLAN

FULL_REPORT=""
if [[ "${FULL_BASELINE_REUSE}" == "1" && -f "${FULL_DEFAULT_ABS}" ]]; then
    if python3 "${VALIDATE_FULL}" \
        --report "${FULL_DEFAULT_ABS}" \
        --tutorial "${TUTORIAL_DIR}" \
        --dts "${DT_ARGS[@]}" \
        --duration-ps "${NVE_DURATION_PS}" \
        --device "${NVE_DEVICE}" \
        --neighbor-search "${NVE_NEIGHBOR_SEARCH}" \
        --precision "${NVE_ML_PRECISION}"; then
        FULL_REPORT="${FULL_DEFAULT_ABS}"
    fi
fi

run_certifier() {
    local variant="$1" out="$2" disable_ml="$3"
    echo
    echo "[VARIANT] ${variant}"
    local cmd=(
        python3 "${CERTIFIER}"
        --pypresso "${PYPRESSO}"
        --model "${TUTORIAL_DIR}/tel22_model.pt"
        --config "${TUTORIAL_DIR}/tel22_training_config.json"
        --priors "${TUTORIAL_DIR}/cg_priors.json"
        --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json"
        --dataset "${TUTORIAL_DIR}/tel22_dataset.bin"
        --checkpoint "${TUTORIAL_DIR}/equilibrated.npz"
        --dts "${DT_ARGS[@]}"
        --duration-ps "${NVE_DURATION_PS}"
        --device "${NVE_DEVICE}"
        --ml-precision "${NVE_ML_PRECISION}"
        --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
        --output-dir "${out}"
        --slope-min "${NVE_SLOPE_MIN}"
        --slope-max "${NVE_SLOPE_MAX}"
        --min-r2 "${NVE_MIN_R2}"
        --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}"
    )
    if [[ "${disable_ml}" == "1" ]]; then
        cmd+=(--disable-ml)
    fi
    case "${MODE}" in
        dry-run) cmd+=(--dry-run) ;;
        overwrite) cmd+=(--overwrite) ;;
        resume) cmd+=(--reuse-existing) ;;
    esac
    set +e
    "${cmd[@]}"
    local rc=$?
    set -e
    if [[ ${rc} -eq 2 ]]; then
        echo "[INFO] ${variant}: strict certification FAIL recorded; comparison will continue."
        return 2
    elif [[ ${rc} -ne 0 ]]; then
        echo "[ERROR] ${variant}: certifier failed operationally with exit code ${rc}" >&2
        return "${rc}"
    fi
    return 0
}

completed_fail=0
if [[ -z "${FULL_REPORT}" ]]; then
    echo "[FULL] validated reusable baseline unavailable; running a fresh full branch."
    set +e
    run_certifier full "${FULL_LOCAL_OUT}" 0
    rc=$?
    set -e
    if [[ ${rc} -eq 2 ]]; then completed_fail=1; elif [[ ${rc} -ne 0 ]]; then exit "${rc}"; fi
    FULL_REPORT="${FULL_LOCAL_OUT}/nve_certification_report.json"
else
    echo "[FULL] reusing validated report: ${FULL_REPORT}"
fi

set +e
run_certifier priors_only "${PRIORS_ONLY_OUT}" 1
rc=$?
set -e
if [[ ${rc} -eq 2 ]]; then completed_fail=1; elif [[ ${rc} -ne 0 ]]; then exit "${rc}"; fi

if [[ "${MODE}" == "dry-run" ]]; then
    echo
    echo "[DRY-RUN] full report: ${FULL_REPORT}"
    echo "[DRY-RUN] priors-only output: ${PRIORS_ONLY_OUT}"
    exit 0
fi

python3 "${COMPARE}" \
    --full "${FULL_REPORT}" \
    --priors-only "${PRIORS_ONLY_OUT}/nve_certification_report.json" \
    --output "${SUMMARY}"

echo
if [[ ${completed_fail} -ne 0 ]]; then
    echo "[DONE] Full/Priors-only A/B completed; at least one branch failed the original strict gate."
else
    echo "[DONE] Full/Priors-only A/B completed; both branches passed the original strict gate."
fi
echo "[SUMMARY] ${SUMMARY}"
