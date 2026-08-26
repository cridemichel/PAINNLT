#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
PREPARE="${SCRIPT_DIR}/12_prepare_angle_ablation.py"
COMPARE="${SCRIPT_DIR}/12_compare_angle_ablation.py"
VALIDATE_BASELINE="${SCRIPT_DIR}/12_validate_baseline_report.py"

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
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_prior_ablation_angles}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.97}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"
ANGLE_ABLATION_BASELINE_REPORT="${ANGLE_ABLATION_BASELINE_REPORT:-diagnostics/nve/nve_prior_ablation_morse_dihedral/baseline/nve_certification_report.json}"
ANGLE_ABLATION_REUSE_BASELINE="${ANGLE_ABLATION_REUSE_BASELINE:-1}"
ANGLE_ABLATION_ALLOW_UNVERIFIED_FP32_REUSE="${ANGLE_ABLATION_ALLOW_UNVERIFIED_FP32_REUSE:-0}"

usage() {
    cat <<'EOF'
Usage:
  12_test_nve_without_angles.sh [--dry-run | --overwrite | --resume]

Compares the production TEL22 baseline with an otherwise identical diagnostic
Hamiltonian where all entries in cg_priors.json:angles are removed. The trained
PaiNN is intentionally not retrained.

By default, for FP32 only, the script tries to reuse the completed baseline from
11_test_nve_without_morse_dihedrals.sh after validating hashes, dt grid, device,
neighbor search and precision from its logs. If validation fails, baseline is
rerun automatically. Set ANGLE_ABLATION_REUSE_BASELINE=0 to force a fresh baseline. If an older
baseline log lacks the precision banner but you independently know it is FP32,
set ANGLE_ABLATION_ALLOW_UNVERIFIED_FP32_REUSE=1 to reuse it explicitly.
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
for path in "${CERTIFIER}" "${PREPARE}" "${COMPARE}" "${VALIDATE_BASELINE}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing diagnostic component: ${path}" >&2; exit 1; }
done
read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} >= 3)) || { echo "[ERROR] NVE_DTS needs at least three values" >&2; exit 1; }

OUT_ABS="${TUTORIAL_DIR}/${NVE_OUTPUT_DIR}"
INPUT_DIR="${OUT_ABS}/inputs"
INPUT_MANIFEST="${INPUT_DIR}/angle_ablation_inputs.json"
NO_ANGLES_OUT="${OUT_ABS}/no_angles"
SUMMARY="${OUT_ABS}/angle_ablation_summary.json"
BASELINE_LOCAL_OUT="${OUT_ABS}/baseline"
BASELINE_DEFAULT_ABS="${TUTORIAL_DIR}/${ANGLE_ABLATION_BASELINE_REPORT}"

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT_ABS}"
elif [[ "${MODE}" == "normal" && -e "${OUT_ABS}" ]]; then
    echo "[ERROR] Output exists: ${OUT_ABS}. Use --overwrite, --resume, or set NVE_OUTPUT_DIR." >&2
    exit 1
fi
mkdir -p "${INPUT_DIR}"
PREP_ARGS=(
    python3 "${PREPARE}"
    --priors "${TUTORIAL_DIR}/cg_priors.json"
    --config "${TUTORIAL_DIR}/tel22_training_config.json"
    --dataset "${TUTORIAL_DIR}/tel22_dataset.bin"
    --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json"
    --model "${TUTORIAL_DIR}/tel22_model.pt"
    --checkpoint "${TUTORIAL_DIR}/equilibrated.npz"
    --output-dir "${INPUT_DIR}"
)
if [[ "${MODE}" == "overwrite" || "${MODE}" == "resume" || "${MODE}" == "dry-run" ]]; then
    PREP_ARGS+=(--overwrite)
fi
"${PREP_ARGS[@]}"

IFS='|' read -r NO_ANGLES_PRIORS NO_ANGLES_CHECKPOINT ANGLE_COUNT <<< "$(python3 - "${INPUT_MANIFEST}" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); v=m["no_angles"]
print(v["priors"] + "|" + v["checkpoint"] + "|" + str(v["removed_angle_entries"]))
PY
)"

cat <<EOF_PLAN

[TEL22 NVE ANGLE ABLATION]
angular priors removed : ${ANGLE_COUNT}
Hamiltonian baseline   : cg_priors.json + tel22_model.pt
Hamiltonian no_angles  : same, but angles=[]; Morse and all other priors retained
checkpoint state       : all mechanical arrays numerically identical
precision              : ${NVE_ML_PRECISION}
device                 : ${NVE_DEVICE}
neighbor search        : ${NVE_NEIGHBOR_SEARCH}
dt grid [ps]           : ${NVE_DTS}
duration / dt          : ${NVE_DURATION_PS} ps
NOTE                    : numerical ablation only; PaiNN is intentionally not retrained.
EOF_PLAN

BASELINE_REPORT=""
if [[ "${ANGLE_ABLATION_REUSE_BASELINE}" == "1" && -f "${BASELINE_DEFAULT_ABS}" ]]; then
    VALIDATE_ARGS=(
        python3 "${VALIDATE_BASELINE}"
        --report "${BASELINE_DEFAULT_ABS}"
        --tutorial "${TUTORIAL_DIR}"
        --dts "${DT_ARGS[@]}"
        --device "${NVE_DEVICE}"
        --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
        --precision "${NVE_ML_PRECISION}"
    )
    if [[ "${ANGLE_ABLATION_ALLOW_UNVERIFIED_FP32_REUSE}" == "1" ]]; then
        VALIDATE_ARGS+=(--allow-unverified-fp32)
    fi
    if "${VALIDATE_ARGS[@]}"; then
        BASELINE_REPORT="${BASELINE_DEFAULT_ABS}"
    fi
fi

run_certifier() {
    local variant="$1" priors="$2" checkpoint="$3" out="$4"
    echo
    echo "[VARIANT] ${variant}"
    local cmd=(
        python3 "${CERTIFIER}"
        --pypresso "${PYPRESSO}"
        --model "${TUTORIAL_DIR}/tel22_model.pt"
        --config "${TUTORIAL_DIR}/tel22_training_config.json"
        --priors "${priors}"
        --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json"
        --dataset "${TUTORIAL_DIR}/tel22_dataset.bin"
        --checkpoint "${checkpoint}"
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
if [[ -z "${BASELINE_REPORT}" ]]; then
    echo "[BASELINE] validated reusable report not available; running a fresh baseline."
    set +e
    run_certifier baseline "${TUTORIAL_DIR}/cg_priors.json" "${TUTORIAL_DIR}/equilibrated.npz" "${BASELINE_LOCAL_OUT}"
    rc=$?
    set -e
    if [[ ${rc} -eq 2 ]]; then completed_fail=1; elif [[ ${rc} -ne 0 ]]; then exit "${rc}"; fi
    BASELINE_REPORT="${BASELINE_LOCAL_OUT}/nve_certification_report.json"
else
    echo "[BASELINE] reusing validated report: ${BASELINE_REPORT}"
fi

set +e
run_certifier no_angles "${NO_ANGLES_PRIORS}" "${NO_ANGLES_CHECKPOINT}" "${NO_ANGLES_OUT}"
rc=$?
set -e
if [[ ${rc} -eq 2 ]]; then completed_fail=1; elif [[ ${rc} -ne 0 ]]; then exit "${rc}"; fi

if [[ "${MODE}" == "dry-run" ]]; then
    echo
    echo "[DRY-RUN] baseline report: ${BASELINE_REPORT}"
    echo "[DRY-RUN] no_angles output: ${NO_ANGLES_OUT}"
    exit 0
fi

python3 "${COMPARE}" \
    --baseline "${BASELINE_REPORT}" \
    --no-angles "${NO_ANGLES_OUT}/nve_certification_report.json" \
    --inputs "${INPUT_MANIFEST}" \
    --output "${SUMMARY}"

echo
if [[ ${completed_fail} -ne 0 ]]; then
    echo "[DONE] Angle A/B diagnostic completed; at least one branch failed the original strict gate."
else
    echo "[DONE] Angle A/B diagnostic completed; all branches passed the original strict gate."
fi
echo "[SUMMARY] ${SUMMARY}"
