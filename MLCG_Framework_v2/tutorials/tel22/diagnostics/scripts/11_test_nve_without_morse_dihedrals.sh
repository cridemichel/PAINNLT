#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
PREPARE="${SCRIPT_DIR}/11_prepare_prior_ablation.py"
COMPARE="${SCRIPT_DIR}/11_compare_prior_ablation.py"

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
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_prior_ablation_morse_dihedral}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.97}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

usage() {
    cat <<'EOF'
Usage:
  11_test_nve_without_morse_dihedrals.sh [--dry-run | --overwrite | --resume]

Diagnostic branches:
  baseline
  no_morse
  no_dihedrals
  no_morse_no_dihedrals

On the current production TEL22 priors, dihedrals=[]; therefore the two
no-dihedral branches are recorded as exact aliases and are not rerun.

Environment overrides use the same names as the standard TEL22 NVE certifier,
e.g. NVE_ML_PRECISION=float64, NVE_DURATION_PS=2.0, NVE_DTS="...".
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
for path in \
    tel22_model.pt \
    tel22_model.pt.manifest.json \
    tel22_training_config.json \
    cg_priors.json \
    rigid_bodies_info.json \
    tel22_dataset.bin \
    equilibrated.npz; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing required input: ${path}" >&2; exit 1; }
done
for path in "${CERTIFIER}" "${PREPARE}" "${COMPARE}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing diagnostic component: ${path}" >&2; exit 1; }
done

read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} >= 3)) || { echo "[ERROR] NVE_DTS needs at least three values" >&2; exit 1; }

OUT_ABS="${TUTORIAL_DIR}/${NVE_OUTPUT_DIR}"
INPUT_DIR="${OUT_ABS}/inputs"
INPUT_MANIFEST="${INPUT_DIR}/ablation_inputs.json"
SUMMARY="${OUT_ABS}/prior_ablation_summary.json"

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

cat <<EOF_PLAN

[TEL22 NVE PRIOR ABLATION]
Hamiltonian base : cg_priors.json + tel22_model.pt (same trained PaiNN)
checkpoint state : same real positions/orientations/velocities; technical Morse markers stripped only when absent
precision        : ${NVE_ML_PRECISION}
device           : ${NVE_DEVICE}
neighbor search  : ${NVE_NEIGHBOR_SEARCH}
dt grid [ps]     : ${NVE_DTS}
duration / dt    : ${NVE_DURATION_PS} ps
sampling         : every integration step
output           : ${NVE_OUTPUT_DIR}
NOTE             : diagnostic Hamiltonian ablation only; PaiNN is intentionally not retrained.
EOF_PLAN

RUN_VARIANTS="$(python3 - "${INPUT_MANIFEST}" <<'PY'
import json, sys
m=json.load(open(sys.argv[1]))
print(" ".join(
    name for name in ("baseline","no_morse","no_dihedrals","no_morse_no_dihedrals")
    if m["variants"][name]["run_required"]
))
PY
)"

completed_fail=0
for variant in ${RUN_VARIANTS}; do
    IFS='|' read -r PRIORS_PATH CHECKPOINT_PATH <<< "$(python3 - "${INPUT_MANIFEST}" "${variant}" <<'PY'
import json, sys
m=json.load(open(sys.argv[1])); v=m["variants"][sys.argv[2]]
print(v["priors"] + "|" + v["checkpoint"])
PY
)"
    VAR_OUT="${OUT_ABS}/${variant}"

    echo
    echo "[VARIANT] ${variant}"
    CERT_CMD=(
        python3 "${CERTIFIER}"
        --pypresso "${PYPRESSO}"
        --model "${TUTORIAL_DIR}/tel22_model.pt"
        --config "${TUTORIAL_DIR}/tel22_training_config.json"
        --priors "${PRIORS_PATH}"
        --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json"
        --dataset "${TUTORIAL_DIR}/tel22_dataset.bin"
        --checkpoint "${CHECKPOINT_PATH}"
        --dts "${DT_ARGS[@]}"
        --duration-ps "${NVE_DURATION_PS}"
        --device "${NVE_DEVICE}"
        --ml-precision "${NVE_ML_PRECISION}"
        --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
        --output-dir "${VAR_OUT}"
        --slope-min "${NVE_SLOPE_MIN}"
        --slope-max "${NVE_SLOPE_MAX}"
        --min-r2 "${NVE_MIN_R2}"
        --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}"
    )
    case "${MODE}" in
        dry-run) CERT_CMD+=(--dry-run) ;;
        overwrite) CERT_CMD+=(--overwrite) ;;
        resume) CERT_CMD+=(--reuse-existing) ;;
    esac

    set +e
    "${CERT_CMD[@]}"
    rc=$?
    set -e
    if [[ ${rc} -eq 2 ]]; then
        # A completed strict NVE FAIL is diagnostic data, not a runner failure.
        completed_fail=1
        echo "[INFO] ${variant}: strict certification FAIL recorded; continuing A/B diagnostic."
    elif [[ ${rc} -ne 0 ]]; then
        echo "[ERROR] ${variant}: certifier failed operationally with exit code ${rc}" >&2
        exit "${rc}"
    fi
done

if [[ "${MODE}" == "dry-run" ]]; then
    echo
    echo "[DRY-RUN] Planned canonical branches: ${RUN_VARIANTS}"
    python3 - "${INPUT_MANIFEST}" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
for name,v in m["variants"].items():
    if v["alias_of"]:
        print(f"[DRY-RUN] alias: {name} = {v['alias_of']}")
PY
    exit 0
fi

python3 "${COMPARE}" \
    --inputs "${INPUT_MANIFEST}" \
    --results-dir "${OUT_ABS}" \
    --output "${SUMMARY}"

echo
if [[ ${completed_fail} -ne 0 ]]; then
    echo "[DONE] A/B diagnostic completed; at least one branch failed the original strict gate."
else
    echo "[DONE] A/B diagnostic completed; all canonical branches passed the original strict gate."
fi
echo "[SUMMARY] ${SUMMARY}"
