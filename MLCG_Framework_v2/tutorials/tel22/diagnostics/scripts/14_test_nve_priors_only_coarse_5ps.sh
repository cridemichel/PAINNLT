#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
SUMMARIZER="${SCRIPT_DIR}/14_summarize_priors_only_coarse_5ps.py"

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYPRESSO="${PYPRESSO:-${DEFAULT_PYPRESSO}}"
NVE_DEVICE="${NVE_DEVICE:-cpu}"
NVE_NEIGHBOR_SEARCH="${NVE_NEIGHBOR_SEARCH:-link-cell}"
NVE_DTS="${NVE_DTS:-0.002 0.003 0.004 0.005}"
NVE_DURATION_PS="${NVE_DURATION_PS:-5.0}"
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_priors_only_coarse_5ps}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.97}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

usage() {
    cat <<'EOF'
Usage:
  14_test_nve_priors_only_coarse_5ps.sh [--dry-run | --overwrite | --resume]

TEL22 classical-control NVE test:
  - production cg_priors.json unchanged
  - identical equilibrated.npz mechanical state
  - trained model retained only for checkpoint/model provenance
  - PaiNN disabled with --disable-ml
  - ESPResSo native classical precision
  - 5 ps per branch
  - dt = 0.002, 0.003, 0.004, 0.005 ps

Important: --ml-precision is intentionally not presented as an FP32/FP64 control
here. With PaiNN disabled it does not affect the force arithmetic. The purpose of
this test is a longer-window, coarse-dt second-order check of ESPResSo + priors.
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
for path in "${CERTIFIER}" "${SUMMARIZER}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing diagnostic component: ${path}" >&2; exit 1; }
done
read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} == 4)) || { echo "[ERROR] This diagnostic requires exactly four dt values" >&2; exit 1; }

OUT_ABS="${TUTORIAL_DIR}/${NVE_OUTPUT_DIR}"
REPORT="${OUT_ABS}/nve_certification_report.json"
SUMMARY="${OUT_ABS}/priors_only_coarse_5ps_summary.json"

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT_ABS}"
elif [[ "${MODE}" == "normal" && -e "${OUT_ABS}" ]]; then
    echo "[ERROR] Output exists: ${OUT_ABS}. Use --overwrite, --resume, or set NVE_OUTPUT_DIR." >&2
    exit 1
fi

cat <<EOF_PLAN

[TEL22 PRIORS-ONLY 5 ps COARSE-DT NVE]
Hamiltonian          : production cg_priors.json; PaiNN disabled
checkpoint state     : exact equilibrated.npz
ESPResSo precision   : native classical numerical path
ML precision         : not applicable (PaiNN disabled)
device               : ${NVE_DEVICE}
neighbor search      : ${NVE_NEIGHBOR_SEARCH}
dt grid [ps]         : ${NVE_DTS}
duration / dt        : ${NVE_DURATION_PS} ps
sampling             : every integration step
NOTE                 : this tests coarse-dt/finite-window scaling, not an ML FP32-vs-FP64 switch.
EOF_PLAN

cmd=(
    python3 "${CERTIFIER}"
    --pypresso "${PYPRESSO}"
    --model "${TUTORIAL_DIR}/tel22_model.pt"
    --config "${TUTORIAL_DIR}/tel22_training_config.json"
    --priors "${TUTORIAL_DIR}/cg_priors.json"
    --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json"
    --dataset "${TUTORIAL_DIR}/tel22_dataset.bin"
    --checkpoint "${TUTORIAL_DIR}/equilibrated.npz"
    --disable-ml
    --dts "${DT_ARGS[@]}"
    --duration-ps "${NVE_DURATION_PS}"
    --device "${NVE_DEVICE}"
    --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
    --output-dir "${OUT_ABS}"
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
rc=$?
set -e
if [[ ${rc} -ne 0 && ${rc} -ne 2 ]]; then
    echo "[ERROR] priors-only coarse-dt certifier failed operationally with exit code ${rc}" >&2
    exit "${rc}"
fi
if [[ "${MODE}" == "dry-run" ]]; then
    echo "[DRY-RUN] output: ${OUT_ABS}"
    exit 0
fi

[[ -f "${REPORT}" ]] || { echo "[ERROR] Missing certification report: ${REPORT}" >&2; exit 1; }
python3 "${SUMMARIZER}" --report "${REPORT}" --output "${SUMMARY}"

if [[ ${rc} -eq 2 ]]; then
    echo "[DONE] Coarse-dt run completed; original broad certification gate reported FAIL."
else
    echo "[DONE] Coarse-dt run completed; original broad certification gate passed."
fi
echo "[SUMMARY] ${SUMMARY}"
