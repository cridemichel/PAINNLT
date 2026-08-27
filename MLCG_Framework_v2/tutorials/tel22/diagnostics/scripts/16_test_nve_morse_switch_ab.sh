#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
VALIDATOR="${SCRIPT_DIR}/16_validate_switched_reference.py"
SUMMARIZER="${SCRIPT_DIR}/16_summarize_morse_switch_ab.py"

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
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_morse_switch_ab_coarse_5ps}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.90}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

usage() {
    cat <<'EOF'
Usage:
  16_test_nve_morse_switch_ab.sh [--dry-run | --overwrite | --resume]

TEL22 Morse-switch isolation test:
  A = existing priors-only 5 ps reference with production switched Morse
  B = same priors/checkpoint/markers/non-bonded path with stock-shifted Morse

Only switch_start changes at runtime:
  A: switch_start = derived r_switch (production default)
  B: switch_start = -1 (ESPResSo stock shifted-Morse branch)

PaiNN is disabled in both branches. No production priors or checkpoints are modified.
The completed switched branch from diagnostic 14 is reused after strict validation.
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
for path in "${CERTIFIER}" "${VALIDATOR}" "${SUMMARIZER}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing diagnostic component: ${path}" >&2; exit 1; }
done
read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} == 4)) || { echo "[ERROR] This diagnostic requires exactly four dt values" >&2; exit 1; }

OUT_ABS="${TUTORIAL_DIR}/${NVE_OUTPUT_DIR}"
VALIDATION="${OUT_ABS}/switched_reference_validation.json"
STOCK_OUT="${OUT_ABS}/stock_shifted"
STOCK_REPORT="${STOCK_OUT}/nve_certification_report.json"
SUMMARY="${OUT_ABS}/morse_switch_ab_summary.json"
SWITCHED_DIR="${TUTORIAL_DIR}/diagnostics/nve/nve_priors_only_coarse_5ps"
SWITCHED_REPORT="${SWITCHED_DIR}/nve_certification_report.json"
SWITCHED_PLAN="${SWITCHED_DIR}/run_plan.json"
NO_MORSE_SUMMARY="${TUTORIAL_DIR}/diagnostics/nve/nve_stock_espresso_coarse_5ps/stock_espresso_coarse_5ps_summary.json"

[[ -f "${SWITCHED_REPORT}" ]] || { echo "[ERROR] Missing completed switched reference: ${SWITCHED_REPORT}" >&2; exit 1; }
[[ -f "${SWITCHED_PLAN}" ]] || { echo "[ERROR] Missing switched reference run plan: ${SWITCHED_PLAN}" >&2; exit 1; }

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT_ABS}"
elif [[ "${MODE}" == "normal" && -e "${OUT_ABS}" ]]; then
    echo "[ERROR] Output exists: ${OUT_ABS}. Use --overwrite, --resume, or set NVE_OUTPUT_DIR." >&2
    exit 1
fi
mkdir -p "${OUT_ABS}"

python3 "${VALIDATOR}" \
    --report "${SWITCHED_REPORT}" \
    --run-plan "${SWITCHED_PLAN}" \
    --config "${TUTORIAL_DIR}/tel22_training_config.json" \
    --priors "${TUTORIAL_DIR}/cg_priors.json" \
    --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json" \
    --dataset "${TUTORIAL_DIR}/tel22_dataset.bin" \
    --checkpoint "${TUTORIAL_DIR}/equilibrated.npz" \
    --model "${TUTORIAL_DIR}/tel22_model.pt" \
    --output "${VALIDATION}"

python3 - "${TUTORIAL_DIR}/cg_priors.json" <<'PY'
import json, statistics, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
ms = [x for x in p.get("bonds", []) if str(x.get("type", "harmonic")).lower() == "morse"]
if len(ms) != 180:
    raise SystemExit(f"expected 180 production pair-specific Morse entries, found {len(ms)}")
if p.get("morse_type_pairs", []):
    raise SystemExit("this diagnostic expects no production morse_type_pairs")
rs = []
for x in ms:
    r0 = float(x["r0"]); rc = float(x.get("r_cut", 15.0))
    rs.append(float(x.get("r_switch", r0 + 0.75 * (rc - r0))))
print(
    "[MORSE INPUTS] "
    f"entries={len(ms)} explicit_r_switch={sum('r_switch' in x for x in ms)} "
    f"explicit_r_cut={sum('r_cut' in x for x in ms)} "
    f"derived_r_switch_nm={min(rs):.6g}..{max(rs):.6g} median={statistics.median(rs):.6g}"
)
PY

cat <<EOF_PLAN

[TEL22 MORSE SWITCH A/B -- 5 ps COARSE-DT NVE]
reference A           : existing production switched-Morse priors-only report
candidate B           : stock-shifted Morse (switch_start=-1)
PaiNN                 : OFF in both branches
Morse mapping         : identical pair-specific endpoint markers
Morse D/a/r0/r_cut    : identical production values
neighbor machinery    : identical hybrid/non-bonded path
dt grid [ps]          : ${NVE_DTS}
duration / dt         : ${NVE_DURATION_PS} ps
device                 : ${NVE_DEVICE}
neighbor search        : ${NVE_NEIGHBOR_SEARCH}
sampling               : every integration step
NOTE                   : diagnostic A/B; stock-shifted mode is not a production recommendation.
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
    --morse-switch-mode stock-shifted
    --dts "${DT_ARGS[@]}"
    --duration-ps "${NVE_DURATION_PS}"
    --device "${NVE_DEVICE}"
    --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
    --output-dir "${STOCK_OUT}"
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
    echo "[ERROR] stock-shifted certifier failed operationally with exit code ${rc}" >&2
    exit "${rc}"
fi
if [[ "${MODE}" == "dry-run" ]]; then
    echo "[DRY-RUN] candidate output: ${STOCK_OUT}"
    exit 0
fi
[[ -f "${STOCK_REPORT}" ]] || { echo "[ERROR] Missing stock-shifted report: ${STOCK_REPORT}" >&2; exit 1; }

summary_cmd=(
    python3 "${SUMMARIZER}"
    --switched-report "${SWITCHED_REPORT}"
    --stock-shifted-report "${STOCK_REPORT}"
    --priors "${TUTORIAL_DIR}/cg_priors.json"
    --reference-validation "${VALIDATION}"
    --output "${SUMMARY}"
)
if [[ -f "${NO_MORSE_SUMMARY}" ]]; then
    summary_cmd+=(--no-morse-summary "${NO_MORSE_SUMMARY}")
fi
"${summary_cmd[@]}"

echo "[DONE] Morse switch isolation test completed."
echo "[SUMMARY] ${SUMMARY}"
exit 0
