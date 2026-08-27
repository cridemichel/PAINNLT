#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
VALIDATOR="${SCRIPT_DIR}/16_validate_switched_reference.py"
SUMMARIZER="${SCRIPT_DIR}/17_summarize_morse_runtime_ab.py"
MORSE_SMOKE="${FRAMEWORK_ROOT}/simulation/espresso_plugin/check_analytic_morse_bond.py"

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
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_morse_runtime_ab_coarse_5ps}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.90}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

usage() {
    cat <<'EOF'
Usage:
  17_test_nve_morse_runtime_ab.sh [--dry-run | --overwrite | --resume]

TEL22 pair-specific Morse runtime isolation:
  A = existing 5 ps priors-only reference: technical marker + non-bonded Morse + hybrid
  B = same priors/checkpoint/particles: analytic MorseBond on physical endpoints + regular decomposition

Candidate B keeps the technical marker particles inert so the exact same equilibrated.npz
and particle bookkeeping are used. The markers carry no Morse interaction in B.
PaiNN is disabled in both branches. Production files are never modified.
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
for path in "${CERTIFIER}" "${VALIDATOR}" "${SUMMARIZER}" "${MORSE_SMOKE}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing diagnostic component: ${path}" >&2; exit 1; }
done
read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} == 4)) || { echo "[ERROR] This diagnostic requires exactly four dt values" >&2; exit 1; }

OUT_ABS="${TUTORIAL_DIR}/${NVE_OUTPUT_DIR}"
VALIDATION="${OUT_ABS}/marker_reference_validation.json"
BONDED_OUT="${OUT_ABS}/bonded_analytic"
BONDED_REPORT="${BONDED_OUT}/nve_certification_report.json"
SUMMARY="${OUT_ABS}/morse_runtime_ab_summary.json"
MARKER_DIR="${TUTORIAL_DIR}/diagnostics/nve/nve_priors_only_coarse_5ps"
MARKER_REPORT="${MARKER_DIR}/nve_certification_report.json"
MARKER_PLAN="${MARKER_DIR}/run_plan.json"
NO_MORSE_SUMMARY="${TUTORIAL_DIR}/diagnostics/nve/nve_stock_espresso_coarse_5ps/stock_espresso_coarse_5ps_summary.json"

[[ -f "${MARKER_REPORT}" ]] || { echo "[ERROR] Missing completed marker/non-bonded reference: ${MARKER_REPORT}" >&2; exit 1; }
[[ -f "${MARKER_PLAN}" ]] || { echo "[ERROR] Missing marker/non-bonded reference run plan: ${MARKER_PLAN}" >&2; exit 1; }

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT_ABS}"
elif [[ "${MODE}" == "normal" && -e "${OUT_ABS}" ]]; then
    echo "[ERROR] Output exists: ${OUT_ABS}. Use --overwrite, --resume, or set NVE_OUTPUT_DIR." >&2
    exit 1
fi
mkdir -p "${OUT_ABS}"

python3 "${VALIDATOR}" \
    --report "${MARKER_REPORT}" \
    --run-plan "${MARKER_PLAN}" \
    --config "${TUTORIAL_DIR}/tel22_training_config.json" \
    --priors "${TUTORIAL_DIR}/cg_priors.json" \
    --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json" \
    --dataset "${TUTORIAL_DIR}/tel22_dataset.bin" \
    --checkpoint "${TUTORIAL_DIR}/equilibrated.npz" \
    --model "${TUTORIAL_DIR}/tel22_model.pt" \
    --output "${VALIDATION}"

python3 - "${TUTORIAL_DIR}/cg_priors.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
ms = [x for x in p.get("bonds", []) if str(x.get("type", "harmonic")).lower() == "morse"]
if len(ms) != 180:
    raise SystemExit(f"expected 180 production pair-specific Morse entries, found {len(ms)}")
if p.get("morse_type_pairs", []):
    raise SystemExit("bonded runtime A/B requires production morse_type_pairs=[]")
if any(float(x.get("r_cut", 15.0)) != 15.0 for x in ms):
    raise SystemExit("unexpected production Morse r_cut; review bonded cutoff safety before running")
print(f"[MORSE INPUTS] explicit_pair_contacts={len(ms)} r_cut=15nm morse_type_pairs=0")
PY

cat <<EOF_PLAN

[TEL22 MORSE RUNTIME A/B -- 5 ps COARSE-DT NVE]
reference A           : marker + non-bonded switched Morse + hybrid/N-square
candidate B           : analytic bonded Morse on physical endpoints + regular decomposition
PaiNN                 : OFF in both branches
Morse contacts        : same 180 explicit endpoint pairs
Morse D/a/r0/r_cut    : identical production values
checkpoint            : exact same equilibrated.npz
particle set          : exact same; technical markers remain inert in candidate B
WCA/harmonic priors   : identical
energy gauge          : bonded Morse adds constant +D/contact; sigma_E and forces unaffected
neighbor search       : ${NVE_NEIGHBOR_SEARCH} (candidate has no Morse hybrid side)
dt grid [ps]          : ${NVE_DTS}
duration / dt         : ${NVE_DURATION_PS} ps
sampling              : every integration step
NOTE                   : diagnostic runtime substitution; not a production promotion.
EOF_PLAN

if [[ "${MODE}" != "dry-run" ]]; then
    "${PYPRESSO}" "${MORSE_SMOKE}"
fi

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
    --morse-switch-mode switched
    --pair-specific-morse-runtime bonded-analytic
    --dts "${DT_ARGS[@]}"
    --duration-ps "${NVE_DURATION_PS}"
    --device "${NVE_DEVICE}"
    --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
    --output-dir "${BONDED_OUT}"
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
    echo "[ERROR] bonded-analytic certifier failed operationally with exit code ${rc}" >&2
    exit "${rc}"
fi
if [[ "${MODE}" == "dry-run" ]]; then
    echo "[DRY-RUN] candidate output: ${BONDED_OUT}"
    exit 0
fi
[[ -f "${BONDED_REPORT}" ]] || { echo "[ERROR] Missing bonded-analytic report: ${BONDED_REPORT}" >&2; exit 1; }

summary_cmd=(
    python3 "${SUMMARIZER}"
    --marker-report "${MARKER_REPORT}"
    --bonded-report "${BONDED_REPORT}"
    --reference-validation "${VALIDATION}"
    --output "${SUMMARY}"
)
if [[ -f "${NO_MORSE_SUMMARY}" ]]; then
    summary_cmd+=(--no-morse-summary "${NO_MORSE_SUMMARY}")
fi
"${summary_cmd[@]}"

echo "[DONE] Morse runtime isolation completed."
echo "[SUMMARY] ${SUMMARY}"
exit 0
