#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
PREPARE="${SCRIPT_DIR}/19_prepare_morse_top10_a_softening.py"
SUMMARIZER="${SCRIPT_DIR}/19_summarize_morse_top10_a_softening.py"

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
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_morse_top10_a_softening_coarse_5ps}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.85}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

usage() {
cat <<'EOF'
Usage:
  19_test_nve_morse_top10_a_softening.sh [--dry-run | --overwrite | --resume]

Diagnostic softening sweep on the fixed top-10% (18) Morse contacts ranked by
checkpoint-local curvature in test 18. The selected subset is identical in all
branches. Only Morse a changes:

  reference : a scale 1.00 (reused full-Morse test 17)
  candidate : a scale 0.90  -> k(r0) ratio 0.81
  candidate : a scale 0.80  -> k(r0) ratio 0.64
  candidate : a scale 0.70  -> k(r0) ratio 0.49

D=50 and r0 are preserved. PaiNN is disabled and the bonded-analytic Morse
runtime is used throughout. Diagnostic only; no production promotion.
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
    echo "[ERROR] Unexpected arguments: $*" >&2
    exit 2
fi

cd "${TUTORIAL_DIR}"
for p in tel22_model.pt tel22_model.pt.manifest.json tel22_training_config.json cg_priors.json rigid_bodies_info.json tel22_dataset.bin equilibrated.npz; do
    [[ -f "${p}" ]] || { echo "[ERROR] Missing required input: ${p}" >&2; exit 1; }
done
for p in "${CERTIFIER}" "${PREPARE}" "${SUMMARIZER}"; do
    [[ -f "${p}" ]] || { echo "[ERROR] Missing diagnostic component: ${p}" >&2; exit 1; }
done
read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} == 4)) || { echo "[ERROR] Requires exactly four dt values" >&2; exit 1; }

OUT_ABS="${TUTORIAL_DIR}/${NVE_OUTPUT_DIR}"
INPUT_DIR="${OUT_ABS}/inputs"
INPUT_MANIFEST="${INPUT_DIR}/morse_top10_a_softening_inputs.json"
SUMMARY="${OUT_ABS}/morse_top10_a_softening_summary.json"
RANKING_MANIFEST="${TUTORIAL_DIR}/diagnostics/nve/nve_morse_curvature_quantiles_coarse_5ps/inputs/curvature_quantile_inputs.json"
REFERENCE_REPORT="${TUTORIAL_DIR}/diagnostics/nve/nve_morse_runtime_ab_coarse_5ps/bonded_analytic/nve_certification_report.json"
ZEROD_SUMMARY="${TUTORIAL_DIR}/diagnostics/nve/nve_morse_curvature_quantiles_coarse_5ps/morse_curvature_quantiles_summary.json"

[[ -f "${RANKING_MANIFEST}" ]] || { echo "[ERROR] Missing test-18 curvature ranking: ${RANKING_MANIFEST}" >&2; exit 1; }
[[ -f "${REFERENCE_REPORT}" ]] || { echo "[ERROR] Missing full-Morse bonded reference from test 17: ${REFERENCE_REPORT}" >&2; exit 1; }

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT_ABS}"
elif [[ "${MODE}" == "normal" && -e "${OUT_ABS}" ]]; then
    echo "[ERROR] Output exists: ${OUT_ABS}; use --overwrite or --resume" >&2
    exit 1
fi
mkdir -p "${INPUT_DIR}"

prepare_mode=(--overwrite)
if [[ ("${MODE}" == "resume" || "${MODE}" == "dry-run") && -f "${INPUT_MANIFEST}" ]]; then
    prepare_mode=(--reuse-existing)
fi
python3 "${PREPARE}" \
    --priors "${TUTORIAL_DIR}/cg_priors.json" \
    --config "${TUTORIAL_DIR}/tel22_training_config.json" \
    --dataset "${TUTORIAL_DIR}/tel22_dataset.bin" \
    --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json" \
    --model "${TUTORIAL_DIR}/tel22_model.pt" \
    --checkpoint "${TUTORIAL_DIR}/equilibrated.npz" \
    --ranking-manifest "${RANKING_MANIFEST}" \
    --output-dir "${INPUT_DIR}" "${prepare_mode[@]}"

# Fail closed unless the reusable reference is exact full production Morse.
python3 - "${REFERENCE_REPORT}" "${TUTORIAL_DIR}/cg_priors.json" <<'PY'
import hashlib, json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
p = sys.argv[2]
d = r.get("definition", {})
if d.get("hamiltonian_mode") != "conservative_classical_model_provenance_ml_disabled":
    raise SystemExit("reference is not priors-only")
runtime = r.get("pair_specific_morse_runtime", d.get("pair_specific_morse_runtime"))
if runtime != "bonded-analytic":
    raise SystemExit(f"reference runtime={runtime!r}, expected bonded-analytic")
sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
if r.get("inputs_sha256", {}).get("priors") != sha:
    raise SystemExit("reference priors hash does not match production cg_priors.json")
runs = sorted(r.get("runs", []), key=lambda x: float(x["dt_ps"]))
if [float(x["dt_ps"]) for x in runs] != [0.002, 0.003, 0.004, 0.005]:
    raise SystemExit("reference dt grid mismatch")
if any(abs(float(x["duration_ps"]) - 5.0) > 0.5 * float(x["dt_ps"]) + 1e-12 for x in runs):
    raise SystemExit("reference duration is not 5 ps")
print("[FULL MORSE REFERENCE] validated bonded-analytic 5 ps coarse report")
PY

cat <<EOF

[TEL22 MORSE TOP10 LOCAL-CURVATURE a-SOFTENING]
reference       : full production Morse, a scale 1.00 (reused test 17)
selected subset : fixed ranks 1..18 from test-18 local-curvature ranking
candidate 0.90  : selected a -> 0.27, D=50 preserved, k(r0) ratio=0.81
candidate 0.80  : selected a -> 0.24, D=50 preserved, k(r0) ratio=0.64
candidate 0.70  : selected a -> 0.21, D=50 preserved, k(r0) ratio=0.49
PaiNN           : OFF in all branches
Morse runtime   : bonded-analytic in all branches
checkpoint      : identical mechanical arrays; derived metadata only updates priors provenance
particle set    : identical; technical markers remain inert
WCA/harmonics   : unchanged
dt grid [ps]    : ${NVE_DTS}
duration / dt   : ${NVE_DURATION_PS} ps
NOTE            : diagnostic parameter softening, not reparameterization/promotion.
EOF

variants=(top10_a0p90 top10_a0p80 top10_a0p70)
reports=()
for variant in "${variants[@]}"; do
    IFS='|' read -r PRIORS CHECKPOINT <<< "$(python3 - "${INPUT_MANIFEST}" "${variant}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
v = m["variants"][sys.argv[2]]
print(v["priors"] + "|" + v["checkpoint"])
PY
)"
    RUN_OUT="${OUT_ABS}/${variant}"
    REPORT="${RUN_OUT}/nve_certification_report.json"
    reports+=("${REPORT}")
    cmd=(
        python3 "${CERTIFIER}"
        --pypresso "${PYPRESSO}"
        --model "${TUTORIAL_DIR}/tel22_model.pt"
        --config "${TUTORIAL_DIR}/tel22_training_config.json"
        --priors "${PRIORS}"
        --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json"
        --dataset "${TUTORIAL_DIR}/tel22_dataset.bin"
        --checkpoint "${CHECKPOINT}"
        --disable-ml
        --morse-switch-mode switched
        --pair-specific-morse-runtime bonded-analytic
        --dts "${DT_ARGS[@]}"
        --duration-ps "${NVE_DURATION_PS}"
        --device "${NVE_DEVICE}"
        --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
        --output-dir "${RUN_OUT}"
        --slope-min "${NVE_SLOPE_MIN}"
        --slope-max "${NVE_SLOPE_MAX}"
        --min-r2 "${NVE_MIN_R2}"
        --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}"
    )
    case "${MODE}" in
        dry-run) cmd+=(--dry-run --reuse-existing) ;;
        overwrite) cmd+=(--overwrite) ;;
        resume) cmd+=(--reuse-existing) ;;
    esac
    set +e
    "${cmd[@]}"
    rc=$?
    set -e
    if [[ ${rc} -ne 0 && ${rc} -ne 2 ]]; then
        echo "[ERROR] ${variant} operational failure rc=${rc}" >&2
        exit "${rc}"
    fi
    if [[ "${MODE}" != "dry-run" && ! -f "${REPORT}" ]]; then
        echo "[ERROR] Missing report: ${REPORT}" >&2
        exit 1
    fi
done

if [[ "${MODE}" == "dry-run" ]]; then
    echo "[DRY-RUN] Planned a=0.90/0.80/0.70 top-10% softening branches; no MD executed."
    exit 0
fi

sumcmd=(
    python3 "${SUMMARIZER}"
    --reference-report "${REFERENCE_REPORT}"
    --inputs "${INPUT_MANIFEST}"
    --top10-a0p90 "${reports[0]}"
    --top10-a0p80 "${reports[1]}"
    --top10-a0p70 "${reports[2]}"
    --output "${SUMMARY}"
)
if [[ -f "${ZEROD_SUMMARY}" ]]; then
    sumcmd+=(--zeroD-summary "${ZEROD_SUMMARY}")
fi
"${sumcmd[@]}"
echo "[DONE] ${SUMMARY}"
