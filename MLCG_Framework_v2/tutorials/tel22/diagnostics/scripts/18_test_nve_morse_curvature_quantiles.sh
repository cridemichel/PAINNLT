#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
PREPARE="${SCRIPT_DIR}/18_prepare_morse_curvature_quantiles.py"
SUMMARIZER="${SCRIPT_DIR}/18_summarize_morse_curvature_quantiles.py"

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
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_morse_curvature_quantiles_coarse_5ps}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.85}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

usage() {
cat <<'EOF'
Usage:
  18_test_nve_morse_curvature_quantiles.sh [--dry-run | --overwrite | --resume]

Diagnostic localization of TEL22 Morse coarse-dt structure.
All 180 production Morse contacts have D=50, a=0.3 and therefore identical
k(r0)=2Da^2=9. Contacts are instead ranked by the spectral local pair-potential
curvature at equilibrated.npz:

  K_local = max(|U''(r_eq)|, |U'(r_eq)/r_eq|)

The top 5%, 10%, and 20% nested subsets (9/18/36 contacts) are made inert by
setting D=0 in derived priors. Endpoint topology, technical marker count,
checkpoint mechanical arrays, WCA/harmonic priors and bonded-analytic runtime
remain unchanged. PaiNN is disabled. Diagnostic only; no production promotion.
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
MANIFEST="${INPUT_DIR}/curvature_quantile_inputs.json"
SUMMARY="${OUT_ABS}/morse_curvature_quantiles_summary.json"
REFERENCE_REPORT="${TUTORIAL_DIR}/diagnostics/nve/nve_morse_runtime_ab_coarse_5ps/bonded_analytic/nve_certification_report.json"
NO_MORSE_SUMMARY="${TUTORIAL_DIR}/diagnostics/nve/nve_stock_espresso_coarse_5ps/stock_espresso_coarse_5ps_summary.json"

[[ -f "${REFERENCE_REPORT}" ]] || { echo "[ERROR] Missing completed full-Morse bonded reference from test 17: ${REFERENCE_REPORT}" >&2; exit 1; }

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT_ABS}"
elif [[ "${MODE}" == "normal" && -e "${OUT_ABS}" ]]; then
    echo "[ERROR] Output exists: ${OUT_ABS}; use --overwrite or --resume" >&2
    exit 1
fi
mkdir -p "${INPUT_DIR}"

python3 "${PREPARE}" \
    --priors "${TUTORIAL_DIR}/cg_priors.json" \
    --config "${TUTORIAL_DIR}/tel22_training_config.json" \
    --dataset "${TUTORIAL_DIR}/tel22_dataset.bin" \
    --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json" \
    --model "${TUTORIAL_DIR}/tel22_model.pt" \
    --checkpoint "${TUTORIAL_DIR}/equilibrated.npz" \
    --output-dir "${INPUT_DIR}" --overwrite

# Fail closed unless the reusable test-17 branch is the exact full-Morse control.
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

[TEL22 MORSE LOCAL-CURVATURE QUANTILE ABLATION]
reference       : full 180 Morse, bonded-analytic runtime (reused test 17)
ranking         : K_local=max(|U''(r_eq)|, |U'(r_eq)/r_eq|) at equilibrated.npz
variant 5%      : top 9 contacts set D=0
variant 10%     : top 18 contacts set D=0
variant 20%     : top 36 contacts set D=0
PaiNN           : OFF in all branches
checkpoint      : identical mechanical arrays; derived metadata only updates priors provenance
particle set    : identical; all technical markers remain inert in bonded runtime
WCA/harmonics   : unchanged
dt grid [ps]    : ${NVE_DTS}
duration / dt   : ${NVE_DURATION_PS} ps
NOTE            : diagnostic Hamiltonian ablation, not reparameterization.
EOF

reports=()
for variant in top_05pct_zeroD top_10pct_zeroD top_20pct_zeroD; do
    IFS='|' read -r PRIORS CHECKPOINT <<< "$(python3 - "${MANIFEST}" "${variant}" <<'PY'
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
        dry-run) cmd+=(--dry-run) ;;
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
    echo "[DRY-RUN] Planned three nested quantile branches; no MD executed."
    exit 0
fi

sumcmd=(
    python3 "${SUMMARIZER}"
    --reference-report "${REFERENCE_REPORT}"
    --inputs "${MANIFEST}"
    --top-05pct-zeroD "${reports[0]}"
    --top-10pct-zeroD "${reports[1]}"
    --top-20pct-zeroD "${reports[2]}"
    --output "${SUMMARY}"
)
if [[ -f "${NO_MORSE_SUMMARY}" ]]; then
    sumcmd+=(--no-morse-summary "${NO_MORSE_SUMMARY}")
fi
"${sumcmd[@]}"
echo "[DONE] ${SUMMARY}"
