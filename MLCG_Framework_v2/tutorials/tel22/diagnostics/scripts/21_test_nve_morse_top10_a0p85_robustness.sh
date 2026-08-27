#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
SUMMARIZER="${SCRIPT_DIR}/21_summarize_morse_top10_a0p85_robustness.py"

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYPRESSO="${PYPRESSO:-${DEFAULT_PYPRESSO}}"
NVE_DEVICE="${NVE_DEVICE:-cpu}"
NVE_NEIGHBOR_SEARCH="${NVE_NEIGHBOR_SEARCH:-link-cell}"
NVE_DTS="${NVE_DTS:-0.001 0.0015 0.002 0.003 0.004 0.005}"
NVE_DURATION_PS="${NVE_DURATION_PS:-10.0}"
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_morse_top10_a0p85_robustness_10ps_fullgrid}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.6}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.4}"
NVE_MIN_R2="${NVE_MIN_R2:-0.85}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

usage() {
cat <<'EOF'
Usage:
  21_test_nve_morse_top10_a0p85_robustness.sh [--dry-run | --overwrite | --resume]

Long/full-grid robustness validation of the test-20 Morse numerical-stabilizer
candidate. Both arms use the production-like marker/non-bonded switched Morse
runtime with PaiNN disabled:

  reference : production Morse (a=0.30 on all 180 contacts)
  candidate : exact test-20 top-10% subset (18 contacts), a scale 0.85

Default grid: 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005 ps; 10 ps per dt.
The Morse terms are evaluated here as TEL22 structural/numerical stabilizers,
not as physically inferred Morse parameters.
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
for p in "${CERTIFIER}" "${SUMMARIZER}"; do
    [[ -f "${p}" ]] || { echo "[ERROR] Missing diagnostic component: ${p}" >&2; exit 1; }
done
read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} == 6)) || { echo "[ERROR] Requires exactly six dt values" >&2; exit 1; }

OUT_ABS="${TUTORIAL_DIR}/${NVE_OUTPUT_DIR}"
SUMMARY="${OUT_ABS}/morse_top10_a0p85_robustness_summary.json"
CANDIDATE_MANIFEST="${TUTORIAL_DIR}/diagnostics/nve/nve_morse_top10_a_refinement_coarse_5ps/inputs/morse_top10_a_refinement_inputs.json"
FIVE_PS_SUMMARY="${TUTORIAL_DIR}/diagnostics/nve/nve_morse_top10_a_refinement_coarse_5ps/morse_top10_a_refinement_summary.json"
for p in "${CANDIDATE_MANIFEST}" "${FIVE_PS_SUMMARY}"; do
    [[ -f "${p}" ]] || { echo "[ERROR] Missing prerequisite test-20 artifact: ${p}" >&2; exit 1; }
done

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT_ABS}"
elif [[ "${MODE}" == "normal" && -e "${OUT_ABS}" ]]; then
    echo "[ERROR] Output exists: ${OUT_ABS}; use --overwrite or --resume" >&2
    exit 1
fi
mkdir -p "${OUT_ABS}"

IFS='|' read -r CANDIDATE_PRIORS CANDIDATE_CHECKPOINT CANDIDATE_PRIORS_SHA <<< "$(python3 - "${CANDIDATE_MANIFEST}" "${TUTORIAL_DIR}/cg_priors.json" <<'PY'
import hashlib, json, os, sys
manifest_path, production_priors = sys.argv[1:]
m = json.load(open(manifest_path, encoding="utf-8"))
if m.get("kind") != "tel22_morse_top10_local_curvature_a_refinement_inputs":
    raise SystemExit("unexpected test-20 manifest kind")
v = m.get("variants", {}).get("top10_a0p850")
if not isinstance(v, dict) or abs(float(v.get("a_scale", -1)) - 0.85) > 1e-15:
    raise SystemExit("test-20 manifest lacks exact a=0.85 candidate")
if int(v.get("selected_count", -1)) != 18 or len(v.get("selected_bond_indices", [])) != 18:
    raise SystemExit("test-20 a=0.85 candidate is not the 18-contact top-10% subset")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if m.get("source", {}).get("priors_sha256") != sha(production_priors):
    raise SystemExit("test-20 source priors no longer match production cg_priors.json")
for key in ("priors", "checkpoint"):
    path = v.get(key)
    if not path or not os.path.isfile(path):
        raise SystemExit(f"missing a=0.85 {key}: {path}")
    if v.get(key + "_sha256") != sha(path):
        raise SystemExit(f"a=0.85 {key} hash mismatch")
print(v["priors"] + "|" + v["checkpoint"] + "|" + v["priors_sha256"])
PY
)"

PROD_PRIORS_SHA="$(python3 - "${TUTORIAL_DIR}/cg_priors.json" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)"

echo "[CANDIDATE] exact test-20 a=0.85 subset validated: ${CANDIDATE_PRIORS}"
cat <<EOF

[TEL22 MORSE a=0.85 ROBUSTNESS -- 10 ps FULL DT GRID]
reference       : production Morse, all 180 contacts at a=0.30
candidate       : fixed test-20 top-10% 18 contacts at a=0.255 (scale 0.85)
Morse role      : numerical/structural TEL22 stabilizer; not physical parameter inference
PaiNN           : OFF in both branches
Morse runtime   : production-like marker + non-bonded switched Morse in both branches
checkpoint      : identical mechanical state; candidate metadata is priors-bound
WCA/harmonics   : unchanged
dt grid [ps]    : ${NVE_DTS}
duration / dt   : ${NVE_DURATION_PS} ps
sampling        : every integration step
NOTE            : diagnostic robustness validation; full-model promotion still requires residual rebuild/retraining.
EOF

variants=(full a0p85)
priors=("${TUTORIAL_DIR}/cg_priors.json" "${CANDIDATE_PRIORS}")
checkpoints=("${TUTORIAL_DIR}/equilibrated.npz" "${CANDIDATE_CHECKPOINT}")
reports=()
for i in 0 1; do
    variant="${variants[$i]}"
    RUN_OUT="${OUT_ABS}/${variant}"
    REPORT="${RUN_OUT}/nve_certification_report.json"
    reports+=("${REPORT}")
    cmd=(
        python3 "${CERTIFIER}"
        --pypresso "${PYPRESSO}"
        --model "${TUTORIAL_DIR}/tel22_model.pt"
        --config "${TUTORIAL_DIR}/tel22_training_config.json"
        --priors "${priors[$i]}"
        --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json"
        --dataset "${TUTORIAL_DIR}/tel22_dataset.bin"
        --checkpoint "${checkpoints[$i]}"
        --disable-ml
        --morse-switch-mode switched
        --pair-specific-morse-runtime marker-nonbonded
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
    echo "[DRY-RUN] Planned full and a=0.85 10 ps full-grid branches; no MD executed."
    exit 0
fi

python3 "${SUMMARIZER}" \
    --full-report "${reports[0]}" \
    --candidate-report "${reports[1]}" \
    --candidate-input-manifest "${CANDIDATE_MANIFEST}" \
    --five-ps-summary "${FIVE_PS_SUMMARY}" \
    --production-priors-sha256 "${PROD_PRIORS_SHA}" \
    --output "${SUMMARY}"
echo "[DONE] ${SUMMARY}"
