#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
PREPARER="${SCRIPT_DIR}/22_prepare_morse_uniform_a0p85.py"
SUMMARIZER="${SCRIPT_DIR}/22_summarize_morse_uniform_abc.py"

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
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_morse_uniform_abc_10ps_fullgrid}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.4}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.6}"
NVE_MIN_R2="${NVE_MIN_R2:-0.85}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

usage() {
cat <<'EOF'
Usage:
  22_test_nve_morse_uniform_abc.sh [--dry-run | --overwrite | --resume]

TEL22 empirical Morse-stabilizer A/B/C comparison on the same 10 ps full-grid
protocol used by test 21:

  A production : all 180 Morse a=0.300       (reuse test 21)
  B selective  : top-18 Morse a=0.255        (reuse test 21)
  C uniform    : all 180 Morse a=0.255        (new run)

PaiNN is disabled in all arms. The production-like marker/non-bonded switched
Morse runtime is used throughout. Only C requires new MD; A and B are validated
and reused from the completed test-21 reports.
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
for p in "${CERTIFIER}" "${PREPARER}" "${SUMMARIZER}"; do
    [[ -f "${p}" ]] || { echo "[ERROR] Missing diagnostic component: ${p}" >&2; exit 1; }
done
read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} == 6)) || { echo "[ERROR] Requires exactly six dt values" >&2; exit 1; }

TEST21_DIR="${TUTORIAL_DIR}/diagnostics/nve/nve_morse_top10_a0p85_robustness_10ps_fullgrid"
FULL_REPORT="${TEST21_DIR}/full/nve_certification_report.json"
SELECTIVE_REPORT="${TEST21_DIR}/a0p85/nve_certification_report.json"
TEST21_SUMMARY="${TEST21_DIR}/morse_top10_a0p85_robustness_summary.json"
SELECTIVE_MANIFEST="${TUTORIAL_DIR}/diagnostics/nve/nve_morse_top10_a_refinement_coarse_5ps/inputs/morse_top10_a_refinement_inputs.json"
for p in "${FULL_REPORT}" "${SELECTIVE_REPORT}" "${TEST21_SUMMARY}" "${SELECTIVE_MANIFEST}"; do
    [[ -f "${p}" ]] || { echo "[ERROR] Missing prerequisite artifact: ${p}" >&2; exit 1; }
done

python3 - "${TEST21_SUMMARY}" "${SELECTIVE_MANIFEST}" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
manifest = json.load(open(sys.argv[2], encoding="utf-8"))
if summary.get("kind") != "tel22_morse_top10_a0p85_robustness_10ps_fullgrid":
    raise SystemExit("unexpected test-21 summary kind")
if summary.get("interpretation") != "a0p85_5ps_gain_not_robust_on_10ps_full_grid":
    raise SystemExit("test-21 prerequisite does not have the expected non-robust selective result")
v = manifest.get("variants", {}).get("top10_a0p850")
if not isinstance(v, dict) or int(v.get("selected_count", -1)) != 18 or abs(float(v.get("a_scale", -1)) - 0.85) > 1e-15:
    raise SystemExit("invalid test-20 selective a=0.85 manifest")
print("[A/B REFERENCES] validated test-21 full and selective 10 ps full-grid evidence")
PY

OUT_ABS="${TUTORIAL_DIR}/${NVE_OUTPUT_DIR}"
INPUT_DIR="${OUT_ABS}/inputs/uniform_a0p85"
UNIFORM_REPORT="${OUT_ABS}/C_uniform_a0p85/nve_certification_report.json"
SUMMARY="${OUT_ABS}/morse_uniform_abc_summary.json"

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT_ABS}"
elif [[ "${MODE}" == "normal" && -e "${OUT_ABS}" ]]; then
    echo "[ERROR] Output exists: ${OUT_ABS}; use --overwrite or --resume" >&2
    exit 1
fi
mkdir -p "${OUT_ABS}"

prepare_cmd=(
    python3 "${PREPARER}"
    --priors "${TUTORIAL_DIR}/cg_priors.json"
    --config "${TUTORIAL_DIR}/tel22_training_config.json"
    --dataset "${TUTORIAL_DIR}/tel22_dataset.bin"
    --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json"
    --model "${TUTORIAL_DIR}/tel22_model.pt"
    --checkpoint "${TUTORIAL_DIR}/equilibrated.npz"
    --output-dir "${INPUT_DIR}"
)
case "${MODE}" in
    overwrite) prepare_cmd+=(--overwrite) ;;
    resume|dry-run) prepare_cmd+=(--reuse-existing) ;;
esac
"${prepare_cmd[@]}"

UNIFORM_MANIFEST="${INPUT_DIR}/morse_uniform_a0p85_inputs.json"
IFS='|' read -r UNIFORM_PRIORS UNIFORM_CHECKPOINT UNIFORM_PRIORS_SHA <<< "$(python3 - "${UNIFORM_MANIFEST}" <<'PY'
import hashlib, json, os, sys
p = sys.argv[1]
m = json.load(open(p, encoding="utf-8"))
if m.get("kind") != "tel22_morse_uniform_a0p85_inputs":
    raise SystemExit("unexpected uniform manifest kind")
if int(m.get("morse_count", -1)) != 180 or abs(float(m.get("a_scale", -1)) - 0.85) > 1e-15:
    raise SystemExit("invalid uniform a=0.85 manifest")
sha = lambda q: hashlib.sha256(open(q, "rb").read()).hexdigest()
for key in ("priors", "checkpoint"):
    path = m.get(key)
    if not path or not os.path.isfile(path):
        raise SystemExit(f"missing uniform {key}: {path}")
    if m.get(key + "_sha256") != sha(path):
        raise SystemExit(f"uniform {key} hash mismatch")
print(m["priors"] + "|" + m["checkpoint"] + "|" + m["priors_sha256"])
PY
)"

PROD_PRIORS_SHA="$(python3 - "${TUTORIAL_DIR}/cg_priors.json" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)"

cat <<EOF

[TEL22 MORSE STABILIZER A/B/C -- 10 ps FULL DT GRID]
A production     : 180/180 Morse a=0.300                      [REUSE test 21]
B selective      : 18 Morse a=0.255 + 162 Morse a=0.300       [REUSE test 21]
C uniform        : 180/180 Morse a=0.255                      [NEW]
Morse role       : empirical numerical/structural TEL22 stabilizers
PaiNN            : OFF in all arms
Morse runtime    : marker + non-bonded switched, production-like
checkpoint       : identical physical/mechanical state; C metadata is priors-bound
WCA/harmonics    : unchanged
dt grid [ps]     : ${NVE_DTS}
duration / dt    : ${NVE_DURATION_PS} ps
sampling         : every integration step
NOTE             : only C performs new MD; no physical-parameter inference is implied.
EOF

cmd=(
    python3 "${CERTIFIER}"
    --pypresso "${PYPRESSO}"
    --model "${TUTORIAL_DIR}/tel22_model.pt"
    --config "${TUTORIAL_DIR}/tel22_training_config.json"
    --priors "${UNIFORM_PRIORS}"
    --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json"
    --dataset "${TUTORIAL_DIR}/tel22_dataset.bin"
    --checkpoint "${UNIFORM_CHECKPOINT}"
    --disable-ml
    --morse-switch-mode switched
    --pair-specific-morse-runtime marker-nonbonded
    --dts "${DT_ARGS[@]}"
    --duration-ps "${NVE_DURATION_PS}"
    --device "${NVE_DEVICE}"
    --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
    --output-dir "${OUT_ABS}/C_uniform_a0p85"
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
    echo "[ERROR] C_uniform_a0p85 operational failure rc=${rc}" >&2
    exit "${rc}"
fi

if [[ "${MODE}" == "dry-run" ]]; then
    echo "[DRY-RUN] A/B references validated; planned only C_uniform_a0p85 MD."
    exit 0
fi
[[ -f "${UNIFORM_REPORT}" ]] || { echo "[ERROR] Missing uniform report: ${UNIFORM_REPORT}" >&2; exit 1; }

python3 "${SUMMARIZER}" \
    --full-report "${FULL_REPORT}" \
    --selective-report "${SELECTIVE_REPORT}" \
    --uniform-report "${UNIFORM_REPORT}" \
    --selective-manifest "${SELECTIVE_MANIFEST}" \
    --uniform-manifest "${UNIFORM_MANIFEST}" \
    --test21-summary "${TEST21_SUMMARY}" \
    --production-priors-sha256 "${PROD_PRIORS_SHA}" \
    --output "${SUMMARY}"
echo "[DONE] ${SUMMARY}"
