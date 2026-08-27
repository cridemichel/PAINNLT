#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
PREPARE="${SCRIPT_DIR}/20_prepare_morse_top10_a_refinement.py"
SUMMARIZER="${SCRIPT_DIR}/20_summarize_morse_top10_a_refinement.py"

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
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_morse_top10_a_refinement_coarse_5ps}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.85}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

usage() {
cat <<'EOF'
Usage:
  20_test_nve_morse_top10_a_refinement.sh [--dry-run | --overwrite | --resume]

Refined diagnostic sweep around the test-19 a=0.90 result on the same fixed
18-contact top-10% checkpoint-local-curvature Morse subset:

  full reference : a scale 1.000 (reused test 17)
  center         : a scale 0.900 (reused test 19)
  new candidates : a scale 0.950, 0.925, 0.875, 0.850

D=50 and r0 are preserved. PaiNN is disabled and bonded-analytic Morse is used
throughout. Diagnostic only; no production promotion.
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
INPUT_MANIFEST="${INPUT_DIR}/morse_top10_a_refinement_inputs.json"
SUMMARY="${OUT_ABS}/morse_top10_a_refinement_summary.json"
RANKING_MANIFEST="${TUTORIAL_DIR}/diagnostics/nve/nve_morse_curvature_quantiles_coarse_5ps/inputs/curvature_quantile_inputs.json"
FULL_REFERENCE_REPORT="${TUTORIAL_DIR}/diagnostics/nve/nve_morse_runtime_ab_coarse_5ps/bonded_analytic/nve_certification_report.json"
CENTER_INPUT_MANIFEST="${TUTORIAL_DIR}/diagnostics/nve/nve_morse_top10_a_softening_coarse_5ps/inputs/morse_top10_a_softening_inputs.json"
CENTER_REFERENCE_REPORT="${TUTORIAL_DIR}/diagnostics/nve/nve_morse_top10_a_softening_coarse_5ps/top10_a0p90/nve_certification_report.json"

for p in "${RANKING_MANIFEST}" "${FULL_REFERENCE_REPORT}" "${CENTER_INPUT_MANIFEST}" "${CENTER_REFERENCE_REPORT}"; do
    [[ -f "${p}" ]] || { echo "[ERROR] Missing prerequisite diagnostic artifact: ${p}" >&2; exit 1; }
done

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

# Fail closed on both reused reference reports and on the identity of the
# a=0.90 center subset/priors produced by test 19.
python3 - "${FULL_REFERENCE_REPORT}" "${CENTER_REFERENCE_REPORT}" "${CENTER_INPUT_MANIFEST}" "${INPUT_MANIFEST}" "${TUTORIAL_DIR}/cg_priors.json" <<'PY'
import hashlib, json, sys
full_path, center_path, center_manifest_path, new_manifest_path, prod_priors = sys.argv[1:]

def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def validate_report(path, expected_priors_sha):
    r = json.load(open(path, encoding="utf-8"))
    d = r.get("definition", {})
    if d.get("hamiltonian_mode") != "conservative_classical_model_provenance_ml_disabled":
        raise SystemExit(f"{path}: not priors-only")
    runtime = r.get("pair_specific_morse_runtime", d.get("pair_specific_morse_runtime"))
    if runtime != "bonded-analytic":
        raise SystemExit(f"{path}: runtime={runtime!r}, expected bonded-analytic")
    if r.get("inputs_sha256", {}).get("priors") != expected_priors_sha:
        raise SystemExit(f"{path}: priors hash mismatch")
    runs = sorted(r.get("runs", []), key=lambda x: float(x["dt_ps"]))
    if [float(x["dt_ps"]) for x in runs] != [0.002, 0.003, 0.004, 0.005]:
        raise SystemExit(f"{path}: dt grid mismatch")
    if any(abs(float(x["duration_ps"]) - 5.0) > 0.5 * float(x["dt_ps"]) + 1e-12 for x in runs):
        raise SystemExit(f"{path}: duration mismatch")

prod_sha = sha(prod_priors)
validate_report(full_path, prod_sha)
cm = json.load(open(center_manifest_path, encoding="utf-8"))
nm = json.load(open(new_manifest_path, encoding="utf-8"))
if cm.get("kind") != "tel22_morse_top10_local_curvature_a_softening_inputs":
    raise SystemExit("unexpected test-19 center manifest kind")
center = cm.get("variants", {}).get("top10_a0p90")
if not isinstance(center, dict) or abs(float(center.get("a_scale", -1)) - 0.90) > 1e-15:
    raise SystemExit("test-19 center reference is not a=0.90")
if center.get("selected_bond_indices") != nm.get("selected_bond_indices"):
    raise SystemExit("test-19 a=0.90 subset differs from test-20 subset")
if cm.get("source", {}).get("priors_sha256") != prod_sha or nm.get("source", {}).get("priors_sha256") != prod_sha:
    raise SystemExit("reference/refinement source priors differ from production")
validate_report(center_path, center.get("priors_sha256"))
print("[REFERENCES] validated full a=1.00 and center a=0.90 reports; identical 18-contact subset")
PY

cat <<EOF

[TEL22 MORSE TOP10 LOCAL-CURVATURE a-REFINEMENT]
full reference   : a scale 1.000 (reused test 17)
center reference : a scale 0.900 (reused test 19)
new candidate    : a scale 0.950 -> a=0.28500, k(r0) ratio=0.902500
new candidate    : a scale 0.925 -> a=0.27750, k(r0) ratio=0.855625
new candidate    : a scale 0.875 -> a=0.26250, k(r0) ratio=0.765625
new candidate    : a scale 0.850 -> a=0.25500, k(r0) ratio=0.722500
selected subset  : fixed ranks 1..18 from test-18 local-curvature ranking
D / r0           : unchanged on every contact
PaiNN             : OFF in all branches
Morse runtime     : bonded-analytic in all branches
checkpoint        : identical mechanical arrays; derived metadata only updates priors provenance
particle set      : identical; technical markers remain inert
WCA/harmonics     : unchanged
dt grid [ps]      : ${NVE_DTS}
duration / dt     : ${NVE_DURATION_PS} ps
NOTE              : diagnostic parameter refinement, not reparameterization/promotion.
EOF

variants=(top10_a0p950 top10_a0p925 top10_a0p875 top10_a0p850)
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
    echo "[DRY-RUN] Planned a=0.950/0.925/0.875/0.850 refinement branches; a=0.900 and full references reused; no MD executed."
    exit 0
fi

python3 "${SUMMARIZER}" \
    --full-reference-report "${FULL_REFERENCE_REPORT}" \
    --center-reference-report "${CENTER_REFERENCE_REPORT}" \
    --inputs "${INPUT_MANIFEST}" \
    --top10-a0p950 "${reports[0]}" \
    --top10-a0p925 "${reports[1]}" \
    --top10-a0p875 "${reports[2]}" \
    --top10-a0p850 "${reports[3]}" \
    --output "${SUMMARY}"
echo "[DONE] ${SUMMARY}"
