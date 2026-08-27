#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
VALIDATE_FULL="${SCRIPT_DIR}/13_validate_full_baseline.py"
SUMMARIZER="${SCRIPT_DIR}/23_summarize_painn_closure_uniform_a0p85.py"

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
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_painn_closure_uniform_a0p85_2ps}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.97}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"
FULL_BASELINE_REPORT="${FULL_BASELINE_REPORT:-}"

usage() {
cat <<'USAGE'
Usage:
  23_test_nve_painn_closure_uniform_a0p85.sh [--dry-run | --overwrite | --resume]

Short TEL22 PaiNN closure after the Morse-stabilizer A/B/C decision.
All arms use the historical six-dt, 2 ps NVE protocol:

  A old priors + old PaiNN       production FP32 baseline; reused and validated
  B uniform a=0.255 + old PaiNN  new diagnostic run
  C uniform a=0.255 + PaiNN OFF  new diagnostic run

B is intentionally NOT a production candidate: the old PaiNN residual was
trained against the old priors. It is run only to measure how the old residual
changes NVE scaling when placed on top of the numerically improved uniform
Morse stabilizers. C isolates the new priors on exactly the same derived state.
USAGE
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
for p in "${CERTIFIER}" "${VALIDATE_FULL}" "${SUMMARIZER}"; do
    [[ -f "${p}" ]] || { echo "[ERROR] Missing diagnostic component: ${p}" >&2; exit 1; }
done
read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} == 6)) || { echo "[ERROR] Requires exactly six dt values" >&2; exit 1; }
if [[ "${NVE_ML_PRECISION}" != "float32" ]]; then
    echo "[ERROR] This closure is intentionally the historical FP32 comparison; NVE_ML_PRECISION must be float32" >&2
    exit 1
fi

TEST22_DIR="${TUTORIAL_DIR}/diagnostics/nve/nve_morse_uniform_abc_10ps_fullgrid"
TEST22_SUMMARY="${TEST22_DIR}/morse_uniform_abc_summary.json"
UNIFORM_MANIFEST="${TEST22_DIR}/inputs/uniform_a0p85/morse_uniform_a0p85_inputs.json"
if [[ -n "${FULL_BASELINE_REPORT}" ]]; then
    FULL_REPORT="${TUTORIAL_DIR}/${FULL_BASELINE_REPORT}"
else
    FULL_REPORT=""
    for candidate in \
        "${TUTORIAL_DIR}/diagnostics/nve/nve_prior_ablation_morse_dihedral/baseline/nve_certification_report.json" \
        "${TUTORIAL_DIR}/diagnostics/nve/nve_scaling_drift_recheck/nve_certification_report.json"; do
        if [[ -f "${candidate}" ]]; then FULL_REPORT="${candidate}"; break; fi
    done
fi
for p in "${TEST22_SUMMARY}" "${UNIFORM_MANIFEST}"; do
    [[ -f "${p}" ]] || { echo "[ERROR] Missing prerequisite artifact: ${p}" >&2; exit 1; }
done
[[ -n "${FULL_REPORT}" && -f "${FULL_REPORT}" ]] || {
    echo "[ERROR] No reusable historical full-PaiNN baseline found; set FULL_BASELINE_REPORT explicitly." >&2
    exit 1
}

IFS='|' read -r UNIFORM_PRIORS UNIFORM_CHECKPOINT UNIFORM_PRIORS_SHA UNIFORM_CHECKPOINT_SHA <<< "$(python3 - "${TEST22_SUMMARY}" "${UNIFORM_MANIFEST}" <<'PY'
import hashlib, json, os, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
manifest = json.load(open(sys.argv[2], encoding="utf-8"))
if summary.get("kind") != "tel22_morse_stabilizer_abc_10ps_fullgrid":
    raise SystemExit("unexpected test-22 summary kind")
if summary.get("interpretation") != "uniform_a0p85_supports_next_painn_closure":
    raise SystemExit("test-22 summary does not support the PaiNN closure")
if summary.get("recommended_numerical_stabilizer_for_next_step") != "C_uniform_a0p85":
    raise SystemExit("test-22 did not recommend C_uniform_a0p85")
if manifest.get("kind") != "tel22_morse_uniform_a0p85_inputs":
    raise SystemExit("unexpected uniform manifest kind")
if int(manifest.get("morse_count", -1)) != 180:
    raise SystemExit("uniform manifest does not contain 180 Morse contacts")
if abs(float(manifest.get("scaled_a", -1.0)) - 0.255) > 1e-15:
    raise SystemExit("uniform manifest does not encode a=0.255")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for key in ("priors", "checkpoint"):
    p = manifest.get(key)
    if not p or not os.path.isfile(p):
        raise SystemExit(f"missing uniform {key}: {p}")
    if manifest.get(key + "_sha256") != sha(p):
        raise SystemExit(f"uniform {key} hash mismatch")
print("|".join((manifest["priors"], manifest["checkpoint"], manifest["priors_sha256"], manifest["checkpoint_sha256"])))
PY
)"
echo "[UNIFORM PRIOR] validated test-22 C: 180 Morse a=0.255"

if ! python3 "${VALIDATE_FULL}" \
    --report "${FULL_REPORT}" \
    --tutorial "${TUTORIAL_DIR}" \
    --dts "${DT_ARGS[@]}" \
    --duration-ps "${NVE_DURATION_PS}" \
    --device "${NVE_DEVICE}" \
    --neighbor-search "${NVE_NEIGHBOR_SEARCH}" \
    --precision "${NVE_ML_PRECISION}"; then
    echo "[ERROR] Historical production full-PaiNN baseline cannot be safely reused." >&2
    echo "        Do not substitute a different baseline silently; set FULL_BASELINE_REPORT to an exact compatible report." >&2
    exit 1
fi

OUT_ABS="${TUTORIAL_DIR}/${NVE_OUTPUT_DIR}"
B_OUT="${OUT_ABS}/B_uniform_a0p85_old_painn"
C_OUT="${OUT_ABS}/C_uniform_a0p85_no_painn"
SUMMARY="${OUT_ABS}/painn_closure_uniform_a0p85_summary.json"

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT_ABS}"
elif [[ "${MODE}" == "normal" && -e "${OUT_ABS}" ]]; then
    echo "[ERROR] Output exists: ${OUT_ABS}; use --overwrite or --resume" >&2
    exit 1
fi
mkdir -p "${OUT_ABS}"

cat <<EOF_PLAN

[TEL22 PAINN CLOSURE -- UNIFORM MORSE a=0.255]
A production       : old priors a=0.300 + old PaiNN FP32       [REUSE]
B changed priors   : all 180 Morse a=0.255 + old PaiNN FP32   [NEW, diagnostic mismatch]
C changed priors   : all 180 Morse a=0.255 + PaiNN OFF        [NEW]
checkpoint state   : same physical/mechanical state; B/C use metadata-rebound test-22 checkpoint
Morse runtime      : marker + non-bonded switched, production-like
neighbor search    : ${NVE_NEIGHBOR_SEARCH}
device / precision : ${NVE_DEVICE} / ${NVE_ML_PRECISION}
dt grid [ps]       : ${NVE_DTS}
duration / dt      : ${NVE_DURATION_PS} ps
sampling           : every integration step
NOTE               : B cannot make accuracy claims because the old residual was trained against A priors.
EOF_PLAN

run_uniform() {
    local label="$1" out="$2" disable_ml="$3"
    echo
    echo "[VARIANT] ${label}"
    local cmd=(
        python3 "${CERTIFIER}"
        --pypresso "${PYPRESSO}"
        --model "${TUTORIAL_DIR}/tel22_model.pt"
        --config "${TUTORIAL_DIR}/tel22_training_config.json"
        --priors "${UNIFORM_PRIORS}"
        --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json"
        --dataset "${TUTORIAL_DIR}/tel22_dataset.bin"
        --checkpoint "${UNIFORM_CHECKPOINT}"
        --provenance-artifact "morse_uniform_abc_summary=${TEST22_SUMMARY}"
        --provenance-artifact "uniform_morse_inputs=${UNIFORM_MANIFEST}"
        --dts "${DT_ARGS[@]}"
        --duration-ps "${NVE_DURATION_PS}"
        --device "${NVE_DEVICE}"
        --ml-precision "${NVE_ML_PRECISION}"
        --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
        --morse-switch-mode switched
        --pair-specific-morse-runtime marker-nonbonded
        --output-dir "${out}"
        --slope-min "${NVE_SLOPE_MIN}"
        --slope-max "${NVE_SLOPE_MAX}"
        --min-r2 "${NVE_MIN_R2}"
        --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}"
    )
    if [[ "${disable_ml}" == "1" ]]; then
        cmd+=(--disable-ml)
    fi
    case "${MODE}" in
        dry-run) cmd+=(--dry-run --reuse-existing) ;;
        overwrite) cmd+=(--overwrite) ;;
        resume) cmd+=(--reuse-existing) ;;
    esac
    set +e
    "${cmd[@]}"
    local rc=$?
    set -e
    if [[ ${rc} -eq 2 ]]; then
        echo "[INFO] ${label}: historical strict gate failed; closure metrics were still recorded."
        return 2
    elif [[ ${rc} -ne 0 ]]; then
        echo "[ERROR] ${label}: operational failure rc=${rc}" >&2
        return "${rc}"
    fi
    return 0
}

completed_fail=0
set +e
run_uniform B_uniform_a0p85_old_painn "${B_OUT}" 0
rc=$?
set -e
if [[ ${rc} -eq 2 ]]; then completed_fail=1; elif [[ ${rc} -ne 0 ]]; then exit "${rc}"; fi

set +e
run_uniform C_uniform_a0p85_no_painn "${C_OUT}" 1
rc=$?
set -e
if [[ ${rc} -eq 2 ]]; then completed_fail=1; elif [[ ${rc} -ne 0 ]]; then exit "${rc}"; fi

if [[ "${MODE}" == "dry-run" ]]; then
    echo
    echo "[DRY-RUN] A baseline validated; planned B and C only."
    echo "[DRY-RUN] new MD budget at defaults: about 11,800 integration steps total."
    exit 0
fi

B_REPORT="${B_OUT}/nve_certification_report.json"
C_REPORT="${C_OUT}/nve_certification_report.json"
for p in "${B_REPORT}" "${C_REPORT}"; do
    [[ -f "${p}" ]] || { echo "[ERROR] Missing closure report: ${p}" >&2; exit 1; }
done

python3 "${SUMMARIZER}" \
    --A-report "${FULL_REPORT}" \
    --B-report "${B_REPORT}" \
    --C-report "${C_REPORT}" \
    --uniform-manifest "${UNIFORM_MANIFEST}" \
    --test22-summary "${TEST22_SUMMARY}" \
    --output "${SUMMARY}"

echo
if [[ ${completed_fail} -ne 0 ]]; then
    echo "[DONE] PaiNN closure completed; at least one new arm failed the historical strict gate (summary remains authoritative for attribution)."
else
    echo "[DONE] PaiNN closure completed; both new arms passed the historical strict gate."
fi
echo "[SUMMARY] ${SUMMARY}"
