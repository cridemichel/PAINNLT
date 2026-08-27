#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
PREPARE="${SCRIPT_DIR}/11_prepare_prior_ablation.py"
SUMMARIZER="${SCRIPT_DIR}/15_summarize_stock_espresso_coarse_5ps.py"

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
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_stock_espresso_coarse_5ps}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.97}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

usage() {
    cat <<'EOF_USAGE'
Usage:
  15_test_nve_stock_espresso_coarse_5ps.sh [--dry-run | --overwrite | --resume]

TEL22 stock-ESPResSo control:
  - derive priors with all pair-specific Morse entries removed
  - strip only technical Morse marker particles from a derived checkpoint
  - disable trained PaiNN with --disable-ml
  - retain production harmonic bonds and harmonic angles
  - retain production WCA, implemented by stock ESPResSo LennardJones
  - retain the production WCA topology exclusions
  - no conservative-spline priors and no dihedrals are present in TEL22 production
  - 5 ps per branch, dt = 0.002 0.003 0.004 0.005 ps

This is a diagnostic Hamiltonian ablation, not a reparameterized production model.
EOF_USAGE
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
for path in "${CERTIFIER}" "${PREPARE}" "${SUMMARIZER}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing diagnostic component: ${path}" >&2; exit 1; }
done
read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} == 4)) || { echo "[ERROR] This diagnostic requires exactly four dt values" >&2; exit 1; }

OUT_ABS="${TUTORIAL_DIR}/${NVE_OUTPUT_DIR}"
INPUT_DIR="${OUT_ABS}/inputs"
INPUT_MANIFEST="${INPUT_DIR}/ablation_inputs.json"
RUN_OUT="${OUT_ABS}/stock_only"
REPORT="${RUN_OUT}/nve_certification_report.json"
SUMMARY="${OUT_ABS}/stock_espresso_coarse_5ps_summary.json"
REFERENCE_SUMMARY="${TUTORIAL_DIR}/diagnostics/nve/nve_priors_only_coarse_5ps/priors_only_coarse_5ps_summary.json"

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
# Preparation is deterministic and cheap; regenerate derived inputs so their
# hashes always correspond to the current production artifacts.
PREP_ARGS+=(--overwrite)
"${PREP_ARGS[@]}"

IFS='|' read -r STOCK_PRIORS STOCK_CHECKPOINT <<< "$(python3 - "${INPUT_MANIFEST}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
v = m["variants"]["no_morse"]
if int(v["removed_morse_entries"]) <= 0 or int(v["remaining_morse_entries"]) != 0:
    raise SystemExit("no_morse input did not remove all Morse entries")
if int(v["remaining_morse_markers"]) != 0:
    raise SystemExit("no_morse checkpoint still contains Morse markers")
print(v["priors"] + "|" + v["checkpoint"])
PY
)"

# Fail closed unless this really is the intended stock-interaction control.
python3 - "${STOCK_PRIORS}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
bond_types = {str(x.get("type", "harmonic")).lower() for x in p.get("bonds", [])}
angle_types = {str(x.get("type", "harmonic")).lower() for x in p.get("angles", [])}
if bond_types != {"harmonic"}:
    raise SystemExit(f"stock control requires harmonic-only bonds after Morse removal; found {sorted(bond_types)}")
if angle_types != {"harmonic"}:
    raise SystemExit(f"stock control requires harmonic-only angles; found {sorted(angle_types)}")
if p.get("dihedrals", []):
    raise SystemExit("stock control expects production TEL22 dihedrals=[]")
if p.get("morse_type_pairs", []):
    raise SystemExit("stock control still contains morse_type_pairs")
if not p.get("wca_pairs", {}):
    raise SystemExit("stock control unexpectedly has no WCA pair table")
print(
    "[STOCK CONTROL INPUTS] "
    f"harmonic_bonds={len(p.get('bonds', []))} "
    f"harmonic_angles={len(p.get('angles', []))} "
    f"wca_pairs={len(p.get('wca_pairs', {}))} morse=0 dihedrals=0"
)
PY

cat <<EOF_PLAN

[TEL22 STOCK ESPRESSO 5 ps COARSE-DT NVE]
Hamiltonian          : stock ESPResSo interactions only; trained PaiNN OFF
bonded priors        : harmonic bonds + harmonic angles
nonbonded prior      : production WCA via stock ESPResSo LennardJones
custom Morse         : OFF; technical marker particles removed
conservative splines : absent
checkpoint state     : physical/runtime prefix identical to equilibrated.npz
ESPResSo precision   : native classical numerical path
ML precision         : not applicable (PaiNN disabled)
device               : ${NVE_DEVICE}
neighbor search      : ${NVE_NEIGHBOR_SEARCH}
dt grid [ps]         : ${NVE_DTS}
duration / dt        : ${NVE_DURATION_PS} ps
sampling             : every integration step
NOTE                 : diagnostic control; removing Morse changes the Hamiltonian.
EOF_PLAN

cmd=(
    python3 "${CERTIFIER}"
    --pypresso "${PYPRESSO}"
    --model "${TUTORIAL_DIR}/tel22_model.pt"
    --config "${TUTORIAL_DIR}/tel22_training_config.json"
    --priors "${STOCK_PRIORS}"
    --rb-info "${TUTORIAL_DIR}/rigid_bodies_info.json"
    --dataset "${TUTORIAL_DIR}/tel22_dataset.bin"
    --checkpoint "${STOCK_CHECKPOINT}"
    --disable-ml
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
    echo "[ERROR] stock-ESPResSo certifier failed operationally with exit code ${rc}" >&2
    exit "${rc}"
fi
if [[ "${MODE}" == "dry-run" ]]; then
    echo "[DRY-RUN] output: ${RUN_OUT}"
    exit 0
fi

[[ -f "${REPORT}" ]] || { echo "[ERROR] Missing certification report: ${REPORT}" >&2; exit 1; }
SUMMARY_CMD=(
    python3 "${SUMMARIZER}"
    --report "${REPORT}"
    --inputs "${INPUT_MANIFEST}"
    --output "${SUMMARY}"
)
if [[ -f "${REFERENCE_SUMMARY}" ]]; then
    SUMMARY_CMD+=(--reference-priors-only "${REFERENCE_SUMMARY}")
fi
"${SUMMARY_CMD[@]}"

if [[ ${rc} -eq 2 ]]; then
    echo "[DONE] Stock-ESPResSo control completed; original broad certification gate reported FAIL."
else
    echo "[DONE] Stock-ESPResSo control completed; original broad certification gate passed."
fi
echo "[SUMMARY] ${SUMMARY}"
