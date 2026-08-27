#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
PREP="${SCRIPT_DIR}/24_prepare_full_pipeline_uniform_morse.py"
SUMMARY_TOOL="${SCRIPT_DIR}/24_summarize_full_pipeline_uniform_morse.py"
BUILDER="${FRAMEWORK_ROOT}/preprocessing/build_cg_dataset.py"
TRAINER="${TRAINER:-${FRAMEWORK_ROOT}/training/build/train_painn}"
MANIFEST_TOOL="${FRAMEWORK_ROOT}/training/create_model_manifest.py"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
EQUILIBRATOR="${FRAMEWORK_ROOT}/simulation/equilibrate.py"
RUNNER="${FRAMEWORK_ROOT}/simulation/run_cg_md.py"

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYPRESSO="${PYPRESSO:-${DEFAULT_PYPRESSO}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUT_REL="${FULL_PIPELINE_OUT:-diagnostics/full_pipeline/tel22_uniform_morse_a0p255}"
OUT="${TUTORIAL_DIR}/${OUT_REL}"
INPUTS="${OUT}/inputs"
ART="${OUT}/artifacts"
TRAIN_DIR="${OUT}/training"
EQUIL_DIR="${OUT}/equilibration"
NVT_DIR="${OUT}/nvt"
NVE32_DIR="${OUT}/nve_fp32"
NVE64_DIR="${OUT}/nve_fp64"

AA_TOPOLOGY="${AA_TOPOLOGY:-md.gro}"
AA_TRAJECTORY="${AA_TRAJECTORY:-md_whole.trr}"
BASE_TOPOLOGY="${BASE_TOPOLOGY:-tel22_topology.json}"
BASE_PRIORS="${BASE_PRIORS:-cg_priors.json}"
BASE_RB_INFO="${BASE_RB_INFO:-rigid_bodies_info.json}"
TRAIN_CONFIG_SRC="${TRAIN_CONFIG_SRC:-tel22_training_config.json}"
NEW_A="${NEW_A:-0.255}"
PIPELINE_DEVICE="${PIPELINE_DEVICE:-cpu}"
PIPELINE_NEIGHBOR_SEARCH="${PIPELINE_NEIGHBOR_SEARCH:-link-cell}"
PIPELINE_KT="${PIPELINE_KT:-2.49}"
PIPELINE_VELOCITY_SEED="${PIPELINE_VELOCITY_SEED:-314159}"
PIPELINE_NVT_STEPS="${PIPELINE_NVT_STEPS:-20000}"
PIPELINE_NVT_DT="${PIPELINE_NVT_DT:-0.001}"
PIPELINE_NVT_LOG_INTERVAL="${PIPELINE_NVT_LOG_INTERVAL:-10}"
PIPELINE_RUN_FP64="${PIPELINE_RUN_FP64:-1}"
NVE_DTS="${NVE_DTS:-0.001 0.0015 0.002 0.003 0.004 0.005}"
NVE_DURATION_PS="${NVE_DURATION_PS:-2.0}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.97}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

ABC_SUMMARY="${ABC_SUMMARY:-diagnostics/nve/nve_morse_uniform_abc_10ps_fullgrid/morse_uniform_abc_summary.json}"
CLOSURE_B="${CLOSURE_B:-diagnostics/nve/nve_painn_closure_uniform_a0p85_2ps/B_uniform_a0p85_old_painn/nve_certification_report.json}"
CLOSURE_C="${CLOSURE_C:-diagnostics/nve/nve_painn_closure_uniform_a0p85_2ps/C_uniform_a0p85_no_painn/nve_certification_report.json}"
OLD_FULL_REPORT="${OLD_FULL_REPORT:-}"

usage() {
cat <<'USAGE'
Usage:
  24_run_full_pipeline_uniform_morse.sh [--dry-run | --overwrite | --resume]

Isolated TEL22 full rebuild using a uniform empirical Morse stabilizer a=0.255:
  exact production priors except Morse a
  -> rebuild residual dataset from the same AA trajectory
  -> train a fresh PaiNN from scratch
  -> fresh full-Hamiltonian equilibration
  -> production-like NVT stability run
  -> six-dt FP32 NVE certification
  -> six-dt FP64 precision closure (default; PIPELINE_RUN_FP64=0 disables it)

Production files in tutorials/tel22 are never overwritten.

Useful overrides:
  FULL_PIPELINE_OUT=diagnostics/full_pipeline/my_candidate
  PIPELINE_NVT_STEPS=20000
  PIPELINE_RUN_FP64=1
  NVE_DURATION_PS=2.0
  PYPRESSO=/path/to/pypresso
  TRAINER=/path/to/train_painn
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
if (($# != 0)); then echo "[ERROR] Unexpected arguments: $*" >&2; exit 2; fi

cd "${TUTORIAL_DIR}"
for p in "${AA_TOPOLOGY}" "${AA_TRAJECTORY}" "${BASE_TOPOLOGY}" "${BASE_PRIORS}" "${BASE_RB_INFO}" "${TRAIN_CONFIG_SRC}" "${ABC_SUMMARY}" "${CLOSURE_B}" "${CLOSURE_C}"; do
    [[ -f "${p}" ]] || { echo "[ERROR] Missing required input: ${p}" >&2; exit 1; }
done
for p in "${PREP}" "${SUMMARY_TOOL}" "${BUILDER}" "${MANIFEST_TOOL}" "${CERTIFIER}" "${EQUILIBRATOR}" "${RUNNER}"; do
    [[ -f "${p}" ]] || { echo "[ERROR] Missing framework component: ${p}" >&2; exit 1; }
done
[[ -x "${TRAINER}" ]] || { echo "[ERROR] train_painn not executable: ${TRAINER}" >&2; exit 1; }
if [[ "${MODE}" != "dry-run" ]]; then
    command -v "${PYPRESSO}" >/dev/null 2>&1 || [[ -x "${PYPRESSO}" ]] || { echo "[ERROR] pypresso not found: ${PYPRESSO}" >&2; exit 1; }
fi
read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} == 6)) || { echo "[ERROR] Full validation requires exactly six dt values" >&2; exit 1; }

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT}"
elif [[ "${MODE}" == "normal" && -e "${OUT}" ]]; then
    echo "[ERROR] Candidate output already exists: ${OUT}" >&2
    echo "        Use --overwrite for a clean rebuild or --resume to reuse completed stages." >&2
    exit 1
fi
mkdir -p "${INPUTS}" "${ART}" "${TRAIN_DIR}" "${EQUIL_DIR}" "${NVT_DIR}"

# Stage 0: deterministic candidate topology/priors; production remains untouched.
"${PYTHON_BIN}" "${PREP}" \
    --topology "${BASE_TOPOLOGY}" \
    --priors "${BASE_PRIORS}" \
    --output-dir "${INPUTS}" \
    --new-a "${NEW_A}" \
    --abc-summary "${ABC_SUMMARY}" \
    --closure-b-report "${CLOSURE_B}" \
    --closure-c-report "${CLOSURE_C}"

CAND_TOPOLOGY="${INPUTS}/tel22_topology_uniform_a0p255.json"
CAND_PRIORS="${INPUTS}/cg_priors.json"
INPUT_MANIFEST="${INPUTS}/full_pipeline_input_manifest.json"
CAND_CONFIG="${ART}/tel22_training_config.json"
DATASET="${ART}/tel22_dataset.bin"
RB_INFO="${ART}/rigid_bodies_info.json"
BUILD_MANIFEST="${ART}/residual_build_manifest.json"
MODEL="${ART}/tel22_model.pt"
MODEL_MANIFEST="${MODEL}.manifest.json"
CHECKPOINT="${ART}/equilibrated.npz"
TRAIN_LOG="${TRAIN_DIR}/cg_training_log.csv"
NVT_ENERGY="${NVT_DIR}/energy.csv"
NVT_CHECKPOINT="${NVT_DIR}/final_state.npz"
FP32_REPORT="${NVE32_DIR}/nve_certification_report.json"
FP64_REPORT="${NVE64_DIR}/nve_certification_report.json"
FINAL_SUMMARY="${OUT}/full_pipeline_summary.json"
cp "${TRAIN_CONFIG_SRC}" "${CAND_CONFIG}"

cat <<EOF_PLAN

[TEL22 FULL REBUILD -- UNIFORM MORSE a=${NEW_A}]
source AA          : ${AA_TOPOLOGY} + ${AA_TRAJECTORY}
source priors      : ${BASE_PRIORS}
candidate change  : all 180 explicit Morse a=0.300 -> ${NEW_A}; all other prior fields preserved
residual dataset  : rebuilt with candidate priors (not post-hoc modified)
PaiNN              : fresh training from scratch, original training config
checkpoint         : fresh full-Hamiltonian equilibration
NVT stability      : ${PIPELINE_NVT_STEPS} steps at dt=${PIPELINE_NVT_DT} ps, FP32
NVE FP32           : ${NVE_DTS}, ${NVE_DURATION_PS} ps per dt
NVE FP64 closure   : ${PIPELINE_RUN_FP64}
output             : ${OUT_REL}
production files   : untouched
EOF_PLAN

if [[ "${MODE}" == "dry-run" ]]; then
    echo "[DRY-RUN] Candidate topology/priors and provenance prepared; no dataset build, training, ESPResSo MD, or NVE executed."
    exit 0
fi

hash_file() { shasum -a 256 "$1" | awk '{print $1}'; }

# Stage 1: residual dataset rebuild.  On resume, validate the recorded hashes.
if [[ "${MODE}" == "resume" && -f "${BUILD_MANIFEST}" && -f "${DATASET}" && -f "${RB_INFO}" ]]; then
    "${PYTHON_BIN}" - "${BUILD_MANIFEST}" "${INPUT_MANIFEST}" "${DATASET}" "${RB_INFO}" "${CAND_PRIORS}" <<'PY'
import hashlib,json,sys
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
m=json.load(open(sys.argv[1])); inp=json.load(open(sys.argv[2]))
if m.get('kind')!='tel22_full_pipeline_residual_build': raise SystemExit('bad build manifest kind')
checks={'dataset_sha256':sys.argv[3], 'rb_info_sha256':sys.argv[4], 'candidate_priors_sha256':sys.argv[5]}
for key,path in checks.items():
    if m.get(key)!=sha(path): raise SystemExit(f'resume hash mismatch: {key}')
if m.get('input_manifest_sha256')!=sha(sys.argv[2]): raise SystemExit('resume input manifest hash mismatch')
if m.get('candidate_priors_sha256')!=inp.get('candidate_priors_sha256'): raise SystemExit('candidate prior provenance mismatch')
print('[REUSE] residual dataset + rb_info validated against build manifest')
PY
else
    rm -f "${DATASET}" "${RB_INFO}" "${BUILD_MANIFEST}"
    echo "[STAGE 1] Rebuilding residual dataset against uniform-a priors..."
    "${PYTHON_BIN}" "${BUILDER}" \
        --topology "${AA_TOPOLOGY}" \
        --trajectory "${AA_TRAJECTORY}" \
        --config "${CAND_TOPOLOGY}" \
        --priors "${CAND_PRIORS}" \
        --output "${DATASET}" \
        --rb-info-output "${RB_INFO}" \
        2>&1 | tee "${OUT}/dataset_build.log"

    "${PYTHON_BIN}" - "${INPUT_MANIFEST}" "${DATASET}" "${RB_INFO}" "${BASE_RB_INFO}" "${CAND_PRIORS}" "${CAND_TOPOLOGY}" "${AA_TOPOLOGY}" "${AA_TRAJECTORY}" "${BUILDER}" "${BUILD_MANIFEST}" <<'PY'
import hashlib,json,sys
from pathlib import Path
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
inp,dataset,rb,base_rb,pri,top,aa_top,aa_traj,builder,out=sys.argv[1:]
if json.load(open(rb)) != json.load(open(base_rb)):
    raise SystemExit('rigid_bodies_info changed semantically although only Morse a was changed')
i=json.load(open(inp))
if sha(pri)!=i['candidate_priors_sha256'] or sha(top)!=i['candidate_topology_sha256']:
    raise SystemExit('candidate inputs changed after preparation')
m={
 'schema_version':1,
 'kind':'tel22_full_pipeline_residual_build',
 'input_manifest':str(Path(inp).resolve()), 'input_manifest_sha256':sha(inp),
 'dataset':str(Path(dataset).resolve()), 'dataset_sha256':sha(dataset),
 'rb_info':str(Path(rb).resolve()), 'rb_info_sha256':sha(rb),
 'baseline_rb_info_sha256':sha(base_rb), 'rb_info_semantically_equal_to_production':True,
 'candidate_priors':str(Path(pri).resolve()), 'candidate_priors_sha256':sha(pri),
 'candidate_topology':str(Path(top).resolve()), 'candidate_topology_sha256':sha(top),
 'aa_topology':str(Path(aa_top).resolve()), 'aa_topology_sha256':sha(aa_top),
 'aa_trajectory':str(Path(aa_traj).resolve()), 'aa_trajectory_sha256':sha(aa_traj),
 'builder':str(Path(builder).resolve()), 'builder_sha256':sha(builder),
 'residual_definition':'reference CG force/torque minus exact candidate priors; all 180 Morse use a=0.255',
}
Path(out).write_text(json.dumps(m,indent=2)+'\n')
print('[PASS] residual build provenance recorded:',out)
PY
fi

# Stage 2: fresh PaiNN training. Never resume the old production model.
training_valid=0
if [[ "${MODE}" == "resume" && -f "${MODEL}" && -f "${MODEL_MANIFEST}" && -f "${TRAIN_LOG}" ]]; then
    if "${PYTHON_BIN}" - "${MODEL}" "${MODEL_MANIFEST}" "${DATASET}" "${CAND_CONFIG}" <<'PY'
import hashlib,json,sys
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
model,man,dataset,config=sys.argv[1:]
m=json.load(open(man))
checks=(m.get('model_sha256')==sha(model),m.get('dataset_sha256')==sha(dataset),m.get('config_sha256')==sha(config))
raise SystemExit(0 if all(checks) else 1)
PY
    then training_valid=1; echo "[REUSE] trained PaiNN + finalized manifest validated"; fi
fi
if [[ ${training_valid} -eq 0 ]]; then
    rm -f "${MODEL}" "${MODEL_MANIFEST}" "${TRAIN_LOG}"
    echo "[STAGE 2] Training fresh PaiNN residual model..."
    (
        cd "${TRAIN_DIR}"
        "${TRAINER}" "${DATASET}" "${MODEL}" "${CAND_CONFIG}"
    )
    [[ -f "${TRAIN_DIR}/cg_training_log.csv" ]] || { echo "[ERROR] trainer did not write cg_training_log.csv" >&2; exit 1; }
    "${PYTHON_BIN}" "${MANIFEST_TOOL}" --model "${MODEL}" --dataset "${DATASET}" --config "${CAND_CONFIG}"
fi

# Stage 3: fresh full-Hamiltonian equilibration.
if [[ "${MODE}" == "resume" && -f "${CHECKPOINT}" ]]; then
    echo "[REUSE] candidate equilibrated checkpoint exists; NVE preflight will revalidate its hashes"
else
    rm -f "${CHECKPOINT}"
    echo "[STAGE 3] Fresh full-Hamiltonian equilibration..."
    "${PYPRESSO}" "${EQUILIBRATOR}" \
        --model "${MODEL}" \
        --config "${CAND_CONFIG}" \
        --priors "${CAND_PRIORS}" \
        --rb_info "${RB_INFO}" \
        --dataset "${DATASET}" \
        --out_checkpoint "${CHECKPOINT}" \
        --device "${PIPELINE_DEVICE}" \
        --neighbor_search "${PIPELINE_NEIGHBOR_SEARCH}" \
        --kT "${PIPELINE_KT}" \
        --velocity_seed "${PIPELINE_VELOCITY_SEED}" \
        2>&1 | tee "${EQUIL_DIR}/equilibrate.log"
fi
[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] missing equilibrated checkpoint" >&2; exit 1; }

# Stage 4: production-like FP32 NVT stability smoke (energy only; no huge VTF).
nvt_valid=0
if [[ "${MODE}" == "resume" && -f "${NVT_ENERGY}" && -f "${NVT_CHECKPOINT}" ]]; then
    if "${PYTHON_BIN}" - "${NVT_ENERGY}" "${PIPELINE_NVT_STEPS}" "${PIPELINE_NVT_DT}" <<'PY'
import csv,math,sys
p,steps,dt=sys.argv[1],int(sys.argv[2]),float(sys.argv[3])
rows=list(csv.DictReader(open(p)))
if not rows: raise SystemExit(1)
last=float(rows[-1]['Time_ps']); expected=steps*dt
raise SystemExit(0 if math.isclose(last,expected,rel_tol=0,abs_tol=max(1e-9,dt*1e-6)) else 1)
PY
    then nvt_valid=1; echo "[REUSE] completed NVT stability run"; fi
fi
if [[ ${nvt_valid} -eq 0 ]]; then
    rm -f "${NVT_ENERGY}" "${NVT_CHECKPOINT}"
    echo "[STAGE 4] Production-like FP32 NVT stability run..."
    "${PYPRESSO}" "${RUNNER}" \
        --model "${MODEL}" \
        --config "${CAND_CONFIG}" \
        --priors "${CAND_PRIORS}" \
        --rb_info "${RB_INFO}" \
        --dataset "${DATASET}" \
        --checkpoint "${CHECKPOINT}" \
        --steps "${PIPELINE_NVT_STEPS}" \
        --dt "${PIPELINE_NVT_DT}" \
        --kT "${PIPELINE_KT}" \
        --device "${PIPELINE_DEVICE}" \
        --ml_precision float32 \
        --neighbor_search "${PIPELINE_NEIGHBOR_SEARCH}" \
        --energy_file "${NVT_ENERGY}" \
        --out_checkpoint "${NVT_CHECKPOINT}" \
        --log_interval "${PIPELINE_NVT_LOG_INTERVAL}" \
        --no_vtf \
        2>&1 | tee "${NVT_DIR}/run.log"
fi

run_nve() {
    local precision="$1" outdir="$2"
    local cmd=(
        "${PYTHON_BIN}" "${CERTIFIER}"
        --pypresso "${PYPRESSO}"
        --model "${MODEL}"
        --config "${CAND_CONFIG}"
        --priors "${CAND_PRIORS}"
        --rb-info "${RB_INFO}"
        --dataset "${DATASET}"
        --checkpoint "${CHECKPOINT}"
        --provenance-artifact "full_pipeline_inputs=${INPUT_MANIFEST}"
        --provenance-artifact "residual_build=${BUILD_MANIFEST}"
        --dts "${DT_ARGS[@]}"
        --duration-ps "${NVE_DURATION_PS}"
        --device cpu
        --ml-precision "${precision}"
        --neighbor-search "${PIPELINE_NEIGHBOR_SEARCH}"
        --morse-switch-mode switched
        --pair-specific-morse-runtime marker-nonbonded
        --output-dir "${outdir}"
        --slope-min "${NVE_SLOPE_MIN}"
        --slope-max "${NVE_SLOPE_MAX}"
        --min-r2 "${NVE_MIN_R2}"
        --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}"
    )
    case "${MODE}" in
        overwrite) cmd+=(--overwrite) ;;
        resume) cmd+=(--reuse-existing) ;;
    esac
    set +e
    "${cmd[@]}"
    local rc=$?
    set -e
    if [[ ${rc} -eq 2 ]]; then
        echo "[INFO] ${precision} NVE completed but historical strict gate failed; retaining report for analysis."
        return 0
    fi
    return "${rc}"
}

echo "[STAGE 5] Full TEL22 FP32 NVE scaling..."
run_nve float32 "${NVE32_DIR}"
[[ -f "${FP32_REPORT}" ]] || { echo "[ERROR] missing FP32 NVE report" >&2; exit 1; }

if [[ "${PIPELINE_RUN_FP64}" == "1" ]]; then
    echo "[STAGE 6] Full TEL22 FP64 precision closure..."
    run_nve float64 "${NVE64_DIR}"
    [[ -f "${FP64_REPORT}" ]] || { echo "[ERROR] missing FP64 NVE report" >&2; exit 1; }
fi

if [[ -z "${OLD_FULL_REPORT}" ]]; then
    for c in \
        "diagnostics/nve/nve_prior_ablation_morse_dihedral/baseline/nve_certification_report.json" \
        "diagnostics/nve/nve_scaling_drift_recheck/nve_certification_report.json"; do
        if [[ -f "${c}" ]]; then OLD_FULL_REPORT="${c}"; break; fi
    done
fi

summary_cmd=(
    "${PYTHON_BIN}" "${SUMMARY_TOOL}"
    --input-manifest "${INPUT_MANIFEST}"
    --build-manifest "${BUILD_MANIFEST}"
    --training-log "${TRAIN_LOG}"
    --model-manifest "${MODEL_MANIFEST}"
    --checkpoint "${CHECKPOINT}"
    --nvt-energy "${NVT_ENERGY}"
    --fp32-report "${FP32_REPORT}"
    --output "${FINAL_SUMMARY}"
)
if [[ "${PIPELINE_RUN_FP64}" == "1" ]]; then summary_cmd+=(--fp64-report "${FP64_REPORT}"); fi
if [[ -n "${OLD_FULL_REPORT}" && -f "${OLD_FULL_REPORT}" ]]; then summary_cmd+=(--old-full-report "${OLD_FULL_REPORT}"); fi
"${summary_cmd[@]}"

echo
printf '[DONE] isolated full TEL22 candidate completed\n'
printf '[SUMMARY] %s\n' "${FINAL_SUMMARY}"
printf '[NOTE] production tel22_dataset.bin/cg_priors.json/tel22_model.pt/equilibrated.npz were not overwritten.\n'
